#!/usr/bin/env python3
"""Test whether existing enhancement renders are explainable by a short FIR.

The output is a descriptive system-identification diagnostic, not a dereverb
quality score. It uses only the existing local listening pack and one public,
paired dry-speech control for a synthetic early-reflection calibration.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy import ndimage, signal
from scipy.sparse.linalg import LinearOperator, cg
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[3]
PACK = ROOT / "results/user-recording-test-2026-10-09/model-family-comparison-02"
RAW = ROOT / "results/user-recording-test-2026-10-09/01_input_mono_48k.wav"
CLARITY = ROOT / ".tools/clarity-multivoice-2026/matrix/stems/T005_CH2_00450-dry-clean.wav"
OUT = PACK
FS = 16_000
FIR_MS = 50
FIR_LEN = round(FIR_MS * FS / 1000) + 1
FRAME = 640
HOP = 160
TRANSFER_FRAME = 1_600  # 100 ms: longer than the calibrated early echoes
TRANSFER_NFFT = 4_096
MAX_LAG_MS = 100
FILES = {
    "Cap60": "CAP60.wav",
    "Adobe V2": "ADOBE.wav",
    "StuPASE": "STUPASE.wav",
    "ROSE raw": "ROSE_RAW.wav",
    "ROSE after Cap60": "ROSE_CAP60.wav",
    "WPE": "WPE.wav",
    "DPDFNet2": "DPDFNET2.wav",
}


def read16(path: Path) -> np.ndarray:
    x, sr = sf.read(path, dtype="float64")
    if x.ndim == 2:
        x = x.mean(axis=1)
    if sr == 48_000:
        return resample_poly(x, 1, 3)
    if sr == FS:
        return x
    raise ValueError(f"Unsupported sample rate {sr}: {path}")


def input_events(raw: np.ndarray) -> list[tuple[int, int]]:
    """Common speech event mask from raw input only, at 16 kHz."""
    frame, hop = 320, 160  # 20 ms / 10 ms
    starts = np.arange(max(0, (len(raw) - frame) // hop + 1)) * hop
    rms = np.asarray([np.sqrt(np.mean(raw[s:s + frame] ** 2)) for s in starts])
    active = rms > 0.035 * np.max(np.abs(raw))
    active = ndimage.binary_closing(active, structure=np.ones(9))
    edges = np.diff(np.r_[False, active, False].astype(np.int8))
    first, last = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return [(int(s * hop), min(len(raw), int((e - 1) * hop + frame)))
            for s, e in zip(first, last) if e - s >= 10]


def sample_mask(n: int, events: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    for start, end in events:
        mask[max(0, start):min(n, end)] = True
    return mask


def voiced_starts(raw: np.ndarray) -> np.ndarray:
    starts = np.arange(max(0, (len(raw) - FRAME) // HOP + 1)) * HOP
    threshold = 0.035 * float(np.max(np.abs(raw)))
    win = np.hanning(FRAME)
    keep = []
    for s in starts:
        x = np.asarray(raw[s:s + FRAME], dtype=np.float64)
        x -= x.mean()
        rms = np.sqrt(np.mean(x * x))
        ac = np.correlate(x * win, x * win, mode="full")[FRAME - 1:]
        ac /= max(float(ac[0]), 1e-12)
        keep.append(rms > threshold and np.max(ac[53:229]) >= 0.35)
    return starts[np.asarray(keep, dtype=bool)]


def align_to_input(x: np.ndarray, y: np.ndarray, active: np.ndarray) -> tuple[np.ndarray, float]:
    """Estimate a global lag on input-active samples and apply fractional shift."""
    n = min(len(x), len(y), len(active))
    x, y, mask = x[:n], y[:n], active[:n]
    corr = signal.correlate(y * mask, x * mask, mode="full", method="fft")
    center = n - 1
    limit = round(MAX_LAG_MS * FS / 1000)
    local = corr[center - limit:center + limit + 1]
    peak = int(np.argmax(np.abs(local)))
    lag = float(peak - limit)
    if 0 < peak < len(local) - 1:
        a, b, c = np.abs(local[peak - 1:peak + 2])
        denom = a - 2 * b + c
        if abs(denom) > 1e-15:
            lag += float(0.5 * (a - c) / denom)
    aligned = np.fft.ifft(ndimage.fourier_shift(np.fft.fft(y), -lag)).real
    return aligned, lag


def transfer_statistics(x: np.ndarray, y: np.ndarray,
                        starts: np.ndarray) -> dict[str, object]:
    starts = starts[starts + TRANSFER_FRAME <= min(len(x), len(y))]
    win = np.hanning(TRANSFER_FRAME)
    X = np.asarray([np.fft.rfft(x[s:s + TRANSFER_FRAME] * win, n=TRANSFER_NFFT)
                    for s in starts])
    Y = np.asarray([np.fft.rfft(y[s:s + TRANSFER_FRAME] * win, n=TRANSFER_NFFT)
                    for s in starts])
    frequencies = np.fft.rfftfreq(TRANSFER_NFFT, 1 / FS)
    pxx = np.mean(np.abs(X) ** 2, axis=0)
    cross = np.mean(np.conj(X) * Y, axis=0)
    pyy = np.mean(np.abs(Y) ** 2, axis=0)
    hmean = cross / np.maximum(pxx, 1e-18)
    denom = np.maximum(pxx * pyy, 1e-18)
    coherence = np.abs(cross) ** 2 / denom
    # Exclude bins where the input has negligible energy in the band.
    band_stats = {}
    for label, low, high in (("300-3000", 300, 3000), ("3000-7900", 3000, 7900)):
        idx = (frequencies >= low) & (frequencies <= high)
        idx &= pxx >= max(float(np.max(pxx[(frequencies >= low) & (frequencies <= high)])) * 1e-4, 1e-18)
        band_stats[label] = float(np.median(coherence[idx])) if np.any(idx) else None

    ratio = Y / np.where(np.abs(X) > 1e-9, X, np.nan)
    logmag = 20 * np.log10(np.maximum(np.abs(ratio), 1e-9))
    phase = np.angle(ratio)
    variability = {}
    for label, low, high in (("300-3000", 300, 3000), ("3000-7900", 3000, 7900)):
        idx = (frequencies >= low) & (frequencies <= high)
        power_cut = max(float(np.max(pxx[(frequencies >= low) & (frequencies <= high)])) * 1e-4, 1e-18)
        idx &= pxx >= power_cut
        mag_mad = np.nanmedian(np.abs(logmag[:, idx] - np.nanmedian(logmag[:, idx], axis=0)), axis=0)
        phase_concentration = np.abs(np.nanmean(np.exp(1j * phase[:, idx]), axis=0))
        variability[label] = {
            "median_magnitude_mad_db": float(np.nanmedian(mag_mad)),
            "median_phase_dispersion": float(np.nanmedian(1 - phase_concentration)),
            "n_bins": int(np.sum(idx)),
        }

    # Early-reflection proxy from the real cepstrum of the regularized mean
    # transfer magnitude; report energy in quefrencies corresponding to 2–50 ms.
    h_db = 20 * np.log10(np.maximum(np.abs(hmean), 1e-8))
    cepstrum = np.fft.irfft(h_db, n=TRANSFER_NFFT)
    early = cepstrum[round(.002 * FS):round(.050 * FS) + 1]
    total = cepstrum[1:1024]
    early_ratio = float(np.sum(early * early) / max(float(np.sum(total * total)), 1e-18))
    return {
        "n_voiced_frames": int(len(starts)),
        "median_complex_coherence": band_stats,
        "transfer_variability": variability,
        "early_cepstral_energy_fraction_2_50ms": early_ratio,
    }


def apply_fir(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    return signal.fftconvolve(x, h, mode="full")[:len(x)]


def adjoint_fir(x: np.ndarray, residual: np.ndarray, length: int) -> np.ndarray:
    corr = signal.correlate(x, residual, mode="full", method="fft")
    n = len(x)
    return corr[n - 1 - np.arange(length)]


def fit_short_fir(x: np.ndarray, y: np.ndarray, train_mask: np.ndarray,
                  length: int = FIR_LEN) -> tuple[np.ndarray, int]:
    """Conjugate-gradient least squares for masked causal FIR identification."""
    n = min(len(x), len(y), len(train_mask))
    x, y, mask = x[:n], y[:n], train_mask[:n].astype(np.float64)
    target = y * mask
    rhs = adjoint_fir(x, target, length)
    ridge = max(float(np.sum(x * x * mask)) * 1e-4, 1e-10)
    diagonal = np.empty(length, dtype=np.float64)
    for lag in range(length):
        diagonal[lag] = float(np.sum(x[:n - lag] ** 2 * mask[lag:])) if lag < n else 0.0
    diagonal += ridge

    def normal(v: np.ndarray) -> np.ndarray:
        pred = apply_fir(x, v)
        return adjoint_fir(x, pred * mask, length) + ridge * v

    operator = LinearOperator((length, length), matvec=normal, dtype=np.float64)
    preconditioner = LinearOperator((length, length), matvec=lambda v: v / diagonal,
                                    dtype=np.float64)
    h, info = cg(operator, rhs, M=preconditioner, rtol=1e-3, atol=0.0, maxiter=180)
    return h, int(info)


def cv_fir_metrics(x: np.ndarray, y: np.ndarray,
                   events: list[tuple[int, int]], folds: int = 3) -> list[dict[str, object]]:
    rows = []
    for fold in range(folds):
        train_events = [event for i, event in enumerate(events) if i % folds != fold]
        test_events = [event for i, event in enumerate(events) if i % folds == fold]
        train_mask = sample_mask(len(x), train_events)
        test_mask = sample_mask(len(x), test_events)
        if np.sum(train_mask) < FIR_LEN * 4 or np.sum(test_mask) < FRAME:
            rows.append({"fold": fold, "status": "insufficient_event_samples"})
            continue
        h, info = fit_short_fir(x, y, train_mask)
        predicted = apply_fir(x, h)
        residual = y[test_mask] - predicted[test_mask]
        observed = y[test_mask]
        sse = float(np.sum(residual * residual))
        total = float(np.sum((observed - np.mean(observed)) ** 2))
        base = float(np.sum((observed - x[test_mask]) ** 2))
        rows.append({
            "fold": fold,
            "status": "ok" if info == 0 else f"cg_info_{info}",
            "train_events": len(train_events),
            "test_events": len(test_events),
            "n_test_samples": int(np.sum(test_mask)),
            "heldout_r2": float(1 - sse / max(total, 1e-18)),
            "heldout_error_db_vs_output": float(10 * np.log10(max(sse, 1e-18) / max(float(np.sum(observed ** 2)), 1e-18))),
            "improvement_db_vs_identity": float(10 * np.log10(max(base, 1e-18) / max(sse, 1e-18))),
            "cg_info": info,
        })
    return rows


def synthetic_early_echo_control(x: np.ndarray) -> dict[str, object]:
    true_h = np.zeros(round(.050 * FS) + 1)
    true_h[0] = 1.0
    true_h[round(.008 * FS)] = 10 ** (-8 / 20)
    true_h[round(.023 * FS)] = 10 ** (-14 / 20)
    y = apply_fir(x, true_h)
    events = [(0, len(x) // 3), (len(x) // 3, 2 * len(x) // 3),
              (2 * len(x) // 3, len(x))]
    rows = cv_fir_metrics(x, y, events, folds=3)
    active = sample_mask(len(x), events)
    aligned, lag = align_to_input(x, y, active)
    stats = transfer_statistics(x, aligned, voiced_starts(x))
    return {"known_taps": [
        {"delay_ms": 0, "gain_db": 0},
        {"delay_ms": 8, "gain_db": -8},
        {"delay_ms": 23, "gain_db": -14}],
        "estimated_lag_ms": lag * 1000 / FS,
        "transfer_statistics": stats,
        "cross_validation": rows,
        "expected": "high short-FIR held-out fit and high input/output coherence; synthetic reflection has no late tail"}


def main() -> None:
    audio = {name: read16(PACK / filename) for name, filename in FILES.items()}
    raw = read16(RAW)
    n = min(len(raw), *(len(x) for x in audio.values()))
    raw = raw[:n]
    audio = {name: x[:n] for name, x in audio.items()}
    events = input_events(raw)
    activity = sample_mask(n, events)
    voiced = voiced_starts(raw)
    x = audio["Cap60"]
    rows = []
    all_cv = {}
    for name, y in audio.items():
        aligned, lag = align_to_input(x, y, activity)
        row = {"candidate": name, "estimated_global_lag_samples": lag,
               "estimated_global_lag_ms": lag * 1000 / FS}
        row.update(transfer_statistics(x, aligned, voiced))
        cv = cv_fir_metrics(x, aligned, events)
        all_cv[name] = cv
        valid = [r for r in cv if r.get("status") == "ok"]
        row["short_fir_cv"] = {
            "length_ms": FIR_MS,
            "folds": cv,
            "median_heldout_r2": float(np.median([r["heldout_r2"] for r in valid])) if valid else None,
            "median_heldout_error_db_vs_output": float(np.median([r["heldout_error_db_vs_output"] for r in valid])) if valid else None,
            "median_improvement_db_vs_identity": float(np.median([r["improvement_db_vs_identity"] for r in valid])) if valid else None,
        }
        rows.append(row)

    dry = read16(CLARITY)
    calibration = synthetic_early_echo_control(dry)
    metadata = {
        "scope": "single consumed user recording plus a synthetic early-echo calibration from a local dry-speech control; no new model inference",
        "sample_rate_hz": FS,
        "input_reference": "Cap60",
        "alignment": "global cross-correlation on raw-input-active samples, lag search +/-100 ms, parabolic sub-sample peak interpolation; fractional Fourier shift applied to candidate",
        "fir": {"length_ms": FIR_MS, "length_samples": FIR_LEN, "method": "masked causal least squares via conjugate gradient with 1e-4 trace-scaled ridge; three event-group cross-validation folds"},
        "coherence": "median magnitude-squared complex coherence over input-selected voiced 100 ms windows (10 ms start grid); energetic bins only; bands 300-3000 and 3000-7900 Hz",
        "variability": "median temporal MAD of 20log10|Y/X| and circular phase dispersion over voiced frames",
        "early_cepstrum": "fraction of real-cepstral log-magnitude transfer energy at 2-50 ms quefrency",
        "events": [{"start_sample": s, "end_sample": e} for s, e in events],
        "n_voiced_frames": int(len(voiced)),
        "synthetic_early_echo_calibration": calibration,
        "candidates": rows,
        "limitations": [
            "Adobe and learned enhancers need not be linear time-invariant; low FIR fit is not evidence of poor audio quality.",
            "High coherence/FIR fit indicates source-preserving stable filtering, not that the filter removed only room reflections.",
            "Speech, phase alignment, denoising and room response are entangled in a single recording.",
            "No automatic candidate selection or claim of Adobe parity is made.",
        ],
    }
    (OUT / "short_filter_explainability.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with (OUT / "short_filter_explainability.csv").open("w", newline="") as f:
        fields = ["candidate", "lag_ms", "coherence_300_3000", "coherence_3000_7900",
                  "magnitude_mad_300_3000_db", "phase_dispersion_300_3000",
                  "early_cepstral_fraction", "fir_cv_r2", "fir_cv_error_db", "fir_gain_vs_identity_db"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            tv = row["transfer_variability"]
            cv = row["short_fir_cv"]
            writer.writerow({
                "candidate": row["candidate"], "lag_ms": row["estimated_global_lag_ms"],
                "coherence_300_3000": row["median_complex_coherence"]["300-3000"],
                "coherence_3000_7900": row["median_complex_coherence"]["3000-7900"],
                "magnitude_mad_300_3000_db": tv["300-3000"]["median_magnitude_mad_db"],
                "phase_dispersion_300_3000": tv["300-3000"]["median_phase_dispersion"],
                "early_cepstral_fraction": row["early_cepstral_energy_fraction_2_50ms"],
                "fir_cv_r2": cv["median_heldout_r2"],
                "fir_cv_error_db": cv["median_heldout_error_db_vs_output"],
                "fir_gain_vs_identity_db": cv["median_improvement_db_vs_identity"],
            })

    labels = [r["candidate"] for r in rows]
    coh = [r["median_complex_coherence"]["300-3000"] for r in rows]
    fir = [r["short_fir_cv"]["median_heldout_r2"] for r in rows]
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    ax.scatter(coh, fir, s=70)
    offsets = {
        "Cap60": (-42, -35), "WPE": (-42, 18), "DPDFNet2": (8, 8),
        "Adobe V2": (8, 8), "ROSE raw": (8, -16),
        "ROSE after Cap60": (8, 8), "StuPASE": (8, 8),
    }
    for name, a, b in zip(labels, coh, fir):
        if b is not None:
            ax.annotate(name, (a, b), xytext=offsets.get(name, (5, 5)),
                        textcoords="offset points", fontsize=8)
    ax.set_xlabel("Cohérence complexe médiane, 300–3000 Hz")
    ax.set_ylabel("R² médian FIR 50 ms, validation par segments")
    ax.set_title("Part de la transformation expliquée par un filtre court\n"
                 "Diagnostic exploratoire — pas un score de qualité")
    ax.grid(alpha=.25)
    fig.savefig(OUT / "short_filter_explainability.png", dpi=170)
    plt.close(fig)
    print(json.dumps({"calibration": calibration, "candidates": [
        {"candidate": r["candidate"], "lag_ms": r["estimated_global_lag_ms"],
         "coherence": r["median_complex_coherence"], "cv": r["short_fir_cv"]}
        for r in rows], "files": ["short_filter_explainability.json",
                                   "short_filter_explainability.csv",
                                   "short_filter_explainability.png"]}, indent=2))


if __name__ == "__main__":
    main()
