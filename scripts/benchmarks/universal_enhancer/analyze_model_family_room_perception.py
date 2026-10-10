#!/usr/bin/env python3
"""Describe post-speech residuals across the existing model-family pack.

This is a single-recording diagnostic. It measures output energy after speech
events, not reverberation ground truth; noise, speech tails and artifacts remain
mixed. Windows overlapped by the next speech event are censored.
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
from scipy import ndimage
from scipy.signal import butter, resample_poly, sosfiltfilt

ROOT = Path(__file__).resolve().parents[3]
PACK = ROOT / "results/user-recording-test-2026-10-09/model-family-comparison-02"
INPUT = ROOT / "results/user-recording-test-2026-10-09/01_input_mono_48k.wav"
OUT = PACK
RATE = 48_000
FRAME = 960  # 20 ms
HOP = 480    # 10 ms
GAP_BRIDGE = 8
MIN_EVENT_FRAMES = 10
WINDOWS_MS = ((50, 150), (150, 300), (300, 600))
BANDS = ((125, 500), (500, 2_000), (2_000, 8_000), (8_000, 19_000))
FILES = {
    "Cap60": "CAP60.wav",
    "Adobe V2": "ADOBE.wav",
    "StuPASE": "STUPASE.wav",
    "ROSE raw": "ROSE_RAW.wav",
    "ROSE after Cap60": "ROSE_CAP60.wav",
    "WPE": "WPE.wav",
    "DPDFNet2": "DPDFNET2.wav",
}
# These engines were inferred at 16 kHz then resampled for the listening pack.
NATIVE_BANDWIDTH_HZ = {"StuPASE": 8_000, "ROSE raw": 8_000, "ROSE after Cap60": 8_000}


def read(name: str | Path) -> np.ndarray:
    path = name if isinstance(name, Path) else PACK / name
    x, sr = sf.read(path, dtype="float64")
    if sr != RATE:
        raise ValueError(f"{name}: expected {RATE} Hz, got {sr}")
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x


def frame_rms(x: np.ndarray) -> np.ndarray:
    n = 1 + max(0, (len(x) - FRAME) // HOP)
    return np.array([
        np.sqrt(np.mean(x[i * HOP:i * HOP + FRAME] ** 2))
        for i in range(n)
    ])


def voiced_frame_mask(x16: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Use input-only periodicity and level to choose common voiced frames."""
    frame = 640  # 40 ms at 16 kHz
    hop = 160    # 10 ms
    starts = np.arange(max(0, (len(x16) - frame) // hop + 1)) * hop
    threshold = 0.035 * float(np.max(np.abs(x16)))
    window = np.hanning(frame)
    voiced = []
    for start in starts:
        chunk = np.asarray(x16[start:start + frame], dtype=np.float64)
        chunk -= chunk.mean()
        rms = float(np.sqrt(np.mean(chunk * chunk)))
        corr_input = chunk * window
        corr = np.correlate(corr_input, corr_input, mode="full")[frame - 1:]
        corr /= max(float(corr[0]), 1e-12)
        # Fundamental range 70–300 Hz, with a conservative periodicity floor.
        periodicity = float(np.max(corr[53:229])) if len(corr) > 229 else 0.0
        voiced.append(rms > threshold and periodicity >= 0.35)
    return starts, np.asarray(voiced, dtype=bool)


def spectral_ripple_db(x16: np.ndarray, starts: np.ndarray,
                       voiced: np.ndarray) -> dict[str, float | int | None]:
    """Describe residual log-spectrum ripple after a broad log-frequency envelope.

    This is only a coloration proxy: speaker formants, phonetic mix, and a
    processor's spectrum can all change it. It is deliberately not called a
    dereverberation score.
    """
    selected = starts[voiced]
    if len(selected) < 5:
        return {"n_voiced_frames": int(len(selected)), "ripple_sd_db": None,
                "ripple_abs_p90_db": None}
    frame = 640
    window = np.hanning(frame)
    spectra = np.asarray([
        np.abs(np.fft.rfft(np.asarray(x16[s:s + frame], dtype=np.float64) * window,
                           n=2048)) ** 2
        for s in selected
    ])
    mean_power = np.mean(spectra, axis=0) + 1e-15
    frequencies = np.fft.rfftfreq(2048, 1 / 16_000)
    log_frequency = np.linspace(np.log(250), np.log(7_000), 1000)
    log_power_db = np.interp(log_frequency, np.log(frequencies[1:]),
                             10 * np.log10(mean_power[1:]))
    sigma = 0.12 * np.log(2) / (log_frequency[1] - log_frequency[0])
    envelope = ndimage.gaussian_filter1d(log_power_db, sigma=sigma)
    ripple = log_power_db - envelope
    return {"n_voiced_frames": int(len(selected)),
            "ripple_sd_db": float(np.std(ripple)),
            "ripple_abs_p90_db": float(np.percentile(np.abs(ripple), 90)),
            "frequency_range_hz": [250, 7_000],
            "smooth_envelope_bandwidth_octaves": 0.12}


def events_from_reference(x: np.ndarray) -> list[tuple[int, int]]:
    env = frame_rms(x)
    active = env > 0.035 * np.max(np.abs(x))
    active = ndimage.binary_closing(active, structure=np.ones(GAP_BRIDGE + 1))
    edges = np.diff(np.r_[False, active, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return [
        (max(0, int(s * HOP)), min(len(x), int((e - 1) * HOP + FRAME)))
        for s, e in zip(starts, stops)
        if e - s >= MIN_EVENT_FRAMES
    ]


def main() -> None:
    audio = {name: read(path) for name, path in FILES.items()}
    n = min(map(len, audio.values()))
    audio = {k: v[:n] for k, v in audio.items()}
    activity_ref = read(INPUT)[:n]
    if len(activity_ref) != n:
        raise ValueError("raw input and listening-pack outputs are not sample-aligned")
    ref = audio["Cap60"]
    ref_env = frame_rms(ref)
    activity_env = frame_rms(activity_ref)
    active = activity_env > 0.035 * np.max(np.abs(activity_ref))
    ref_speech = float(np.sqrt(np.mean(ref_env[active] ** 2)))
    events = events_from_reference(activity_ref)
    activity16 = resample_poly(activity_ref, 1, 3)
    voiced_starts, voiced_mask = voiced_frame_mask(activity16)

    # One fixed gain per output: common Cap60 activity mask, never a tail gain.
    matched: dict[str, np.ndarray] = {}
    for name, x in audio.items():
        env = frame_rms(x)
        gain = ref_speech / max(float(np.sqrt(np.mean(env[active[:len(env)]] ** 2))), 1e-12)
        matched[name] = x * gain

    filtered: dict[str, dict[str, np.ndarray]] = {}
    for name, x in matched.items():
        filtered[name] = {"broadband": x}
        for lo, hi in BANDS:
            sos = butter(4, (lo, hi), btype="bandpass", fs=RATE, output="sos")
            filtered[name][f"{lo}-{hi}Hz"] = sosfiltfilt(sos, x)

    # SRMRpy is an optional, locally cloned research metric. Its reference
    # implementation is tested at 8/16 kHz, so downsample the 48 kHz pack to
    # 16 kHz instead of asking it to process unsupported rates. Keep the
    # metric separate from pause residuals: it is sensitive to noise and
    # processing artifacts and is not a perceptual ground truth.
    srmr_rows = []
    try:
        import sys
        sys.path.insert(0, str(ROOT / ".tools/SRMRpy"))
        from srmrpy import srmr
    except ImportError:
        srmr = None
    if srmr is not None:
        for name, x in matched.items():
            x16 = resample_poly(x, 1, 3)
            raw_score, _ = srmr(x16, 16_000, fast=True, norm=False)
            norm_score, _ = srmr(x16, 16_000, fast=True, norm=True)
            srmr_rows.append({"candidate": name,
                              "srmr_original": float(raw_score),
                              "srmr_normalized": float(norm_score),
                              "sample_rate_hz": 16_000,
                              "duration_s": len(x16) / 16_000})

    ripple_rows = []
    for name, x in matched.items():
        x16 = resample_poly(x, 1, 3)
        ripple_rows.append({"candidate": name,
                            **spectral_ripple_db(x16, voiced_starts, voiced_mask)})

    rows = []
    for event_id, (_, end) in enumerate(events, 1):
        next_start = events[event_id][0] if event_id < len(events) else n
        for wstart_ms, wend_ms in WINDOWS_MS:
            a = end + round(wstart_ms * RATE / 1000)
            b = min(n, end + round(wend_ms * RATE / 1000))
            complete = b - a == round((wend_ms - wstart_ms) * RATE / 1000)
            unoverlapped = next_start >= b
            valid_geometry = complete and unoverlapped
            for name, bands in filtered.items():
                for band, x in bands.items():
                    supported = not (band == "8000-19000Hz" and name in NATIVE_BANDWIDTH_HZ)
                    valid = valid_geometry and supported
                    value = None
                    if valid:
                        value = float(20 * np.log10(max(np.sqrt(np.mean(x[a:b] ** 2)) / ref_speech, 1e-12)))
                    rows.append({
                        "candidate": name,
                        "event": event_id,
                        "event_end_s": round(end / RATE, 4),
                        "window_ms": f"{wstart_ms}-{wend_ms}",
                        "band": band,
                        "valid": valid,
                        "censor_reason": "" if valid else (
                            "outside_native_bandwidth" if not supported else
                            "overlapped_by_next_speech" if complete and not unoverlapped else
                            "file_ended_before_window"
                        ),
                        "post_event_rms_db_relative_to_common_speech": value,
                    })

    summary = []
    for name in FILES:
        for window in (f"{a}-{b}" for a, b in WINDOWS_MS):
            for band in ["broadband", *(f"{lo}-{hi}Hz" for lo, hi in BANDS)]:
                values = [r["post_event_rms_db_relative_to_common_speech"] for r in rows
                          if r["candidate"] == name and r["window_ms"] == window
                          and r["band"] == band and r["valid"]]
                summary.append({
                    "candidate": name, "window_ms": window, "band": band,
                    "n_valid_offsets": len(values),
                    "median_db_relative_to_common_speech": float(np.median(values)) if values else None,
                    "p10_db_relative_to_common_speech": float(np.percentile(values, 10)) if values else None,
                    "p90_db_relative_to_common_speech": float(np.percentile(values, 90)) if values else None,
                })

    metadata = {
        "scope": "single consumed recording; descriptive post-speech residual, not dereverberation ground truth",
        "input_pack": str(PACK.relative_to(ROOT)),
        "sample_rate_hz": RATE,
        "speech_activity": "20 ms RMS > 3.5% of raw-input sample peak; gaps up to 80 ms bridged; minimum event 100 ms",
        "alignment_gain": "one constant per candidate matching RMS over the shared raw-input active-frame mask; common level reference is Cap60 active RMS",
        "windows_ms_after_event_end": WINDOWS_MS,
        "bands_hz": BANDS,
        "native_bandwidth_hz": NATIVE_BANDWIDTH_HZ,
        "srmr": {
            "implementation": "SRMRpy official reference implementation, optional local clone",
            "repository": "https://github.com/jfsantos/SRMRpy",
            "commit": "fee009779cef96bed34db3a7e31d10f3ad1ea133",
            "sample_rate_hz": 16000,
            "rows": srmr_rows,
            "limitations": [
                "Non-intrusive modulation-based proxy; not a direct RT60 or room-depth measurement.",
                "Can change with noise reduction, compression, bandwidth, and artifacts as well as reverberation.",
                "A single 6.23 s user clip cannot establish a universal ranking or Adobe parity.",
                "Scores are reported only when optional local SRMRpy is available.",
            ],
        },
        "voiced_spectral_ripple": {
            "definition": "SD and absolute p90 of log-power residual from a 0.12-octave Gaussian-smoothed log-frequency envelope, averaged across input-selected periodic frames",
            "mask": "40 ms frames, 10 ms hop, raw-input RMS > 3.5% of raw peak and normalized autocorrelation peak >= 0.35 for 70-300 Hz periodicity; candidates share exact frame indices",
            "rows": ripple_rows,
            "limitations": [
                "Proxy for spectral coloration, not an isolated early-reflection measurement.",
                "Changes in phonetic balance, formants, bandwidth, and model spectrum also change the index.",
                "Interpret only beside listening and known dry/reverberant controls; not as a product score.",
            ],
        },
        "offsets": [{"event": i, "start_s": round(s / RATE, 4), "end_s": round(e / RATE, 4)} for i, (s, e) in enumerate(events, 1)],
        "limitations": [
            "Residual contains room response, noise, unvoiced speech, and model artifacts; there is no dry stem.",
            "A short phrase does not provide independent repeated room-tail observations.",
            "Early reflections during speech and spectral combing are not identified by these pause windows.",
            "A lower number is not automatically better: aggressive gating can lower it while damaging speech.",
        ],
        "summary": summary,
    }
    (OUT / "multi_offset_room_residual.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with (OUT / "multi_offset_room_residual.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    colors = plt.get_cmap("tab10").colors
    candidates = list(FILES)
    for ax, (wstart, wend) in zip(axes, WINDOWS_MS[:2]):
        x = np.arange(len(BANDS))
        width = 0.105
        window_rows = [r for r in summary if r["window_ms"] == f"{wstart}-{wend}" and r["band"] == f"{BANDS[0][0]}-{BANDS[0][1]}Hz"]
        if not any(r["n_valid_offsets"] for r in window_rows):
            ax.text(.5, .5, "Aucune fenêtre d’offset assez longue et sans parole suivante : données censurées.",
                    transform=ax.transAxes, ha="center", va="center", fontsize=11)
            ax.set_axis_off()
            continue
        for i, name in enumerate(candidates):
            vals = []
            for lo, hi in BANDS:
                vals.extend(r["median_db_relative_to_common_speech"] for r in summary
                            if r["candidate"] == name and r["window_ms"] == f"{wstart}-{wend}"
                            and r["band"] == f"{lo}-{hi}Hz")
            if any(v is not None for v in vals):
                ax.bar(x + (i - (len(candidates)-1)/2) * width, [v if v is not None else np.nan for v in vals],
                       width=width, label=name, color=colors[i])
        ax.set_xticks(x, [f"{lo}-{hi} Hz" for lo, hi in BANDS])
        ax.set_ylabel("dB vs speech")
        ax.set_title(f"Residual après événement : {wstart}–{wend} ms (barres sans offsets valides = données censurées)")
        ax.grid(axis="y", alpha=.25)
        if ax.containers:
            ax.legend(ncol=4, fontsize=8)
    fig.suptitle("Énergie résiduelle multi-bandes — une voix, une pièce, alignement commun\nMesure descriptive, pas un score de déréverbération\n300–600 ms : aucun offset complet et non recouvert (n=0, censuré)")
    fig.savefig(OUT / "multi_offset_room_residual.png", dpi=170)
    plt.close(fig)
    ripple_sd = {r["candidate"]: r["ripple_sd_db"] for r in ripple_rows}
    ripple_order = list(FILES)
    ripple_fig, ripple_ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    ripple_values = [ripple_sd[name] for name in ripple_order]
    bars = ripple_ax.bar(ripple_order, ripple_values,
                         color=colors[:len(ripple_order)])
    ripple_ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ripple_ax.set_ylabel("Écart-type du résidu spectral (dB)")
    ripple_ax.set_title("Ripple spectrale sur 230 trames voisées communes\n"
                        "Indice de coloration; le lissage peut aussi le réduire")
    ripple_ax.tick_params(axis="x", rotation=24)
    ripple_ax.grid(axis="y", alpha=.25)
    ripple_fig.savefig(OUT / "voiced_spectral_ripple.png", dpi=170)
    plt.close(ripple_fig)
    print(json.dumps({"events": metadata["offsets"], "srmr": srmr_rows,
                      "voiced_spectral_ripple": ripple_rows,
                      "valid_counts": {
        f"{a}-{b}ms": sum(1 for r in rows if r["candidate"] == "Cap60" and r["window_ms"] == f"{a}-{b}" and r["band"] == "broadband" and r["valid"])
        for a, b in WINDOWS_MS
    }, "files": ["multi_offset_room_residual.json", "multi_offset_room_residual.csv", "multi_offset_room_residual.png", "voiced_spectral_ripple.png"]}, indent=2))


if __name__ == "__main__":
    main()
