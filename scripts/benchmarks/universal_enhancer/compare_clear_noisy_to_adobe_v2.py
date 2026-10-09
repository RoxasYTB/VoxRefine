#!/usr/bin/env python3
"""Compare matched Adobe v2 and VoxRefine renders on three clean-referenced mixes."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import stft

ROOT = Path(__file__).resolve().parents[3]
CORPUS = ROOT / "corpus/samples/clear-noisy-mix-01"
OUT = ROOT / "results/adobe-v2-pair-audit-2026-10-09"
EXPORTS = OUT / "exports"
RATE = 48_000
FRAME = 960  # 20 ms
SAMPLES = ("emy_mixed_ambience18", "remi_crowd18", "stephanie_fan18")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path: Path) -> np.ndarray:
    x, rate = sf.read(path, dtype="float64", always_2d=True)
    if rate != RATE or x.shape[1] != 1:
        raise ValueError(f"expected mono {RATE} Hz WAV: {path} ({rate} Hz, {x.shape})")
    if not np.isfinite(x).all():
        raise ValueError(f"non-finite samples: {path}")
    return x[:, 0]


def rms_frames(x: np.ndarray) -> np.ndarray:
    n = len(x) // FRAME
    return np.sqrt(np.mean(x[:n * FRAME].reshape(n, FRAME) ** 2, axis=1) + 1e-20)


def db(value: float) -> float:
    return float(20 * np.log10(max(float(value), 1e-12)))


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def speech_spectrum(x: np.ndarray, speech_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    freq, times, z = stft(x, fs=RATE, window="hann", nperseg=2048,
                          noverlap=2048 - FRAME, boundary="zeros", padded=True)
    idx = np.clip(np.rint(times / (FRAME / RATE)).astype(int), 0, len(speech_mask) - 1)
    active = speech_mask[idx]
    if not np.any(active):
        raise ValueError("no speech-active STFT frames")
    power = np.mean(np.abs(z[:, active]) ** 2, axis=1)
    smooth = np.empty_like(power)
    bin_hz = RATE / 2048
    for i, hz in enumerate(freq):
        width_hz = max(hz, 80.0) * (2 ** (1 / 12) - 2 ** (-1 / 12))
        width = max(1, round(width_hz / bin_hz))
        smooth[i] = np.mean(power[max(0, i - width):min(len(power), i + width + 1)])
    return freq, 10 * np.log10(smooth + 1e-20)


def main() -> None:
    rows: list[dict] = []
    curves: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    spectrogram_audio: dict[str, dict[str, np.ndarray]] = {}
    hashes: dict[str, dict[str, str]] = {}
    for sample_id in SAMPLES:
        paths = {
            "noisy_input": CORPUS / "noisy" / f"{sample_id}-noisy.wav",
            "clean_speech": CORPUS / "clean" / f"{sample_id}-clean-reference.wav",
            "noise_stem": CORPUS / "noise" / (f"{sample_id}-" + {
                "emy_mixed_ambience18": "fan+crowd-stem.wav",
                "remi_crowd18": "crowd-stem.wav",
                "stephanie_fan18": "fan-stem.wav",
            }[sample_id]),
            "NFE64-C": EXPORTS / f"{sample_id}-nfe64-C.wav",
            "Adobe v2": EXPORTS / f"{sample_id}-adobe-v2.wav",
        }
        audio = {name: read(path) for name, path in paths.items()}
        n = min(map(len, audio.values()))
        audio = {name: value[:n] for name, value in audio.items()}
        spectrogram_audio[sample_id] = {name: audio[name] for name in ("noisy_input", "NFE64-C", "Adobe v2")}
        clean_env = rms_frames(audio["clean_speech"])
        # Use the corpus's 30 dB below p95 clean-reference activity rule.
        threshold = np.percentile(clean_env, 95) * 10 ** (-30 / 20)
        speech = clean_env >= threshold
        pad = 6  # ±120 ms guard around source speech at 20 ms frame rate.
        padded = np.pad(speech.astype(np.int8), pad)
        speech = np.convolve(padded, np.ones(2 * pad + 1, dtype=np.int32), mode="same")[pad:-pad] > 0
        gap = ~speech
        weak = speech & (clean_env <= np.percentile(clean_env[speech], 25))
        clean_speech_level = float(np.median(clean_env[speech]))
        noise_env = rms_frames(audio["noise_stem"])
        input_snr_db = db(float(np.mean(clean_env[speech] ** 2) / max(np.mean(noise_env[speech] ** 2), 1e-20)) ** .5)
        clean_curve_frequency, clean_curve = speech_spectrum(audio["clean_speech"], speech)
        adobe_env = rms_frames(audio["Adobe v2"])
        adobe_gain = clean_speech_level / max(float(np.median(adobe_env[speech])), 1e-12)
        f_adobe, curve_adobe = speech_spectrum(audio["Adobe v2"] * adobe_gain, speech)
        use = (clean_curve_frequency >= 80) & (clean_curve_frequency <= 12_000)
        curves[sample_id] = {"clean": (clean_curve_frequency, clean_curve),
                             "Adobe v2": (f_adobe, curve_adobe)}
        hashes[sample_id] = {name: sha256(path) for name, path in paths.items()}

        for name in ("noisy_input", "NFE64-C", "Adobe v2"):
            candidate = audio[name]
            env = rms_frames(candidate)[:len(clean_env)]
            frame_delta = 20 * np.log10((env + 1e-12) / (clean_env + 1e-12))
            active_level_delta = float(np.median(frame_delta[speech]))
            candidate_active_rms = float(np.median(env[speech]))
            fixed_gain = clean_speech_level / max(candidate_active_rms, 1e-12)
            fixed = candidate * fixed_gain
            freq, curve = speech_spectrum(fixed, speech)
            if name != "Adobe v2":
                f_adobe, curve_adobe = speech_spectrum(audio["Adobe v2"] * (
                    clean_speech_level / max(float(np.median(rms_frames(audio["Adobe v2"])[speech])), 1e-12)), speech)
            curves[sample_id][name] = (freq, curve)
            gap_level = float(np.median(env[gap])) if np.any(gap) else None
            rows.append({
                "sample_id": sample_id,
                "noise_type": {"emy_mixed_ambience18": "fan+crowd", "remi_crowd18": "crowd",
                               "stephanie_fan18": "fan"}[sample_id],
                "candidate": name,
                "duration_s": n / RATE,
                "speech_frames": int(speech.sum()),
                "weak_speech_frames": int(weak.sum()),
                "input_active_snr_db": input_snr_db,
                "speech_log_rms_envelope_corr_vs_clean": correlation(
                    np.log(clean_env[speech] + 1e-12), np.log(env[speech] + 1e-12)),
                "speech_level_delta_median_db_vs_clean": active_level_delta,
                "weak_frame_level_delta_p10_db_vs_clean": float(np.percentile(frame_delta[weak], 10)),
                "gap_rms_median_dbfs": db(gap_level) if gap_level is not None else None,
                "speech_spectrum_mae_80_12k_db_vs_adobe": float(np.mean(np.abs(curve[use] - curve_adobe[use]))),
                "adobe_delta_0_250_hz_db": None,
                "adobe_delta_250_1000_hz_db": None,
                "adobe_delta_1000_4000_hz_db": None,
                "adobe_delta_4000_8000_hz_db": None,
                "adobe_delta_8000_12000_hz_db": None,
                "sample_peak_dbfs": db(float(np.max(np.abs(candidate)))),
            })
            row = rows[-1]
            for key, lo, hi in (("0_250", 0, 250), ("250_1000", 250, 1_000),
                                ("1000_4000", 1_000, 4_000), ("4000_8000", 4_000, 8_000),
                                ("8000_12000", 8_000, 12_000)):
                band = (freq >= lo) & (freq < hi)
                if name == "Adobe v2":
                    delta = 0.0
                else:
                    ref_curve = curves[sample_id]["Adobe v2"][1]
                    delta = float(np.mean(curve[band] - ref_curve[band]))
                    row[f"adobe_delta_{key}_hz_db"] = delta

    OUT.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / "clear-noisy-adobe-comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    labels = {"emy_mixed_ambience18": "Emy · ventilateur + foule",
              "remi_crowd18": "Rémi · foule", "stephanie_fan18": "Stéphanie · ventilateur"}
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), constrained_layout=True)
    for axis, sample_id in zip(axes, SAMPLES):
        for candidate, style in (("clean", "k--"), ("noisy_input", "#94a3b8"),
                                 ("NFE64-C", "#e76f51"), ("Adobe v2", "#111827")):
            f, curve = curves[sample_id][candidate]
            axis.plot(f, curve, style, linewidth=1.2, label=candidate)
        axis.set_xscale("log"); axis.set_xlim(80, 12_000); axis.set_ylabel("Power (dB)")
        axis.set_title(labels[sample_id]); axis.grid(True, which="both", alpha=.2); axis.legend(ncol=4, fontsize=8)
    axes[-1].set_xlabel("Frequency (Hz)")
    fig.suptitle("Adobe v2 and NFE64-C on three 15 s controlled-noise pairs · speech RMS matched to clean")
    fig.savefig(OUT / "clear-noisy-adobe-spectra.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(15, 10), constrained_layout=True, sharex=True, sharey=True)
    mesh = None
    for row_idx, sample_id in enumerate(SAMPLES):
        for col_idx, candidate in enumerate(("noisy_input", "NFE64-C", "Adobe v2")):
            _, times, z = stft(spectrogram_audio[sample_id][candidate], fs=RATE, window="hann",
                               nperseg=1024, noverlap=768, boundary=None, padded=False)
            freq = np.fft.rfftfreq(1024, 1 / RATE)
            level = 20 * np.log10(np.abs(z) + 1e-8)
            axes[row_idx, col_idx].pcolormesh(times, freq, level, shading="auto", cmap="magma",
                                               vmin=-100, vmax=-25)
            axes[row_idx, col_idx].set_yscale("log"); axes[row_idx, col_idx].set_ylim(80, 12_000)
            axes[row_idx, col_idx].set_title(f"{labels[sample_id]} · {candidate}")
            axes[row_idx, col_idx].grid(False)
            mesh = axes[row_idx, col_idx].collections[-1]
    axes[-1, 1].set_xlabel("Time (s)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Frequency (Hz)")
    if mesh is not None:
        fig.colorbar(mesh, ax=axes, label="Magnitude (dBFS)", shrink=.8)
    fig.suptitle("15 s spectrograms · common 48 kHz mono inputs and Adobe / VoxRefine outputs")
    fig.savefig(OUT / "clear-noisy-adobe-spectrograms.png", dpi=160)
    plt.close(fig)

    summary_rows = []
    for name in ("noisy_input", "NFE64-C", "Adobe v2"):
        group = [r for r in rows if r["candidate"] == name]
        summary_rows.append({
            "candidate": name,
            "median_envelope_corr": float(np.median([r["speech_log_rms_envelope_corr_vs_clean"] for r in group])),
            "median_active_level_delta_db": float(np.median([r["speech_level_delta_median_db_vs_clean"] for r in group])),
            "median_weak_frame_p10_db": float(np.median([r["weak_frame_level_delta_p10_db_vs_clean"] for r in group])),
            "median_gap_rms_dbfs": (float(np.median([r["gap_rms_median_dbfs"] for r in group
                                                       if r["gap_rms_median_dbfs"] is not None]))
                                    if any(r["gap_rms_median_dbfs"] is not None for r in group) else None),
            "median_spectral_mae_vs_adobe_db": float(np.median([r["speech_spectrum_mae_80_12k_db_vs_adobe"] for r in group])),
        })
    report = {
        "protocol": "Three same-source 15 s mono 48 kHz public-domain speech clips mixed with documented CC0 fan/crowd noise at 18 dB active SNR; same input files processed in Adobe Podcast v2 at visible default Voice 50%, Music 10%, Background 10%, and local Resemble NFE64-C.",
        "alignment": "All six enhanced WAVs are 15.000 s / 720000 frames at 48 kHz; frame metrics use same-start alignment with no per-candidate time warp.",
        "speech_mask": "Known clean reference; 20 ms RMS threshold 30 dB below its p95, expanded by ±120 ms.",
        "spectral_method": "Each candidate is fixed-gain matched to median clean active-speech RMS before 1/6-octave spectrum comparison; active 80 Hz–12 kHz MAE is descriptive, not quality.",
        "samples": SAMPLES,
        "audio_sha256": hashes,
        "summary": summary_rows,
        "csv": str((OUT / "clear-noisy-adobe-comparison.csv").relative_to(ROOT)),
        "figure": str((OUT / "clear-noisy-adobe-spectra.png").relative_to(ROOT)),
        "spectrogram_figure": str((OUT / "clear-noisy-adobe-spectrograms.png").relative_to(ROOT)),
        "limitations": [
            "Three read-speech voices, one recording style, three controlled backgrounds at one SNR; not universal coverage.",
            "Frame-envelope and gap RMS are proxies; gap includes breaths and reference activity-mask leakage.",
            "Generative NFE64-C changes waveform phase, so no sample-level SI-SDR/STOI/PESQ claim is made here.",
            "Adobe output is a comparison reference, not clean ground truth; the paired clean stem is the speech-level reference.",
            "The Adobe export used the visible free/default controls and does not test paid strength sliders or separate stems.",
        ],
    }
    (OUT / "clear-noisy-adobe-report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    main()
