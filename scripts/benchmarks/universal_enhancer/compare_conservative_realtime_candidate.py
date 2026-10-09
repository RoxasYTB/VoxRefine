#!/usr/bin/env python3
"""Compare speech preservation on the exact controlled Adobe-reference crop.

Requires numpy, scipy, soundfile, matplotlib, pyloudnorm and pystoi. The report is
descriptive for this one six-second sample; STOI is an intelligibility proxy,
not a substitute for a listening test or a multi-speaker evaluation.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from pystoi import stoi
from scipy.signal import butter, correlate, correlation_lags, sosfiltfilt, stft

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/adobe-curve-eq-2026-10-09"
PAIR = ROOT / "results/resemble-enhance-01/controlled-challenger-03/audio"
RATE = 48_000
TARGET_LUFS = -24.0


def read(path: Path) -> np.ndarray:
    x, rate = sf.read(path, dtype="float64", always_2d=True)
    if rate != RATE:
        raise ValueError(f"{path}: expected {RATE} Hz, got {rate}")
    if x.shape[1] != 1:
        raise ValueError(f"{path}: expected mono audio")
    return x[:, 0]


def frame_rms(x: np.ndarray, frame: int = 480) -> np.ndarray:
    count = len(x) // frame
    values = x[:count * frame].reshape(count, frame)
    return np.sqrt(np.mean(values * values, axis=1) + 1e-15)


def db(value: float) -> float:
    return float(20 * np.log10(max(value, 1e-15)))


def main() -> None:
    dry_path = PAIR / "noise-snr10-adobe-dry.wav"
    paths = {
        "Noisy input": PAIR / "noise-snr10-adobe-input.wav",
        "Resemble NFE16": OUT / "controlled-nfe16-flat-cpu.wav",
        "Resemble NFE32": OUT / "controlled-nfe32-flat-cpu.wav",
        "Resemble NFE64": OUT / "controlled-nfe64-flat-cpu.wav",
        "DPDFNet2 stream": OUT / "controlled-dpdfnet2-live-sim.wav",
        "Adobe Podcast v2": PAIR / "adobe-v2-reference.wav",
    }
    missing = [str(p) for p in [dry_path, *paths.values()] if not p.is_file()]
    if missing:
        raise SystemExit("Missing benchmark renders:\n" + "\n".join(missing))

    dry = read(dry_path)
    count = min(len(dry), *(len(read(path)) for path in paths.values()))
    dry = dry[:count]
    signals = {name: read(path)[:count] for name, path in paths.items()}
    meter = pyln.Meter(RATE)
    dry_lufs = float(meter.integrated_loudness(dry))
    dry_frames = frame_rms(dry)
    active = dry_frames >= float(np.max(dry_frames)) * 10 ** (-35 / 20)
    guard = 6  # ±60 ms around clean-reference speech frames.
    active = np.convolve(np.pad(active.astype(np.int8), guard),
                         np.ones(2 * guard + 1, dtype=np.int32), mode="same")[guard:-guard] > 0
    gap = ~active

    # One transparent gain per render: equal integrated loudness to the clean stem.
    matched: dict[str, np.ndarray] = {}
    rows: list[dict[str, float | str]] = []
    filter_sos = butter(4, [200, 4_000], btype="bandpass", fs=RATE, output="sos")
    ref_band = sosfiltfilt(filter_sos, dry)
    max_lag = RATE // 10
    for name, x in signals.items():
        raw_lufs = float(meter.integrated_loudness(x))
        y = x * (10 ** ((dry_lufs - raw_lufs) / 20))
        matched[name] = y
        levels = frame_rms(y)
        # Find one small global alignment for STOI. This does not time-warp speech.
        filtered = sosfiltfilt(filter_sos, y)
        lag_values = correlation_lags(len(filtered), len(ref_band), mode="full")
        cross = correlate(filtered, ref_band, mode="full", method="fft")
        allowed = np.abs(lag_values) <= max_lag
        lag = int(lag_values[allowed][np.argmax(cross[allowed])])
        candidate_start, reference_start = max(lag, 0), max(-lag, 0)
        shared = min(len(y) - candidate_start, len(dry) - reference_start)
        score = float(stoi(dry[reference_start:reference_start + shared],
                           y[candidate_start:candidate_start + shared], RATE))
        rows.append({
            "render": name,
            "stoi_best_global_lag": score,
            "best_lag_ms": lag / RATE * 1000,
            "raw_lufs": raw_lufs,
            "gain_to_clean_lufs_db": dry_lufs - raw_lufs,
            "active_frame_median_dbfs": db(float(np.median(levels[active]))),
            "gap_frame_median_dbfs": db(float(np.median(levels[gap]))),
            "sample_peak_dbfs_after_loudness_match": db(float(np.max(np.abs(y)))),
        })

    # Same short-time speech activity mask for every spectrogram/curve.
    frequencies, times, _ = stft(dry, RATE, nperseg=2048, noverlap=1536,
                                 boundary=None, padded=False)
    active_times = active[np.clip(np.rint(times / 0.01).astype(int), 0, len(active) - 1)]
    spectra: dict[str, np.ndarray] = {}
    spectrograms: dict[str, np.ndarray] = {}
    for name, y in matched.items():
        _, _, z = stft(y, RATE, nperseg=2048, noverlap=1536,
                       boundary=None, padded=False)
        power = np.abs(z) ** 2
        spectra[name] = np.mean(power[:, active_times], axis=1)
        spectrograms[name] = 10 * np.log10(power + 1e-12)

    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "purpose": "Same-input speech-preservation screen after the NFE16 render was reported to lose words and crackle.",
        "sample_rate_hz": RATE,
        "duration_seconds": count / RATE,
        "source_condition": "Existing six-second LibriVox crop mixed with classroom noise at nominal +10 dB SNR; clean speech stem and Adobe v2 reference are available.",
        "reference_lufs": dry_lufs,
        "speech_mask": "Known clean stem, 10 ms RMS threshold at peak -35 dB, expanded by ±60 ms.",
        "method": "Constant LUFS match to clean reference; best global lag within ±100 ms for STOI; no time warping.",
        "metrics": rows,
        "limitations": [
            "STOI is an intelligibility proxy, not a word-error rate or subjective quality score.",
            "One voice and one synthetic noise mix cannot establish universality.",
            "Gap RMS includes breaths and any speech-mask leakage; it is not pure noise power.",
            "Global delay search can improve STOI slightly; the reported lag is shown for auditability.",
        ],
        "dpdfnet_live_sim": {
            "mode": "48 kHz ONNX streaming backend on CPU, wall-clock paced virtual device, preserve-all output",
            "report": "controlled-dpdfnet2-live-sim.json",
        },
    }
    (OUT / "speech-preservation-report.json").write_text(json.dumps(report, indent=2) + "\n")
    with (OUT / "speech-preservation-metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Public-domain source excerpt, normalized copies for direct app playback.
    listening_dir = OUT / "speech-preservation-listening"
    listening_dir.mkdir(exist_ok=True)
    for name, y in {"Clean reference": dry, **matched}.items():
        safe = name.lower().replace(" ", "-")
        wav = listening_dir / f"{safe}.wav"
        sf.write(wav, y * (10 ** ((TARGET_LUFS - float(meter.integrated_loudness(y))) / 20)),
                 RATE, subtype="PCM_24")
        subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                        "-i", str(wav), "-codec:a", "libmp3lame", "-b:a", "192k",
                        str(wav.with_suffix(".mp3"))], check=True)

    labels = list(matched)
    colors = {"Noisy input": "#64748b", "Resemble NFE16": "#e76f51",
              "Resemble NFE32": "#f4a261", "Resemble NFE64": "#9b5de5",
              "DPDFNet2 stream": "#2a9d8f", "Adobe Podcast v2": "#111827"}
    fig, axes = plt.subplots(4, 1, figsize=(12, 16), constrained_layout=True)
    frame_times = np.arange(len(dry_frames)) * 0.01
    axes[0].plot(frame_times, 20 * np.log10(dry_frames + 1e-12), color="black", lw=1.2,
                 label="Clean stem")
    for name in labels:
        env = frame_rms(matched[name])
        axes[0].plot(frame_times, 20 * np.log10(env + 1e-12), color=colors[name], lw=0.8,
                     alpha=0.85, label=name)
    axes[0].set_title("10 ms RMS envelope after constant loudness matching")
    axes[0].set_ylabel("dBFS"); axes[0].set_xlabel("Time (s)"); axes[0].legend(ncol=3)
    for name in labels:
        curve = 10 * np.log10(spectra[name] + 1e-20)
        axes[1].plot(frequencies, curve, color=colors[name], label=name)
    axes[1].plot(frequencies, 10 * np.log10(np.mean(np.abs(stft(dry, RATE, nperseg=2048,
                 noverlap=1536, boundary=None, padded=False)[2][:, active_times]) ** 2,
                 axis=1) + 1e-20), color="black", ls="--", label="Clean stem")
    axes[1].set_xscale("log"); axes[1].set_xlim(80, 16_000)
    axes[1].set_title("Active-speech mean spectrum (STFT, same reference mask)")
    axes[1].set_ylabel("Power (dB, relative)"); axes[1].set_xlabel("Frequency (Hz)")
    axes[1].grid(True, which="both", alpha=0.2); axes[1].legend(ncol=3)
    keep = frequencies <= 12_000
    for spec_ax, name in zip(axes[2:], ["Resemble NFE16", "DPDFNet2 stream"]):
        spec = spectrograms[name]
        mesh = spec_ax.pcolormesh(times, frequencies[keep] / 1000, spec[keep], shading="auto",
                                  vmin=-105, vmax=-35, cmap="magma")
        spec_ax.set_title(f"{name} output spectrogram; same input and reference mask")
        spec_ax.set_ylabel("Frequency (kHz)"); spec_ax.set_xlabel("Time (s)")
        fig.colorbar(mesh, ax=spec_ax, label="dB")
    fig.savefig(OUT / "speech-preservation-comparison.png", dpi=170)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
