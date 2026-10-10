#!/usr/bin/env python3
"""Level-match existing model-family renders for the consumed test.wav only."""
from __future__ import annotations
import csv, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt, stft

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "results/user-recording-test-2026-10-09"
OUT = BASE / "model-family-comparison-02"
SR = 48000
FRAME, HOP = 960, 480
TAIL = (5.79, 5.94)

SOURCES = {
    "CAP60": BASE / "cap60-dereverb-screen-02/deepfilternet-cap60-aligned.wav",
    "ADOBE": BASE / "06_adobe_v2.wav",
    "STUPASE": BASE / "stupase-testwav-cfg025-01/stupase-cap60-raw-48k.wav",
    "ROSE_RAW": BASE / "rose-cd-testwav-exploratory-01/raw-input/rose-cd-cap60-raw-48k.wav",
    "ROSE_CAP60": BASE / "rose-cd-testwav-exploratory-01/rose-cd-cap60-raw-48k.wav",
    "WPE": BASE / "cap60-dereverb-screen-02/wpe-t32-d3-r1e-05-aligned.wav",
    "DPDFNET2": BASE / "cap60-dereverb-screen-02/cap60-dpdfnet2-aligned.wav",
}


def read(path: Path) -> tuple[np.ndarray, int]:
    x, sr = sf.read(path, dtype="float64")
    if x.ndim == 2:
        x = np.mean(x, axis=1)
    return x, sr


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2) + 1e-30))


def db(x: float) -> float:
    return float(20 * np.log10(max(float(x), 1e-15)))


def env(x: np.ndarray) -> np.ndarray:
    n = 1 + (len(x) - FRAME) // HOP
    return np.asarray([rms(x[i * HOP:i * HOP + FRAME]) for i in range(max(n, 0))])


def band_active(x: np.ndarray, active: np.ndarray, low: float, high: float, ref: float) -> float:
    sos = butter(4, [low, high], btype="bandpass", fs=SR, output="sos")
    z = sosfiltfilt(sos, x)
    e = env(z)
    n = min(len(e), len(active))
    return db(rms(e[:n][active[:n]]) / ref)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    audio = {}
    for name, path in SOURCES.items():
        x, sr = read(path)
        if sr != SR:
            raise ValueError(f"{name}: expected {SR} Hz, got {sr} Hz")
        audio[name] = x
    n = min(map(len, audio.values()))
    audio = {k: v[:n] for k, v in audio.items()}
    cap = audio["CAP60"]
    cap_env = env(cap)
    active = cap_env > .035 * np.max(np.abs(cap))
    target = rms(cap_env[active])
    rows, audition, native = [], {}, {}
    for name, x in audio.items():
        e = env(x)
        m = min(len(e), len(active))
        raw_active = rms(e[:m][active[:m]])
        gain = target / max(raw_active, 1e-15)
        y = x * gain
        native[name] = x
        audition[name] = y
        sf.write(OUT / f"{name}.wav", y.astype(np.float32), SR, subtype="PCM_24")
        ey = env(y)
        tail = y[round(TAIL[0] * SR):round(TAIL[1] * SR)]
        ratio = 20 * np.log10(np.maximum(ey[:m], 1e-12) / np.maximum(cap_env[:m], 1e-12))
        row = {"id": name, "source": str(SOURCES[name].relative_to(ROOT)),
               "level_match_gain_db": db(gain), "tail_db_relative_to_speech": db(rms(tail) / target),
               "active_envelope_p10_vs_cap60_db": float(np.percentile(ratio[active[:m]], 10)),
               "peak_dbfs": db(np.max(np.abs(y))), "clipped_samples": int(np.count_nonzero(np.abs(y) >= 1))}
        for low, high, tag in [(80, 140, "80_140_hz"), (140, 250, "140_250_hz"), (4000, 8000, "4_8khz"), (8000, 12000, "8_12khz")]:
            if high <= SR / 2:
                row[tag + "_speech_db"] = band_active(y, active, low, high, target)
        rows.append(row)

    report = {"scope": "one already-consumed user recording; existing local renders only; exploratory, not a holdout",
              "sample_rate_hz": SR, "duration_s": n / SR,
              "common_mask": "Cap60 20 ms RMS, 10 ms hop, active above 3.5% of Cap60 peak",
              "level_match": "one scalar to shared Cap60 active-speech RMS for every file",
              "tail_window_s": TAIL,
              "candidates": rows,
              "limits": ["Adobe is descriptive only", "one short tail window mixes room decay, noise, and speech", "metrics do not establish perceptual ranking", "ROSE-CD is stochastic despite fixed local seed"]}
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / "metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

    colors = {"CAP60": "#555555", "ADOBE": "#d18b00", "STUPASE": "#6a4c93", "ROSE_RAW": "#e76f51", "ROSE_CAP60": "#2a9d8f", "WPE": "#457b9d", "DPDFNET2": "#bc4749"}
    fig, axs = plt.subplots(2, 1, figsize=(12, 9), constrained_layout=True)
    for name, y in audition.items():
        f, t, z = stft(y, fs=SR, nperseg=2048, noverlap=1536, boundary=None)
        am = np.interp(t, np.arange(len(active)) * HOP / SR, active.astype(float), left=0, right=0) > .5
        p = np.mean(np.abs(z[:, am]) ** 2, axis=1)
        axs[0].plot(f, 10 * np.log10(np.maximum(p, 1e-16)), label=name, color=colors[name], lw=1.25)
        e = env(y)
        axs[1].plot(np.arange(len(e)) * HOP / SR, 20 * np.log10(np.maximum(e / target, 1e-9)), label=name, color=colors[name], lw=1.15)
    axs[0].set(xlim=(60, 14000), ylabel="Active PSD (dB)", title="Model-family screen — descriptive active spectrum")
    axs[0].grid(alpha=.25); axs[0].legend(fontsize=8, ncol=2)
    axs[1].axvspan(*TAIL, color="gold", alpha=.2)
    axs[1].set(xlim=(5.2, n / SR), ylim=(-90, 5), xlabel="Time (s)", ylabel="RMS / active RMS (dB)", title="Envelope and single short terminal window")
    axs[1].grid(alpha=.25); axs[1].legend(fontsize=8, ncol=2)
    fig.savefig(OUT / "model-family-comparison.png", dpi=170); plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
