#!/usr/bin/env python3
"""Screen gentle output-level and peak-control variants on existing WAV renders.

Requires ffmpeg, numpy, and matplotlib. Writes PCM24 audition files and a
CSV/PNG report. The compressor is deliberately conservative and operates on
the already-enhanced signal; it does not alter denoising or spectral balance.
"""
from __future__ import annotations

import argparse
import csv
import math
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def decode(path: Path) -> tuple[np.ndarray, int]:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=sample_rate,channels", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout.strip().split(",")
    rate, channels = int(probe[0]), int(probe[1])
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-acodec",
         "pcm_f32le", "-ac", str(channels), "-ar", str(rate), "pipe:1"],
        check=True, capture_output=True,
    ).stdout
    samples = np.frombuffer(raw, dtype="<f4").reshape(-1, channels).astype(np.float64)
    return samples, rate


def write(path: Path, audio: np.ndarray, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(audio, -1.0, 1.0).astype("<f4").tobytes()
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(rate),
         "-ac", str(audio.shape[1]), "-i", "pipe:0", "-c:a", "pcm_s24le", str(path)],
        input=pcm, check=True,
    )


def smooth_peak_control(x: np.ndarray, rate: int, threshold_db: float = -15.0,
                        ratio: float = 1.5, attack_ms: float = 8.0,
                        release_ms: float = 100.0) -> tuple[np.ndarray, np.ndarray]:
    """Gentle feed-forward compressor, RMS detector with attack/release smoothing."""
    mono = np.mean(x, axis=1)
    win = max(1, round(rate * 0.020))
    padded = np.pad(mono * mono, (win - 1, 0))
    energy = np.convolve(padded, np.ones(win) / win, mode="valid")
    level_db = 10.0 * np.log10(np.maximum(energy, 1e-12))
    over = np.maximum(level_db - threshold_db, 0.0)
    desired_db = -over * (1.0 - 1.0 / ratio)
    attack = math.exp(-1.0 / max(1.0, rate * attack_ms / 1000.0))
    release = math.exp(-1.0 / max(1.0, rate * release_ms / 1000.0))
    gain_db = np.empty_like(desired_db)
    state = 0.0
    for i, target in enumerate(desired_db):
        coeff = attack if target < state else release
        state = coeff * state + (1.0 - coeff) * target
        gain_db[i] = state
    y = x * np.power(10.0, gain_db[:, None] / 20.0)
    return y, gain_db


def measures(x: np.ndarray, rate: int) -> dict[str, float]:
    mono = np.mean(x, axis=1)
    peak = float(np.max(np.abs(mono)))
    rms = float(np.sqrt(np.mean(mono * mono)))
    n = max(1, round(rate * 0.020))
    count = len(mono) // n
    frame_rms = np.sqrt(np.mean(mono[:count * n].reshape(count, n) ** 2, axis=1))
    db = 20 * np.log10(np.maximum(frame_rms, 1e-12))
    return {
        "rms_dbfs": 20 * math.log10(max(rms, 1e-12)),
        "sample_peak_dbfs": 20 * math.log10(max(peak, 1e-12)),
        "crest_db": 20 * math.log10(max(peak, 1e-12) / max(rms, 1e-12)),
        "rms20_p50_dbfs": float(np.percentile(db, 50)),
        "rms20_p90_dbfs": float(np.percentile(db, 90)),
        "rms20_p99_dbfs": float(np.percentile(db, 99)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", action="append", required=True, help="input WAV; repeatable")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--reference-adobe", help="optional same-source Adobe WAV for the first input")
    a = p.parse_args()
    out = Path(a.output_dir)
    rows = []
    plot_data = []
    adobe = decode(Path(a.reference_adobe)) if a.reference_adobe else None
    for name in a.input:
        src = Path(name)
        x, sr = decode(src)
        variants: list[tuple[str, np.ndarray, np.ndarray | None, float]] = [
            ("A-original", x, None, 0.0),
            ("B-minus1.5dB", x, None, -1.5),
        ]
        compressed, gain_db = smooth_peak_control(x, sr)
        variants.append(("C-soft-1.5to1", compressed, gain_db, -1.0))
        for label, audio, gd, master_gain_db in variants:
            audio = audio * 10 ** (master_gain_db / 20)
            dest = out / "audio" / f"{src.stem}-{label}.wav"
            write(dest, audio, sr)
            row = {"sample": src.stem, "variant": label, "master_gain_db": master_gain_db, "path": str(dest)}
            row.update(measures(audio, sr))
            if gd is not None:
                row["max_compressor_reduction_db"] = float(np.min(gd))
                row["mean_compressor_reduction_db"] = float(np.mean(gd))
            rows.append(row)
            plot_data.append((src.stem, label, audio, sr))
        if adobe is not None and src == Path(a.input[0]):
            adobe_audio, adobe_sr = adobe
            if adobe_sr != sr or len(adobe_audio) != len(x):
                raise ValueError("Adobe reference must have the same sample rate and duration as first input")
            row = {"sample": src.stem, "variant": "Adobe-Podcast-v2-reference", "path": a.reference_adobe}
            row.update(measures(adobe_audio, adobe_sr))
            rows.append(row)
            plot_data.append((src.stem, "Adobe-Podcast-v2-reference", adobe_audio, adobe_sr))
    out.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row})
    with (out / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(len(a.input), 2, figsize=(13, 3.7 * len(a.input)), squeeze=False)
    colors = {"A-original": "#777777", "B-minus1.5dB": "#2374ab", "C-soft-1.5to1": "#d1495b", "Adobe-Podcast-v2-reference": "#111111"}
    for i, src in enumerate(a.input):
        stem = Path(src).stem
        group = [(label, audio, sr) for s, label, audio, sr in plot_data if s == stem]
        for label, audio, sr in group:
            mono = np.mean(audio, axis=1)
            t = np.arange(len(mono)) / sr
            stride = max(1, round(sr / 2000))
            axes[i, 0].plot(t[::stride], mono[::stride], lw=.45, alpha=.8,
                            color=colors[label], label=label)
            win = round(sr * .020)
            n = len(mono) // win
            env = np.sqrt(np.mean(mono[:n * win].reshape(n, win) ** 2, axis=1))
            axes[i, 1].plot((np.arange(n) + .5) * .020,
                            20 * np.log10(np.maximum(env, 1e-12)),
                            color=colors[label], label=label)
        axes[i, 0].set_title(f"{stem}: forme d'onde")
        axes[i, 0].set_ylabel("amplitude")
        axes[i, 1].set_title("Enveloppe RMS, fenêtres de 20 ms")
        axes[i, 1].set_ylabel("dBFS")
        axes[i, 1].set_ylim(-90, 0)
        for ax in axes[i]:
            ax.set_xlabel("temps (s)")
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "waveform-envelope.png", dpi=160)


if __name__ == "__main__":
    main()
