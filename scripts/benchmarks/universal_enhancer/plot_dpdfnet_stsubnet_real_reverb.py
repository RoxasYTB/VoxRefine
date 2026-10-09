#!/usr/bin/env python3
"""Plot descriptive comparisons for the REVERB / STSubNet screen."""
from __future__ import annotations

import csv
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, stft

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / ".tools/stsubnet-screen/fullband/reverb_challenge"
OUT_BASE = ROOT / "results"
SR = 16_000


def get_signals(st_path: Path, out: Path, model: str):
    key = st_path.name.removesuffix("_stsubnet.wav")
    x, _ = sf.read(st_path.with_name(key + ".wav"), dtype="float32")
    st, _ = sf.read(st_path, dtype="float32")
    dp, _ = sf.read(out / model / f"{key}-{model}.wav", dtype="float32")
    return key, (x, resample_poly(st, 1, 3), resample_poly(dp, 1, 3))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("dpdfnet2", "dpdfnet8"), default="dpdfnet2")
    args = parser.parse_args()
    out = OUT_BASE / ("stsubnet-real-reverb-dpdfnet-compare-2026-10-09" if args.model == "dpdfnet2"
                       else f"stsubnet-real-reverb-{args.model}-compare-2026-10-09")
    paths = sorted(SOURCE.glob("*/**/*_stsubnet.wav"))
    # Choose a near-room example and a longer far-room example.
    chosen = [next(p for p in paths if "T21c020i_stsubnet" in p.name),
              next(p for p in paths if "T21c020u_stsubnet" in p.name)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), constrained_layout=True)
    names = ["Entrée réverbérée", "STSubNet publié", f"{args.model.upper()} VoxRefine"]
    for row, path in enumerate(chosen):
        key, signals = get_signals(path, out, args.model)
        for col, (name, y) in enumerate(zip(names, signals)):
            f, t, z = stft(y, fs=SR, nperseg=512, noverlap=384,
                           boundary=None, padded=False)
            db = 20 * np.log10(np.abs(z) + 1e-7)
            axes[row, col].pcolormesh(t, f, db, shading="auto", cmap="magma", vmin=-95, vmax=-25)
            axes[row, col].set_ylim(0, 8000)
            axes[row, col].set_title(f"{key}\n{name}")
            axes[row, col].set_xlabel("Temps (s)")
            if col == 0:
                axes[row, col].set_ylabel("Fréquence (Hz)")
    fig.colorbar(axes[0, 0].collections[0], ax=axes, label="Magnitude STFT (dB)", shrink=.8)
    fig.savefig(out / f"real-reverb-spectrograms-{args.model}.png", dpi=170)
    plt.close(fig)

    rows = list(csv.DictReader((out / "metrics.csv").open()))
    labels = ["Entrée", "STSubNet", args.model.upper()]
    keys = ["input_rms_dbfs", "stsubnet_rms_dbfs", f"{args.model}_rms_dbfs"]
    levels = [[float(r[k]) for r in rows] for k in keys]
    keys = ["input_frame_envelope_std_db", "stsubnet_frame_envelope_std_db", f"{args.model}_frame_envelope_std_db"]
    env = [[float(r[k]) for r in rows] for k in keys]
    keys = ["input_low_energy_frame_fraction", "stsubnet_low_energy_frame_fraction", f"{args.model}_low_energy_frame_fraction"]
    low = [[float(r[k]) for r in rows] for k in keys]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, data, title, ylabel in zip(
            axes, (levels, env, low),
            ("Niveau RMS", "Écart-type enveloppe 20 ms", "Trames faibles (< seuil relatif)"),
            ("dBFS", "dB", "Fraction")):
        ax.boxplot(data, tick_labels=labels, showmeans=True)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=.25)
    fig.suptitle("10 exemples REVERB réels — indicateurs descriptifs, sans référence sèche")
    fig.savefig(out / f"real-reverb-summary-{args.model}.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
