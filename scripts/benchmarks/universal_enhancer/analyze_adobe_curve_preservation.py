#!/usr/bin/env python3
"""Check speech level/envelope/tail tradeoffs for the fitted Adobe-style EQ.

Reads six existing NFE64-C baseline/EQ/Adobe pairs and dry speech targets.
No enhancement inference or audio modification is performed.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import stft
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
PAIR = ROOT / "results/noise-rir-truth-01/rir-tail-truth-02"
OUT = ROOT / "results/adobe-curve-eq-2026-10-09/preservation"
RATE = 48_000
FRAME = 960  # 20 ms


def read(path: Path) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float64", always_2d=True)
    if rate != RATE or audio.shape[1] != 1:
        raise ValueError(f"expected mono {RATE} Hz WAV: {path} ({rate} Hz, {audio.shape})")
    if not np.isfinite(audio).all():
        raise ValueError(f"non-finite audio samples: {path}")
    return audio[:, 0]


def db(value: float) -> float:
    return float(20 * np.log10(max(float(value), 1e-12)))


def frames(x: np.ndarray) -> np.ndarray:
    count = len(x) // FRAME
    return np.sqrt(np.mean(x[:count * FRAME].reshape(count, FRAME) ** 2, axis=1) + 1e-20)


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spectral_curve(x: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    freq, times, z = stft(x, fs=RATE, window="hann", nperseg=2048,
                          noverlap=2048 - FRAME, boundary="zeros", padded=True)
    indices = np.clip(np.rint(times / (FRAME / RATE)).astype(int), 0, len(mask) - 1)
    keep = mask[indices]
    power = np.mean(np.abs(z[:, keep]) ** 2, axis=1)
    smooth = np.empty_like(power)
    bin_hz = RATE / 2048
    for i, hz in enumerate(freq):
        width_hz = max(hz, 80.0) * (2 ** (1 / 12) - 2 ** (-1 / 12))
        width = max(1, round(width_hz / bin_hz))
        smooth[i] = np.mean(power[max(0, i-width):min(len(power), i+width+1)])
    return freq, 10 * np.log10(smooth + 1e-20)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    rows: list[dict] = []
    for event in ("christiane-event-1", "christiane-event-2", "naf-event-2"):
        dry = read(PAIR / "inputs" / f"{event}-dry-control.wav")
        dry_env = frames(dry)
        time = (np.arange(len(dry_env)) + .5) * FRAME / RATE
        speech = (time < 1.45) & (dry_env >= np.max(dry_env) * 10 ** (-35 / 20))
        weak = speech & (dry_env <= np.percentile(dry_env[speech], 25))
        for condition in ("dry-control", "rir-only"):
            case = f"{event}-{condition}"
            adobe = read(ROOT / "results/noise-rir-truth-01" / f"adobe-pair-{case}" / "adobe-v2-output.wav")
            baseline = read(OUT.parent / f"{case}-baseline.wav")
            eq = read(OUT.parent / f"{case}-curve-eq.wav")
            x = read(PAIR / "inputs" / f"{case}.wav")
            n = min(map(len, (dry, adobe, baseline, eq, x)))
            ref = dry[:n]
            ref_env = frames(ref)
            signals = {"Adobe v2": adobe[:n], "NFE64-C baseline": baseline[:n], "NFE64-C + fitted EQ": eq[:n]}
            active = speech[:len(ref_env)]
            low = weak[:len(ref_env)]
            dry_speech_rms = float(np.median(ref_env[active]))
            # Use the exact source-derived active mask and fixed dry RMS gain for spectra.
            freq, _ = spectral_curve(ref, active)
            use = (freq >= 80) & (freq <= 12_000)
            adobe_env = frames(signals["Adobe v2"])
            adobe_gain = dry_speech_rms / max(float(np.median(adobe_env[active])), 1e-12)
            _, adobe_curve = spectral_curve(signals["Adobe v2"] * adobe_gain, active)
            for label, signal in signals.items():
                env = frames(signal)
                env = env[:len(ref_env)]
                active_env_corr = corr(np.log(ref_env[active] + 1e-12), np.log(env[active] + 1e-12))
                weak_delta = 20 * np.log10((env[low] + 1e-12) / (ref_env[low] + 1e-12))
                active_delta = 20 * np.log10((env[active] + 1e-12) / (ref_env[active] + 1e-12))
                signal_rms = float(np.median(env[active]))
                fixed = signal * (dry_speech_rms / max(signal_rms, 1e-12))
                _, curve = spectral_curve(fixed, active)
                spectrum_mae = float(np.mean(np.abs(curve[use] - adobe_curve[use])))
                tail_db = None
                if condition == "rir-only":
                    tail_mask = (time[:len(env)] >= 1.80) & (time[:len(env)] < 2.10)
                    tail_energy_db = db(float(np.median(env[tail_mask])))
                    speech_level_db = db(float(np.median(env[active])))
                    tail_db = tail_energy_db - speech_level_db
                rows.append({
                    "event": event,
                    "condition": condition,
                    "candidate": label,
                    "duration_s": n / RATE,
                    "active_frame_count": int(active.sum()),
                    "weak_frame_count": int(low.sum()),
                    "active_envelope_corr_vs_dry": active_env_corr,
                    "active_frame_level_delta_median_db": float(np.median(active_delta)),
                    "active_frame_level_delta_p10_db": float(np.percentile(active_delta, 10)),
                    "weak_frame_level_delta_median_db": float(np.median(weak_delta)) if len(weak_delta) else None,
                    "weak_frame_level_delta_p10_db": float(np.percentile(weak_delta, 10)) if len(weak_delta) else None,
                    "speech_spectrum_mae_vs_adobe_db_80_12k": spectrum_mae,
                    "tail_300_600ms_db_relative_to_active": tail_db,
                    "sample_peak_dbfs": db(float(np.max(np.abs(signal)))),
                })

    args.out.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with (args.out / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "purpose": "Compare fitted stationary EQ against NFE64-C and Adobe v2 using dry-speech activity/level references.",
        "method": "20 ms RMS frames, speech mask from paired dry stem above peak -35 dB before the known 1.5 s endpoint; weak frames are the bottom quartile within active speech; spectrum uses fixed active RMS gain and 1/6-octave curves.",
        "groups": 3,
        "conditions_per_group": ["dry-control", "rir-only"],
        "rows": len(rows),
        "metrics_csv": str((args.out / "metrics.csv").relative_to(ROOT)),
        "limitations": [
            "Three events cover only two voices and synthetic room responses.",
            "Frame-level energy and envelope are not phoneme recognition or subjective quality.",
            "Adobe is a reference render, not clean ground truth; the dry speech stem is the preservation reference.",
            "Spectral proximity is reported beside speech and tail measurements and must not be optimized alone.",
        ],
    }
    (args.out / "report.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    cases = [(event, condition) for event in ("christiane-event-1", "christiane-event-2", "naf-event-2")
             for condition in ("dry-control", "rir-only")]
    labels = [f"{event.replace('-event-', ' ')} {condition.replace('-control', '').replace('-only', '')}"
              for event, condition in cases]
    candidates = ("Adobe v2", "NFE64-C baseline", "NFE64-C + fitted EQ")
    colors = {"Adobe v2": "#111827", "NFE64-C baseline": "#e76f51", "NFE64-C + fitted EQ": "#2a9d8f"}
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), constrained_layout=True)
    xloc = np.arange(len(cases)); width = .24
    descriptors = (
        ("speech_spectrum_mae_vs_adobe_db_80_12k", "Active speech spectrum MAE vs Adobe (dB; lower is closer)"),
        ("active_envelope_corr_vs_dry", "20 ms speech-envelope correlation vs dry speech"),
        ("active_frame_level_delta_p10_db", "Speech-frame level delta p10 vs dry target (dB)"),
    )
    for axis, (key, ylabel) in zip(axes, descriptors):
        for offset, candidate in enumerate(candidates):
            values = []
            for event, condition in cases:
                match = next(row for row in rows if row["event"] == event and row["condition"] == condition and row["candidate"] == candidate)
                values.append(match[key])
            axis.bar(xloc + (offset - 1) * width, values, width, label=candidate, color=colors[candidate])
        axis.set_ylabel(ylabel); axis.set_xticks(xloc, labels, rotation=15, ha="right")
        axis.grid(axis="y", alpha=.25); axis.legend(ncol=3, fontsize=8)
    axes[1].set_ylim(0, 1.02)
    axes[2].axhline(0, color="black", linewidth=.8)
    fig.suptitle("Adobe-derived static EQ: spectral proximity and speech-preservation tradeoffs")
    fig.savefig(args.out / "weak-speech-tradeoff.png", dpi=170)
    plt.close(fig)
    summary["figure"] = str((args.out / "weak-speech-tradeoff.png").relative_to(ROOT))
    (args.out / "report.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"rows={len(rows)} output={args.out.relative_to(ROOT)}")
    for condition in ("dry-control", "rir-only"):
        for candidate in ("Adobe v2", "NFE64-C baseline", "NFE64-C + fitted EQ"):
            group = [r for r in rows if r["condition"] == condition and r["candidate"] == candidate]
            print(condition, candidate,
                  "median envelope r", round(float(np.median([r["active_envelope_corr_vs_dry"] for r in group])), 3),
                  "weak p10 dB", round(float(np.median([r["weak_frame_level_delta_p10_db"] for r in group])), 2),
                  "spectrum MAE", round(float(np.median([r["speech_spectrum_mae_vs_adobe_db_80_12k"] for r in group])), 2))


if __name__ == "__main__":
    main()
