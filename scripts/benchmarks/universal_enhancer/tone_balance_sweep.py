#!/usr/bin/env python3
"""Measure gentle low/high shelf candidates against a same-source Adobe render.

This is an offline diagnostic. It uses known dry-stem activity for masking,
fixed LUFS matching for spectral-shape comparisons, and exports audition files
at matched level. No candidate is silently made the product default.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import scipy
from scipy.signal import resample_poly, stft

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from voxrefine.toneshape import apply_shelves

RUN = ROOT / "results/resemble-enhance-01/controlled-challenger-03"
RATE = 48_000
BANDS = ((20, 80), (80, 250), (250, 1_000), (1_000, 4_000), (4_000, 8_000), (8_000, 12_000))


def read(path: Path) -> np.ndarray:
    x, rate = sf.read(path, dtype="float64", always_2d=True)
    if rate != RATE:
        raise ValueError(f"{path}: expected {RATE} Hz, got {rate}")
    if x.shape[1] != 1:
        raise ValueError(f"{path}: expected mono, got {x.shape[1]} channels")
    return x[:, 0]


def shelf(x: np.ndarray, low_db: float = 0.0, low_hz: float = 200.0,
          high_db: float = 0.0, high_hz: float = 3_500.0) -> np.ndarray:
    return apply_shelves(x, RATE, bass_db=low_db, bass_corner_hz=low_hz,
                         treble_db=high_db, treble_corner_hz=high_hz)


def frame_rms(x: np.ndarray, size: int = 480) -> np.ndarray:
    n = len(x) // size
    blocks = x[:n * size].reshape(n, size)
    return np.sqrt(np.mean(blocks * blocks, axis=1) + 1e-20)


def db(v: float) -> float:
    return float(20.0 * np.log10(max(float(v), 1e-12)))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def spectral_curve(x: np.ndarray, speech_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    freqs, times, z = stft(x, fs=RATE, window="hann", nperseg=2048,
                            noverlap=2048 - 480, boundary="zeros", padded=True)
    frame_index = np.clip(np.rint(times / .01).astype(int), 0, len(speech_mask) - 1)
    keep = speech_mask[frame_index]
    if not np.any(keep):
        raise ValueError("activity mask selected no STFT frames")
    power = np.mean(np.abs(z[:, keep]) ** 2, axis=1)
    smooth = np.empty_like(power)
    for i, hz in enumerate(freqs):
        width_hz = max(hz, 80.0) * (2 ** (1 / 12) - 2 ** (-1 / 12))
        width = max(1, round(width_hz / (RATE / 2048)))
        smooth[i] = np.mean(power[max(0, i - width):min(len(power), i + width + 1)])
    return freqs, 10 * np.log10(smooth + 1e-20)


def macroband(x: np.ndarray, speech_mask: np.ndarray) -> dict[str, float]:
    freqs, times, z = stft(x, fs=RATE, window="hann", nperseg=2048,
                            noverlap=2048 - 480, boundary="zeros", padded=True)
    frame_index = np.clip(np.rint(times / .01).astype(int), 0, len(speech_mask) - 1)
    power = np.mean(np.abs(z[:, speech_mask[frame_index]]) ** 2, axis=1)
    result = {}
    for lo, hi in BANDS:
        take = (freqs >= lo) & (freqs < hi)
        result[f"{lo}-{hi}Hz_db"] = float(10 * np.log10(np.mean(power[take]) + 1e-20))
    return result


def metrics(x: np.ndarray, speech: np.ndarray, gap: np.ndarray, ref_curve: np.ndarray,
            adobe_bands: dict[str, float], dry_lufs: float, source_name: str,
            variant: str, gain_db: float) -> dict[str, float | str]:
    rms_frames = frame_rms(x)
    speech_rms = float(np.median(rms_frames[speech]))
    gap_rms = float(np.median(rms_frames[gap]))
    peak = float(np.max(np.abs(x)))
    true_peak = float(np.max(np.abs(resample_poly(x, 4, 1))))
    lufs = float(pyln.Meter(RATE).integrated_loudness(x))
    _, curve = spectral_curve(x, speech)
    take = (np.fft.rfftfreq(2048, d=1 / RATE) >= 20) & (np.fft.rfftfreq(2048, d=1 / RATE) <= 12_000)
    bands = macroband(x, speech)
    row: dict[str, float | str] = {
        "source": source_name,
        "variant": variant,
        "lufs_i_db": lufs,
        "gain_to_dry_lufs_db": gain_db,
        "sample_peak_dbfs": db(peak),
        "true_peak_4x_dbfs": db(true_peak),
        "crest_factor_db": db(peak / max(float(np.sqrt(np.mean(x * x))), 1e-12)),
        "speech_rms_median_dbfs": db(speech_rms),
        "gap_rms_median_dbfs": db(gap_rms),
        "speech_gap_contrast_db": db(speech_rms / max(gap_rms, 1e-12)),
        "spectrum_mae_vs_adobe_db_20_12k": float(np.mean(np.abs(curve[take] - ref_curve[take]))),
    }
    for band, value in bands.items():
        row[band] = value
        row[f"adobe_delta_{band}"] = value - adobe_bands[band]
    return row


def make_current_sample_screen(sample_paths: list[Path], output_dir: Path) -> list[dict]:
    """Apply the Adobe-inspired sub-bass/treble profile to current NFE64-C renders."""
    rows: list[dict] = []
    plot_items: list[tuple[str, dict[str, tuple[np.ndarray, np.ndarray]]]] = []
    for source_path in sample_paths:
        base = read(source_path)
        sample_id = source_path.stem.removesuffix("-resemble-nfe64-C")
        clean_path = ROOT / "corpus/samples/clear-noisy-mix-01/clean" / f"{sample_id}-clean-reference.wav"
        clean = read(clean_path)
        if len(clean) != len(base):
            raise ValueError(f"Paired clean reference length mismatch for {sample_id}")
        clean_frames = frame_rms(clean)
        speech = clean_frames >= np.max(clean_frames) * 10 ** (-35 / 20)
        pad = 6
        expanded = np.pad(speech.astype(np.int8), pad)
        speech = np.convolve(expanded, np.ones(2 * pad + 1, dtype=np.int32), mode="same")[pad:-pad] > 0
        gap = ~speech
        clean_active_rms = float(np.median(clean_frames[speech]))
        clean_freq, clean_curve = spectral_curve(clean, speech)
        clean_bands = macroband(clean, speech)
        options = [
            ("A-original-NFE64-C", base),
            ("B-soft-edges-subbass-minus3-treble-minus2p5", shelf(base, low_db=-3.0,
                low_hz=100.0, high_db=-2.5, high_hz=3_500.0)),
            ("C-treble-minus2p5-only", shelf(base, high_db=-2.5, high_hz=3_500.0)),
        ]
        sample_curves: dict[str, tuple[np.ndarray, np.ndarray]] = {"Clean reference": (clean_freq, clean_curve)}
        for name, audio in options:
            rms = frame_rms(audio)
            active_rms = float(np.median(rms[speech]))
            match_gain = db(clean_active_rms / max(active_rms, 1e-12))
            matched = audio * 10 ** (match_gain / 20)
            freq, curve = spectral_curve(matched, speech)
            sample_curves[name] = (freq, curve)
            bands = macroband(matched, speech)
            rms_full = float(np.sqrt(np.mean(audio * audio)))
            peak = float(np.max(np.abs(audio)))
            true_peak = float(np.max(np.abs(resample_poly(audio, 4, 1))))
            gap_rms = float(np.median(rms[gap])) if np.any(gap) else None
            row: dict[str, float | str] = {
                "sample": sample_id, "variant": name, "duration_s": len(audio) / RATE,
                "lufs_i_db": float(pyln.Meter(RATE).integrated_loudness(audio)),
                "rms_dbfs": db(rms_full), "sample_peak_dbfs": db(peak),
                "true_peak_4x_dbfs": db(true_peak),
                "crest_factor_db": db(peak / max(rms_full, 1e-12)),
                "speech_rms_median_dbfs": db(active_rms),
                "gap_rms_median_dbfs": db(gap_rms) if gap_rms is not None else None,
                "speech_gap_contrast_db": db(active_rms / max(gap_rms, 1e-12)) if gap_rms is not None else None,
                "speech_gain_to_clean_reference_db": match_gain,
                "spectrum_mae_vs_clean_equal_speech_rms_db_20_12k": float(np.mean(np.abs(curve - clean_curve))),
            }
            for band, value in bands.items():
                row[band] = value
                row[f"clean_delta_{band}"] = value - clean_bands[band]
            rows.append(row)

            sample_out = output_dir / sample_id
            sample_out.mkdir(parents=True, exist_ok=True)
            slug = name.lower()
            wav = sample_out / f"{slug}.wav"
            mp3 = sample_out / f"{slug}.mp3"
            sf.write(wav, audio, RATE, subtype="PCM_24")
            subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                            "-i", str(wav), "-codec:a", "libmp3lame", "-b:a", "192k", str(mp3)], check=True)
        plot_items.append((sample_id, sample_curves))
    if plot_items:
        fig, axes = plt.subplots(len(plot_items), 2, figsize=(13, 3.7 * len(plot_items)),
                                 squeeze=False, constrained_layout=True)
        colors = {"Clean reference": "#111827", "A-original-NFE64-C": "#64748b",
                  "B-soft-edges-subbass-minus3-treble-minus2p5": "#d1495b",
                  "C-treble-minus2p5-only": "#2374ab"}
        for i, (sample_id, curves_for_sample) in enumerate(plot_items):
            ref = curves_for_sample["Clean reference"][1]
            for label, (freqs, curve) in curves_for_sample.items():
                axes[i, 0].plot(freqs, curve, label=label, color=colors[label], linewidth=1.3)
                if label != "Clean reference":
                    axes[i, 1].plot(freqs, curve - ref, label=label, color=colors[label], linewidth=1.4)
            axes[i, 0].set_title(f"{sample_id}: speech-active spectrum")
            axes[i, 0].set_ylabel("dB, speech RMS matched")
            axes[i, 1].axhline(0, color="black", linewidth=.8)
            axes[i, 1].set_title("Difference vs clean reference spectrum")
            axes[i, 1].set_ylabel("dB")
            axes[i, 1].set_ylim(-12, 12)
            for axis in axes[i]:
                axis.set_xscale("log"); axis.set_xlim(20, 12_000); axis.grid(True, which="both", alpha=.2)
                axis.set_xlabel("Fréquence (Hz)"); axis.legend(fontsize=7)
        fig.savefig(output_dir / "current-sample-tone-spectra.png", dpi=160)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=RUN)
    ap.add_argument("--output-dir", type=Path, default=ROOT / "results/tone-balance-adobe-01")
    ap.add_argument("--current-sample", type=Path, action="append", default=[],
                    help="optional current NFE64-C render; repeatable; writes A/B/C listening derivatives")
    args = ap.parse_args()
    run, out = args.run_dir.resolve(), args.output_dir.resolve()
    audio_dir = run / "audio"
    paths = {
        "dry": audio_dir / "noise-snr10-adobe-dry.wav",
        "input": audio_dir / "noise-snr10-adobe-input.wav",
        "base": audio_dir / "noise-snr10-adobe-resemble-nfe16-C.wav",
        "adobe": audio_dir / "adobe-v2-reference.wav",
    }
    x = {key: read(path) for key, path in paths.items()}
    n = min(len(v) for v in x.values())
    x = {key: value[:n] for key, value in x.items()}
    if n < RATE:
        raise ValueError("paired example must be at least one second")

    dry_frames = frame_rms(x["dry"])
    speech = dry_frames >= np.max(dry_frames) * 10 ** (-35 / 20)
    guard = 6
    padded = np.pad(speech.astype(np.int8), guard)
    speech = np.convolve(padded, np.ones(2 * guard + 1, dtype=np.int32), mode="same")[guard:-guard] > 0
    gap = ~speech
    meter = pyln.Meter(RATE)
    dry_lufs = float(meter.integrated_loudness(x["dry"]))
    adobe_lufs = float(meter.integrated_loudness(x["adobe"]))
    adobe_common = x["adobe"] * 10 ** ((dry_lufs - adobe_lufs) / 20)
    adobe_bands = macroband(adobe_common, speech)
    freq, adobe_curve = spectral_curve(adobe_common, speech)

    specs = [
        ("NFE16-C baseline", 0.0, 0.0, 200.0, 3_500.0),
        ("Bass cut -1.5 dB below 200 Hz", -1.5, 0.0, 200.0, 3_500.0),
        ("Sub-bass cut -3 dB below 100 Hz", -3.0, 0.0, 100.0, 3_500.0),
        ("Treble cut -2.5 dB above 3.5 kHz", 0.0, -2.5, 200.0, 3_500.0),
        ("Both cuts: bass -1.5 / treble -2.5 dB", -1.5, -2.5, 200.0, 3_500.0),
        ("Soft edges: sub-bass -3 / treble -2.5 dB", -3.0, -2.5, 100.0, 3_500.0),
        ("Adobe-guided: bass +2 / treble -3 dB", 2.0, -3.0, 200.0, 3_500.0),
    ]
    result_rows = []
    audio_variants: dict[str, np.ndarray] = {"Adobe v2": adobe_common}
    curves: dict[str, np.ndarray] = {"Adobe v2": adobe_curve}
    matched_for_listening: dict[str, np.ndarray] = {}
    base_lufs = float(meter.integrated_loudness(x["base"]))

    for name, low_db, high_db, low_hz, high_hz in specs:
        processed = shelf(x["base"], low_db=low_db, low_hz=low_hz,
                          high_db=high_db, high_hz=high_hz)
        raw_lufs = float(meter.integrated_loudness(processed))
        dry_gain_db = dry_lufs - raw_lufs
        common = processed * 10 ** (dry_gain_db / 20)
        # Every audition item gets the baseline's integrated loudness so EQ is
        # not mistaken for a loudness improvement during subjective comparison.
        listen_gain_db = base_lufs - raw_lufs
        matched_for_listening[name] = processed * 10 ** (listen_gain_db / 20)
        audio_variants[name] = common
        _, curve = spectral_curve(common, speech)
        curves[name] = curve
        result_rows.append(metrics(common, speech, gap, adobe_curve, adobe_bands,
                                   dry_lufs, "controlled-noise-SNR10", name, dry_gain_db))

    result_rows.append(metrics(adobe_common, speech, gap, adobe_curve, adobe_bands,
                               dry_lufs, "controlled-noise-SNR10", "Adobe v2 reference", dry_lufs - adobe_lufs))
    current_rows = make_current_sample_screen(args.current_sample, out / "current-samples") if args.current_sample else []
    out.mkdir(parents=True, exist_ok=True)
    audio_out = out / "listening"
    audio_out.mkdir(exist_ok=True)
    for name, sig in matched_for_listening.items():
        slug = name.lower().replace(" ", "-").replace("/", "-").replace(":", "")
        wav = audio_out / f"{slug}.wav"
        mp3 = audio_out / f"{slug}.mp3"
        sf.write(wav, sig, RATE, subtype="PCM_24")
        subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                        "-i", str(wav), "-codec:a", "libmp3lame", "-b:a", "192k", str(mp3)], check=True)
    adobe_listen_wav = audio_out / "adobe-v2-reference.wav"
    sf.write(adobe_listen_wav, adobe_common * 10 ** ((base_lufs - dry_lufs) / 20), RATE, subtype="PCM_24")
    subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(adobe_listen_wav), "-codec:a", "libmp3lame", "-b:a", "192k",
                    str(audio_out / "adobe-v2-reference.mp3")], check=True)

    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        keys = list(dict.fromkeys(k for row in result_rows for k in row))
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(result_rows)
    report = {
        "protocol": "6 s exact same-source controlled-noise pair; activity from known dry stem with +/-60 ms guard; each spectrum fixed-gain matched to dry integrated LUFS; listening renders matched to NFE16-C integrated LUFS",
        "sample_rate_hz": RATE,
        "duration_seconds": n / RATE,
        "input_sha256": {key: sha256_file(path) for key, path in paths.items()},
        "source_analysis_report_sha256": sha256_file(run / "report.json"),
        "software_versions": {"numpy": np.__version__, "scipy": scipy.__version__,
                               "soundfile": sf.__version__, "matplotlib": matplotlib.__version__},
        "dry_reference_lufs_i_db": dry_lufs,
        "adobe_reference_lufs_i_db": adobe_lufs,
        "nfe16_c_baseline_lufs_i_db": base_lufs,
        "speech_frames": int(np.sum(speech)),
        "gap_frames": int(np.sum(gap)),
        "filters": {name: {"low_shelf_db": low, "low_shelf_corner_hz": low_hz,
                            "high_shelf_db": high, "high_shelf_corner_hz": high_hz,
                            "implementation": "causal first-order shelves; no look-ahead"}
                    for name, low, high, low_hz, high_hz in specs},
        "limitations": [
            "one 6 s LibriVox voice, one synthetic classroom-noise mixture at nominal 10 dB SNR, one Adobe version and one slider configuration",
            "broad shelf measurements do not identify narrow resonances or establish listener preference",
            "Adobe is a perceptual reference, not clean ground truth",
            "one-pole shelves are diagnostic candidates; no default product tuning is selected by this single pair",
            "true-peak estimate is 4x oversampling and not a standards certification",
        ],
        "rows": result_rows,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    if current_rows:
        with (out / "current-sample-metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            keys = list(dict.fromkeys(k for row in current_rows for k in row))
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(current_rows)
        report["current_nfe64_c_samples"] = current_rows
        report["current_sample_note"] = "These are later NFE64-C recordings without paired Adobe outputs; they show the same tone filters' level/spectral effect and cannot be scored for Adobe proximity."
        report["current_sample_figure"] = "current-samples/current-sample-tone-spectra.png"
        report["current_sample_input_sha256"] = {
            path.stem: {
                "enhanced": sha256_file(path),
                "clean_reference": sha256_file(ROOT / "corpus/samples/clear-noisy-mix-01/clean" /
                                                 f"{path.stem.removesuffix('-resemble-nfe64-C')}-clean-reference.wav"),
            } for path in args.current_sample
        }
        (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")

    labels = [name for name, *_ in specs] + ["Adobe v2"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), constrained_layout=True)
    colors = ["#6b7280", "#f59e0b", "#06b6d4", "#ef4444", "#8b5cf6", "#84cc16", "#10b981", "#111827"]
    for label, color in zip(labels, colors):
        axes[0].plot(freq, curves[label], label=label, color=color, linewidth=1.6)
    axes[0].set_xscale("log"); axes[0].set_xlim(20, 12_000); axes[0].set_ylim(-115, -20)
    axes[0].set_ylabel("Power (dB, fixed-LUFS shape)"); axes[0].set_title("Active-speech spectrum: Resemble tone variants vs Adobe v2")
    axes[0].grid(True, which="both", alpha=.2); axes[0].legend(fontsize=8, ncol=2)
    for label, color in zip(labels[:-1], colors[:-1]):
        axes[1].plot(freq, curves[label] - adobe_curve, label=label, color=color, linewidth=1.4)
    axes[1].axhline(0, color="black", linewidth=.9); axes[1].set_xscale("log"); axes[1].set_xlim(20, 12_000)
    axes[1].set_ylim(-12, 12); axes[1].set_ylabel("Écart à Adobe (dB)"); axes[1].set_xlabel("Fréquence (Hz)")
    axes[1].set_title("Distance spectrale après gain constant — descripteur, pas score de qualité")
    axes[1].grid(True, which="both", alpha=.2); axes[1].legend(fontsize=8, ncol=2)
    fig.savefig(out / "spectral-sweep.png", dpi=170)

    bands_labels = [f"{lo}-{hi}" for lo, hi in BANDS]
    fig2, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    xloc = np.arange(len(BANDS)); width = .14
    for j, (row, color) in enumerate(zip(result_rows, colors)):
        deltas = [row[f"adobe_delta_{lo}-{hi}Hz_db"] for lo, hi in BANDS]
        ax.bar(xloc + (j - (len(result_rows) - 1) / 2) * width, deltas, width,
               label=str(row["variant"]), color=color)
    ax.axhline(0, color="black", linewidth=.8); ax.set_xticks(xloc, bands_labels)
    ax.set_ylabel("Candidate − Adobe energy (dB)"); ax.set_xlabel("Speech-active frequency band (Hz)")
    ax.set_title("Macroband differences after fixed-LUFS matching"); ax.grid(axis="y", alpha=.2); ax.legend(fontsize=8)
    fig2.savefig(out / "macroband-deltas.png", dpi=170)
    print(json.dumps({"report": str(out / "report.json"), "metrics": str(out / "metrics.csv"),
                      "figures": [str(out / "spectral-sweep.png"), str(out / "macroband-deltas.png")],
                      "rows": result_rows, "current_sample_rows": current_rows}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
