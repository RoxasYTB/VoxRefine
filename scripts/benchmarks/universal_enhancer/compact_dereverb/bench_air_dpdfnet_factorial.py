#!/usr/bin/env python3
"""Frozen AIR factorial screen for compact dereverb and DPDFNet2.

This is a diagnostic benchmark, not a model-selection loop. It reuses the
measured-RIR preparation and fixed-reference speech/tail metrics from
screen_measured_rirs.py. Input speech and renders stay in the local .tools
directory; the public report contains aggregate measurements only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import butter, fftconvolve, resample_poly, sosfiltfilt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from screen_measured_rirs import (  # noqa: E402
    SR, convolve_pair, db_ratio, infer, load_model, make_inserted_pause_pair,
    pause_metrics, prepare_rir, speech_bounds,
    speech_metrics, rms_frames,
)

DEFAULT_OUTPUT = ROOT / ".tools/compact-dereverb/air-dpdfnet-factorial-2026-10-09"
DEFAULT_SPEECH = ROOT / ".tools/compact-dereverb/data/LibriSpeech"
DEFAULT_RIR = ROOT / ".tools/compact-dereverb/data/air/extracted"
DEFAULT_DEREVERB = ROOT / ".tools/compact-dereverb/training-measured-mix-20261009/checkpoints/best-dev.pt"
DEFAULT_DPDF = ROOT / ".tools/deepvqe-screen/dpdfnet2_48khz_hr.onnx"
DEFAULT_SDK = ROOT / ".tools/deepvqe-screen"
SR48 = 48_000
HOP48 = 480
ALIGNMENT48 = 1_920
NOISE_CONDITIONS = (("fan20", 20.0), ("fan10", 10.0))
ARMS = ("input", "dereverb", "dpdfnet2", "dpdfnet2_then_dereverb")
UTTERANCE_CROP_SAMPLES = 4 * SR
SPEECH_WARMUP_SAMPLES = int(0.20 * SR)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rms(x: np.ndarray) -> float:
    a = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(a * a) + 1e-24)) if a.size else 0.0


def active_sample_indices(clean: np.ndarray) -> np.ndarray:
    frame = 320
    hop = 160
    levels = np.lib.stride_tricks.sliding_window_view(clean, frame)[::hop]
    levels = np.sqrt(np.mean(levels.astype(np.float64) ** 2, axis=1) + 1e-24)
    active = levels >= max(float(levels.max()) * 0.02, 1e-5)
    # Use every other overlapping frame so each selected sample contributes
    # once to the SNR calibration.
    starts = np.arange(levels.size) * hop
    selected = np.flatnonzero(active & ((np.arange(levels.size) % 2) == 0))
    if selected.size == 0:
        raise ValueError("no clean active frames available for SNR calibration")
    return np.concatenate([np.arange(starts[i], min(starts[i] + frame, len(clean)))
                           for i in selected])


def fan_noise(n: int, seed: int) -> np.ndarray:
    """Fixed-seed broadband fan-like noise with weak 120/240 Hz components."""
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n).astype(np.float64)
    sos = butter(4, [90, 5_800], btype="bandpass", fs=SR, output="sos")
    noise = sosfiltfilt(sos, white)
    t = np.arange(n, dtype=np.float64) / SR
    noise += 0.10 * np.sin(2 * np.pi * 120 * t + 0.31)
    noise += 0.035 * np.sin(2 * np.pi * 240 * t + 1.17)
    noise /= max(rms(noise), 1e-12)
    return noise.astype(np.float32)


def mix_at_snr(clean: np.ndarray, wet: np.ndarray, seed: int,
               snr_db: float) -> tuple[np.ndarray, np.ndarray, float]:
    n = min(len(clean), len(wet))
    clean, wet = clean[:n], wet[:n]
    noise_unit = fan_noise(n, seed)
    active = active_sample_indices(clean)
    speech_rms = rms(wet[active])
    noise_rms = rms(noise_unit[active])
    scale = speech_rms / max(noise_rms * (10 ** (snr_db / 20)), 1e-12)
    noise = (noise_unit * scale).astype(np.float32)
    measured_snr = 20 * math.log10(max(speech_rms, 1e-12) /
                                   max(rms(noise[active]), 1e-12))
    return (wet + noise).astype(np.float32), noise, measured_snr


def make_dpdfnet(sdk: Path, model_path: Path):
    sys.path.insert(0, str(sdk))
    from dpdfnet.stream import StreamEnhancer
    from voxrefine.live.backends.dpdfnet import DpdfNetLiveBackend

    model = StreamEnhancer("dpdfnet2_48khz_hr", onnx_path=model_path, verbose=False)
    return DpdfNetLiveBackend(model, alignment_samples=ALIGNMENT48)


def run_dpdfnet(backend, audio: np.ndarray) -> tuple[np.ndarray, float]:
    """Render through the exact local 48 kHz stream adapter and return 16 kHz."""
    started = time.perf_counter()
    x48 = resample_poly(np.asarray(audio, dtype=np.float32), 3, 1).astype(np.float32)
    logical_n = len(x48)
    padded_n = int(math.ceil(logical_n / HOP48) * HOP48)
    x48 = np.pad(x48, (0, padded_n - logical_n)).astype(np.float32, copy=False)
    backend.reset()
    pieces = []
    for start in range(0, len(x48), HOP48):
        out = backend.process(x48[start:start + HOP48])
        if out.size:
            pieces.append(out)
    tail = backend.finalize()
    if tail.size:
        pieces.append(tail)
    y48 = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
    if len(y48) != padded_n:
        raise RuntimeError(f"DPDFNet duration mismatch: expected {padded_n}, got {len(y48)}")
    y16 = resample_poly(y48[:logical_n], 1, 3).astype(np.float32)
    if len(y16) != len(audio):
        raise RuntimeError(f"resampler duration mismatch: expected {len(audio)}, got {len(y16)}")
    return y16, time.perf_counter() - started


def branch_outputs(audio: np.ndarray, *, dereverb_model, device,
                   dpdf_backend) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    result = {"input": audio.astype(np.float32, copy=False)}
    elapsed: dict[str, float] = {}
    result["dereverb"], elapsed["dereverb"] = infer(dereverb_model, audio, device)
    if len(result["dereverb"]) != len(audio):
        raise RuntimeError("compact dereverberator changed the clip duration")
    result["dpdfnet2"], elapsed["dpdfnet2"] = run_dpdfnet(dpdf_backend, audio)
    result["dpdfnet2_then_dereverb"], dereverb_elapsed = infer(
        dereverb_model, result["dpdfnet2"], device)
    elapsed["dpdfnet2_then_dereverb"] = elapsed["dpdfnet2"] + dereverb_elapsed
    if len(result["dpdfnet2_then_dereverb"]) != len(audio):
        raise RuntimeError("dereverb cascade changed the clip duration")
    for key, value in result.items():
        if not np.isfinite(value).all():
            raise RuntimeError(f"{key} produced non-finite samples")
    return result, elapsed


def weak_speech_p10(clean: np.ndarray, output: np.ndarray,
                    pauses: list[dict]) -> float | None:
    clean = clean[SPEECH_WARMUP_SAMPLES:]
    output = output[SPEECH_WARMUP_SAMPLES:]
    pauses = [{**pause,
               "clip_relative_start_sample": pause["clip_relative_start_sample"] - SPEECH_WARMUP_SAMPLES,
               "second_speech_start_sample": pause["second_speech_start_sample"] - SPEECH_WARMUP_SAMPLES}
              for pause in pauses]
    c, y = rms_frames(clean), rms_frames(output)
    n = min(c.size, y.size)
    c, y = c[:n], y[:n]
    active = c >= max(float(c.max()) * 0.02, 1e-5)
    for pause in pauses:
        start = int(pause["clip_relative_start_sample"])
        end = start + int(pause["duration_s"] * SR)
        active[max(0, (start - 320) // 160):min(n, (end + 159) // 160)] = False
    ids = np.flatnonzero(active)
    if not ids.size:
        return None
    weak = ids[np.argsort(c[ids])[:max(1, int(np.ceil(.2 * ids.size)))]]
    gain = 20 * np.log10(np.maximum(y[weak], 1e-10) /
                         np.maximum(c[weak], 1e-10))
    return float(np.percentile(gain, 10))


def steady_speech_metrics(clean: np.ndarray, wet: np.ndarray, output: np.ndarray,
                          pauses: list[dict]) -> dict:
    """Exclude the first 200 ms to avoid scoring model startup transients."""
    start = SPEECH_WARMUP_SAMPLES
    adjusted_pauses = [{**pause,
                        "clip_relative_start_sample": pause["clip_relative_start_sample"] - start,
                        "second_speech_start_sample": pause["second_speech_start_sample"] - start}
                       for pause in pauses]
    metrics = speech_metrics(clean[start:], wet[start:], output[start:], adjusted_pauses)
    metrics["startup_excluded_seconds"] = start / SR
    metrics["weak_speech_gain_db_p10"] = weak_speech_p10(clean, output, pauses)
    return metrics


def add_floor_metrics(rows: list[dict], clean: np.ndarray, wet: np.ndarray,
                      noises: dict[str, dict[str, np.ndarray]],
                      dry_outputs: dict[str, np.ndarray], pauses: list[dict]) -> None:
    for row in rows:
        arm = row["arm"]
        output = row.pop("_output")
        measured = pause_metrics(clean, wet, output, dry_outputs[arm], pauses)
        if row["condition"] == "rir_only":
            noise_floor_rows = []
        else:
            noise_floor_rows = pause_metrics(clean, wet, noises[row["condition"]][arm],
                                             dry_outputs[arm], pauses)
        by_band = {entry["band_ms"]: entry for entry in noise_floor_rows}
        for metric in measured:
            noise_row = by_band.get(metric["band_ms"])
            noise_floor = (noise_row["output_tail_vs_same_fixed_input_speech_db"]
                           if noise_row else metric["dry_model_floor_vs_fixed_input_speech_db"])
            common_floor = max(metric["dry_model_floor_vs_fixed_input_speech_db"], noise_floor)
            metric["noise_only_output_floor_vs_fixed_input_speech_db"] = noise_floor
            metric["common_output_floor_vs_fixed_input_speech_db"] = common_floor
            metric["output_margin_above_common_floor_db"] = (
                metric["output_tail_vs_same_fixed_input_speech_db"] - common_floor)
            metric["common_floor_censored"] = bool(
                metric["output_tail_vs_same_fixed_input_speech_db"] <= common_floor + 3.0)
        row["tail_metrics"] = measured


def load_air_rirs(root: Path) -> list[dict]:
    manifest = json.loads((root / "air_subset_manifest.json").read_text())
    rows = []
    for row in manifest["rirs"]:
        item = dict(row)
        item.update({"source_url": manifest["source_page"],
                     "dataset_version": "Aachen Impulse Response Database AIR v1.4",
                     "license": manifest["license"], "attribution": manifest["attribution"]})
        rows.append(item)
    if len(rows) != 8:
        raise RuntimeError(f"expected the frozen eight AIR RIRs; found {len(rows)}")
    return rows


def bootstrap_speaker_median(values: list[tuple[str, float]], *, seed: int = 20261009,
                             samples: int = 5000) -> list[float] | None:
    by_speaker: dict[str, list[float]] = {}
    for speaker, value in values:
        by_speaker.setdefault(speaker, []).append(value)
    speaker_values = np.asarray([np.median(x) for x in by_speaker.values()], dtype=np.float64)
    if not speaker_values.size:
        return None
    rng = np.random.default_rng(seed)
    boot = np.median(rng.choice(speaker_values, size=(samples, len(speaker_values)),
                                replace=True), axis=1)
    return [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


def summarize(output_dir: Path, manifest: dict, rows: list[dict]) -> dict:
    indexed = {}
    for row in rows:
        if row["condition"] == "dry_control":
            continue
        tail = {metric["band_ms"]: metric for metric in row["tail_metrics"]}
        indexed[(row["speaker"], row["room"], row["rir"], row["condition"], row["arm"])] = {
            "row": row, "tail": tail}

    comparisons = {}
    for condition in ("rir_only", "fan20", "fan10"):
        for band in ("50_150", "150_300", "300_600"):
            variants = {
                "dereverb_on_input": ("input", "dereverb"),
                "dereverb_after_dpdfnet2": ("dpdfnet2", "dpdfnet2_then_dereverb"),
            }
            for comparison, (base_arm, enhanced_arm) in variants.items():
                paired: list[tuple[str, str, str, float]] = []
                censored_pairs = 0
                for key in sorted({(s, r, rir) for s, r, rir, c, a in indexed
                                   if c == condition and a == base_arm}):
                    speaker, room, rir = key
                    base = indexed.get((speaker, room, rir, condition, base_arm))
                    enh = indexed.get((speaker, room, rir, condition, enhanced_arm))
                    if not base or not enh or band not in base["tail"] or band not in enh["tail"]:
                        continue
                    bm, em = base["tail"][band], enh["tail"][band]
                    if bm["common_floor_censored"] or em["common_floor_censored"]:
                        censored_pairs += 1
                        continue
                    gain = (bm["output_tail_vs_same_fixed_input_speech_db"] -
                            em["output_tail_vs_same_fixed_input_speech_db"])
                    paired.append((speaker, room, base["row"]["distance_role"], float(gain)))
                values = [(speaker, gain) for speaker, _, _, gain in paired]
                room_medians = {}
                for room in sorted({room for _, room, _, _ in paired}):
                    room_medians[room] = float(np.median(
                        [gain for _, item_room, _, gain in paired if item_room == room]))
                distance_medians = {}
                for role in ("short_distance", "long_distance"):
                    role_values = [gain for _, _, item_role, gain in paired if item_role == role]
                    distance_medians[role] = float(np.median(role_values)) if role_values else None
                key = f"{condition}/{band}/{comparison}"
                comparisons[key] = {
                    "valid_pair_count": len(paired), "censored_pair_count": censored_pairs,
                    "speaker_count": len({speaker for speaker, _, _, _ in paired}),
                    "median_gain_db": float(np.median([x[3] for x in paired])) if paired else None,
                    "speaker_cluster_bootstrap_95ci_db": bootstrap_speaker_median(values),
                    "positive_location_count_of_four": sum(v > 0 for v in room_medians.values()),
                    "location_medians_db": room_medians,
                    "distance_role_medians_db": distance_medians,
                }

    floor_comparisons = {}
    for condition in ("fan20", "fan10"):
        for band in ("150_300", "300_600"):
            for comparison, enhanced_arm in (("dereverb_on_input", "dereverb"),
                                             ("dpdfnet2_vs_input", "dpdfnet2"),
                                             ("cascade_vs_input", "dpdfnet2_then_dereverb"),
                                             ("dereverb_after_dpdfnet2", "dpdfnet2_then_dereverb")):
                paired_floor = []
                base_arm = "dpdfnet2" if comparison == "dereverb_after_dpdfnet2" else "input"
                for speaker, room, rir, item_condition, arm in sorted(indexed):
                    if item_condition != condition or arm != base_arm:
                        continue
                    base = indexed[(speaker, room, rir, condition, base_arm)]["tail"].get(band)
                    enhanced = indexed.get((speaker, room, rir, condition, enhanced_arm))
                    if not base or not enhanced or band not in enhanced["tail"]:
                        continue
                    base_floor = base.get("noise_only_output_floor_vs_fixed_input_speech_db")
                    enhanced_floor = enhanced["tail"][band].get(
                        "noise_only_output_floor_vs_fixed_input_speech_db")
                    if base_floor is None or enhanced_floor is None:
                        continue
                    # Positive values mean the processed noise-only output
                    # floor is lower than the paired base route.
                    paired_floor.append((speaker, float(base_floor - enhanced_floor)))
                key = f"{condition}/{band}/{comparison}"
                floor_comparisons[key] = {
                    "valid_pair_count": len(paired_floor),
                    "speaker_count": len({speaker for speaker, _ in paired_floor}),
                    "median_floor_reduction_db": (float(np.median([v for _, v in paired_floor]))
                                                   if paired_floor else None),
                    "speaker_cluster_bootstrap_95ci_db": bootstrap_speaker_median(paired_floor),
                }

    dry_by_key = {(row["speaker"], row["room"], row["rir"], row["arm"]): row
                  for row in rows if row["condition"] == "dry_control"}
    speech_deltas = {"active_p50_db": [], "onset_p10_db": [], "weak_p10_db": []}
    for (speaker, room, rir, arm), cascade in dry_by_key.items():
        if arm != "dpdfnet2_then_dereverb":
            continue
        dfn = dry_by_key.get((speaker, room, rir, "dpdfnet2"))
        if not dfn:
            continue
        ca, da = cascade["speech_metrics"], dfn["speech_metrics"]
        speech_deltas["active_p50_db"].append((speaker,
            ca["active_output_vs_clean_db_p10_p50_p90"][1] -
            da["active_output_vs_clean_db_p10_p50_p90"][1]))
        speech_deltas["onset_p10_db"].append((speaker,
            ca["rising_output_vs_clean_db_p10_p50_p90"][0] -
            da["rising_output_vs_clean_db_p10_p50_p90"][0]))
        speech_deltas["weak_p10_db"].append((speaker,
            ca["weak_speech_gain_db_p10"] - da["weak_speech_gain_db_p10"]))
    speech_summary = {name: {"median_db": float(np.median([v for _, v in values])) if values else None,
                             "speaker_cluster_bootstrap_95ci_db": bootstrap_speaker_median(values)}
                      for name, values in speech_deltas.items()}

    gate_results = {}
    for band in ("150_300", "300_600"):
        wet = comparisons[f"rir_only/{band}/dereverb_on_input"]
        cascade = comparisons[f"rir_only/{band}/dereverb_after_dpdfnet2"]
        gate_results[band] = {
            "wet_dereverb_gain_ge_2_db": (wet["median_gain_db"] is not None and
                                           wet["median_gain_db"] >= 2.0),
            "post_dpdfnet_gain_ge_1_5_db": (cascade["median_gain_db"] is not None and
                                             cascade["median_gain_db"] >= 1.5),
            "post_dpdfnet_positive_in_at_least_3_locations":
                cascade["positive_location_count_of_four"] >= 3,
        }
    preserve = {
        "active_speech_gt_minus_1_db": (speech_summary["active_p50_db"]["median_db"] is not None and
                                         speech_summary["active_p50_db"]["median_db"] > -1.0),
        "onset_p10_gt_minus_1_db": (speech_summary["onset_p10_db"]["median_db"] is not None and
                                     speech_summary["onset_p10_db"]["median_db"] > -1.0),
        "weak_speech_p10_gt_minus_1_5_db": (speech_summary["weak_p10_db"]["median_db"] is not None and
                                             speech_summary["weak_p10_db"]["median_db"] > -1.5),
    }
    passed = any(g["wet_dereverb_gain_ge_2_db"] and g["post_dpdfnet_gain_ge_1_5_db"] and
                 g["post_dpdfnet_positive_in_at_least_3_locations"] for g in gate_results.values())
    passed = passed and all(preserve.values())
    snr_summary = {}
    for condition in ("fan20", "fan10"):
        vals = [row["measured_snr_db"] for row in rows
                if row["condition"] == condition and row["arm"] == "input"]
        snr_summary[condition] = {"nominal_db": dict(NOISE_CONDITIONS)[condition],
                                  "median_measured_db": float(np.median(vals)) if vals else None,
                                  "p10_measured_db": float(np.percentile(vals, 10)) if vals else None,
                                  "p90_measured_db": float(np.percentile(vals, 90)) if vals else None}
    summary = {"comparisons": comparisons, "noise_floor_comparisons": floor_comparisons,
               "measured_snr": snr_summary, "dry_control_incremental_speech": speech_summary,
               "predeclared_gates": {"by_tail_window": gate_results,
                                     "speech_preservation": preserve,
                                     "overall_candidate_pass": bool(passed)},
               "decision": ("B stays experimental; proceed to a streaming integration screen." if passed else
                            "Do not claim the cascade passes. Inspect which tail and speech gates failed before proposing any retraining."),
               "interpretation": "Positive tail gain means a lower tail level than the paired base route, and only uncensored pairs are included. Bootstrap resamples speakers as clusters; AIR locations are reported separately.",
               "input_rows": len(rows), "air_locations": sorted({r["room"] for r in manifest["rir_rirs"]}),
               "speakers": [s["speaker"] for s in manifest["speakers"]]}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    plot_summary(output_dir, comparisons)
    return summary


def plot_summary(output_dir: Path, comparisons: dict) -> None:
    import matplotlib.pyplot as plt

    bands = ("150_300", "300_600")
    conditions = ("rir_only", "fan20", "fan10")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    colors = {"dereverb_on_input": "#2a788e", "dereverb_after_dpdfnet2": "#d1495b"}
    for ax, condition in zip(axes, conditions):
        x = np.arange(len(bands))
        for offset, comparison in ((-.12, "dereverb_on_input"), (.12, "dereverb_after_dpdfnet2")):
            vals, lows, highs = [], [], []
            censored = []
            for band in bands:
                row = comparisons[f"{condition}/{band}/{comparison}"]
                val = row["median_gain_db"]
                ci = row["speaker_cluster_bootstrap_95ci_db"]
                vals.append(np.nan if val is None else val)
                lows.append(0.0 if val is None or ci is None else val - ci[0])
                highs.append(0.0 if val is None or ci is None else ci[1] - val)
                censored.append(row["censored_pair_count"])
            ax.errorbar(x + offset, vals, yerr=np.asarray([lows, highs]), fmt="o",
                        capsize=4, color=colors[comparison], label=comparison.replace("_", " "))
            for point, value, count in zip(x + offset, vals, censored):
                if np.isnan(value):
                    ax.annotate(f"not measurable\n{count} censored", (point, 0),
                                xytext=(0, 15 if comparison == "dereverb_on_input" else -33),
                                textcoords="offset points", ha="center", va="center",
                                fontsize=7, color=colors[comparison])
        ax.axhline(0, color="#555", linewidth=0.8)
        ax.axhline(1.5, color="#d1495b", linestyle=":", linewidth=0.9)
        ax.set_xticks(x, ["150–300 ms", "300–600 ms"])
        ax.set_title(condition.replace("_", " "))
        ax.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("Tail reduction vs paired base route (dB)\nuncensored pairs only")
    axes[0].set_ylim(-58, 19)
    axes[-1].legend(loc="best", fontsize=8)
    fig.suptitle("Compact dereverb factorial: incremental tail effect")
    fig.tight_layout()
    fig.savefig(output_dir / "factorial-tail-gains.png", dpi=180)
    plt.close(fig)


def choose_speakers(speech_root: Path, prior_exclusion_file: Path,
                    speaker_limit: int) -> list[tuple[str, tuple[Path, Path], np.ndarray, list[dict]]]:
    """Select fresh sorted train-clean-360 speakers and deterministic utterance pairs."""
    from data import read_librispeech

    excluded = {line.strip() for line in prior_exclusion_file.read_text().splitlines()
                if line.strip() and not line.lstrip().startswith("#")}
    for split in ("train-clean-100", "dev-clean", "dev-other", "test-clean", "test-other"):
        excluded.update(s for _, s in read_librispeech(speech_root, split))
    current_air_manifest = DEFAULT_OUTPUT.parent / "air-external-2026-10-09" / "manifest.json"
    if current_air_manifest.exists():
        air = json.loads(current_air_manifest.read_text())
        excluded.update(str(row["speaker"]) for row in air["speech_speakers"])

    root = speech_root / "train-clean-360"
    eligible = []
    for speaker_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        speaker = speaker_dir.name
        if speaker in excluded:
            continue
        paths = sorted(speaker_dir.glob("*/*.flac"))
        for first, second_path in zip(paths, paths[1:]):
            try:
                clip, pause = make_inserted_pause_pair(first, second_path)
            except ValueError:
                continue
            # Keep the first four seconds of each natural utterance so the
            # full factorial remains runnable on CPU-only DPDFNet runtimes.
            gap = int(pause["inserted_gap_start_sample"])
            first_crop = clip[:min(gap, UTTERANCE_CROP_SAMPLES)]
            second_audio = clip[gap + int(.8 * SR):]
            second_crop = second_audio[:min(len(second_audio), UTTERANCE_CROP_SAMPLES)]
            try:
                first_end = speech_bounds(first_crop)[1]
                second_start = len(first_crop) + int(.8 * SR) + speech_bounds(second_crop)[0]
            except ValueError:
                continue
            if first_end + int(.60 * SR) > second_start - int(.20 * SR):
                continue
            clip = np.concatenate((first_crop, np.zeros(int(.8 * SR), np.float32),
                                   second_crop)).astype(np.float32)
            pause = {"clip_relative_start_sample": first_end,
                     "second_speech_start_sample": second_start,
                     "duration_s": (second_start - first_end) / SR,
                     "kind": "inserted_controlled_pause_800ms",
                     "first_utterance": first.name, "second_utterance": second_path.name,
                     "first_crop_seconds": len(first_crop) / SR,
                     "second_crop_seconds": len(second_crop) / SR}
            eligible.append((speaker, (first, second_path), clip, [pause]))
            break
        if len(eligible) >= speaker_limit:
            break
    if len(eligible) != speaker_limit:
        raise RuntimeError(f"requested {speaker_limit} fresh speakers; found {len(eligible)}")
    overlap = {entry[0] for entry in eligible} & excluded
    if overlap:
        raise RuntimeError(f"fresh speaker selection overlaps previous speakers: {sorted(overlap)}")
    return eligible


def render(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing result directory {args.output_dir}")
    for path in (args.checkpoint, args.dpdfnet_model, args.speech_root, args.rir_root, args.sdk):
        if not path.exists():
            raise FileNotFoundError(path)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    rirs = load_air_rirs(args.rir_root)
    eligible = choose_speakers(args.speech_root, args.prior_exclusion_file, args.speaker_limit)
    model = load_model(args.checkpoint, device)
    dpdf = make_dpdfnet(args.sdk, args.dpdfnet_model)
    speakers = [{"speaker": speaker,
                 "paths": [str(path.relative_to(ROOT)) for path in paths],
                 "sha256": [file_sha256(path) for path in paths],
                 "pause": pauses[0]}
                for speaker, paths, _, pauses in eligible]
    manifest = {
        "protocol": "Frozen paired factorial; 12 fresh train-clean-360 speakers × 8 measured AIR RIRs; input/dereverb/DPDFNet2/DPDFNet2→dereverb; first 4 s of two successive natural utterances (controlled-offset clips), joined with an 800 ms inserted pause; RIR-only and fan-like additive noise at 20/10 dB SNR; 20 ms RMS / 10 ms hop and fixed 2% clean-peak activity threshold; full RIR convolution; no loudness normalization.",
        "speech_dataset": "LibriSpeech train-clean-360", "speech_license": "CC BY 4.0",
        "speech_license_url": "https://www.openslr.org/12/", "speakers": speakers,
        "rir_dataset": "AIR v1.4", "rir_rirs": rirs,
        "noise": {"type": "fixed-seed bandpassed Gaussian with weak 120/240 Hz tones (fan-like proxy)",
                  "snr_db": [20, 10], "limitation": "procedural noise proxy; not a real fan recording"},
        "branches": list(ARMS), "device": str(device), "sample_rate": SR,
        "model_execution": {"compact_dereverb": str(device),
                             "dpdfnet2": ",".join(dpdf.model._runtime.session.get_providers())},
        "dereverb_checkpoint": {"path": str(args.checkpoint.relative_to(ROOT)),
                                 "sha256": file_sha256(args.checkpoint),
                                 "parameter_count": sum(p.numel() for p in model.parameters())},
        "dpdfnet2_model": {"path": str(args.dpdfnet_model.relative_to(ROOT)),
                           "sha256": file_sha256(args.dpdfnet_model),
                           "native_sample_rate": SR48,
                           "stream_alignment_samples": ALIGNMENT48},
        "metrics": {"speech": "paired RMS output/clean per 20 ms frame, clean-only activity; active, weak-frame p10/p50/p90 and rising/onset p10; inserted silence and first 200 ms startup excluded",
                    "tail": "50–150/150–300/300–600 ms after clean speech end, all normalized to fixed wet speech-active input reference; floor is max(dry processed control, noise-only processed floor); censored at floor +3 dB",
                    "noise_floor": "noise-only branch output evaluated in the same pause window relative to the same wet speech reference",
                    "timing": "model forward time per audio pass; excludes model initialization and reports real-time factor"},
        "gates": {"dereverb_on_wet_median_gain_db": 2.0,
                  "incremental_after_dpdfnet_median_gain_db": 1.5,
                  "positive_rooms_min_of_4": 3,
                  "extra_active_speech_db_gt": -1.0,
                  "extra_onset_p10_db_gt": -1.0,
                  "extra_weak_speech_p10_db_gt": -1.5},
        "limitations": ["AIR RIRs have appeared in a previous compact-dereverb screen; this experiment uses fresh speakers to isolate the cascade interaction, not a new room-level external holdout.",
                        "Four AIR locations represent four environments and eight measured responses; repeated speaker-room pairs are correlated.",
                        "Fan-like noise is synthetic and cannot validate real microphone or fan noise.",
                        "Each voice is represented by the first four seconds of two successive utterances, so the 4 s cutoff is a controlled offset and not necessarily a natural phrase ending. A fixed 800 ms digital pause is an acoustic diagnostic, not conversational reverb intelligibility.",
                        "This does not measure Adobe Podcast, live latency, or universal hardware performance."],
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    rows = []
    case_path = args.output_dir / "cases.jsonl"
    case_stream = case_path.open("w")
    times: dict[str, list[tuple[float, float]]] = {arm: [] for arm in ARMS}

    for speaker, paths, clean_raw, pauses in eligible:
        for rir_item in rirs:
            rir, rir_meta = prepare_rir(rir_item)
            pair = convolve_pair(clean_raw, rir)
            n = len(pair["clean"])
            clean = pair["clean"][:n]
            wet = pair["reverberant"][:n]
            dry_outputs, dry_times = branch_outputs(clean, dereverb_model=model, device=device,
                                                    dpdf_backend=dpdf)
            dry_rows = []
            for arm in ARMS:
                dry_speech = steady_speech_metrics(clean, clean, dry_outputs[arm], pauses)
                dry_rows.append({"speaker": speaker, "utterance": [p.name for p in paths],
                             "room": rir_item["room"], "rir": Path(rir_item["path"]).name,
                             "distance_role": rir_item["distance_role"],
                             "distance_m": rir_item["distance_m"],
                             "rir_sha256": rir_item["sha256"], "condition": "dry_control",
                             "nominal_snr_db": None, "measured_snr_db": None, "arm": arm,
                             "duration_seconds": len(clean) / SR,
                             "runtime_seconds": dry_times.get(arm, 0.0),
                             "rtf": dry_times.get(arm, 0.0) / (len(clean) / SR),
                             "speech_metrics": {**dry_speech,
                                                "weak_speech_gain_db_p10": weak_speech_p10(
                                                    clean, dry_outputs[arm], pauses)},
                             "tail_metrics": pause_metrics(clean, clean, dry_outputs[arm],
                                                           dry_outputs[arm], pauses),
                             "rir_processing": rir_meta})
            for row in dry_rows:
                rows.append(row)
                case_stream.write(json.dumps(row, allow_nan=False) + "\n")
            case_stream.flush()
            # The branch helper includes an unchanged input route; retain only
            # its model routes for paired dry-control floor/voice measurements.
            case_inputs: list[tuple[str, np.ndarray, np.ndarray | None, float | None]] = [
                ("rir_only", wet, None, None)]
            for condition, target_snr in NOISE_CONDITIONS:
                seed_material = f"{speaker}:{rir_item['sha256']}:{condition}".encode()
                seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:4], "little")
                noisy, noise, actual_snr = mix_at_snr(clean, wet, seed, target_snr)
                case_inputs.append((condition, noisy, noise, actual_snr))

            for condition, mixture, noise, actual_snr in case_inputs:
                outputs, elapsed = branch_outputs(mixture, dereverb_model=model, device=device,
                                                  dpdf_backend=dpdf)
                noise_outputs = None
                if noise is not None:
                    noise_outputs, _ = branch_outputs(noise, dereverb_model=model, device=device,
                                                      dpdf_backend=dpdf)
                case_rows = []
                for arm in ARMS:
                    output = outputs[arm]
                    speech = steady_speech_metrics(clean, wet, output, pauses)
                    input_seconds = len(mixture) / SR
                    row = {"speaker": speaker, "utterance": [p.name for p in paths],
                           "room": rir_item["room"], "rir": Path(rir_item["path"]).name,
                           "distance_role": rir_item["distance_role"],
                           "distance_m": rir_item["distance_m"],
                           "rir_sha256": rir_item["sha256"], "condition": condition,
                           "nominal_snr_db": (None if condition == "rir_only" else
                                               dict(NOISE_CONDITIONS)[condition]),
                           "measured_snr_db": actual_snr,
                           "arm": arm, "duration_seconds": input_seconds,
                           "runtime_seconds": elapsed.get(arm, 0.0),
                           "rtf": elapsed.get(arm, 0.0) / max(input_seconds, 1e-12),
                           "speech_metrics": speech,
                           "rir_processing": rir_meta,
                           "_output": output}
                    if noise_outputs is not None:
                        row["_noise_output"] = noise_outputs[arm]
                    case_rows.append(row)
                    times[arm].append((elapsed.get(arm, 0.0), input_seconds))
                floor_inputs = {condition: {arm: case_rows[i]["_noise_output"]
                                           for i, arm in enumerate(ARMS)}
                                for condition in ([condition] if noise_outputs is not None else [])}
                add_floor_metrics(case_rows, clean, wet, floor_inputs,
                                  dry_outputs, pauses)
                for row in case_rows:
                    row.pop("_noise_output", None)
                    rows.append(row)
                    case_stream.write(json.dumps(row, allow_nan=False) + "\n")
                case_stream.flush()
                print(f"DONE {speaker}/{rir_item['room']}/{condition} ", flush=True)

    case_stream.close()
    manifest["case_count"] = len(rows)
    manifest["model_timing"] = {
        arm: {"median_seconds_per_pass": float(np.median([v[0] for v in values])),
              "median_input_duration_seconds": float(np.median([v[1] for v in values])),
              "median_rtf": float(np.median([v[0] / v[1] for v in values if v[1] > 0]))}
        for arm, values in times.items()}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    summary = summarize(args.output_dir, manifest, rows)
    return {"output_dir": str(args.output_dir), "case_rows": len(rows),
            "speakers": [row["speaker"] for row in speakers], "rooms": sorted({r["room"] for r in rirs}),
            "device": str(device), "timing": manifest["model_timing"],
            "predeclared_gates": summary["predeclared_gates"], "decision": summary["decision"]}


def complete_dry_controls(args: argparse.Namespace) -> dict:
    """Add deterministic dry-control rows if the benchmark file lacks them."""
    output_dir = args.output_dir
    manifest_path = output_dir / "manifest.json"
    cases_path = output_dir / "cases.jsonl"
    if not manifest_path.is_file() or not cases_path.is_file():
        raise FileNotFoundError(f"benchmark manifest/cases not found under {output_dir}")
    manifest = json.loads(manifest_path.read_text())
    rows = [json.loads(line) for line in cases_path.read_text().splitlines() if line.strip()]
    expected = len(manifest["speakers"]) * len(manifest["rir_rirs"]) * len(ARMS)
    existing = [row for row in rows if row["condition"] == "dry_control"]
    if len(existing) == expected:
        return {"dry_controls": "already complete", "case_rows": len(rows)}
    if existing:
        raise RuntimeError(f"partial dry controls: {len(existing)} of {expected}; refusing ambiguous repair")

    eligible = choose_speakers(args.speech_root, args.prior_exclusion_file, len(manifest["speakers"]))
    recorded = [entry["speaker"] for entry in manifest["speakers"]]
    if [row[0] for row in eligible] != recorded:
        raise RuntimeError("deterministic speaker selection differs from the saved manifest")
    for (_, paths, _, _), saved in zip(eligible, manifest["speakers"]):
        if [file_sha256(path) for path in paths] != saved["sha256"]:
            raise RuntimeError(f"speech file hash changed for speaker {saved['speaker']}")

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    model = load_model(args.checkpoint, device)
    dpdf = make_dpdfnet(args.sdk, args.dpdfnet_model)
    missing: set[tuple[str, str, str, str, str]] = set()
    for row in rows:
        missing.add((row["speaker"], row["room"], row["rir"], row["condition"], row["arm"]))
    appended = 0
    with cases_path.open("a") as stream:
        for speaker, paths, clean_raw, pauses in eligible:
            for rir_item in manifest["rir_rirs"]:
                rir, rir_meta = prepare_rir(rir_item)
                pair = convolve_pair(clean_raw, rir)
                clean = pair["clean"]
                dry_outputs, dry_times = branch_outputs(clean, dereverb_model=model,
                                                        device=device, dpdf_backend=dpdf)
                for arm in ARMS:
                    key = (speaker, rir_item["room"], Path(rir_item["path"]).name,
                           "dry_control", arm)
                    if key in missing:
                        continue
                    speech = steady_speech_metrics(clean, clean, dry_outputs[arm], pauses)
                    seconds = len(clean) / SR
                    row = {"speaker": speaker, "utterance": [p.name for p in paths],
                           "room": rir_item["room"], "rir": Path(rir_item["path"]).name,
                           "distance_role": rir_item["distance_role"],
                           "distance_m": rir_item["distance_m"],
                           "rir_sha256": rir_item["sha256"], "condition": "dry_control",
                           "nominal_snr_db": None, "measured_snr_db": None, "arm": arm,
                           "duration_seconds": seconds,
                           "runtime_seconds": dry_times.get(arm, 0.0),
                           "rtf": dry_times.get(arm, 0.0) / seconds,
                           "speech_metrics": speech,
                           "tail_metrics": pause_metrics(clean, clean, dry_outputs[arm],
                                                         dry_outputs[arm], pauses),
                           "rir_processing": rir_meta}
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
                    rows.append(row)
                    missing.add(key)
                    appended += 1
    if appended != expected:
        raise RuntimeError(f"appended {appended} dry-control rows; expected {expected}")
    manifest["case_count"] = len(rows)
    manifest["supplemental_dry_controls"] = {
        "reconstructed_from_frozen_manifest": True,
        "same_checkpoint_and_speech_sha256": True,
        "reason": "dry-control JSONL rows were omitted by the original streaming writer; their in-memory rows had been included in the initial summary"}
    manifest["model_execution"] = {
        "compact_dereverb": str(device),
        "dpdfnet2": ",".join(dpdf.model._runtime.session.get_providers())}
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    summary = summarize(output_dir, manifest, rows)
    return {"appended_dry_controls": appended, "case_rows": len(rows),
            "predeclared_gates": summary["predeclared_gates"], "decision": summary["decision"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_DEREVERB)
    parser.add_argument("--dpdfnet-model", type=Path, default=DEFAULT_DPDF)
    parser.add_argument("--sdk", type=Path, default=DEFAULT_SDK)
    parser.add_argument("--speech-root", type=Path, default=DEFAULT_SPEECH)
    parser.add_argument("--rir-root", type=Path, default=DEFAULT_RIR)
    parser.add_argument("--prior-exclusion-file", type=Path,
                        default=DEFAULT_OUTPUT.parent / "air-external-2026-10-09" / "excluded_prior-speakers.txt")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--speaker-limit", type=int, default=12)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--complete-dry-controls", action="store_true",
                        help="Append missing dry controls to an existing completed benchmark and regenerate its summary")
    args = parser.parse_args()
    print(json.dumps(complete_dry_controls(args) if args.complete_dry_controls else render(args), indent=2))


if __name__ == "__main__":
    main()
