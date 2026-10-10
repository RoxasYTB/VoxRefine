#!/usr/bin/env python3
"""Cached-only attribution of weak/onset losses to Cap60 versus common scaling."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
from prepare_f2_tailbank import crop, load_schedule, resolve_source  # noqa: E402

V1_ROOT = ROOT / ".tools/compact-dereverb/f2-strong-rir-preflight-2026-10-10"
V2_ROOT = ROOT / ".tools/compact-dereverb/f2-strong-rir-integrity-v2-2026-10-10"
OUT = ROOT / ".tools/compact-dereverb/f2-cap60-weak-frame-diagnostic-2026-10-10"
SCHEDULE_INDICES = (7, 10, 15, 18, 20, 23, 27, 31)
EPS = 1e-8
BANDS = (("02_05_pct", .02, .05), ("05_10_pct", .05, .10),
         ("10_35_pct", .10, .35), ("over_35_pct", .35, float("inf")))


def frame_rms(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    frames = np.lib.stride_tricks.sliding_window_view(x, 320)[::160]
    return np.sqrt(np.mean(frames * frames, axis=1) + 1e-24)


def metric(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"frame_count": 0, "p10_db": None, "median_db": None,
                "fraction_below_minus_3_db": None,
                "fraction_below_minus_6_db": None,
                "fraction_below_minus_20_db": None,
                "max_consecutive_below_minus_6_frames": None,
                "max_consecutive_below_minus_6_ms": None}
    below = values < -6.0
    padded = np.r_[False, below, False]
    transitions = np.flatnonzero(padded[1:] != padded[:-1])
    max_run = int(np.max(transitions[1::2] - transitions[::2])) if len(transitions) else 0
    return {"frame_count": int(len(values)), "p10_db": float(np.percentile(values, 10)),
            "median_db": float(np.median(values)),
            "fraction_below_minus_3_db": float(np.mean(values < -3.0)),
            "fraction_below_minus_6_db": float(np.mean(values < -6.0)),
            "fraction_below_minus_20_db": float(np.mean(values < -20.0)),
            "max_consecutive_below_minus_6_frames": max_run,
            "max_consecutive_below_minus_6_ms": float(max_run * 10.0)}


def summarize_records(records: list[dict], signal: str) -> dict:
    output: dict[str, dict] = {}
    for band in [x[0] for x in BANDS] + ["active", "weak", "onset"]:
        selected = [r[signal][band] for r in records]
        selected = [x for x in selected if x["frame_count"] > 0]
        if not selected:
            output[f"{signal}:{band}"] = {"record_count": 0}
            continue
        output[f"{signal}:{band}"] = {
            "record_count": len(selected),
            "p10_db_median_across_records": float(np.median([x["p10_db"] for x in selected])),
            "p10_db_min_across_records": float(np.min([x["p10_db"] for x in selected])),
            "median_db_median_across_records": float(np.median([x["median_db"] for x in selected])),
            "fraction_below_minus_3_db_median": float(np.median([x["fraction_below_minus_3_db"] for x in selected])),
            "fraction_below_minus_6_db_median": float(np.median([x["fraction_below_minus_6_db"] for x in selected])),
            "fraction_below_minus_20_db_median": float(np.median([x["fraction_below_minus_20_db"] for x in selected])),
            "max_run_below_minus_6_ms_max": float(np.max([x["max_consecutive_below_minus_6_ms"] for x in selected])),
        }
    return output


def run(args: argparse.Namespace) -> dict:
    v1 = json.loads((args.v1_root / "report.json").read_text())
    v2 = json.loads((args.v2_root / "report.json").read_text())
    if len(v1.get("records", [])) != 64 or len(v2.get("equivariance_cases", [])) != 8:
        raise RuntimeError("expected the completed 64-case preflight and 8 unscaled Cap60 cache entries")
    schedule = load_schedule()
    rows = [schedule[index] for index in SCHEDULE_INDICES]
    records_by_key = {(int(x["schedule_index"]), int(x["candidate_index"])): x for x in v1["records"]}
    unscaled_by_slot = {int(x["schedule_index"]): x for x in v2["equivariance_cases"]}
    if len(records_by_key) != 64 or set(unscaled_by_slot) != set(SCHEDULE_INDICES):
        raise RuntimeError("case manifests do not match the frozen schedule")

    intrinsic_records: list[dict] = []
    scaling_records: list[dict] = []
    for row in rows:
        index = int(row["index"])
        joined, pause_meta, _ = resolve_source(row, args.data_root / "train-clean-100")
        crop_start = int(pause_meta["clip_relative_start_sample"]) - int(.75 * 16_000)
        clean = crop(joined, crop_start)
        clean_rms = frame_rms(clean)
        peak = float(clean_rms.max())
        bands = {name: (clean_rms >= lo * peak) & (clean_rms < hi * peak)
                 for name, lo, hi in BANDS}
        active = clean_rms > .02 * peak
        bands["active"] = active
        bands["weak"] = active & (clean_rms < .35 * peak)
        previous = np.r_[0.0, clean_rms[:-1]]
        onset = active & (clean_rms > 1.5 * previous)
        bands["onset"] = onset

        unscaled_key = f"row-{index:03d}-candidate-00-clean-unscaled"
        unscaled_audio_path = args.v2_root / "cap60-cache" / f"{unscaled_key}.npy"
        unscaled_meta_path = unscaled_audio_path.with_suffix(".json")
        if not unscaled_audio_path.is_file() or not unscaled_meta_path.is_file():
            raise FileNotFoundError(unscaled_audio_path)
        unscaled_meta = json.loads(unscaled_meta_path.read_text())
        if unscaled_meta.get("source_sha256") != hashlib.sha256(joined.tobytes()).hexdigest():
            raise RuntimeError(f"unscaled clean cache does not match source for row {index}")
        unscaled = np.load(unscaled_audio_path, allow_pickle=False).astype(np.float32)
        u_crop = crop(unscaled, crop_start)
        u_rms = frame_rms(u_crop)
        if len(u_rms) != len(clean_rms):
            raise RuntimeError(f"unscaled frame alignment mismatch for row {index}")
        d_cap = 20 * np.log10((u_rms + EPS) / (clean_rms + EPS))
        intrinsic_records.append({"schedule_index": index, "speaker": str(row["speaker"]),
            "intrinsic_cap60": {name: metric(d_cap[mask]) for name, mask in bands.items()}})

        for candidate in range(8):
            rec = records_by_key[(index, candidate)]
            cache_key = f"row-{index:03d}-candidate-{candidate:02d}-clean"
            scaled_path = args.v1_root / "cap60-cache" / f"{cache_key}.npy"
            scaled_meta_path = scaled_path.with_suffix(".json")
            if not scaled_path.is_file() or not scaled_meta_path.is_file():
                raise FileNotFoundError(scaled_path)
            scaled_meta = json.loads(scaled_meta_path.read_text())
            if scaled_meta.get("source_sha256") != rec["cap60_clean_source_sha256"]:
                raise RuntimeError(f"scaled clean cache metadata mismatch for {cache_key}")
            scaled = np.load(scaled_path, allow_pickle=False).astype(np.float32)
            s = float(rec["shared_scale"])
            scaled_crop = crop(scaled, crop_start) / max(s, EPS)
            scaled_rms = frame_rms(scaled_crop)
            if len(scaled_rms) != len(clean_rms):
                raise RuntimeError(f"scaled frame alignment mismatch for {cache_key}")
            d_scale = 20 * np.log10((scaled_rms + EPS) / (u_rms + EPS))
            scaling_records.append({"schedule_index": index, "speaker": str(row["speaker"]),
                "candidate_index": candidate, "shared_scale": s,
                "scaling_effect": {name: metric(d_scale[mask]) for name, mask in bands.items()}})

    intrinsic_summary = summarize_records(intrinsic_records, "intrinsic_cap60")
    scaling_summary = summarize_records(scaling_records, "scaling_effect")
    scale_weak_p10 = scaling_summary["scaling_effect:weak"]["p10_db_median_across_records"]
    cap_weak_p10 = intrinsic_summary["intrinsic_cap60:weak"]["p10_db_median_across_records"]
    cap_onset_p10 = intrinsic_summary["intrinsic_cap60:onset"]["p10_db_median_across_records"]
    intrinsic_cap60 = bool(scale_weak_p10 > -1.0 and
                           (cap_weak_p10 <= -1.5 or cap_onset_p10 <= -1.0))
    scaling_responsible = bool(scale_weak_p10 <= -1.0 and
                               cap_weak_p10 > -1.5 and cap_onset_p10 > -1.0)
    result = {"name": "F2-Cap60-weak-frame-attribution-diagnostic-v1",
        "analysis_mode": "cached-only; no new Cap60 inference",
        "schedule_indices": list(SCHEDULE_INDICES),
        "speakers": [str(row["speaker"]) for row in rows],
        "intrinsic_cap60_case_count": len(intrinsic_records),
        "scaling_effect_case_count": len(scaling_records),
        "masks": {"frame_ms": 20, "hop_ms": 10,
            "clean_relative_rms_bands": {name: [lo, hi if np.isfinite(hi) else "infinity"]
                                         for name, lo, hi in BANDS},
            "active_threshold": ">2% of each clean crop RMS peak",
            "onset_threshold": "active and current clean-frame RMS > 1.5x previous frame",
            "epsilon_amplitude": EPS},
        "aggregation": "per voice/candidate p10, median, fractions and run length; report medians across records; no frame pooling across records",
        "intrinsic_cap60_summary": intrinsic_summary,
        "scaling_effect_summary": scaling_summary,
        "pre_registered_interpretation": {
            "median_p10_scaling_effect_weak_above_minus_1_db": scale_weak_p10 > -1.0,
            "median_p10_intrinsic_cap60_weak_at_or_below_minus_1_5_db": cap_weak_p10 <= -1.5,
            "median_p10_intrinsic_cap60_onset_at_or_below_minus_1_db": cap_onset_p10 <= -1.0,
            "cap60_intrinsic_suppression_confirmed": intrinsic_cap60,
            "shared_scale_responsible_by_frozen_rule": scaling_responsible,
            "strong_rir_training_remains_blocked": True},
        "intrinsic_records": intrinsic_records,
        "scaling_records": scaling_records,
        "new_cap60_runs": 0, "selection_or_training_data_modified": False,
        "test_wav_accessed": False}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return {k: v for k, v in result.items() if k not in ("intrinsic_records", "scaling_records")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path,
        default=ROOT / ".tools/compact-dereverb/data/LibriSpeech")
    parser.add_argument("--v1-root", type=Path, default=V1_ROOT)
    parser.add_argument("--v2-root", type=Path, default=V2_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
