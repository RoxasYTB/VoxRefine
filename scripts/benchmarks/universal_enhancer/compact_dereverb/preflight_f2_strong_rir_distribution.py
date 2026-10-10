#!/usr/bin/env python3
"""Full-exact, non-selecting preflight for a proposed strong-reverb distribution."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts/benchmarks/universal_enhancer"))
from data import make_procedural_rir  # noqa: E402
from prepare_cap60_conditioned_pairs import (  # noqa: E402
    CROP_SAMPLES, DEEP_FILTER, LIBRISPEECH, SR, cap60_one,
    cap_target_speech_ratio, sha256,
)
from prepare_f2_tailbank import crop, load_schedule, resolve_source, seed_for, tail_input_levels  # noqa: E402
from train_measured_mix import measured_pair  # noqa: E402

DEFAULT_OUT = ROOT / ".tools/compact-dereverb/f2-strong-rir-preflight-2026-10-10"
SCHEDULE_INDICES = (7, 10, 15, 18, 20, 23, 27, 31)
EXPECTED_SPEAKERS = ("6563", "696", "8123", "3112", "6181", "1743", "6925", "7367")
T60_RANGE = (1.65, 1.85)
DRR_RANGE = (-14.5, -12.5)
ACTIVE_DELTA_MEDIAN_FLOOR_DB = -6.0
ACTIVE_DELTA_P10_FLOOR_DB = -12.0
SCALE_DB_MEDIAN_FLOOR_DB = -6.0
SCALE_DB_P10_FLOOR_DB = -12.0
SCALE_LOW_EXTREME = 0.25
SCALE_LOW_EXTREME_MAX_COUNT = 6
TAIL_THRESHOLD_DB = -50.0


def active_sample_mask(clean_crop: np.ndarray) -> np.ndarray:
    frame, hop = 320, 160
    frames = np.lib.stride_tricks.sliding_window_view(clean_crop, frame)[::hop]
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-24)
    active = rms >= max(float(rms.max()) * .02, 1e-5)
    centers = np.arange(len(active)) * hop + frame // 2
    sample_frames = np.minimum(np.arange(len(clean_crop), dtype=np.int64) // hop,
                                len(active) - 1)
    return active[sample_frames]


def rms_masked(audio: np.ndarray, mask: np.ndarray) -> float:
    values = np.asarray(audio, dtype=np.float64)[mask]
    if not len(values):
        raise RuntimeError("active mask has no samples")
    return float(np.sqrt(np.mean(values * values) + 1e-24))


def run(args) -> dict:
    if not args.deep_filter.is_file():
        raise FileNotFoundError(args.deep_filter)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"refusing to overwrite nonempty {args.output_dir}; pass --resume")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = args.output_dir / "cap60-cache"
    cache.mkdir(exist_ok=True)
    schedule = load_schedule()
    rows = [schedule[index] for index in SCHEDULE_INDICES]
    speakers = tuple(str(row["speaker"]) for row in rows)
    if speakers != EXPECTED_SPEAKERS or any(row.get("kind") != "rir_only" for row in rows):
        raise RuntimeError(f"preselected clean slots changed: {SCHEDULE_INDICES}, {speakers}")
    records = []
    started = time.perf_counter()
    for slot_order, row in enumerate(rows):
        schedule_index = int(row["index"])
        clean, pause_meta, _ = resolve_source(row, args.data_root / "train-clean-100")
        global_pause = int(pause_meta["clip_relative_start_sample"])
        crop_start = global_pause - int(.75 * SR)
        if crop_start < 0 or crop_start + CROP_SAMPLES > len(clean):
            raise RuntimeError(f"schedule row {schedule_index} crop does not fit")
        clean_crop = crop(clean, crop_start)
        mask = active_sample_mask(clean_crop)
        pause_in_crop = global_pause - crop_start
        measure_end = global_pause + 9_600
        if measure_end > len(clean):
            raise RuntimeError(f"schedule row {schedule_index} tail windows do not fit")
        gain = float(row["post_cap_gain"])
        for candidate_index in range(8):
            parameter_seed = seed_for(slot_order, candidate_index,
                namespace="F2-strong-RIR-preflight-parameters-2026-10-10")
            rng = np.random.default_rng(parameter_seed)
            t60 = float(rng.uniform(*T60_RANGE))
            drr = float(rng.uniform(*DRR_RANGE))
            rir_seed = seed_for(slot_order, candidate_index,
                namespace="F2-strong-RIR-preflight-geometry-2026-10-10")
            rir, _, _ = make_procedural_rir(SR, rir_seed, t60, drr,
                max_t60_s=T60_RANGE[1], min_direct_to_reverb_db=DRR_RANGE[0])
            pair = measured_pair(clean, rir)
            shared_scale = float(np.max(np.abs(pair["clean"]))) / max(
                float(np.max(np.abs(clean))), 1e-8)
            key = f"row-{schedule_index:03d}-candidate-{candidate_index:02d}"
            wet_out, wet_meta = cap60_one(args.deep_filter, pair["reverberant"],
                SR, args.output_dir, key + "-wet", cache)
            clean_out, clean_meta = cap60_one(args.deep_filter, pair["clean"],
                SR, args.output_dir, key + "-clean", cache)
            levels = tail_input_levels(
                clean_out[crop_start:measure_end] * gain,
                wet_out[crop_start:measure_end] * gain,
                pause_in_crop, activity_clean_crop=clean_crop)
            wet_crop = wet_out[crop_start:crop_start + CROP_SAMPLES]
            target_crop = clean_out[crop_start:crop_start + CROP_SAMPLES]
            delta_active_db = float(20 * np.log10(
                rms_masked(wet_crop, mask) / rms_masked(target_crop, mask)))
            retention = float(cap_target_speech_ratio(
                crop(pair["clean"], crop_start), target_crop))
            input_peak = max(float(np.max(np.abs(pair["reverberant"]))),
                             float(np.max(np.abs(pair["clean"]))))
            output_peak = max(float(np.max(np.abs(wet_out))),
                              float(np.max(np.abs(clean_out))))
            clipped = input_peak >= 1.0 or output_peak >= 1.0
            record = {"schedule_index": schedule_index,
                "speaker": str(row["speaker"]), "candidate_index": candidate_index,
                "parameter_seed": parameter_seed, "geometry_seed": rir_seed,
                "t60_s": t60, "direct_to_reverb_db": drr,
                "shared_scale": shared_scale,
                "shared_scale_db": float(20 * np.log10(max(shared_scale, 1e-12))),
                "w1_w2_tail_db": levels.get("window_db"),
                "tail_eligible": bool(levels.get("eligible")),
                "delta_active_db": delta_active_db,
                "cap_target_speech_retention_ratio": retention,
                "input_peak": input_peak, "output_peak": output_peak,
                "clipped": clipped,
                "cap60_wet_source_sha256": wet_meta["source_sha256"],
                "cap60_clean_source_sha256": clean_meta["source_sha256"],
                "cap60_wet_elapsed_s": float(wet_meta["elapsed_s"]),
                "cap60_clean_elapsed_s": float(clean_meta["elapsed_s"])}
            records.append(record)
            print(json.dumps({k: record[k] for k in
                ("schedule_index", "candidate_index", "t60_s", "direct_to_reverb_db",
                 "shared_scale", "w1_w2_tail_db", "delta_active_db",
                 "cap_target_speech_retention_ratio", "clipped")},
                allow_nan=False), flush=True)

    slot_counts = {str(index): sum(r["schedule_index"] == index and r["tail_eligible"]
        for r in records) for index in SCHEDULE_INDICES}
    eligible_count = sum(r["tail_eligible"] for r in records)
    active_delta = np.array([r["delta_active_db"] for r in records], dtype=np.float64)
    scale_db = np.array([r["shared_scale_db"] for r in records], dtype=np.float64)
    scales = np.array([r["shared_scale"] for r in records], dtype=np.float64)
    retention = np.array([r["cap_target_speech_retention_ratio"] for r in records], dtype=np.float64)
    clipped_count = sum(r["clipped"] for r in records)
    active_median = float(np.median(active_delta))
    active_p10 = float(np.percentile(active_delta, 10))
    scale_db_median = float(np.median(scale_db))
    scale_db_p10 = float(np.percentile(scale_db, 10))
    scale_extreme_count = int(np.sum(scales < SCALE_LOW_EXTREME))
    slots_at_least6 = sum(count >= 6 for count in slot_counts.values())
    min_slot_count = min(slot_counts.values())
    gates = {"tail_pass_rate": eligible_count >= 48,
        "six_slots_at_least_six": slots_at_least6 >= 6,
        "no_slot_below_four": min_slot_count >= 4,
        "no_clipping": clipped_count == 0,
        "active_delta_median": active_median > ACTIVE_DELTA_MEDIAN_FLOOR_DB,
        "active_delta_p10": active_p10 > ACTIVE_DELTA_P10_FLOOR_DB,
        "scale_db_median": scale_db_median > SCALE_DB_MEDIAN_FLOOR_DB,
        "scale_db_p10": scale_db_p10 > SCALE_DB_P10_FLOOR_DB,
        "scale_below_0_25_count": scale_extreme_count <= SCALE_LOW_EXTREME_MAX_COUNT,
        "target_speech_retention": bool(np.all(retention >= .05))}
    report = {"name": "F2-strong-RIR-distribution-preflight-v1",
        "schedule_indices": list(SCHEDULE_INDICES), "speaker_ids": list(EXPECTED_SPEAKERS),
        "source_schedule_sha256": sha256(ROOT / ".tools/compact-dereverb/cap60-conditioned-2026-10-10/pairs.jsonl"),
        "deep_filter_sha256": sha256(args.deep_filter),
        "t60_uniform_range_s": list(T60_RANGE), "drr_uniform_range_db": list(DRR_RANGE),
        "namespace": "F2-strong-RIR-preflight-parameters/geometry-2026-10-10",
        "full_exact_cases": len(records), "full_exact_cap60_runs": len(records) * 2,
        "surrogate_used": False, "no_candidate_selection": True,
        "selection_criterion_db": TAIL_THRESHOLD_DB,
        "eligible_count": int(eligible_count), "eligible_rate": float(eligible_count / len(records)),
        "eligible_by_schedule_index": slot_counts,
        "active_delta_db": {"median": active_median, "p10": active_p10,
            "p90": float(np.percentile(active_delta, 90)),
            "min": float(np.min(active_delta)), "max": float(np.max(active_delta))},
        "shared_scale": {"median": float(np.median(scales)),
            "p10": float(np.percentile(scales, 10)),
            "p90": float(np.percentile(scales, 90)),
            "min": float(np.min(scales)), "max": float(np.max(scales)),
            "log_db_median": scale_db_median, "log_db_p10": scale_db_p10,
            "count_below_0_25": scale_extreme_count,
            "fraction_below_0_25": float(scale_extreme_count / len(scales))},
        "cap_target_speech_retention_ratio": {"min": float(np.min(retention)),
            "median": float(np.median(retention)), "max": float(np.max(retention))},
        "clipped_case_count": int(clipped_count), "maximum_output_peak": float(max(r["output_peak"] for r in records)),
        "gates": gates, "preflight_pass": all(gates.values()),
        "selection_or_training_data_modified": False,
        "elapsed_s": time.perf_counter() - started,
        "records": records, "test_wav_accessed": False}
    (args.output_dir / "report.json").write_text(json.dumps(report,
        indent=2, allow_nan=False) + "\n")
    return {k: v for k, v in report.items() if k != "records"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=LIBRISPEECH)
    parser.add_argument("--deep-filter", type=Path, default=DEEP_FILTER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--resume", action="store_true",
        help="reuse exact-hash outputs from this preflight only")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
