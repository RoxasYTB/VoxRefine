#!/usr/bin/env python3
"""Map a full-exact strong-reverb region across three frozen F2 speakers."""
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
    CROP_SAMPLES, DEEP_FILTER, LIBRISPEECH, SR, cap60_one, sha256,
)
from prepare_f2_tailbank import crop, load_schedule, resolve_source, seed_for, tail_input_levels  # noqa: E402
from train_measured_mix import measured_pair  # noqa: E402

DEFAULT_OUT = ROOT / ".tools/compact-dereverb/f2-rir-region-three-slots-2026-10-10"
SCHEDULE_INDICES = (3, 5, 6)
EXPECTED_SPEAKERS = ("3857", "2159", "8098")
T60_VALUES = (1.6, 1.9, 2.2, 2.5)
DRR_VALUES = (-9.0, -12.0, -15.0, -18.0)


def contiguous_rectangle_counts(records: list[dict]) -> list[dict]:
    by_slot = {}
    for row in records:
        by_slot.setdefault(int(row["schedule_index"]), {})[
            (float(row["t60_s"]), float(row["direct_to_reverb_db"]))] = bool(row["full_eligible"])
    rectangles = []
    for ti in range(len(T60_VALUES) - 1):
        for di in range(len(DRR_VALUES) - 1):
            cells = [(T60_VALUES[ti], DRR_VALUES[di]),
                (T60_VALUES[ti + 1], DRR_VALUES[di]),
                (T60_VALUES[ti], DRR_VALUES[di + 1]),
                (T60_VALUES[ti + 1], DRR_VALUES[di + 1])]
            counts = {str(slot): sum(grid.get(cell, False) for cell in cells)
                      for slot, grid in by_slot.items()}
            rectangles.append({"t60_bounds_s": [T60_VALUES[ti], T60_VALUES[ti + 1]],
                "drr_bounds_db": [DRR_VALUES[di], DRR_VALUES[di + 1]],
                "eligible_cells_per_schedule_index": counts,
                "passes_two_slot_rectangle_gate": sum(value >= 3 for value in counts.values()) >= 2})
    return rectangles


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
        raise RuntimeError(f"preselected schedule slots changed: {SCHEDULE_INDICES}, {speakers}")
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
        pause_in_crop = global_pause - crop_start
        measure_end = global_pause + 9_600
        if measure_end > len(clean):
            raise RuntimeError(f"schedule row {schedule_index} tail windows do not fit")
        gain = float(row["post_cap_gain"])
        # Same random geometry, reflection signs/delays, noise realization, and
        # damping draw across each slot's 4 × 4 RT60 × DRR surface.
        seed = seed_for(slot_order, 0, namespace="F2-RIR-region-map-three-slots-2026-10-10")
        for t60 in T60_VALUES:
            for drr in DRR_VALUES:
                rir, _, _ = make_procedural_rir(SR, seed, t60, drr,
                    max_t60_s=max(T60_VALUES),
                    min_direct_to_reverb_db=min(DRR_VALUES))
                pair = measured_pair(clean, rir)
                shared_scale = float(np.max(np.abs(pair["clean"]))) / max(
                    float(np.max(np.abs(clean))), 1e-8)
                key = f"row-{schedule_index:03d}-t60-{t60:.1f}-drr-{drr:+.0f}"
                wet_out, wet_meta = cap60_one(args.deep_filter,
                    pair["reverberant"], SR, args.output_dir, key + "-wet", cache)
                clean_out, clean_meta = cap60_one(args.deep_filter,
                    pair["clean"], SR, args.output_dir, key + "-clean", cache)
                levels = tail_input_levels(
                    clean_out[crop_start:measure_end] * gain,
                    wet_out[crop_start:measure_end] * gain,
                    pause_in_crop, activity_clean_crop=clean_crop)
                record = {"schedule_index": schedule_index,
                    "speaker": str(row["speaker"]), "t60_s": t60,
                    "direct_to_reverb_db": drr, "geometry_seed": seed,
                    "shared_scale": shared_scale,
                    "w1_w2_tail_db": levels.get("window_db"),
                    "reference_valid": bool(levels.get("reference_valid")),
                    "full_eligible": bool(levels.get("eligible")),
                    "cap60_wet_source_sha256": wet_meta["source_sha256"],
                    "cap60_clean_source_sha256": clean_meta["source_sha256"],
                    "cap60_wet_elapsed_s": float(wet_meta["elapsed_s"]),
                    "cap60_clean_elapsed_s": float(clean_meta["elapsed_s"])}
                records.append(record)
                print(json.dumps({k: record[k] for k in
                    ("schedule_index", "t60_s", "direct_to_reverb_db",
                     "w1_w2_tail_db", "full_eligible")}, allow_nan=False), flush=True)
    successes_by_slot = {str(index): sum(r["schedule_index"] == index and r["full_eligible"]
        for r in records) for index in SCHEDULE_INDICES}
    eligible_records = [r for r in records if r["full_eligible"]]
    interior_success = any(r["t60_s"] < max(T60_VALUES) and
        r["direct_to_reverb_db"] > min(DRR_VALUES) for r in eligible_records)
    rectangles = contiguous_rectangle_counts(records)
    rectangle_gate = any(r["passes_two_slot_rectangle_gate"] for r in rectangles)
    slot_counts = list(successes_by_slot.values())
    per_slot_gate = sum(count >= 6 for count in slot_counts) >= 2 and min(slot_counts) >= 3
    result = {"name": "F2-strong-reverb-region-map-three-slots-v1",
        "schedule_indices": list(SCHEDULE_INDICES),
        "speaker_ids": list(EXPECTED_SPEAKERS),
        "source_schedule_sha256": sha256(ROOT / ".tools/compact-dereverb/cap60-conditioned-2026-10-10/pairs.jsonl"),
        "deep_filter_sha256": sha256(args.deep_filter),
        "t60_values_s": list(T60_VALUES), "drr_values_db": list(DRR_VALUES),
        "geometry_seed_policy": "one deterministic seed per preselected slot, reused over its entire 4x4 grid",
        "grid_points": len(records), "full_exact_runs": len(records) * 2,
        "surrogate_used": False, "selection_criterion_db": -50.0,
        "eligible_by_schedule_index": successes_by_slot,
        "per_slot_gate": per_slot_gate,
        "interior_success_exists": interior_success,
        "rectangle_gate_any_two_slots_at_least_3_of_4": rectangle_gate,
        "contiguous_2x2_regions": rectangles,
        "region_gate_pass": per_slot_gate and interior_success and rectangle_gate,
        "selection_or_training_data_modified": False,
        "elapsed_s": time.perf_counter() - started,
        "records": records, "test_wav_accessed": False}
    (args.output_dir / "report.json").write_text(json.dumps(result,
        indent=2, allow_nan=False) + "\n")
    return {key: value for key, value in result.items()
            if key not in ("records", "contiguous_2x2_regions")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=LIBRISPEECH)
    parser.add_argument("--deep-filter", type=Path, default=DEEP_FILTER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--resume", action="store_true",
        help="reuse exact-hash outputs from this probe only")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
