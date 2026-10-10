#!/usr/bin/env python3
"""Full-exact RT60 × DRR probes for an unresolved frozen F2 tail slot."""
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
from prepare_f2_tailbank import (  # noqa: E402
    crop, load_schedule, resolve_source, seed_for, tail_input_levels,
)
from train_measured_mix import measured_pair  # noqa: E402

DEFAULT_OUT = ROOT / ".tools/compact-dereverb/f2-targeted-rir-grid-2026-10-10"
T60_VALUES = (1.10, 1.40, 1.70)
DRR_VALUES = (-3.0, -6.0, -9.0, -12.0)


def run(args) -> dict:
    if not args.deep_filter.is_file():
        raise FileNotFoundError(args.deep_filter)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"refusing to overwrite nonempty {args.output_dir}; pass --resume")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = args.output_dir / "cap60-cache"
    cache.mkdir(exist_ok=True)
    schedule = load_schedule()
    row = schedule[args.schedule_index]
    if row.get("kind") != "rir_only":
        raise RuntimeError("targeted full-exact probe must use an RIR-only scheduled slot")
    clean, pause_meta, _ = resolve_source(row, args.data_root / "train-clean-100")
    global_pause = int(pause_meta["clip_relative_start_sample"])
    crop_start = global_pause - int(.75 * SR)
    if crop_start < 0 or crop_start + CROP_SAMPLES > len(clean):
        raise RuntimeError("fixed speaker crop does not fit source utterance")
    clean_crop = crop(clean, crop_start)
    pause_in_crop = global_pause - crop_start
    measure_end = global_pause + 9_600
    if measure_end > len(clean):
        raise RuntimeError("full-exact probe measurement window exceeds utterance")
    gain = float(row["post_cap_gain"])
    # One seed freezes reflection delays, signs, late-noise realization, and
    # damping draw across the entire RT60 × DRR grid.
    common_seed = seed_for(1, 0, namespace="F2-targeted-RIR-probe-2026-10-10")
    records = []
    started = time.perf_counter()
    for t60 in T60_VALUES:
        for drr in DRR_VALUES:
            rir, _, _ = make_procedural_rir(SR, common_seed, t60, drr,
                max_t60_s=max(T60_VALUES), min_direct_to_reverb_db=min(DRR_VALUES))
            pair = measured_pair(clean, rir)
            shared_scale = float(np.max(np.abs(pair["clean"]))) / max(
                float(np.max(np.abs(clean))), 1e-8)
            key = f"t60-{t60:.2f}-drr-{drr:+.0f}"
            wet_out, wet_meta = cap60_one(args.deep_filter, pair["reverberant"],
                SR, args.output_dir, key + "-wet", cache)
            clean_out, clean_meta = cap60_one(args.deep_filter, pair["clean"],
                SR, args.output_dir, key + "-clean", cache)
            levels = tail_input_levels(
                clean_out[crop_start:measure_end] * gain,
                wet_out[crop_start:measure_end] * gain,
                pause_in_crop, activity_clean_crop=clean_crop)
            record = {"t60_s": t60, "direct_to_reverb_db": drr,
                "seed": common_seed, "shared_scale": shared_scale,
                "w1_w2_tail_db": levels["window_db"],
                "reference_valid": bool(levels.get("reference_valid")),
                "full_eligible": bool(levels.get("eligible")),
                "cap60_wet_source_sha256": wet_meta["source_sha256"],
                "cap60_clean_source_sha256": clean_meta["source_sha256"],
                "cap60_wet_elapsed_s": float(wet_meta["elapsed_s"]),
                "cap60_clean_elapsed_s": float(clean_meta["elapsed_s"])}
            records.append(record)
            print(json.dumps({k: record[k] for k in
                ("t60_s", "direct_to_reverb_db", "shared_scale", "w1_w2_tail_db",
                 "full_eligible", "cap60_wet_elapsed_s", "cap60_clean_elapsed_s")},
                allow_nan=False), flush=True)
    successes = [r for r in records if r["full_eligible"]]
    result = {"name": "F2-targeted-RT60-DRR-full-exact-grid-v1",
        "schedule_index": int(args.schedule_index),
        "source_schedule_sha256": sha256(ROOT / ".tools/compact-dereverb/cap60-conditioned-2026-10-10/pairs.jsonl"),
        "source_pair_sha256": row["source_sha256"],
        "deep_filter_sha256": sha256(args.deep_filter),
        "seed_shared_across_grid": common_seed,
        "t60_values_s": list(T60_VALUES), "drr_values_db": list(DRR_VALUES),
        "full_exact_runs": len(records) * 2,
        "eligible_probe_count": len(successes),
        "eligible_probes": [{"t60_s": r["t60_s"],
            "direct_to_reverb_db": r["direct_to_reverb_db"]} for r in successes],
        "at_least_3_of_12_eligible": len(successes) >= 3,
        "same_random_geometry_seed": True,
        "surrogate_used": False,
        "selection_or_training_data_modified": False,
        "elapsed_s": time.perf_counter() - started,
        "records": records, "test_wav_accessed": False}
    (args.output_dir / "report.json").write_text(json.dumps(result,
        indent=2, allow_nan=False) + "\n")
    return {k: v for k, v in result.items() if k != "records"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=LIBRISPEECH)
    parser.add_argument("--deep-filter", type=Path, default=DEEP_FILTER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--schedule-index", type=int, default=5)
    parser.add_argument("--resume", action="store_true",
        help="reuse exact-hash outputs from this targeted probe only")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
