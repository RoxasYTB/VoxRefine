#!/usr/bin/env python3
"""Rescore cached F2 prefix outputs using one fixed activity/reference horizon."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts/benchmarks/universal_enhancer"))

from data import make_procedural_rir  # noqa: E402
from prepare_cap60_conditioned_pairs import (  # noqa: E402
    CROP_SAMPLES, DEEP_FILTER, LIBRISPEECH,
)
from prepare_f2_tailbank import (  # noqa: E402
    candidate_parameters, load_schedule, resolve_source, seed_for, tail_input_levels,
)
from calibrate_f2_cap60_prefix import hash_array  # noqa: E402
from train_measured_mix import measured_pair  # noqa: E402

BASE = ROOT / ".tools/compact-dereverb/e2-f2-tailbank-2026-10-10/prefix-calibration"
OUT = BASE / "rescored-fixed2s-activity"
CANDIDATES = (0, 1, 3, 7, 15, 31, 63, 127)
WINDOW_END = 9_600


def load_exact(cache: Path, signal: np.ndarray, exe_hash: str) -> np.ndarray:
    key = hash_array(np.asarray(signal, dtype=np.float32))
    audio_path, meta_path = cache / f"{key}.npy", cache / f"{key}.json"
    if not audio_path.is_file() or not meta_path.is_file():
        raise RuntimeError(f"missing cached Cap60 signal {key} in {cache}")
    meta = json.loads(meta_path.read_text())
    if meta.get("source_sha256") != key or meta.get("deep_filter_sha256") != exe_hash:
        raise RuntimeError(f"invalid cached Cap60 signal {key}")
    return np.load(audio_path, allow_pickle=False).astype(np.float32)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def run() -> dict:
    initial_report = json.loads((BASE / "report.json").read_text())
    exe_hash = hashlib.sha256(DEEP_FILTER.read_bytes()).hexdigest()
    if exe_hash != initial_report.get("deep_filter_sha256"):
        raise RuntimeError("Cap60 binary differs from the frozen calibration")
    old_rows = [json.loads(line) for line in (BASE / "results.jsonl").read_text().splitlines() if line]
    old_by_key = {(int(x["tail_slot"]), int(x["candidate_index"])): x for x in old_rows}
    if len(old_by_key) != 64:
        raise RuntimeError("initial prefix calibration is incomplete")
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    schedule = [r for r in load_schedule() if r["kind"] == "rir_only"]
    full_cache = BASE / "exact-hash-cache"
    # Both full and prefix outputs are stored in the shared exact-byte cache;
    # their source hashes distinguish the different input lengths.
    prefix_cache = full_cache
    new_rows = []

    for slot in range(8):
        row = schedule[slot]
        clean, pause_meta, _ = resolve_source(row, LIBRISPEECH / "train-clean-100")
        global_pause = int(pause_meta["clip_relative_start_sample"])
        crop_start = global_pause - int(.75 * 16_000)
        measure_end = global_pause + WINDOW_END
        pause_in_crop = global_pause - crop_start
        activity = clean[crop_start:crop_start + CROP_SAMPLES]
        gain = float(row["post_cap_gain"])

        for candidate in CANDIDATES:
            old = old_by_key[(slot, candidate)]
            seed = seed_for(slot, candidate)
            t60, drr = candidate_parameters(seed)
            rir, _, _ = make_procedural_rir(16_000, seed, t60, drr)
            pair = measured_pair(clean, rir)
            prefix_end = measure_end
            full_wet = load_exact(full_cache, pair["reverberant"], exe_hash)
            full_clean = load_exact(full_cache, pair["clean"], exe_hash)
            prefix_wet = load_exact(prefix_cache, pair["reverberant"][:prefix_end], exe_hash)
            prefix_clean = load_exact(prefix_cache, pair["clean"][:prefix_end], exe_hash)
            full = tail_input_levels(full_clean[crop_start:measure_end] * gain,
                full_wet[crop_start:measure_end] * gain, pause_in_crop,
                activity_clean_crop=activity)
            prefix = tail_input_levels(prefix_clean[crop_start:measure_end] * gain,
                prefix_wet[crop_start:measure_end] * gain, pause_in_crop,
                activity_clean_crop=activity)
            result = {"tail_slot": slot, "candidate_index": candidate,
                "full_window_db": full["window_db"], "prefix_window_db": prefix["window_db"],
                "window_level_delta_db": {w: (prefix["window_db"][w] - full["window_db"][w]
                    if prefix["window_db"][w] is not None and full["window_db"][w] is not None else None)
                    for w in ("150_300", "300_600")},
                "full_eligible": bool(full.get("eligible", False)),
                "prefix_eligible": bool(prefix.get("eligible", False)),
                "waveform_error_reference_plus_windows": old["waveform_error_reference_plus_windows"],
                "near_threshold_windows": [w for w, v in full["window_db"].items()
                    if v is not None and -55 <= v <= -45]}
            new_rows.append(result)

    results_path = OUT / "results.jsonl"
    results_path.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in new_rows))
    max_wave = max(x["max_abs"] for r in new_rows
                   for x in r["waveform_error_reference_plus_windows"])
    max_rms = max(x["rms"] for r in new_rows
                  for x in r["waveform_error_reference_plus_windows"])
    max_delta = max(abs(v) for r in new_rows
                    for v in r["window_level_delta_db"].values() if v is not None)
    disagreements = sum(r["full_eligible"] != r["prefix_eligible"] for r in new_rows)
    report = {"name": "F2-Cap60-prefix-common-mask-rescore-v1",
        "source_schedule_sha256": initial_report["source_schedule_sha256"],
        "deep_filter_sha256": exe_hash,
        "prefix_end_after_pause_ms": 600,
        "measurement_end_after_pause_ms": 600,
        "activity_mask_source": "raw clean on the frozen 2s training crop",
        "frame_mapping": "10ms sample-time bins",
        "completed_candidates": len(new_rows),
        "max_waveform_error": max_wave, "max_waveform_rms_error": max_rms,
        "max_abs_window_level_delta_db": max_delta,
        "eligibility_disagreements": disagreements,
        "near_threshold_count": sum(bool(r["near_threshold_windows"]) for r in new_rows),
        "empirical_prefix_gate_pass": (len(new_rows) == 64 and max_wave < 1e-7 and
            max_delta < .01 and disagreements == 0),
        "test_wav_accessed": False}
    atomic_json(OUT / "report.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, allow_nan=False))
