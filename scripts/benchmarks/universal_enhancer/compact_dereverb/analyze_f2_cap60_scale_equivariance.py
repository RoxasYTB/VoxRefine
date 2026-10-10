#!/usr/bin/env python3
"""Measure Cap60 target-reference scale equivariance from cached 64 cases."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from prepare_cap60_conditioned_pairs import CROP_SAMPLES, LIBRISPEECH, SR, DEEP_FILTER, sha256
from prepare_f2_tailbank import load_schedule, resolve_source

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / ".tools/compact-dereverb/e2-f2-tailbank-2026-10-10/prefix-calibration"
ASSET = ROOT / "docs/benchmarking/assets/compact-dereverb/f2-prefix-calibration-2026-10-10"
FINAL_CACHE = BASE / "prefix-margin-1600ms-common-mask-release/exact-hash-cache"
FINAL_RESULTS = ASSET / "prefix-1600ms-common-activity-results.jsonl"
BASE_RESULTS = BASE / "results.jsonl"
OUT = BASE / "scale-equivariance-report.json"


def sample_mask(clean: np.ndarray, crop_start: int, measure_end: int,
                pause: int) -> np.ndarray:
    activity = np.asarray(clean[crop_start:crop_start + CROP_SAMPLES], np.float32)
    frame, hop = 320, 160
    frames = np.lib.stride_tricks.sliding_window_view(activity, frame)[::hop]
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-24)
    active = rms >= max(float(rms.max()) * .02, 1e-5)
    centers = np.arange(len(active)) * hop + frame // 2
    reference = active & (centers >= pause - 8_000) & (centers < pause - 800)
    if int(reference.sum()) * hop < 3_200:
        raise RuntimeError("fixed active reference has less than 200ms speech")
    count = measure_end - crop_start
    nearest = np.minimum(np.arange(count, dtype=np.int64) // hop, len(reference) - 1)
    return reference[nearest]


def load_cached_output(source_hash: str) -> np.ndarray:
    audio_path = FINAL_CACHE / f"{source_hash}.npy"
    meta_path = FINAL_CACHE / f"{source_hash}.json"
    if not audio_path.is_file() or not meta_path.is_file():
        raise RuntimeError(f"missing prefix clean output for {source_hash}")
    meta = json.loads(meta_path.read_text())
    if (meta.get("source_sha256") != source_hash or
            meta.get("deep_filter_sha256") != sha256(DEEP_FILTER)):
        raise RuntimeError(f"invalid prefix clean output for {source_hash}")
    return np.load(audio_path, allow_pickle=False).astype(np.float32)


def run() -> dict:
    base = [json.loads(line) for line in BASE_RESULTS.read_text().splitlines() if line]
    margin = [json.loads(line) for line in FINAL_RESULTS.read_text().splitlines() if line]
    base_map = {(int(x["tail_slot"]), int(x["candidate_index"])): x for x in base}
    margin_map = {(int(x["tail_slot"]), int(x["candidate_index"])): x for x in margin}
    if len(base_map) != 64 or len(margin_map) != 64:
        raise RuntimeError("scale experiment requires the frozen 8x8 cache")
    schedule = [row for row in load_schedule() if row["kind"] == "rir_only"]
    rows = []
    global_errors = []
    for slot, schedule_row in enumerate(schedule[:8]):
        clean, pause_meta, _ = resolve_source(schedule_row, LIBRISPEECH / "train-clean-100")
        global_pause = int(pause_meta["clip_relative_start_sample"])
        crop_start = global_pause - int(.75 * SR)
        measure_end = global_pause + 9_600
        pause_in_crop = global_pause - crop_start
        mask = sample_mask(clean, crop_start, measure_end, pause_in_crop)
        gain = float(schedule_row["post_cap_gain"])
        slot_items = []
        for candidate in (0, 1, 3, 7, 15, 31, 63, 127):
            source_row = base_map[(slot, candidate)]
            cached_row = margin_map[(slot, candidate)]
            scale = float(source_row["shared_scale"])
            source_hash = cached_row["clean_prefix_source_sha256"]
            output = load_cached_output(source_hash)
            if len(output) < measure_end:
                raise RuntimeError("cached Cap60 prefix is shorter than measurement region")
            reference = output[crop_start:measure_end].astype(np.float64) * gain
            energy = float(np.mean(reference[mask] ** 2))
            if not np.isfinite(energy) or energy <= 0:
                raise RuntimeError("invalid Cap60-clean active reference energy")
            slot_items.append({"slot": slot, "candidate_index": candidate,
                "shared_scale": scale, "eref": energy,
                "source_sha256": source_hash,
                "normalized_output": (reference / scale).astype(np.float32)})
        reference_item = slot_items[0]
        reference_scale = float(reference_item["shared_scale"])
        reference_energy = float(reference_item["eref"])
        normalized_ref = reference_item["normalized_output"].astype(np.float64)
        slot_errors = []
        for item in slot_items:
            predicted_energy = reference_energy * (float(item["shared_scale"]) / reference_scale) ** 2
            error_db = 10 * np.log10(float(item["eref"]) / predicted_energy)
            diff = item["normalized_output"].astype(np.float64) - normalized_ref
            row = {key: value for key, value in item.items() if key != "normalized_output"}
            row.update({"predicted_eref": float(predicted_energy),
                "eref_error_db": float(error_db),
                "normalized_wave_max_abs": float(np.max(np.abs(diff))),
                "normalized_wave_rms_error": float(np.sqrt(np.mean(diff * diff)))})
            rows.append(row)
            slot_errors.append(row)
            global_errors.append(abs(float(error_db)))
        scales = [float(item["shared_scale"]) for item in slot_items]
        slot_max = max(abs(float(item["eref_error_db"])) for item in slot_errors)
        slot_wave_max = max(float(item["normalized_wave_max_abs"]) for item in slot_errors)
        print(json.dumps({"slot": slot, "scale_min": min(scales),
            "scale_max": max(scales), "max_abs_eref_error_db": slot_max,
            "max_normalized_wave_error": slot_wave_max}, allow_nan=False), flush=True)

    max_error = max(global_errors)
    report = {"name": "F2-Cap60-scale-equivariance-v1",
        "source_schedule_sha256": sha256(ROOT / ".tools/compact-dereverb/cap60-conditioned-2026-10-10/pairs.jsonl"),
        "deep_filter_sha256": sha256(DEEP_FILTER), "slots": 8,
        "candidate_indices": [0, 1, 3, 7, 15, 31, 63, 127],
        "reference_policy": "Cap60(clean*s0) reference-energy scaled by (si/s0)^2; diagnostic/rejection-only hypothesis",
        "candidate_scale_min": min(row["shared_scale"] for row in rows),
        "candidate_scale_max": max(row["shared_scale"] for row in rows),
        "max_abs_eref_error_db": max_error,
        "mean_abs_eref_error_db": float(np.mean(global_errors)),
        "max_normalized_wave_error": max(row["normalized_wave_max_abs"] for row in rows),
        "cases": rows, "test_wav_accessed": False}
    # Check the rejection-only rule against exact full scores using the measured
    # Eref error bound plus 1dB. Any passing full candidate may not be screened.
    full_eligibility = {(int(x["tail_slot"]), int(x["candidate_index"])): bool(x["full_eligible"])
                        for x in margin}
    for row in rows:
        key = (int(row["slot"]), int(row["candidate_index"]))
        conservative = max_error + 1.0
        actual_tail = margin_map[key]["prefix_margin_window_db"]
        predicted_tail = {window: (float(value) + float(row["eref_error_db"])
                            if value is not None else None)
                          for window, value in actual_tail.items()}
        row["predicted_prefix_tail_db_from_shared_scale"] = predicted_tail
        row["conservative_reject"] = any(value is not None and
            float(value) <= -50.0 - conservative for value in predicted_tail.values())
        row["full_eligible"] = full_eligibility[key]
    report["false_negative_count_at_bound_plus_1db"] = sum(
        bool(row["full_eligible"]) and bool(row["conservative_reject"]) for row in rows)
    report["no_false_negative_at_1db_margin"] = report["false_negative_count_at_bound_plus_1db"] == 0
    OUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return {k: v for k, v in report.items() if k != "cases"}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, allow_nan=False))
