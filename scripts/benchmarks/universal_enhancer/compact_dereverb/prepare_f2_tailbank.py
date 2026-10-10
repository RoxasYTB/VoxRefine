#!/usr/bin/env python3
"""Prepare E″/F″ from the fixed C pair schedule with input-only tailbank RIRs."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
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
    CROP_SAMPLES, DEEP_FILTER, LIBRISPEECH, BUT_ROOT, SR, cap60_one,
    make_fan_mix, ratio_db, sha256,
)
from screen_measured_rirs import (  # noqa: E402
    make_inserted_pause_pair, prepare_rir, select_rirs,
)
from train_measured_mix import measured_pair  # noqa: E402

C_DATA = ROOT / ".tools/compact-dereverb/cap60-conditioned-2026-10-10"
DEFAULT_OUT = ROOT / ".tools/compact-dereverb/e2-f2-tailbank-2026-10-10/data"
MAX_CANDIDATES = 256
TAIL_THRESHOLD_DB = -50.0
PREFIX_CONTEXT_AFTER_PAUSE_SAMPLES = 25_600
PREFIX_SCREEN_MARGIN_DB = 1.0
CALIBRATION_ASSETS = (ROOT / "docs/benchmarking/assets/compact-dereverb/"
    "f2-prefix-calibration-2026-10-10")
PREFIX_CALIBRATION = CALIBRATION_ASSETS / "prefix-1600ms-common-activity-report.json"
SCALE_CALIBRATION = CALIBRATION_ASSETS / "scale-equivariance-report.json"
SCALE_POSITIVE_CONTROL = CALIBRATION_ASSETS / "positive-scale-surrogate-check.json"
SCALE_SURROGATE_OBSERVED_ERROR_DB = 0.03952134427315273
SCALE_SURROGATE_SAFETY_DB = 1.0
SCALE_SURROGATE_OPERATIONAL_BOUND_DB = SCALE_SURROGATE_OBSERVED_ERROR_DB + SCALE_SURROGATE_SAFETY_DB


def seed_for(slot: int, candidate: int, namespace: str = "F2-tailbank") -> int:
    payload = f"{namespace}|{slot}|{candidate}".encode()
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def candidate_parameters(seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    return float(rng.uniform(.45, 1.10)), float(rng.uniform(-6.0, 18.0))


def crop(audio: np.ndarray, start: int) -> np.ndarray:
    out = np.asarray(audio[start:start + CROP_SAMPLES], dtype=np.float32)
    if len(out) != CROP_SAMPLES:
        raise RuntimeError("crop falls outside the precomputed utterance")
    return out


def tail_input_levels(reference_crop: np.ndarray, cap_input_crop: np.ndarray,
                      pause: int, activity_clean_crop: np.ndarray | None = None) -> dict:
    """Measure input tails against Cap60(clean-scaled), masking from raw clean."""
    activity_clean_crop = (reference_crop if activity_clean_crop is None
                           else activity_clean_crop)
    if len(reference_crop) != len(cap_input_crop) or len(activity_clean_crop) < len(reference_crop):
        raise ValueError("reference/input lengths must match and clean activity crop must cover them")
    frame, hop = 320, 160
    frames = np.lib.stride_tricks.sliding_window_view(activity_clean_crop, frame)[::hop]
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-24)
    active = rms >= max(float(rms.max()) * .02, 1e-5)
    centers = np.arange(len(active)) * hop + frame // 2
    reference = active & (centers >= pause - 8_000) & (centers < pause - 800)
    if int(reference.sum()) * hop < 3_200:
        return {"reference_valid": False, "reference_active_samples": int(reference.sum()) * hop,
                "window_db": {"150_300": None, "300_600": None}}
    # Activity comes from a fixed raw-clean crop, which may extend beyond the
    # prefix being scored. Map each sample to its 10ms frame by time, rather
    # than stretching the activity mask to the shorter prefix length.
    nearest = np.minimum(np.arange(len(reference_crop), dtype=np.int64) // hop,
                         max(len(reference) - 1, 0))
    sample_mask = reference[nearest]
    eref = float(np.mean(np.asarray(reference_crop[sample_mask], np.float64) ** 2))
    eps = 1e-12 * eref
    levels = {}
    for key, start_ms, end_ms in (("150_300", 150, 300), ("300_600", 300, 600)):
        start, end = pause + start_ms * 16, pause + end_ms * 16
        section = np.asarray(cap_input_crop[start:end], dtype=np.float64)
        energy = float(np.mean(section * section))
        levels[key] = float(10 * np.log10((energy + eps) / (eref + eps)))
    return {"reference_valid": True,
        "reference_active_samples": int(reference.sum()) * hop,
        "window_db": levels,
        "reference_energy": eref,
        "eligible": all(value > TAIL_THRESHOLD_DB for value in levels.values())}


def resolve_source(row: dict, speech_root: Path) -> tuple[np.ndarray, dict, tuple[Path, Path]]:
    speaker = str(row["speaker"])
    paths_by_name = {path.name: path for path in (speech_root / speaker).glob("*/*.flac")}
    names = row["source"]["source_pair"]
    paths = tuple(paths_by_name[name] for name in names)
    if [sha256(path) for path in paths] != row["source_sha256"]:
        raise RuntimeError("fixed C schedule source hash mismatch")
    joined, pause_meta = make_inserted_pause_pair(*paths)
    joined = joined.astype(np.float32)
    joined *= .80 / max(float(np.max(np.abs(joined))), 1e-8)
    return joined, pause_meta, paths


def load_schedule() -> list[dict]:
    path = C_DATA / "pairs.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    expected = {"identity": 32, "rir_only": 48, "fan20": 24, "fan10": 24}
    actual = {kind: sum(row["kind"] == kind for row in rows) for kind in expected}
    if len(rows) != 128 or actual != expected or len({str(r["speaker"]) for r in rows}) != 128:
        raise RuntimeError(f"fixed C schedule is not the expected 128 unique-speaker mix: {actual}")
    return rows


def load_prefix_screening_policy(schedule_path: Path, executable: Path) -> tuple[dict, str]:
    """Fail closed unless prefix and clean-reference surrogate calibrations match."""
    for path in (PREFIX_CALIBRATION, SCALE_CALIBRATION, SCALE_POSITIVE_CONTROL):
        if not path.is_file():
            raise RuntimeError(f"missing screening calibration artifact: {path}")
    report = json.loads(PREFIX_CALIBRATION.read_text())
    scale = json.loads(SCALE_CALIBRATION.read_text())
    positive = json.loads(SCALE_POSITIVE_CONTROL.read_text())
    schedule_hash = sha256(schedule_path)
    exe_hash = sha256(executable)
    if (report.get("empirical_prefix_gate_pass") is not True or
            report.get("completed_candidates") != 64 or
            report.get("source_schedule_sha256") != schedule_hash or
            report.get("deep_filter_sha256") != exe_hash or
            report.get("prefix_end_after_pause_ms") != 1600 or
            report.get("max_waveform_error", float("inf")) >= 1e-7 or
            report.get("max_abs_window_level_delta_db", float("inf")) >= .01 or
            report.get("eligibility_disagreements") != 0):
        raise RuntimeError("prefix screen is not backed by its matching 64-case calibration")
    margins = report.get("per_window_screening_margin_db", {})
    if any(float(margins.get(key, 0)) < PREFIX_SCREEN_MARGIN_DB
           for key in ("150_300", "300_600")):
        raise RuntimeError("prefix screening safety margin is below 1dB")
    observed_error = float(scale.get("max_abs_eref_error_db", float("inf")))
    if (scale.get("name") != "F2-Cap60-scale-equivariance-v1" or
            scale.get("source_schedule_sha256") != schedule_hash or
            scale.get("deep_filter_sha256") != exe_hash or
            scale.get("test_wav_accessed") is not False or
            scale.get("no_false_negative_at_1db_margin") is not True or
            scale.get("false_negative_count_at_bound_plus_1db") != 0 or
            abs(observed_error - SCALE_SURROGATE_OBSERVED_ERROR_DB) > 1e-12):
        raise RuntimeError("scale surrogate is not backed by its matching frozen calibration")
    if (positive.get("passed") is not True or
            positive.get("full_eligible") is not True or
            positive.get("surrogate_would_reject") is not False or
            positive.get("deep_filter_sha256") != exe_hash or
            positive.get("test_wav_accessed") is not False):
        raise RuntimeError("scale surrogate positive control failed")
    if (float(scale.get("candidate_scale_min", 0)) <= 0 or
            float(scale.get("candidate_scale_max", 0)) <= 0):
        raise RuntimeError("invalid scale calibration envelope")
    policy = {"prefix": sha256(PREFIX_CALIBRATION),
        "scale": sha256(SCALE_CALIBRATION), "positive": sha256(SCALE_POSITIVE_CONTROL),
        "observed_eref_error_bound_db": observed_error,
        "operational_eref_error_bound_db": SCALE_SURROGATE_OPERATIONAL_BOUND_DB,
        "scale_min": float(scale["candidate_scale_min"]),
        "scale_max": float(scale["candidate_scale_max"]),
        "threshold_db": TAIL_THRESHOLD_DB, "prefix_context_after_pause_ms": 1600}
    policy_hash = hashlib.sha256(json.dumps(policy, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()
    return {"prefix": report, "scale": scale, "positive": positive,
        "policy": policy}, policy_hash


def index_cap60_cache(cache_dir: Path) -> dict[str, tuple[Path, Path]]:
    """Index prior Cap60 outputs by exact source bytes for safe local reuse."""
    index: dict[str, tuple[Path, Path]] = {}
    if not cache_dir.is_dir():
        return index
    for meta_path in sorted(cache_dir.glob("*.json")):
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        audio_path = meta_path.with_suffix(".npy")
        source_hash = meta.get("source_sha256")
        if source_hash and audio_path.is_file():
            index.setdefault(str(source_hash), (audio_path, meta_path))
    return index


def cap60_cached(executable: Path, x: np.ndarray, work: Path, key: str,
                 cache_dir: Path, reuse_index: dict[str, tuple[Path, Path]]) -> tuple[np.ndarray, dict]:
    """Reuse prior preprocessing only when the exact unmodified source hash matches."""
    source_hash = hashlib.sha256(np.asarray(x).tobytes()).hexdigest()
    target_audio, target_meta = cache_dir / f"{key}.npy", cache_dir / f"{key}.json"
    if not target_audio.exists() and source_hash in reuse_index:
        source_audio, source_meta = reuse_index[source_hash]
        if source_audio.is_file() and source_meta.is_file():
            shutil.copy2(source_audio, target_audio)
            shutil.copy2(source_meta, target_meta)
        else:
            reuse_index.pop(source_hash, None)
    result = cap60_one(executable, x, SR, work, key, cache_dir)
    # Exact-byte matches within this run may reuse their canonical Cap60
    # output too. Candidate-specific clean scales remain distinct hashes.
    reuse_index.setdefault(source_hash, (target_audio, target_meta))
    return result


def prepare(args) -> dict:
    if not args.deep_filter.is_file():
        raise FileNotFoundError(args.deep_filter)
    schedule = load_schedule()
    schedule_path = C_DATA / "pairs.jsonl"
    prefix_policy, prefix_policy_hash = load_prefix_screening_policy(
        schedule_path, args.deep_filter)
    old_manifest = json.loads((C_DATA / "training-data-manifest.json").read_text())
    if old_manifest.get("test_wav_accessed") is not False:
        raise RuntimeError("fixed C schedule violates sealed test.wav policy")
    if old_manifest.get("deep_filter_sha256") != sha256(args.deep_filter):
        raise RuntimeError("fixed C schedule was generated with a different Cap60 binary")
    by_configuration = {item["configuration"]: item for item in select_rirs(args.rir_root)}
    output = args.output_dir
    progress_path = output / "progress.json"
    if output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {output}; use --resume for its own partial cache")
    if output.exists() and (output / "pairs.jsonl").exists():
        raise FileExistsError("completed pair manifest exists; refusing to replace it")
    output.mkdir(parents=True, exist_ok=True)
    arrays = output / "arrays"
    arrays.mkdir(exist_ok=True)
    cache = output / "cap60-cache"
    cache.mkdir(exist_ok=True)
    reuse_index = index_cap60_cache(C_DATA / "cap60-cache")
    for source_hash, paths in index_cap60_cache(cache).items():
        reuse_index.setdefault(source_hash, paths)
    schedule_sha = sha256(C_DATA / "pairs.jsonl")
    if args.resume and progress_path.is_file():
        progress = json.loads(progress_path.read_text())
        if (progress.get("source_schedule_manifest_sha256") != schedule_sha or
                progress.get("deep_filter_sha256") != sha256(args.deep_filter)):
            raise RuntimeError("partial cache belongs to a different frozen schedule or Cap60 binary")
        if progress.get("screen_surrogate_invalidated"):
            raise RuntimeError("screen surrogate was invalidated; this partial cache must be regenerated")
        result_rows = progress.get("completed_rows", [])
        if progress.get("selection_policy_sha256") != prefix_policy_hash:
            tail_rows = [row for row in result_rows if row.get("kind") == "rir_only"]
            unsafe = any(any(item.get("screening_reference_mode") == "scale_surrogate"
                for item in row.get("rir_recipe", {}).get("candidate_history", []))
                for row in tail_rows)
            if unsafe:
                raise RuntimeError("partial cache contains surrogate rows from a different policy")
            progress["selection_policy_sha256"] = prefix_policy_hash
            progress["prefix_calibration_sha256"] = sha256(PREFIX_CALIBRATION)
            progress["scale_calibration_sha256"] = sha256(SCALE_CALIBRATION)
            progress["positive_control_sha256"] = sha256(SCALE_POSITIVE_CONTROL)
            progress["selection_policy_migration"] = "safe: existing tail slots used exact clean-prefix/full selection"
            progress_tmp = progress_path.with_suffix(".json.tmp")
            progress_tmp.write_text(json.dumps(progress, allow_nan=False) + "\n")
            progress_tmp.replace(progress_path)
        if [int(row["index"]) for row in result_rows] != list(range(len(result_rows))):
            raise RuntimeError("partial cache is not a contiguous prefix of the frozen schedule")
    else:
        result_rows = []
    progress = {"source_schedule_manifest_sha256": schedule_sha,
        "deep_filter_sha256": sha256(args.deep_filter),
        "selection_policy_sha256": prefix_policy_hash,
        "prefix_calibration_sha256": sha256(PREFIX_CALIBRATION),
        "scale_calibration_sha256": sha256(SCALE_CALIBRATION),
        "positive_control_sha256": sha256(SCALE_POSITIVE_CONTROL),
        "completed_rows": result_rows, "test_wav_accessed": False}
    completed_indices = {int(row["index"]) for row in result_rows}
    selection_counts = [{"slot": int(row["rir_recipe"]["tail_slot"]),
        "resolved": True, "candidates_tested": int(row["rir_recipe"]["candidates_tested"])}
        for row in result_rows if row.get("kind") == "rir_only"]
    started = time.perf_counter()
    slot = sum(row["kind"] == "rir_only" for row in result_rows)

    for row in schedule:
        index = int(row["index"])
        if index in completed_indices:
            continue
        clean, pause_meta, paths = resolve_source(row, args.data_root / "train-clean-100")
        global_pause = int(pause_meta["clip_relative_start_sample"])
        if row["kind"] == "rir_only":
            crop_start = global_pause - int(.75 * SR)
            crop_policy = "pause-centered-750ms-before"
        else:
            crop_start = int(row["crop_start_sample"])
            crop_policy = "fixed-C-schedule-crop"
        if crop_start < 0 or crop_start + CROP_SAMPLES > len(clean):
            raise RuntimeError(f"fixed slot {index} cannot fit the preregistered crop")
        clean_crop = crop(clean, crop_start)
        pause_in_crop = global_pause - crop_start
        gain = float(row["post_cap_gain"])
        kind = row["kind"]
        selected_recipe = None
        branch_scale = float(row.get("shared_branch_scale", 1.0))

        if kind == "identity":
            input_full, cap_meta = cap60_cached(args.deep_filter, clean, output,
                f"input-fixed-{index:03d}", cache, reuse_index)
            target_full = input_full
            clean_for_ratio = clean
            rir_kind, rir_room = "identity", None
        elif kind == "rir_only":
            input_full = None
            cap_meta = None
            selected_pair = None
            selected_target = None
            candidate_history = []
            tail_slot = slot
            scale_report = prefix_policy["scale"]
            scale_min = float(scale_report["candidate_scale_min"])
            scale_max = float(scale_report["candidate_scale_max"])
            scale_reference = None
            reference_candidate_index = None
            target_reference = None
            target_reference_meta = None
            for candidate in range(MAX_CANDIDATES):
                candidate_seed = seed_for(tail_slot, candidate)
                t60_s, drr_db = candidate_parameters(candidate_seed)
                rir, _, _ = make_procedural_rir(SR, candidate_seed, t60_s, drr_db)
                pair_data = measured_pair(clean, rir)
                candidate_key = f"tailbank-{tail_slot:02d}-candidate-{candidate:03d}"
                target_key = f"tailbank-{tail_slot:02d}-target-candidate-{candidate:03d}"
                measure_end = global_pause + 9_600
                measure_count = measure_end - crop_start
                if measure_count <= 0 or measure_end > len(clean):
                    raise RuntimeError("tail measurement horizon falls outside the utterance")
                prefix_end = global_pause + PREFIX_CONTEXT_AFTER_PAUSE_SAMPLES
                if prefix_end > len(clean):
                    raise RuntimeError(f"prefix screening context falls outside tail slot {tail_slot}")
                shared_scale = float(np.max(np.abs(pair_data["clean"]))) / max(
                    float(np.max(np.abs(clean))), 1e-8)
                scale_in_range = scale_min <= shared_scale <= scale_max
                prefix_input_key = f"tailbank-screen-{tail_slot:02d}-candidate-{candidate:03d}"
                cap_prefix, prefix_meta = cap60_cached(args.deep_filter,
                    pair_data["reverberant"][:prefix_end], output,
                    prefix_input_key, cache, reuse_index)

                # Candidate 0 is an exact full-length Cap60(clean) reference.
                # Out-of-envelope candidates also use exact full-length targets;
                # the surrogate is never extrapolated beyond the calibration.
                exact_target_reason = ("reference_exact_full" if target_reference is None
                    else "scale_outside_calibration_full_fallback" if not scale_in_range
                    else None)
                cap_target_candidate = None
                target_candidate_meta = None
                if exact_target_reason is not None:
                    cap_target_candidate, target_candidate_meta = cap60_cached(
                        args.deep_filter, pair_data["clean"], output,
                        target_key, cache, reuse_index)
                    target_prefix = cap_target_candidate[:prefix_end]
                    if scale_in_range and target_reference is None:
                        scale_reference = shared_scale
                        reference_candidate_index = candidate
                        target_reference = cap_target_candidate
                        target_reference_meta = target_candidate_meta
                        exact_target_reason = "reference_exact_full"
                    reference_mode = "exact_full_target"
                    reference_source_hash = target_candidate_meta["source_sha256"]
                    reference_scale = shared_scale
                else:
                    ratio = shared_scale / float(scale_reference)
                    target_prefix = target_reference[:prefix_end] * np.float32(ratio)
                    reference_mode = "scale_surrogate"
                    exact_target_reason = None
                    reference_source_hash = target_reference_meta["source_sha256"]
                    reference_scale = float(scale_reference)

                prefix_measurements = tail_input_levels(
                    target_prefix[crop_start:measure_end] * gain,
                    cap_prefix[crop_start:measure_end] * gain,
                    pause_in_crop, activity_clean_crop=clean_crop)
                prefix_levels = prefix_measurements.get("window_db", {})
                reject_threshold_value = TAIL_THRESHOLD_DB - PREFIX_SCREEN_MARGIN_DB
                if reference_mode == "scale_surrogate":
                    reject_threshold_value = TAIL_THRESHOLD_DB - SCALE_SURROGATE_OPERATIONAL_BOUND_DB
                reject_thresholds = {key: reject_threshold_value
                    for key in ("150_300", "300_600")}
                screened_out = any(prefix_levels.get(key) is not None and
                    float(prefix_levels[key]) <= reject_thresholds[key]
                    for key in reject_thresholds)
                candidate_record = {"candidate_index": candidate,
                    "seed": candidate_seed, "t60_s": t60_s,
                    "direct_to_reverb_db": drr_db,
                    "target_source_sha256": (target_candidate_meta["source_sha256"]
                        if target_candidate_meta else None),
                    "prefix_target_source_sha256": (target_candidate_meta["source_sha256"]
                        if target_candidate_meta else None),
                    "prefix_input_source_sha256": prefix_meta["source_sha256"],
                    "shared_scale": shared_scale,
                    "scale_in_calibration_range": scale_in_range,
                    "screening_reference_mode": reference_mode,
                    "reference_candidate_index": (candidate if reference_mode == "exact_full_target"
                        and scale_reference == shared_scale else reference_candidate_index),
                    "reference_scale": reference_scale,
                    "reference_source_sha256": reference_source_hash,
                    "scale_ratio": (1.0 if reference_mode == "exact_full_target"
                        else shared_scale / float(scale_reference)),
                    "scale_calibration_error_bound_db": (
                        SCALE_SURROGATE_OPERATIONAL_BOUND_DB
                        if reference_mode == "scale_surrogate" else 0.0),
                    "prefix_input_tail_db": prefix_levels,
                    "prefix_reject_threshold_db": reject_thresholds,
                    "prefix_screened_out": bool(screened_out),
                    "input_tail_db": None,
                    "eligible": False,
                    "reason": "prefix_screen_reject" if screened_out else "full_not_run_yet"}
                if screened_out:
                    candidate_history.append(candidate_record)
                    continue

                # Full wet and clean signals decide every candidate that could
                # pass. Exact targets already computed above are reused.
                cap_candidate, candidate_meta = cap60_cached(args.deep_filter,
                    pair_data["reverberant"], output, candidate_key, cache, reuse_index)
                if cap_target_candidate is None:
                    cap_target_candidate, target_candidate_meta = cap60_cached(
                        args.deep_filter, pair_data["clean"], output,
                        target_key, cache, reuse_index)
                measurements = tail_input_levels(
                    cap_target_candidate[crop_start:measure_end] * gain,
                    cap_candidate[crop_start:measure_end] * gain,
                    pause_in_crop, activity_clean_crop=clean_crop)
                eligible = bool(measurements.get("eligible", False))
                candidate_record.update({"target_source_sha256": target_candidate_meta["source_sha256"],
                    "input_tail_db": measurements.get("window_db"),
                    "full_target_reference_energy": measurements.get("reference_energy"),
                    "eligible": eligible,
                    "reason": "accept_full" if eligible else "full_reject"})
                if reference_mode == "scale_surrogate":
                    surrogate_energy = float(prefix_measurements["reference_energy"])
                    exact_energy = float(measurements["reference_energy"])
                    eref_error_db = float(10 * np.log10(exact_energy / surrogate_energy))
                    exact_prefix_levels = tail_input_levels(
                        cap_target_candidate[crop_start:measure_end] * gain,
                        cap_prefix[crop_start:measure_end] * gain,
                        pause_in_crop, activity_clean_crop=clean_crop).get("window_db", {})
                    prefix_full_delta = {key: float(exact_prefix_levels[key] -
                        measurements["window_db"][key]) for key in ("150_300", "300_600")}
                    candidate_record["eref_surrogate_error_db"] = eref_error_db
                    candidate_record["prefix_vs_full_exact_target_window_delta_db"] = prefix_full_delta
                    if abs(eref_error_db) > SCALE_SURROGATE_OBSERVED_ERROR_DB:
                        candidate_record["eref_bound_warning"] = True
                    severe_error = abs(eref_error_db) > SCALE_SURROGATE_OPERATIONAL_BOUND_DB
                    prefix_drift = max(abs(value) for value in prefix_full_delta.values()) > .01
                    if severe_error or prefix_drift:
                        invalidation = {"reason": "surrogate_calibration_bound_exceeded" if severe_error
                            else "prefix_full_exact_drift_exceeded",
                            "tail_slot": tail_slot, "candidate_index": candidate,
                            "eref_surrogate_error_db": eref_error_db,
                            "operational_eref_bound_db": SCALE_SURROGATE_OPERATIONAL_BOUND_DB,
                            "prefix_full_exact_window_delta_db": prefix_full_delta,
                            "test_wav_accessed": False}
                        progress["screen_surrogate_invalidated"] = invalidation
                        progress_tmp = progress_path.with_suffix(".json.tmp")
                        progress_tmp.write_text(json.dumps(progress, allow_nan=False) + "\n")
                        progress_tmp.replace(progress_path)
                        raise RuntimeError("scale surrogate calibration invalidated; stop and re-audit all generated tail slots")
                candidate_history.append(candidate_record)
                if eligible:
                    input_full, cap_meta = cap_candidate, candidate_meta
                    selected_pair = pair_data
                    selected_target = (cap_target_candidate, target_candidate_meta)
                    selected_recipe = {"seed": candidate_seed, "t60_s": t60_s,
                        "direct_to_reverb_db": drr_db,
                        "tail_slot": tail_slot,
                        "candidate_index": candidate,
                        "candidates_tested": candidate + 1,
                        "input_tail_db": measurements["window_db"],
                        "prefix_context_after_pause_ms": 1600,
                        "prefix_screening_margin_db": PREFIX_SCREEN_MARGIN_DB,
                        "prefix_calibration_sha256": sha256(PREFIX_CALIBRATION),
                        "scale_calibration_sha256": sha256(SCALE_CALIBRATION),
                        "positive_control_sha256": sha256(SCALE_POSITIVE_CONTROL),
                        "screening_policy_sha256": prefix_policy_hash,
                        "scale_surrogate_operational_error_bound_db": SCALE_SURROGATE_OPERATIONAL_BOUND_DB,
                        "input_only_selection": True,
                        "candidate_history": candidate_history}
                    break
                # Discard full-run rejects after their measurements; exact-hash
                # reuse keeps canonical outputs alive when referenced elsewhere.
                (cache / f"{candidate_key}.npy").unlink(missing_ok=True)
                (cache / f"{candidate_key}.json").unlink(missing_ok=True)
                canonical_target = reuse_index.get(target_candidate_meta["source_sha256"])
                if canonical_target is None or canonical_target[0] != cache / f"{target_key}.npy":
                    (cache / f"{target_key}.npy").unlink(missing_ok=True)
                    (cache / f"{target_key}.json").unlink(missing_ok=True)
            if selected_pair is None:
                selection_counts.append({"slot": tail_slot, "resolved": False,
                    "candidate_limit": MAX_CANDIDATES,
                    "candidate_history": candidate_history})
                (output / "selection-audit.json").write_text(json.dumps({
                    "name": "E2-F2-tailbank-selection-audit-v2",
                    "resolved_tail_slots": slot,
                    "required_tail_slots": 48,
                    "unresolved_slot": tail_slot,
                    "tail_slot_candidate_counts": selection_counts,
                    "candidate_limit": MAX_CANDIDATES,
                    "threshold_db": TAIL_THRESHOLD_DB,
                    "prefix_screening_margin_db": PREFIX_SCREEN_MARGIN_DB,
                    "prefix_context_after_pause_ms": 1600,
                    "prefix_calibration_sha256": sha256(PREFIX_CALIBRATION),
                    "scale_calibration_sha256": sha256(SCALE_CALIBRATION),
                    "positive_control_sha256": sha256(SCALE_POSITIVE_CONTROL),
                    "screening_policy_sha256": prefix_policy_hash,
                    "input_only_selection": True,
                    "test_wav_accessed": False}, indent=2) + "\n")
                raise RuntimeError(f"F2 tailbank slot {tail_slot} unresolved after {MAX_CANDIDATES} candidates")
            target_full, target_meta = selected_target
            clean_for_ratio = selected_pair["clean"]
            rir_kind, rir_room = "procedural_tailbank", None
            selection_counts.append({"slot": tail_slot, "resolved": True,
                "candidates_tested": selected_recipe["candidates_tested"]})
            slot += 1
        else:
            if row["rir_kind"] == "procedural":
                recipe = row["rir_recipe"]
                rir, _, _ = make_procedural_rir(SR, int(recipe["seed"]),
                    float(recipe["t60_s"]), float(recipe["direct_to_reverb_db"]))
            elif row["rir_kind"] == "but_measured":
                recipe = row["rir_recipe"]
                item = by_configuration.get(recipe["configuration"])
                if item is None or item["sha256"] != recipe["rir_sha256"]:
                    raise RuntimeError("fixed C measured-RIR schedule hash mismatch")
                rir, _ = prepare_rir(item)
            else:
                raise RuntimeError(f"unexpected fixed C RIR kind: {row['rir_kind']}")
            pair_data = measured_pair(clean, rir)
            wet = pair_data["reverberant"]
            noise_seed = int(row["noise_seed"])
            mixture = make_fan_mix(clean, wet, noise_seed,
                                   20 if kind == "fan20" else 10)
            mixture = (mixture * branch_scale).astype(np.float32)
            clean_for_ratio = (pair_data["clean"] * branch_scale).astype(np.float32)
            input_full, cap_meta = cap60_cached(args.deep_filter, mixture, output,
                f"input-fixed-{index:03d}", cache, reuse_index)
            target_full, target_meta = cap60_cached(args.deep_filter, clean_for_ratio,
                output, f"target-fixed-{index:03d}", cache, reuse_index)
            rir_kind, rir_room = row["rir_kind"], row.get("rir_room")

        c_audio = clean_crop * gain
        x_audio = crop(input_full, crop_start) * gain
        t_audio = crop(target_full, crop_start) * gain
        clean_ratio_crop = crop(clean_for_ratio, crop_start)
        target_ratio_crop = crop(target_full, crop_start)
        from prepare_cap60_conditioned_pairs import (
            cap_target_speech_ratio, paired_clean_scale_ratio,
        )
        ratio = cap_target_speech_ratio(clean_ratio_crop, target_ratio_crop)
        if ratio < .05:
            raise RuntimeError(f"fixed C speaker/pair slot {index} fails target preservation")
        scale_ratio = paired_clean_scale_ratio(clean, clean_for_ratio)
        for letter, audio in (("x", x_audio), ("t", t_audio), ("c", c_audio)):
            np.save(arrays / f"{letter}-{index:06d}.npy", audio.astype(np.float32),
                    allow_pickle=False)
        source = {**row["source"], "crop_start": crop_start,
            "crop_policy": crop_policy,
            "pause_start_in_crop": pause_in_crop,
            "pause_end_in_crop": int(pause_meta["second_speech_start_sample"]) - crop_start}
        output_row = {"index": index, "kind": kind, "speaker": str(row["speaker"]),
            "source": source, "rir_kind": rir_kind, "rir_room": rir_room,
            "rir_recipe": selected_recipe if kind == "rir_only" else row["rir_recipe"],
            "noise_seed": row.get("noise_seed"), "source_sha256": row["source_sha256"],
            "post_cap_gain": gain, "shared_branch_scale": branch_scale,
            "cap_target_speech_to_clean_rms_ratio": ratio,
            "cap_target_speech_to_clean_db": ratio_db(ratio),
            "cap_input_clean_to_original_rms_ratio": scale_ratio,
            "cap_input_clean_to_original_db": ratio_db(scale_ratio),
            "crop_start_sample": crop_start,
            "pause_start_in_crop": pause_in_crop,
            "x": f"arrays/x-{index:06d}.npy", "c": f"arrays/c-{index:06d}.npy",
            "t": f"arrays/t-{index:06d}.npy",
            "input_sha256": sha256(arrays / f"x-{index:06d}.npy"),
            "clean_sha256": sha256(arrays / f"c-{index:06d}.npy"),
            "target_sha256": sha256(arrays / f"t-{index:06d}.npy"),
            "cap60_input": cap_meta,
            "cap60_target": None if kind == "identity" else target_meta}
        result_rows.append(output_row)
        progress_tmp = progress_path.with_suffix(".json.tmp")
        progress.update({"source_schedule_manifest_sha256": schedule_sha,
            "deep_filter_sha256": sha256(args.deep_filter),
            "selection_policy_sha256": prefix_policy_hash,
            "prefix_calibration_sha256": sha256(PREFIX_CALIBRATION),
            "scale_calibration_sha256": sha256(SCALE_CALIBRATION),
            "positive_control_sha256": sha256(SCALE_POSITIVE_CONTROL),
            "completed_rows": result_rows, "test_wav_accessed": False})
        progress_tmp.write_text(json.dumps(progress, allow_nan=False) + "\n")
        progress_tmp.replace(progress_path)
        print(json.dumps({"row": index, "kind": kind,
            "resolved_tail_slots": slot,
            "candidates_tested": (selected_recipe.get("candidates_tested")
                                  if selected_recipe else None),
            "elapsed_s": time.perf_counter() - started}), flush=True)

    pair_path = output / "pairs.jsonl"
    pair_tmp = pair_path.with_suffix(".jsonl.tmp")
    pair_tmp.write_text("".join(json.dumps(item, allow_nan=False) + "\n"
        for item in result_rows))
    pair_tmp.replace(pair_path)
    counts = {kind: sum(row["kind"] == kind for row in result_rows)
              for kind in ("identity", "rir_only", "fan20", "fan10")}
    manifest = {"name": "E2-F2-tailbank-shared-training-data-v2",
        "source_schedule_manifest_sha256": schedule_sha,
        "pair_manifest_sha256": sha256(pair_path),
        "sample_rate": SR, "crop_samples": CROP_SAMPLES,
        "crop_policy": "C schedule for identity/fan; fixed 750ms pre-transition for all 48 tailbank rows",
        "candidate_namespace": "F2-tailbank|slot|candidate-index",
        "candidate_max_per_slot": MAX_CANDIDATES,
        "candidate_parameter_policy": "RT60 U[0.45,1.10]s; DRR U[-6,18]dB using existing procedural generator",
        "selection_policy": "reject-only 1600ms Cap60 wet prefix; clean Eref uses a guarded scale surrogate in the frozen calibration envelope, with an operational 1.039521dB bound; out-of-envelope targets use full exact Cap60(clean); full exact wet+clean alone can accept",
        "prefix_context_after_pause_ms": 1600,
        "prefix_screening_margin_db": PREFIX_SCREEN_MARGIN_DB,
        "prefix_reject_db_exact_target": {"150_300": -51.0, "300_600": -51.0},
        "scale_surrogate_reject_db": {"150_300": TAIL_THRESHOLD_DB - SCALE_SURROGATE_OPERATIONAL_BOUND_DB,
            "300_600": TAIL_THRESHOLD_DB - SCALE_SURROGATE_OPERATIONAL_BOUND_DB},
        "scale_surrogate_observed_eref_error_db": SCALE_SURROGATE_OBSERVED_ERROR_DB,
        "scale_surrogate_operational_eref_error_bound_db": SCALE_SURROGATE_OPERATIONAL_BOUND_DB,
        "prefix_calibration_report_sha256": sha256(PREFIX_CALIBRATION),
        "scale_calibration_report_sha256": sha256(SCALE_CALIBRATION),
        "scale_positive_control_sha256": sha256(SCALE_POSITIVE_CONTROL),
        "selection_policy_sha256": prefix_policy_hash,
        "prefix_calibration_cases": prefix_policy["prefix"].get("completed_candidates"),
        "prefix_calibration_gate_pass": prefix_policy["prefix"].get("empirical_prefix_gate_pass"),
        "activity_mask_crop_samples": CROP_SAMPLES,
        "tail_measurement_end_after_pause_ms": 600,
        "input_only_selection": True, "resolved_tail_slots": slot,
        "tail_slot_candidate_counts": selection_counts,
        "train_speaker_count": len({row["speaker"] for row in result_rows}),
        "kind_counts": counts, "deep_filter_sha256": sha256(args.deep_filter),
        "cap60_elapsed_s": time.perf_counter() - started,
        "test_wav_accessed": False}
    (output / "training-data-manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    progress_path.unlink(missing_ok=True)
    print(json.dumps({k: v for k, v in manifest.items()
                      if k != "tail_slot_candidate_counts"}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=LIBRISPEECH)
    parser.add_argument("--rir-root", type=Path, default=BUT_ROOT)
    parser.add_argument("--deep-filter", type=Path, default=DEEP_FILTER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--resume", action="store_true",
                        help="resume this experiment's own incomplete local cache")
    args = parser.parse_args()
    prepare(args)


if __name__ == "__main__":
    main()
