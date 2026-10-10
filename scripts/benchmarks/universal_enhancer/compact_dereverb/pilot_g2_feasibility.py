#!/usr/bin/env python3
"""Input-only feasibility pilot for the separately preregistered G2 tail gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
from data import make_procedural_rir  # noqa: E402
from freeze_f2_holdout import used_speakers  # noqa: E402
from prepare_cap60_conditioned_pairs import (  # noqa: E402
    CROP_SAMPLES, DEEP_FILTER, LIBRISPEECH, SR, cap60_one, sha256,
)
from prepare_f2_tailbank import crop, tail_input_levels  # noqa: E402
from screen_measured_rirs import make_inserted_pause_pair  # noqa: E402
from train_measured_mix import measured_pair  # noqa: E402

EXPERIMENT = ROOT / ".tools/compact-dereverb/g2-feasibility-2026-10-10"
POOL = LIBRISPEECH / "train-clean-360"
TRAIN_DATA = ROOT / ".tools/compact-dereverb/g-target-ablation-2026-10-10/data"
TRAIN_PAIRS = TRAIN_DATA / "pairs.jsonl"
TRAIN_MANIFEST = TRAIN_DATA / "training-data-manifest.json"
TRAIN_COVERAGE = TRAIN_DATA / "coverage-audit.json"
SPEAKER_COUNT = 16
PAIRS_PER_SPEAKER = 4
MAX_CANDIDATES = 32
TAIL_WINDOWS_MS = (("150_300", 150, 300), ("300_600", 300, 600))
RT60_RANGE = (1.65, 1.85)
DRR_RANGE = (-14.5, -12.5)
SEED_NAMESPACE = "G2-feasibility-dev-v1|speaker|pair|candidate"


def hash_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def digest_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(value, dtype=np.float32).tobytes()).hexdigest()


def free_cache(cache: Path, keys: list[str]) -> None:
    for key in keys:
        (cache / f"{key}.npy").unlink(missing_ok=True)
        (cache / f"{key}.json").unlink(missing_ok=True)


def _selection_design(experiment_dir: Path, deep_filter: Path) -> dict:
    for path in (TRAIN_PAIRS, TRAIN_MANIFEST, TRAIN_COVERAGE):
        if not path.is_file():
            raise FileNotFoundError(f"G training artifacts are missing: {path}")
    train_manifest = json.loads(TRAIN_MANIFEST.read_text())
    coverage = json.loads(TRAIN_COVERAGE.read_text())
    if (train_manifest.get("test_wav_accessed") is not False or
            coverage.get("executable") is not True or
            coverage.get("resolved_tail_slots") != 48):
        raise RuntimeError("shared G training dataset is not safe and fully frozen")
    if not POOL.is_dir() or not deep_filter.is_file():
        raise FileNotFoundError("LibriSpeech train-clean-360 or Cap60 binary is missing")

    # A frozen design is authoritative on resume. Its speakers are intentionally
    # already consumed, so they must not be reselected from the global pool.
    design_path = experiment_dir / "design.json"
    if design_path.is_file():
        design = json.loads(design_path.read_text())
        if (design.get("name") != "G2-feasibility-dev-v1" or
                design.get("seed_namespace") != SEED_NAMESPACE or
                design.get("training_pair_manifest_sha256") != sha256(TRAIN_PAIRS) or
                design.get("cap60_binary_sha256") != sha256(deep_filter) or
                design.get("test_wav_accessed") is not False):
            raise RuntimeError("existing G2 feasibility design does not match current inputs")
        return design

    all_used = used_speakers(ROOT / ".tools/compact-dereverb")
    train_rows = [json.loads(line) for line in TRAIN_PAIRS.read_text().splitlines() if line.strip()]
    train_ids = {str(row["speaker"]) for row in train_rows}
    if len(train_ids) != 128:
        raise RuntimeError("the frozen G schedule must contain 128 unique training speakers")
    excluded = all_used | train_ids
    folders = [p for p in POOL.iterdir() if p.is_dir() and p.name.isdigit()
               and p.name not in excluded]
    folders.sort(key=lambda p: hashlib.sha256(("dev|" + p.name).encode()).hexdigest())
    selected = []
    for folder in folders:
        files = sorted(folder.glob("*/*.flac"))
        if len(files) >= 8:
            selected.append((folder.name, files[:8]))
        if len(selected) == SPEAKER_COUNT:
            break
    if len(selected) != SPEAKER_COUNT:
        raise RuntimeError("could not reproduce the 16-speaker G-v1 diagnostic DEV selection")

    speakers = []
    for speaker_id, files in selected:
        pair_rows = []
        for pair_index in range(PAIRS_PER_SPEAKER):
            first, second = files[pair_index * 2:pair_index * 2 + 2]
            pair_rows.append({
                "pair_index": pair_index,
                "first": str(first), "first_sha256": sha256(first),
                "second": str(second), "second_sha256": sha256(second),
            })
        speakers.append({"speaker_id": speaker_id, "pairs": pair_rows})
    design = {
        "name": "G2-feasibility-dev-v1",
        "purpose": "input-only two-window feasibility; no model fitting or outputs",
        "speakers": speakers,
        "speaker_count": SPEAKER_COUNT,
        "pair_count": SPEAKER_COUNT * PAIRS_PER_SPEAKER,
        "seed_namespace": SEED_NAMESPACE,
        "candidate_limit": MAX_CANDIDATES,
        "candidate_parameters": {
            "rt60_seconds_uniform": list(RT60_RANGE),
            "direct_to_reverb_db_uniform": list(DRR_RANGE),
        },
        "selection_rule": (
            "first exact Cap60 candidate j=0..31 for which each W in 150-300ms "
            "and 300-600ms satisfies Lx(W) > max(-60 dB, Ldry(W)+6 dB)"
        ),
        "training_pair_manifest_sha256": sha256(TRAIN_PAIRS),
        "training_data_manifest_sha256": sha256(TRAIN_MANIFEST),
        "cap60_binary_sha256": sha256(deep_filter),
        "cap60_guard_policy": "exact full Cap60; guard and delay compensation from cap60_one",
        "sample_rate": SR,
        "frozen_before_inference": True,
        "training_started": False,
        "model_outputs_accessed": False,
        "test_wav_accessed": False,
        "public_report_omits_speaker_ids": True,
    }
    atomic_json(design_path, design)
    return design


def active_dry_levels(reference_crop: np.ndarray, pause: int, eref: float) -> dict[str, float]:
    eps = 1e-12 * eref
    values = {}
    for name, start_ms, end_ms in TAIL_WINDOWS_MS:
        start, end = pause + start_ms * 16, pause + end_ms * 16
        section = np.asarray(reference_crop[start:end], dtype=np.float64)
        energy = float(np.mean(section * section))
        values[name] = float(10 * np.log10((energy + eps) / (eref + eps)))
    return values


def candidate_params(seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    return (float(rng.uniform(*RT60_RANGE)), float(rng.uniform(*DRR_RANGE)))


def prepare_pair(row: dict) -> tuple[np.ndarray, dict, np.ndarray, int]:
    first, second = Path(row["first"]), Path(row["second"])
    if sha256(first) != row["first_sha256"] or sha256(second) != row["second_sha256"]:
        raise RuntimeError("frozen LibriSpeech source hash mismatch")
    joined, pause_meta = make_inserted_pause_pair(first, second)
    joined = np.asarray(joined, dtype=np.float32)
    joined *= .80 / max(float(np.max(np.abs(joined))), 1e-8)
    crop_start = int(pause_meta["clip_relative_start_sample"]) - int(.75 * SR)
    pause_crop = int(pause_meta["clip_relative_start_sample"]) - crop_start
    clean_crop = crop(joined, crop_start)
    if crop_start < 0 or crop_start + CROP_SAMPLES > len(joined):
        raise RuntimeError("frozen feasibility crop extends beyond joined utterance")
    return joined, pause_meta, clean_crop, pause_crop


def process_slot(name: str, speaker_index: int, speaker: dict, pair_row: dict,
                 args, cache: Path, work: Path, progress_dir: Path,
                 result_dir: Path) -> dict:
    pair_index = int(pair_row["pair_index"])
    base = f"{name}-{speaker_index:02d}-{pair_index:02d}"
    result_path = result_dir / f"{base}.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        if (result.get("speaker_id") != speaker["speaker_id"] or
                result.get("pair_index") != pair_index or
                result.get("design_sha256") != sha256(args.experiment_dir / "design.json")):
            raise RuntimeError(f"existing feasibility result does not match design: {result_path}")
        return result

    joined, pause_meta, clean_crop, pause_crop = prepare_pair(pair_row)
    progress_path = progress_dir / f"{base}.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress.get("design_sha256") != sha256(args.experiment_dir / "design.json"):
            raise RuntimeError(f"slot progress belongs to a different design: {progress_path}")
        history = list(progress.get("history", []))
    else:
        history = []
    expected_indices = list(range(len(history)))
    if [int(item["candidate_index"]) for item in history] != expected_indices:
        raise RuntimeError(f"candidate progress is not a contiguous deterministic prefix: {progress_path}")
    if len(history) > MAX_CANDIDATES:
        raise RuntimeError("candidate progress exceeds the frozen search limit")

    accepted = next((item for item in history if item["eligible"]), None)
    started = time.perf_counter()
    candidate_start = len(history) if accepted is None else MAX_CANDIDATES
    for candidate_index in range(candidate_start, MAX_CANDIDATES):
        candidate_seed = hash_seed(
            f"{SEED_NAMESPACE}|{speaker['speaker_id']}|{pair_index}|{candidate_index}")
        t60_s, drr_db = candidate_params(candidate_seed)
        rir, _early, _late = make_procedural_rir(
            SR, candidate_seed, t60_s, drr_db,
            max_t60_s=2.0, min_direct_to_reverb_db=-20.0)
        pair = measured_pair(joined, rir)
        input_key = f"{base}-candidate-{candidate_index:02d}-input"
        dry_key = f"{base}-candidate-{candidate_index:02d}-dry"
        source_x_hash = digest_array(pair["reverberant"])
        source_dry_hash = digest_array(pair["clean"])
        cap_input, input_meta = cap60_one(
            args.deep_filter, pair["reverberant"], SR, work, input_key, cache)
        cap_dry, dry_meta = cap60_one(
            args.deep_filter, pair["clean"], SR, work, dry_key, cache)
        cap_input_crop = crop(cap_input, int(pause_meta["clip_relative_start_sample"]) - int(.75 * SR))
        cap_dry_crop = crop(cap_dry, int(pause_meta["clip_relative_start_sample"]) - int(.75 * SR))
        input_levels = tail_input_levels(
            cap_dry_crop, cap_input_crop, pause_crop, activity_clean_crop=clean_crop)
        reference_valid = bool(input_levels.get("reference_valid"))
        if reference_valid:
            eref = float(input_levels["reference_energy"])
            dry_levels = active_dry_levels(cap_dry_crop, pause_crop, eref)
            wet_levels = input_levels["window_db"]
            per_window = {}
            eligible = True
            for band, _start, _end in TAIL_WINDOWS_MS:
                threshold = max(-60.0, float(dry_levels[band]) + 6.0)
                margin = float(wet_levels[band] - dry_levels[band])
                passed = bool(np.isfinite(wet_levels[band]) and
                              np.isfinite(dry_levels[band]) and
                              float(wet_levels[band]) > threshold)
                per_window[band] = {
                    "lx_db": float(wet_levels[band]),
                    "ldry_db": float(dry_levels[band]),
                    "margin_lx_minus_ldry_db": margin,
                    "strict_threshold_db": threshold,
                    "eligible": passed,
                }
                eligible &= passed
            reason = "eligible" if eligible else "one_or_more_windows_below_threshold"
        else:
            per_window = {band: {"lx_db": None, "ldry_db": None,
                "margin_lx_minus_ldry_db": None, "strict_threshold_db": None,
                "eligible": False} for band, _start, _end in TAIL_WINDOWS_MS}
            eligible = False
            reason = "dry_speech_reference_invalid"

        peak_joined = max(float(np.max(np.abs(joined))), 1e-8)
        shared_scale = float(np.max(np.abs(pair["clean"]))) / peak_joined
        candidate_doc = {
            "candidate_index": candidate_index,
            "seed": candidate_seed,
            "rt60_s": t60_s,
            "direct_to_reverb_db": drr_db,
            "shared_scale": shared_scale,
            "clean_scaled_source_sha256": source_dry_hash,
            "wet_scaled_source_sha256": source_x_hash,
            "cap60_clean_source_sha256": dry_meta["source_sha256"],
            "cap60_wet_source_sha256": input_meta["source_sha256"],
            "cap60_clean_output_sha256": digest_array(cap_dry),
            "cap60_wet_output_sha256": digest_array(cap_input),
            "cap60_clean_elapsed_s": float(dry_meta["elapsed_s"]),
            "cap60_wet_elapsed_s": float(input_meta["elapsed_s"]),
            "reference_valid": reference_valid,
            "reference_active_samples": int(input_levels.get("reference_active_samples", 0)),
            "windows": per_window,
            "eligible": bool(eligible),
            "rejection_reason": reason,
        }
        history.append(candidate_doc)
        atomic_json(progress_path, {"design_sha256": sha256(args.experiment_dir / "design.json"),
            "speaker_id": speaker["speaker_id"], "pair_index": pair_index,
            "history": history, "test_wav_accessed": False})
        free_cache(cache, [input_key, dry_key])
        if eligible:
            accepted = candidate_doc
            break

    result = {
        "speaker_id": speaker["speaker_id"],
        "pair_index": pair_index,
        "source_pair": [Path(pair_row["first"]).name, Path(pair_row["second"]).name],
        "source_pair_sha256": [pair_row["first_sha256"], pair_row["second_sha256"]],
        "pause_start_sample": int(pause_meta["clip_relative_start_sample"]),
        "crop_start_sample": int(pause_meta["clip_relative_start_sample"]) - int(.75 * SR),
        "pause_start_in_crop_sample": pause_crop,
        "candidates_tested": len(history),
        "eligible": accepted is not None,
        "selected_candidate": accepted,
        "history": history,
        "elapsed_s": time.perf_counter() - started,
        "design_sha256": sha256(args.experiment_dir / "design.json"),
        "training_started": False,
        "model_outputs_accessed": False,
        "test_wav_accessed": False,
    }
    atomic_json(result_path, result)
    return result


def run(args) -> dict:
    args.experiment_dir.mkdir(parents=True, exist_ok=True)
    design = _selection_design(args.experiment_dir, args.deep_filter)
    design_hash = sha256(args.experiment_dir / "design.json")
    cache = args.experiment_dir / "cap60-cache"
    work = args.experiment_dir / "cap60-work"
    progress_dir = args.experiment_dir / "progress"
    result_dir = args.experiment_dir / "slots"
    for path in (cache, work, progress_dir, result_dir):
        path.mkdir(parents=True, exist_ok=True)

    results = []
    started = time.perf_counter()
    for speaker_index, speaker in enumerate(design["speakers"]):
        for pair_row in speaker["pairs"]:
            result = process_slot("dev", speaker_index, speaker, pair_row,
                args, cache, work, progress_dir, result_dir)
            results.append(result)
            resolved = sum(bool(row["eligible"]) for row in results)
            print(json.dumps({"speakers": speaker_index + 1,
                "pair_index": pair_row["pair_index"], "resolved": resolved,
                "processed": len(results), "eligible": result["eligible"],
                "candidates_tested": result["candidates_tested"],
                "elapsed_s": time.perf_counter() - started}), flush=True)

    by_window = {}
    for band, _start, _end in TAIL_WINDOWS_MS:
        selected = [row["selected_candidate"] for row in results if row["selected_candidate"]]
        lx = [float(candidate["windows"][band]["lx_db"]) for candidate in selected]
        ldry = [float(candidate["windows"][band]["ldry_db"]) for candidate in selected]
        margins = [float(candidate["windows"][band]["margin_lx_minus_ldry_db"])
                   for candidate in selected]
        by_window[band] = {
            "selected_pair_count": len(selected),
            "lx_db_median": float(np.median(lx)) if lx else None,
            "lx_db_min": float(np.min(lx)) if lx else None,
            "ldry_db_median": float(np.median(ldry)) if ldry else None,
            "margin_db_median": float(np.median(margins)) if margins else None,
            "margin_db_min": float(np.min(margins)) if margins else None,
        }
    summary = {
        "name": "G2-feasibility-dev-v1-results",
        "design_sha256": design_hash,
        "pair_count": len(results),
        "eligible_pair_count": sum(bool(row["eligible"]) for row in results),
        "candidate_limit": MAX_CANDIDATES,
        "two_window_feasibility_pass": len(results) == 64 and all(row["eligible"] for row in results),
        "unresolved_slots": [{"speaker_index": index // PAIRS_PER_SPEAKER,
            "pair_index": row["pair_index"], "candidates_tested": row["candidates_tested"]}
            for index, row in enumerate(results) if not row["eligible"]],
        "candidate_index_median": float(np.median([
            row["selected_candidate"]["candidate_index"] for row in results if row["eligible"]
        ])) if any(row["eligible"] for row in results) else None,
        "candidate_index_max": max((row["selected_candidate"]["candidate_index"]
            for row in results if row["eligible"]), default=None),
        "per_window": by_window,
        "elapsed_s": time.perf_counter() - started,
        "training_started": False,
        "model_outputs_accessed": False,
        "test_wav_accessed": False,
        "speaker_ids_in_public_report": False,
    }
    atomic_json(args.experiment_dir / "results.json", summary)
    print(json.dumps(summary, indent=2, allow_nan=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, default=EXPERIMENT)
    parser.add_argument("--deep-filter", type=Path, default=DEEP_FILTER)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
