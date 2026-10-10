#!/usr/bin/env python3
"""Fresh-speaker feasibility pilot for W1-only G-early training."""
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
from pilot_g2_feasibility import (  # noqa: E402
    RT60_RANGE, DRR_RANGE, active_dry_levels, atomic_json, candidate_params,
    digest_array, free_cache, hash_seed, prepare_pair,
)
from prepare_cap60_conditioned_pairs import (  # noqa: E402
    CROP_SAMPLES, DEEP_FILTER, LIBRISPEECH, SR, cap60_one, sha256,
)
from prepare_f2_tailbank import crop, tail_input_levels  # noqa: E402
from screen_measured_rirs import make_inserted_pause_pair  # noqa: E402
from train_measured_mix import measured_pair  # noqa: E402

EXPERIMENT = ROOT / ".tools/compact-dereverb/g-early-feasibility-2026-10-10"
POOL = LIBRISPEECH / "train-clean-360"
TRAIN_PAIRS = ROOT / ".tools/compact-dereverb/g-target-ablation-2026-10-10/data/pairs.jsonl"
SPEAKER_COUNT = 12
PAIRS_PER_SPEAKER = 4
MAX_CANDIDATES = 32
WINDOWS_MS = (("150_300", 150, 300), ("300_600", 300, 600))
SEED_NAMESPACE = "G-early-feasibility-v1|speaker|pair|candidate"


def _freeze_design(experiment: Path, deep_filter: Path) -> dict:
    if not TRAIN_PAIRS.is_file() or not POOL.is_dir() or not deep_filter.is_file():
        raise FileNotFoundError("fixed G training schedule, LibriSpeech, or Cap60 is missing")
    design_path = experiment / "design.json"
    if design_path.is_file():
        design = json.loads(design_path.read_text())
        if (design.get("name") != "G-early-feasibility-v1" or
                design.get("seed_namespace") != SEED_NAMESPACE or
                design.get("training_pair_manifest_sha256") != sha256(TRAIN_PAIRS) or
                design.get("cap60_binary_sha256") != sha256(deep_filter) or
                design.get("test_wav_accessed") is not False):
            raise RuntimeError("existing G-early feasibility design does not match this run")
        return design

    all_used = used_speakers(ROOT / ".tools/compact-dereverb")
    train_rows = [json.loads(line) for line in TRAIN_PAIRS.read_text().splitlines() if line.strip()]
    train_ids = {str(row["speaker"]) for row in train_rows}
    if len(train_ids) != 128:
        raise RuntimeError("the fixed G training schedule must contain 128 unique speakers")
    excluded = all_used | train_ids
    folders = [p for p in POOL.iterdir() if p.is_dir() and p.name.isdigit()
               and p.name not in excluded]
    folders.sort(key=lambda p: hashlib.sha256(("G-early-feasibility|" + p.name).encode()).hexdigest())
    selected = []
    for folder in folders:
        files = sorted(folder.glob("*/*.flac"))
        if len(files) >= 8:
            selected.append((folder.name, files[:8]))
        if len(selected) == SPEAKER_COUNT:
            break
    if len(selected) != SPEAKER_COUNT:
        raise RuntimeError("could not select 12 fresh G-early feasibility speakers")

    speakers = []
    for speaker_id, files in selected:
        pairs = []
        for pair_index in range(PAIRS_PER_SPEAKER):
            first, second = files[pair_index * 2:pair_index * 2 + 2]
            pairs.append({"pair_index": pair_index,
                "first": str(first), "first_sha256": sha256(first),
                "second": str(second), "second_sha256": sha256(second)})
        speakers.append({"speaker_id": speaker_id, "pairs": pairs})
    design = {
        "name": "G-early-feasibility-v1",
        "purpose": "fresh-speaker feasibility for W1-only tail supervision; no model fit",
        "speakers": speakers,
        "speaker_count": SPEAKER_COUNT,
        "pair_count": SPEAKER_COUNT * PAIRS_PER_SPEAKER,
        "seed_namespace": SEED_NAMESPACE,
        "candidate_limit": MAX_CANDIDATES,
        "candidate_parameters": {"rt60_seconds_uniform": list(RT60_RANGE),
            "direct_to_reverb_db_uniform": list(DRR_RANGE)},
        "tail_eligible_rule": (
            "first exact Cap60 candidate j=0..31 satisfying Lx(150-300ms) "
            "> max(-60 dB, Ldry(150-300ms)+6 dB)"
        ),
        "base_only_rule": "if no candidate passes, retain candidate j=0 as BASE_ONLY_CONTROL; no tail loss",
        "w2_rule": "300-600ms is descriptive only and never affects selection",
        "training_pair_manifest_sha256": sha256(TRAIN_PAIRS),
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


def _process_slot(speaker_index: int, speaker: dict, pair_row: dict, args,
                  cache: Path, work: Path, progress_dir: Path, result_dir: Path) -> dict:
    pair_index = int(pair_row["pair_index"])
    key = f"dev-{speaker_index:02d}-{pair_index:02d}"
    result_path = result_dir / f"{key}.json"
    design_hash = sha256(args.experiment_dir / "design.json")
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        if (result.get("design_sha256") != design_hash or
                result.get("speaker_id") != speaker["speaker_id"] or
                result.get("pair_index") != pair_index):
            raise RuntimeError("existing G-early slot result does not match the frozen design")
        return result

    joined, pause_meta, clean_crop, pause_crop = prepare_pair(pair_row)
    progress_path = progress_dir / f"{key}.json"
    history = []
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text())
        if progress.get("design_sha256") != design_hash:
            raise RuntimeError("existing G-early candidate progress belongs to another design")
        history = list(progress["history"])
    if [int(item["candidate_index"]) for item in history] != list(range(len(history))):
        raise RuntimeError("candidate progress must be a contiguous prefix starting at j=0")
    if len(history) > MAX_CANDIDATES:
        raise RuntimeError("candidate progress exceeds the frozen 32-candidate limit")
    accepted = next((item for item in history if item["tail_eligible"]), None)
    started = time.perf_counter()
    candidate_start = len(history) if accepted is None else MAX_CANDIDATES
    for candidate_index in range(candidate_start, MAX_CANDIDATES):
        seed = hash_seed(f"{SEED_NAMESPACE}|{speaker['speaker_id']}|{pair_index}|{candidate_index}")
        rt60_s, drr_db = candidate_params(seed)
        rir, _early, _late = make_procedural_rir(
            SR, seed, rt60_s, drr_db,
            max_t60_s=2.0, min_direct_to_reverb_db=-20.0)
        pair = measured_pair(joined, rir)
        input_key = f"{key}-candidate-{candidate_index:02d}-input"
        dry_key = f"{key}-candidate-{candidate_index:02d}-dry"
        wet_source_hash, dry_source_hash = digest_array(pair["reverberant"]), digest_array(pair["clean"])
        cap_wet, wet_meta = cap60_one(args.deep_filter, pair["reverberant"], SR,
                                      work, input_key, cache)
        cap_dry, dry_meta = cap60_one(args.deep_filter, pair["clean"], SR,
                                      work, dry_key, cache)
        crop_start = int(pause_meta["clip_relative_start_sample"]) - int(.75 * SR)
        wet_crop, dry_cap_crop = crop(cap_wet, crop_start), crop(cap_dry, crop_start)
        levels = tail_input_levels(dry_cap_crop, wet_crop, pause_crop,
                                   activity_clean_crop=clean_crop)
        valid_ref = bool(levels.get("reference_valid"))
        windows = {}
        if valid_ref:
            dry_levels = active_dry_levels(dry_cap_crop, pause_crop,
                                           float(levels["reference_energy"]))
            for band, _start, _end in WINDOWS_MS:
                lx = float(levels["window_db"][band])
                ldry = float(dry_levels[band])
                threshold = max(-60.0, ldry + 6.0)
                windows[band] = {
                    "lx_db": lx, "ldry_db": ldry,
                    "margin_lx_minus_ldry_db": lx - ldry,
                    "strict_threshold_db": threshold,
                    "above_input_floor_threshold": bool(lx > threshold),
                }
        else:
            for band, _start, _end in WINDOWS_MS:
                windows[band] = {"lx_db": None, "ldry_db": None,
                    "margin_lx_minus_ldry_db": None, "strict_threshold_db": None,
                    "above_input_floor_threshold": False}
        eligible = bool(valid_ref and windows["150_300"]["above_input_floor_threshold"])
        shared_scale = float(np.max(np.abs(pair["clean"]))) / max(
            float(np.max(np.abs(joined))), 1e-8)
        item = {
            "candidate_index": candidate_index, "seed": seed,
            "rt60_s": rt60_s, "direct_to_reverb_db": drr_db,
            "shared_scale": shared_scale,
            "clean_scaled_source_sha256": dry_source_hash,
            "wet_scaled_source_sha256": wet_source_hash,
            "cap60_clean_source_sha256": dry_meta["source_sha256"],
            "cap60_wet_source_sha256": wet_meta["source_sha256"],
            "cap60_clean_output_sha256": digest_array(cap_dry),
            "cap60_wet_output_sha256": digest_array(cap_wet),
            "cap60_clean_elapsed_s": float(dry_meta["elapsed_s"]),
            "cap60_wet_elapsed_s": float(wet_meta["elapsed_s"]),
            "reference_valid": valid_ref,
            "reference_active_samples": int(levels.get("reference_active_samples", 0)),
            "windows": windows,
            "tail_eligible": eligible,
            "rejection_reason": ("eligible_w1" if eligible else
                "invalid_speech_reference" if not valid_ref else "w1_below_threshold"),
        }
        history.append(item)
        atomic_json(progress_path, {"design_sha256": design_hash,
            "speaker_id": speaker["speaker_id"], "pair_index": pair_index,
            "history": history, "test_wav_accessed": False})
        free_cache(cache, [input_key, dry_key])
        if eligible:
            accepted = item
            break

    if accepted is None:
        if len(history) != MAX_CANDIDATES:
            raise RuntimeError("BASE_ONLY_CONTROL requires a fully exhausted j=0..31 search")
        control = history[0]
        classification = "BASE_ONLY_CONTROL"
        tail_loss_enabled = False
    else:
        control = None
        classification = "TAIL_ELIGIBLE"
        tail_loss_enabled = True
    result = {
        "speaker_id": speaker["speaker_id"], "pair_index": pair_index,
        "source_pair": [Path(pair_row["first"]).name, Path(pair_row["second"]).name],
        "source_pair_sha256": [pair_row["first_sha256"], pair_row["second_sha256"]],
        "pause_start_sample": int(pause_meta["clip_relative_start_sample"]),
        "crop_start_sample": int(pause_meta["clip_relative_start_sample"]) - int(.75 * SR),
        "pause_start_in_crop_sample": pause_crop,
        "classification": classification,
        "tail_eligible": accepted is not None,
        "tail_loss_enabled": tail_loss_enabled,
        "candidates_tested": len(history),
        "selected_candidate": accepted if accepted is not None else control,
        "history": history,
        "elapsed_s": time.perf_counter() - started,
        "design_sha256": design_hash,
        "training_started": False,
        "model_outputs_accessed": False,
        "test_wav_accessed": False,
    }
    atomic_json(result_path, result)
    return result


def run(args) -> dict:
    args.experiment_dir.mkdir(parents=True, exist_ok=True)
    design = _freeze_design(args.experiment_dir, args.deep_filter)
    design_hash = sha256(args.experiment_dir / "design.json")
    cache = args.experiment_dir / "cap60-cache"
    work = args.experiment_dir / "cap60-work"
    progress_dir = args.experiment_dir / "progress"
    result_dir = args.experiment_dir / "slots"
    for path in (cache, work, progress_dir, result_dir):
        path.mkdir(parents=True, exist_ok=True)
    rows = []
    started = time.perf_counter()
    for speaker_index, speaker in enumerate(design["speakers"]):
        for pair_row in speaker["pairs"]:
            row = _process_slot(speaker_index, speaker, pair_row, args,
                                cache, work, progress_dir, result_dir)
            rows.append(row)
            eligible_count = sum(item["tail_eligible"] for item in rows)
            print(json.dumps({"speakers": speaker_index + 1,
                "pair_index": pair_row["pair_index"], "processed": len(rows),
                "tail_eligible": row["tail_eligible"],
                "classification": row["classification"],
                "candidates_tested": row["candidates_tested"],
                "eligible_count": eligible_count,
                "elapsed_s": time.perf_counter() - started}), flush=True)

    per_speaker = []
    for speaker_index in range(SPEAKER_COUNT):
        group = rows[speaker_index * PAIRS_PER_SPEAKER:(speaker_index + 1) * PAIRS_PER_SPEAKER]
        per_speaker.append(sum(row["tail_eligible"] for row in group))
    eligible_count = sum(row["tail_eligible"] for row in rows)
    speakers_ge3 = sum(count >= 3 for count in per_speaker)
    speakers_zero = sum(count == 0 for count in per_speaker)
    result = {
        "name": "G-early-feasibility-v1-results",
        "design_sha256": design_hash,
        "pair_count": len(rows),
        "eligible_pair_count": eligible_count,
        "base_only_control_count": sum(not row["tail_eligible"] for row in rows),
        "speaker_eligible_counts_private": per_speaker,
        "speakers_with_at_least_3_of_4": speakers_ge3,
        "speakers_with_0_of_4": speakers_zero,
        "gate_thresholds": {"eligible_pairs_minimum": 36,
            "speakers_at_least_3_minimum": 9, "speakers_at_0_maximum": 0},
        "feasibility_pass": (len(rows) == 48 and eligible_count >= 36 and
            speakers_ge3 >= 9 and speakers_zero == 0),
        "w2_selection_used": False,
        "candidate_limit": MAX_CANDIDATES,
        "candidate_total": sum(row["candidates_tested"] for row in rows),
        "elapsed_s": time.perf_counter() - started,
        "training_started": False,
        "model_outputs_accessed": False,
        "test_wav_accessed": False,
        "speaker_ids_in_public_report": False,
    }
    atomic_json(args.experiment_dir / "results.json", result)
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, default=EXPERIMENT)
    parser.add_argument("--deep-filter", type=Path, default=DEEP_FILTER)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
