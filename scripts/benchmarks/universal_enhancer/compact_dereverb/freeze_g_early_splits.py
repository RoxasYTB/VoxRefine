#!/usr/bin/env python3
"""Freeze fresh G-early DEV/HOLDOUT RIR recipes and input-only coverage gates."""
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
    DRR_RANGE, RT60_RANGE, active_dry_levels, atomic_json, free_cache, hash_seed,
)
from prepare_cap60_conditioned_pairs import (  # noqa: E402
    DEEP_FILTER, LIBRISPEECH, SR, cap60_one, sha256,
)
from prepare_f2_tailbank import crop, tail_input_levels  # noqa: E402
from screen_measured_rirs import make_inserted_pause_pair  # noqa: E402
from train_measured_mix import measured_pair  # noqa: E402

EXPERIMENT = ROOT / ".tools/compact-dereverb/g-early-2026-10-10"
POOL = LIBRISPEECH / "train-clean-360"
MAX_CANDIDATES = 32
WINDOWS = ("150_300", "300_600")


def slot_result(split: str, speaker_index: int, speaker_id: str, pair_index: int,
                files: list[Path], root: Path) -> dict:
    result_dir = root / "slots"
    progress_dir = root / "progress"
    cache, work = root / "cap60-cache", root / "cap60-work"
    for p in (result_dir, progress_dir, cache, work):
        p.mkdir(parents=True, exist_ok=True)
    key = f"{split}-{speaker_index:02d}-{pair_index:02d}"
    design_hash = sha256(root / "manifest.design.json")
    result_path = result_dir / f"{key}.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        if result.get("design_sha256") != design_hash:
            raise RuntimeError("slot result belongs to a different frozen split design")
        return result
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    source_hashes = [sha256(p) for p in files]
    progress_path = progress_dir / f"{key}.json"
    history = []
    if progress_path.is_file():
        old = json.loads(progress_path.read_text())
        if old.get("design_sha256") != design_hash or old.get("source_sha256") != source_hashes:
            raise RuntimeError("partial split slot does not match its frozen design/source")
        history = old["history"]
    if [int(x["candidate_index"]) for x in history] != list(range(len(history))):
        raise RuntimeError("partial candidate history is not a contiguous prefix")

    joined, pause_meta = make_inserted_pause_pair(files[0], files[1])
    joined = np.asarray(joined, dtype=np.float32)
    joined *= .80 / max(float(np.max(np.abs(joined))), 1e-8)
    pause_start = int(pause_meta["clip_relative_start_sample"])
    crop_start = pause_start - int(.75 * SR)
    pause_crop = pause_start - crop_start
    dry_clean_crop = crop(joined, crop_start)
    accepted = next((item for item in history if item.get("tail_eligible")), None)
    base = None
    elapsed_start = time.perf_counter()
    start_candidate = len(history) if accepted is None else MAX_CANDIDATES
    for candidate_index in range(start_candidate, MAX_CANDIDATES):
        seed = hash_seed(f"G-early-{split}-v1|{speaker_id}|{pair_index}|{candidate_index}")
        rng = np.random.default_rng(seed)
        t60, drr = float(rng.uniform(*RT60_RANGE)), float(rng.uniform(*DRR_RANGE))
        rir, _, _ = make_procedural_rir(SR, seed, t60, drr,
            max_t60_s=2.0, min_direct_to_reverb_db=-20.0)
        pair = measured_pair(joined, rir)
        wet_key, dry_key = f"{key}-{candidate_index:02d}-wet", f"{key}-{candidate_index:02d}-dry"
        cap_wet, wet_meta = cap60_one(DEEP_FILTER, pair["reverberant"], SR, work, wet_key, cache)
        cap_dry, dry_meta = cap60_one(DEEP_FILTER, pair["clean"], SR, work, dry_key, cache)
        wet_crop, dry_crop = crop(cap_wet, crop_start), crop(cap_dry, crop_start)
        levels = tail_input_levels(dry_crop, wet_crop, pause_crop,
                                   activity_clean_crop=dry_clean_crop)
        windows = {}
        valid = bool(levels.get("reference_valid"))
        if valid:
            dry_levels = active_dry_levels(dry_crop, pause_crop, float(levels["reference_energy"]))
            for window in WINDOWS:
                lx, ldry = float(levels["window_db"][window]), float(dry_levels[window])
                threshold = max(-60.0, ldry + 6.0)
                windows[window] = {"lx_db": lx, "ldry_db": ldry,
                    "threshold_db": threshold, "passes": lx > threshold}
        else:
            windows = {w: {"lx_db": None, "ldry_db": None,
                "threshold_db": None, "passes": False} for w in WINDOWS}
        eligible = bool(valid and windows["150_300"]["passes"])
        item = {"candidate_index": candidate_index, "seed": seed,
            "t60_s": t60, "direct_to_reverb_db": drr,
            "wet_source_sha256": wet_meta["source_sha256"],
            "dry_source_sha256": dry_meta["source_sha256"],
            "wet_cap60_output_sha256": hashlib.sha256(cap_wet.tobytes()).hexdigest(),
            "dry_cap60_output_sha256": hashlib.sha256(cap_dry.tobytes()).hexdigest(),
            "wet_cap60_elapsed_s": float(wet_meta["elapsed_s"]),
            "dry_cap60_elapsed_s": float(dry_meta["elapsed_s"]),
            "reference_valid": valid, "windows": windows,
            "tail_eligible": eligible}
        history.append(item)
        atomic_json(progress_path, {"design_sha256": design_hash,
            "source_sha256": source_hashes, "history": history,
            "test_wav_accessed": False})
        # Preserve the deterministic control j=0 and the first selected pair;
        # discard failed candidates that cannot participate in evaluation.
        if candidate_index != 0 and not eligible:
            free_cache(cache, [wet_key, dry_key])
        if candidate_index == 0:
            base = {"candidate": item, "rir": rir,
                "wet": pair["reverberant"], "dry": pair["clean"],
                "cap_wet": cap_wet, "cap_dry": cap_dry}
        if eligible:
            accepted = item
            break
    if accepted:
        chosen = accepted
        classification = "TAIL_ELIGIBLE"
        tail_loss_enabled = True
    else:
        if len(history) != MAX_CANDIDATES:
            raise RuntimeError("unresolved slot without all 32 frozen candidates")
        chosen = history[0]
        classification = "BASE_ONLY_CONTROL"
        tail_loss_enabled = False
    # Store only recipe and measures; exact full arrays remain in the ignored Cap60 cache/progress hashes.
    slot = {"split": split, "speaker_id": str(speaker_id),
        "pair_index": pair_index, "source_pair": [p.name for p in files],
        "source_sha256": source_hashes, "pause_start_sample": pause_start,
        "crop_start_sample": crop_start, "pause_start_in_crop": pause_crop,
        "classification": classification, "tail_loss_enabled": tail_loss_enabled,
        "candidate_index": int(chosen["candidate_index"]),
        "candidates_tested": len(history), "selected": chosen,
        "history": history, "elapsed_s": time.perf_counter()-elapsed_start,
        "design_sha256": design_hash, "training_started": False,
        "model_outputs_accessed": False, "test_wav_accessed": False}
    atomic_json(result_path, slot)
    return slot


def freeze_split(split: str, n_speakers: int, excluded: set[str], args) -> tuple[dict, set[str]]:
    root = args.experiment_dir / "splits" / split
    root.mkdir(parents=True, exist_ok=True)
    design_path = root / "manifest.design.json"
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        ids = {str(s["speaker_id"]) for s in manifest["speakers"]}
        if (manifest.get("speaker_count") != n_speakers or len(ids) != n_speakers or
                ids & excluded or manifest.get("test_wav_accessed") is not False or
                not manifest.get("frozen_before_training")):
            raise RuntimeError(f"existing {split} manifest invalid or overlapping")
        return manifest, ids
    if design_path.is_file():
        design = json.loads(design_path.read_text())
        if (design.get("split") != split or design.get("speaker_count") != n_speakers or
                design.get("excluded_speaker_ids_sha256") != hashlib.sha256(
                    "\n".join(sorted(excluded)).encode()).hexdigest() or
                design.get("test_wav_accessed") is not False):
            raise RuntimeError(f"existing {split} selection design differs from requested exclusions")
    else:
        folders = [p for p in POOL.iterdir() if p.is_dir() and p.name.isdigit()
                   and p.name not in excluded]
        folders.sort(key=lambda p: hashlib.sha256((f"G-early-{split}-v1|"+p.name).encode()).hexdigest())
        chosen = []
        for folder in folders:
            files = sorted(folder.glob("*/*.flac"))
            if len(files) >= 8:
                chosen.append((folder.name, files[:8]))
            if len(chosen) == n_speakers:
                break
        if len(chosen) != n_speakers:
            raise RuntimeError(f"could not select {n_speakers} fresh speakers for {split}")
        design = {"split": split, "speaker_count": n_speakers,
            "pair_count": n_speakers*4,
            "speakers": [{"speaker_id": sid, "pairs": [
                {"pair_index": i, "first": str(fs[2*i]), "first_sha256": sha256(fs[2*i]),
                 "second": str(fs[2*i+1]), "second_sha256": sha256(fs[2*i+1])}
                for i in range(4)]} for sid, fs in chosen],
            "candidate_namespace": f"G-early-{split}-v1|speaker|pair|candidate",
            "candidate_limit": MAX_CANDIDATES, "rt60_s_uniform": list(RT60_RANGE),
            "drr_db_uniform": list(DRR_RANGE),
            "eligibility": "first candidate with Lx(W1)>max(-60dB,Ldry(W1)+6dB); W2 descriptive only; j0 control if none",
            "excluded_speaker_ids_sha256": hashlib.sha256("\n".join(sorted(excluded)).encode()).hexdigest(),
            "cap60_binary_sha256": sha256(DEEP_FILTER),
            "frozen_before_inference": True, "frozen_before_training": True,
            "opened_for_metrics": False, "test_wav_accessed": False}
        atomic_json(design_path, design)
    if design.get("cap60_binary_sha256") != sha256(DEEP_FILTER):
        raise RuntimeError("Cap60 binary differs from frozen split design")
    design_sha = sha256(design_path)
    if design_path.name != "manifest.design.json":
        raise RuntimeError("split design path mismatch")
    # slot_result expects the design alongside its results/cache.
    atomic_json(design_path, design)
    rows, speaker_docs = [], []
    started = time.perf_counter()
    for si, speaker in enumerate(design["speakers"]):
        speaker_rows = []
        for pair in speaker["pairs"]:
            files = [Path(pair["first"]), Path(pair["second"])]
            slot = slot_result(split, si, str(speaker["speaker_id"]),
                               int(pair["pair_index"]), files, root)
            if [sha256(p) for p in files] != [pair["first_sha256"], pair["second_sha256"]]:
                raise RuntimeError("source changed after split design was frozen")
            recipe = {**slot["selected"], "input_only_selection": True,
                "classification": slot["classification"],
                "tail_loss_enabled": slot["tail_loss_enabled"],
                "candidate_index": slot["candidate_index"],
                "candidates_tested": slot["candidates_tested"]}
            speaker_rows.append({"pair_index": pair["pair_index"],
                "first": pair["first"], "first_sha256": pair["first_sha256"],
                "second": pair["second"], "second_sha256": pair["second_sha256"],
                "pause_start_sample": slot["pause_start_sample"],
                "crop_start_sample": slot["crop_start_sample"],
                "pause_start_in_crop": slot["pause_start_in_crop"],
                "classification": slot["classification"],
                "tail_loss_enabled": slot["tail_loss_enabled"],
                "candidate_index": slot["candidate_index"],
                "candidates_tested": slot["candidates_tested"],
                "procedural_rir": recipe,
                "input_tail_selection_history": slot["history"]})
            rows.append(slot)
            count = sum(x["tail_loss_enabled"] for x in speaker_rows)
            print(json.dumps({"split": split, "speakers": si+1,
                "pairs": len(rows), "speaker_eligible": count,
                "eligible": sum(x["tail_loss_enabled"] for x in rows),
                "elapsed_s": time.perf_counter()-started}), flush=True)
        speaker_docs.append({"speaker_id": str(speaker["speaker_id"]), "pairs": speaker_rows})
    counts = [sum(pair["tail_loss_enabled"] for pair in speaker["pairs"])
              for speaker in speaker_docs]
    eligible = sum(counts)
    if split == "dev":
        gate = eligible >= 48 and sum(c >= 3 for c in counts) >= 12 and all(c > 0 for c in counts)
        thresholds = {"eligible_pairs_minimum": 48, "speakers_at_least_3_minimum": 12,
                      "speakers_at_0_maximum": 0}
    else:
        gate = eligible >= 36 and sum(c >= 3 for c in counts) >= 9 and all(c > 0 for c in counts)
        thresholds = {"eligible_pairs_minimum": 36, "speakers_at_least_3_minimum": 9,
                      "speakers_at_0_maximum": 0}
    manifest = {"name": f"G-early-{split}-v1", "split": split,
        "speakers": speaker_docs, "speaker_count": n_speakers,
        "pair_count": n_speakers*4, "eligible_pair_count": eligible,
        "base_only_control_count": n_speakers*4-eligible,
        "speaker_eligible_counts_private": counts,
        "speakers_with_at_least_3_of_4": sum(c >= 3 for c in counts),
        "speakers_with_0_of_4": sum(c == 0 for c in counts),
        "gate_thresholds": thresholds, "coverage_gate_pass": gate,
        "design_sha256": design_sha,
        "candidate_namespace": design["candidate_namespace"],
        "cap60_binary_sha256": sha256(DEEP_FILTER),
        "frozen_before_training": True, "opened_for_metrics": False,
        "test_wav_accessed": False, "speaker_ids_in_public_report": False}
    atomic_json(manifest_path, manifest)
    return manifest, {str(s["speaker_id"]) for s in speaker_docs}


def freeze(args) -> dict:
    train_manifest_path = args.experiment_dir / "data/training-data-manifest.json"
    pairs_path = args.experiment_dir / "data/pairs.jsonl"
    coverage_path = args.experiment_dir / "data/coverage-audit.json"
    for p in (train_manifest_path, pairs_path, coverage_path):
        if not p.is_file():
            raise FileNotFoundError(f"prepare G-early training data first: {p}")
    train = json.loads(train_manifest_path.read_text())
    coverage = json.loads(coverage_path.read_text())
    if (train.get("test_wav_accessed") is not False or
            train.get("pair_manifest_sha256") != sha256(pairs_path) or
            not coverage.get("executable") or coverage.get("resolved_tail_slots") != 48 or
            int(coverage.get("eligible_tail_slots", 0)) < 36):
        raise RuntimeError("G-early training coverage/hash gate failed")
    rows = [json.loads(line) for line in pairs_path.read_text().splitlines() if line.strip()]
    train_ids = {str(row["speaker"]) for row in rows}
    if len(rows) != 128 or len(train_ids) != 128:
        raise RuntimeError("G-early requires exactly 128 unique fixed training speakers")
    all_used = used_speakers(ROOT / ".tools/compact-dereverb")
    # Resume-safe: known G-early selections are re-added explicitly in order below.
    for split in ("dev", "sealed"):
        p = args.experiment_dir / "splits" / split / "manifest.json"
        if p.is_file():
            ids = {str(s["speaker_id"]) for s in json.loads(p.read_text()).get("speakers", [])}
            all_used.difference_update(ids)
    excluded = all_used | train_ids
    dev, dev_ids = freeze_split("dev", 16, excluded, args)
    holdout, holdout_ids = freeze_split("sealed", 12, excluded | dev_ids, args)
    if (dev_ids & holdout_ids or (dev_ids | holdout_ids) & train_ids or
            not dev.get("coverage_gate_pass") or not holdout.get("coverage_gate_pass")):
        raise RuntimeError("G-early DEV/HOLDOUT speaker disjointness or coverage gate failed; no training")
    for manifest in (dev, holdout):
        if manifest.get("test_wav_accessed") is not False or manifest.get("opened_for_metrics"):
            raise RuntimeError("split manifest violated sealed evaluation policy")
    index_path = args.experiment_dir / "splits/manifest.json"
    if index_path.exists():
        raise FileExistsError("G-early split index already frozen")
    index = {"name": "G-early-speaker-splits-v1",
        "training_pair_manifest_sha256": sha256(pairs_path),
        "training_eligible_tail_slots": int(coverage["eligible_tail_slots"]),
        "dev_manifest_sha256": sha256(args.experiment_dir / "splits/dev/manifest.json"),
        "holdout_manifest_sha256": sha256(args.experiment_dir / "splits/sealed/manifest.json"),
        "dev_speaker_count": 16, "dev_pair_count": 64,
        "holdout_speaker_count": 12, "holdout_pair_count": 48,
        "dev_coverage_gate_pass": True, "holdout_coverage_gate_pass": True,
        "dev_holdout_disjoint": True, "disjoint_from_training": True,
        "frozen_before_training": True, "opened_for_metrics": False,
        "holdout_opened": False, "test_wav_accessed": False}
    atomic_json(index_path, index)
    return {"dev_eligible": dev["eligible_pair_count"],
        "holdout_eligible": holdout["eligible_pair_count"],
        "dev_manifest_sha256": index["dev_manifest_sha256"],
        "holdout_manifest_sha256": index["holdout_manifest_sha256"],
        "split_manifest_sha256": sha256(index_path), "test_wav_accessed": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, default=EXPERIMENT)
    args = parser.parse_args()
    print(json.dumps(freeze(args), indent=2))


if __name__ == "__main__":
    main()
