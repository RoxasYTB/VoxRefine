#!/usr/bin/env python3
"""Freeze fresh speaker DEV and HOLDOUT-G recipes before either G fit."""
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
from evaluate_e_synthetic_sealed import rms  # noqa: E402
from freeze_f2_holdout import used_speakers  # noqa: E402
from prepare_cap60_conditioned_pairs import (  # noqa: E402
    DEEP_FILTER, LIBRISPEECH, SR, cap60_one, sha256,
)
from prepare_f2_tailbank import crop, tail_input_levels  # noqa: E402
from screen_measured_rirs import make_inserted_pause_pair  # noqa: E402
from train_measured_mix import measured_pair  # noqa: E402

EXPERIMENT = ROOT / ".tools/compact-dereverb/g-target-ablation-2026-10-10"
POOL = LIBRISPEECH / "train-clean-360"
TAIL_T60, TAIL_DRR = (1.65, 1.85), (-14.5, -12.5)
MAX_CANDIDATES = 32


def hash_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def unit(text: str) -> float:
    return hash_seed(text) / 0xFFFFFFFF


def candidate(seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    return float(rng.uniform(*TAIL_T60)), float(rng.uniform(*TAIL_DRR))


def free_cache(cache: Path, keys: list[str]) -> None:
    for key in keys:
        (cache / f"{key}.npy").unlink(missing_ok=True)
        (cache / f"{key}.json").unlink(missing_ok=True)


def freeze_one(name: str, count: int, excluded: set[str], args) -> tuple[dict, set[str]]:
    out = args.experiment_dir / "splits" / name
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        ids = {str(doc["speaker_id"]) for doc in manifest.get("speakers", [])}
        if (manifest.get("speaker_count") != count or len(ids) != count or
                manifest.get("pair_count") != count * 4 or
                not manifest.get("frozen_before_training") or
                manifest.get("opened_for_metrics") or
                manifest.get("test_wav_accessed") is not False or ids & excluded):
            raise RuntimeError(f"existing G split is invalid or overlaps exclusions: {manifest_path}")
        return ({"name": manifest["name"], "speaker_count": count,
            "pair_count": count * 4, "manifest_sha256": sha256(manifest_path),
            "opened_for_metrics": False, "test_wav_accessed": False}, ids)
    out.mkdir(parents=True, exist_ok=True)
    cache, work = out / "cap60-cache", out / "cap60-work"
    cache.mkdir(exist_ok=True)
    work.mkdir(exist_ok=True)
    if not POOL.is_dir():
        raise FileNotFoundError(POOL)
    folders = [p for p in POOL.iterdir() if p.is_dir() and p.name.isdigit()
               and p.name not in excluded]
    folders.sort(key=lambda p: hashlib.sha256((name + "|" + p.name).encode()).hexdigest())
    selected = []
    for folder in folders:
        files = sorted(folder.glob("*/*.flac"))
        if len(files) >= 8:
            selected.append((folder.name, files[:8]))
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(f"could not select {count} fresh {name} speakers")

    speakers = []
    all_speaker_ids = set()
    stats = []
    started = time.perf_counter()
    for speaker_index, (speaker_id, files) in enumerate(selected):
        all_speaker_ids.add(speaker_id)
        pairs = []
        for pair_index in range(4):
            first, second = files[pair_index * 2:pair_index * 2 + 2]
            joined, pause = make_inserted_pause_pair(first, second)
            joined = joined.astype(np.float32)
            joined *= .80 / max(float(np.max(np.abs(joined))), 1e-8)
            crop_start = int(pause["clip_relative_start_sample"]) - int(.75 * SR)
            pause_crop = int(pause["clip_relative_start_sample"]) - crop_start
            clean_crop = crop(joined, crop_start)
            key_base = f"{name}-{speaker_index:02d}-{pair_index:02d}"
            recipe = None
            history = []
            kept_keys: list[str] = []
            for candidate_i in range(MAX_CANDIDATES):
                candidate_seed = hash_seed(f"G-target-ablation-{name}-rir-v1|{speaker_id}|{pair_index}|{candidate_i}")
                t60_s, drr_db = candidate(candidate_seed)
                rir, _early, _late = make_procedural_rir(SR, candidate_seed,
                    t60_s, drr_db, max_t60_s=2.0, min_direct_to_reverb_db=-20.0)
                pair = measured_pair(joined, rir)
                input_key = f"{key_base}-candidate-{candidate_i:02d}-input"
                clean_key = f"{key_base}-candidate-{candidate_i:02d}-clean"
                cap_input, input_meta = cap60_one(args.deep_filter,
                    pair["reverberant"], SR, work, input_key, cache)
                cap_clean, clean_meta = cap60_one(args.deep_filter,
                    pair["clean"], SR, work, clean_key, cache)
                levels = tail_input_levels(crop(cap_clean, crop_start),
                    crop(cap_input, crop_start), pause_crop,
                    activity_clean_crop=clean_crop)
                eligible = bool(levels.get("reference_valid") and all(
                    value is not None and value > -50.0
                    for value in levels.get("window_db", {}).values()))
                history.append({"candidate_index": candidate_i,
                    "seed": candidate_seed, "t60_s": t60_s, "drr_db": drr_db,
                    "input_tail_db": levels.get("window_db"),
                    "input_only_eligible": eligible})
                if eligible:
                    kept_keys = [input_key, clean_key]
                    recipe = {"seed": candidate_seed, "t60_s": t60_s,
                        "direct_to_reverb_db": drr_db,
                        "candidate_index": candidate_i,
                        "candidates_tested": candidate_i + 1,
                        "input_tail_db": levels["window_db"],
                        "input_only_selection": True,
                        "cap60_input": input_meta, "cap60_clean": clean_meta}
                    break
                free_cache(cache, [input_key, clean_key])
            if recipe is None:
                audit = {"name": f"{name}-selection-audit-v1",
                    "resolved_pairs": len(stats), "required_pairs": count * 4,
                    "unresolved_speaker_slot": pair_index,
                    "candidate_limit": MAX_CANDIDATES, "history": history,
                    "input_only_selection": True, "test_wav_accessed": False}
                (out / "selection-audit.json").write_text(json.dumps(audit,
                    indent=2, allow_nan=False) + "\n")
                raise RuntimeError(f"unresolved {name} pair after {MAX_CANDIDATES} RIR candidates")
            key = f"{name}|{speaker_id}|{pair_index}"
            pairs.append({"pair_index": pair_index,
                "first": first.name, "first_sha256": sha256(first),
                "second": second.name, "second_sha256": sha256(second),
                "procedural_rir": recipe,
                "fan20_seed": hash_seed(key + "|fan20"),
                "fan10_seed": hash_seed(key + "|fan10"),
                "noise20_seed": hash_seed(key + "|noise20"),
                "noise10_seed": hash_seed(key + "|noise10"),
                "common_post_cap_gain": 1.0,
                "crop_start_sample": crop_start,
                "pause_start_in_crop": pause_crop,
                "input_tail_selection_history": history})
            stats.append({"speaker_slot": speaker_index, "pair_slot": pair_index,
                "candidates_tested": recipe["candidates_tested"]})
            print(json.dumps({"split": name, "speakers": speaker_index + 1,
                "pairs": len(stats), "candidates_tested": recipe["candidates_tested"],
                "elapsed_s": time.perf_counter() - started}), flush=True)
        speakers.append({"speaker_id": speaker_id, "pairs": pairs})

    manifest = {"name": f"G-target-ablation-{name}-v1", "speakers": speakers,
        "speaker_count": count, "pair_count": count * 4,
        "rir_count": count * 4,
        "selection": "hash order among train-clean-360 speakers absent from all prior manifests and earlier G splits",
        "candidate_namespace": f"G-target-ablation-{name}-rir-v1|speaker|pair|candidate",
        "candidate_limit": MAX_CANDIDATES,
        "candidate_parameters": "RT60 U[1.65,1.85] seconds; DRR U[-14.5,-12.5] dB",
        "selection_policy": "first exact full Cap60 input with both tail windows > -50dB against exact Cap60 dry reference",
        "input_only_selection": True, "common_post_cap_gain": 1.0,
        "slot_resolution": stats, "frozen_before_training": True,
        "opened_for_metrics": False, "test_wav_accessed": False,
        "speaker_ids_in_public_report": False}
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return {"name": manifest["name"], "speaker_count": count,
        "pair_count": count * 4, "manifest_sha256": sha256(manifest_path),
        "opened_for_metrics": False, "test_wav_accessed": False}, all_speaker_ids


def freeze(args) -> dict:
    train_manifest = args.experiment_dir / "data/training-data-manifest.json"
    train_pairs_path = args.experiment_dir / "data/pairs.jsonl"
    coverage_path = args.experiment_dir / "data/coverage-audit.json"
    for path in (train_manifest, train_pairs_path, coverage_path):
        if not path.is_file():
            raise FileNotFoundError(f"G training data must be complete before split freeze: {path}")
    train = json.loads(train_manifest.read_text())
    coverage = json.loads(coverage_path.read_text())
    if train.get("test_wav_accessed") is not False or not coverage.get("executable") or coverage.get("resolved_tail_slots") != 48:
        raise RuntimeError("G training data is not safe/executable")
    all_used = used_speakers(ROOT / ".tools/compact-dereverb")
    # A repeat after an interrupted split-freeze may see its own already-frozen
    # speaker IDs while collecting prior IDs. Remove only these known G IDs;
    # they are added back explicitly as DEV exclusions for HOLDOUT-G below.
    for own_path in (args.experiment_dir / "splits/dev/manifest.json",
                     args.experiment_dir / "splits/sealed/manifest.json"):
        if own_path.is_file():
            own_doc = json.loads(own_path.read_text())
            all_used.difference_update(str(d["speaker_id"])
                                       for d in own_doc.get("speakers", []))
    train_rows = [json.loads(line) for line in train_pairs_path.read_text().splitlines() if line.strip()]
    train_ids = {str(row["speaker"]) for row in train_rows}
    if len(train_ids) != 128:
        raise RuntimeError("G training split must contain 128 unique speakers")
    excluded = all_used | train_ids
    dev, dev_ids = freeze_one("dev", 16, excluded, args)
    holdout, holdout_ids = freeze_one("sealed", 12, excluded | dev_ids, args)
    if dev_ids & holdout_ids or (dev_ids | holdout_ids) & train_ids:
        raise RuntimeError("G speaker splits overlap")
    split_manifest = {"name": "G-target-ablation-speaker-splits-v1",
        "training_pair_manifest_sha256": sha256(train_pairs_path),
        "dev_manifest_sha256": dev["manifest_sha256"],
        "holdout_manifest_sha256": holdout["manifest_sha256"],
        "dev_speaker_count": 16, "dev_pair_count": 64,
        "holdout_speaker_count": 12, "holdout_pair_count": 48,
        "dev": dev, "holdout": holdout,
        "disjoint_from_training": True, "dev_holdout_disjoint": True,
        "frozen_before_training": True, "opened_for_metrics": False,
        "holdout_opened": False,
        "exclusion_set_sha256": hashlib.sha256("\n".join(sorted(excluded)).encode()).hexdigest(),
        "test_wav_accessed": False}
    parent = args.experiment_dir / "splits"
    target = parent / "manifest.json"
    if target.exists():
        raise FileExistsError(f"G split index already exists: {target}")
    target.write_text(json.dumps(split_manifest, indent=2, allow_nan=False) + "\n")
    return {"dev_speakers": 16, "holdout_speakers": 12,
        "dev_manifest_sha256": dev["manifest_sha256"],
        "holdout_manifest_sha256": holdout["manifest_sha256"],
        "split_manifest_sha256": sha256(target), "test_wav_accessed": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, default=EXPERIMENT)
    parser.add_argument("--deep-filter", type=Path, default=DEEP_FILTER)
    args = parser.parse_args()
    print(json.dumps(freeze(args), indent=2))


if __name__ == "__main__":
    main()
