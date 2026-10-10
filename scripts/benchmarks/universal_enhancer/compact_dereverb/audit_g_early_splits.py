#!/usr/bin/env python3
"""Verify frozen G-early speaker splits, input-only W1 gates, and HOLDOUT lock."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
from prepare_cap60_conditioned_pairs import DEEP_FILTER, sha256  # noqa: E402

EXPERIMENT = ROOT / ".tools/compact-dereverb/g-early-2026-10-10"
EXPECTED = {"dev": (16, 64, 48, 12), "sealed": (12, 48, 36, 9)}


def verify_pair(pair: dict, label: str) -> bool:
    recipe = pair["procedural_rir"]
    history = pair["input_tail_selection_history"]
    if not recipe.get("input_only_selection") or not history:
        raise RuntimeError(f"missing input-only W1 trace: {label}")
    if [int(row["candidate_index"]) for row in history] != list(range(len(history))):
        raise RuntimeError(f"candidate history is not a contiguous prefix: {label}")
    eligible_indices = []
    for item in history:
        w1 = item["windows"]["150_300"]
        eligible = bool(item["reference_valid"] and w1["lx_db"] is not None and
            float(w1["lx_db"]) > max(-60.0, float(w1["ldry_db"]) + 6.0))
        if eligible != bool(item["tail_eligible"]):
            raise RuntimeError(f"W1 threshold differs from frozen history: {label}")
        if eligible:
            eligible_indices.append(int(item["candidate_index"]))
    expected_eligible = pair["classification"] == "TAIL_ELIGIBLE"
    if expected_eligible:
        selected = int(pair["candidate_index"])
        if (not eligible_indices or selected != eligible_indices[0] or
                len(history) != selected + 1 or not pair["tail_loss_enabled"]):
            raise RuntimeError(f"slot is not the first passing W1 candidate: {label}")
    elif (pair["classification"] != "BASE_ONLY_CONTROL" or
            len(history) != 32 or pair["candidate_index"] != 0 or eligible_indices or
            pair["tail_loss_enabled"]):
        raise RuntimeError(f"base-only control rule failed: {label}")
    if (not 1.65 <= float(recipe["t60_s"]) <= 1.85 or
            not -14.5 <= float(recipe["direct_to_reverb_db"]) <= -12.5):
        raise RuntimeError(f"RIR parameters escaped frozen bounds: {label}")
    return expected_eligible


def audit() -> dict:
    split_root = EXPERIMENT / "splits"
    index_path = split_root / "manifest.json"
    index = json.loads(index_path.read_text())
    train_pair_path = EXPERIMENT / "data/pairs.jsonl"
    train_manifest_path = EXPERIMENT / "data/training-data-manifest.json"
    train_rows = [json.loads(line) for line in train_pair_path.read_text().splitlines() if line.strip()]
    train_ids = {str(row["speaker"]) for row in train_rows}
    expected_train_hash = hashlib.sha256(train_pair_path.read_bytes()).hexdigest()
    if (index.get("training_pair_manifest_sha256") != expected_train_hash or
            index.get("test_wav_accessed") is not False or
            index.get("opened_for_metrics") is not False or
            index.get("holdout_opened") is not False or
            not index.get("frozen_before_training")):
        raise RuntimeError("invalid or opened G-early split index")
    if json.loads(train_manifest_path.read_text()).get("test_wav_accessed") is not False:
        raise RuntimeError("training manifest accessed reserved test.wav")

    ids_by_split, hashes_by_split, summary = {}, {}, {}
    for name, (speakers_expected, pairs_expected, eligible_min, speakers_min) in EXPECTED.items():
        path = split_root / name / "manifest.json"
        doc = json.loads(path.read_text())
        hash_key = "dev_manifest_sha256" if name == "dev" else "holdout_manifest_sha256"
        if (hashlib.sha256(path.read_bytes()).hexdigest() != index.get(hash_key) or
                doc.get("speaker_count") != speakers_expected or
                doc.get("pair_count") != pairs_expected or
                not doc.get("coverage_gate_pass") or
                not doc.get("frozen_before_training") or
                doc.get("opened_for_metrics") is not False or
                doc.get("test_wav_accessed") is not False or
                doc.get("cap60_binary_sha256") != sha256(DEEP_FILTER)):
            raise RuntimeError(f"invalid frozen split manifest: {name}")
        speaker_ids = {str(speaker["speaker_id"]) for speaker in doc["speakers"]}
        if len(speaker_ids) != speakers_expected or speaker_ids & train_ids:
            raise RuntimeError(f"speaker duplication or train overlap: {name}")
        counts, hashes, pair_count = [], [], 0
        for speaker in doc["speakers"]:
            if len(speaker["pairs"]) != 4:
                raise RuntimeError(f"speaker does not have four frozen RIR slots: {name}")
            count = 0
            for pair in speaker["pairs"]:
                pair_count += 1
                count += int(verify_pair(pair,
                    f"{name}:{speaker['speaker_id']}:{pair['pair_index']}"))
                hashes.extend((pair["first_sha256"], pair["second_sha256"]))
            counts.append(count)
        eligible = sum(counts)
        at_least_three = sum(count >= 3 for count in counts)
        if (pair_count != pairs_expected or eligible != doc.get("eligible_pair_count") or
                eligible < eligible_min or at_least_three < speakers_min or
                any(count == 0 for count in counts) or len(set(hashes)) != len(hashes)):
            raise RuntimeError(f"split coverage/source uniqueness gate failed: {name}")
        ids_by_split[name], hashes_by_split[name] = speaker_ids, set(hashes)
        summary[name] = {"speakers": speakers_expected, "pairs": pair_count,
            "w1_eligible": eligible, "speakers_ge_3_of_4": at_least_three,
            "speakers_zero_of_4": sum(count == 0 for count in counts),
            "gate": "PASS"}

    if (ids_by_split["dev"] & ids_by_split["sealed"] or
            hashes_by_split["dev"] & hashes_by_split["sealed"] or
            index.get("dev_holdout_disjoint") is not True or
            index.get("disjoint_from_training") is not True or
            index.get("dev_coverage_gate_pass") is not True or
            index.get("holdout_coverage_gate_pass") is not True):
        raise RuntimeError("G-early splits overlap or index gate flags are invalid")
    result = {"name": "G-early-split-integrity-audit-v1", "splits": summary,
        "train_speakers": len(train_ids), "dev_holdout_disjoint": True,
        "disjoint_from_training": True, "holdout_opened": False,
        "opened_for_metrics": False, "test_wav_accessed": False,
        "split_index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest()}
    output = EXPERIMENT / "split-integrity-audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    audit()
