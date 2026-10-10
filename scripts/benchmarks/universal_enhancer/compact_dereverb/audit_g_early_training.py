#!/usr/bin/env python3
"""Fail-closed integrity audit for frozen G-early training arrays."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import stft

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
from prepare_cap60_conditioned_pairs import CROP_SAMPLES, DEEP_FILTER, sha256  # noqa: E402

DATA = ROOT / ".tools/compact-dereverb/g-early-2026-10-10/data"
OUT = ROOT / ".tools/compact-dereverb/g-early-2026-10-10/training-data-audit.json"
EXPECTED_KINDS = {"identity": 32, "rir_only": 48, "fan20": 24, "fan10": 24}


def audit() -> dict:
    manifest_path, pair_path, coverage_path = (DATA / "training-data-manifest.json",
        DATA / "pairs.jsonl", DATA / "coverage-audit.json")
    manifest, coverage = json.loads(manifest_path.read_text()), json.loads(coverage_path.read_text())
    rows = [json.loads(line) for line in pair_path.read_text().splitlines() if line.strip()]
    if sha256(pair_path) != manifest.get("pair_manifest_sha256") or sha256(pair_path) != coverage.get("pair_manifest_sha256"):
        raise RuntimeError("pair manifest hash mismatch")
    if manifest.get("deep_filter_sha256") != sha256(DEEP_FILTER):
        raise RuntimeError("Cap60 binary hash mismatch")
    if any(doc.get("test_wav_accessed") is not False for doc in (manifest, coverage)):
        raise RuntimeError("test.wav access flag must remain false")
    if manifest.get("training_started") is not False or manifest.get("model_outputs_accessed") is not False:
        raise RuntimeError("training data artifact unexpectedly records model activity")
    counts = {kind: sum(row["kind"] == kind for row in rows) for kind in EXPECTED_KINDS}
    speaker_ids = {str(row["speaker"]) for row in rows}
    if len(rows) != 128 or counts != EXPECTED_KINDS or len(speaker_ids) != 128:
        raise RuntimeError(f"unexpected fixed training schedule: {len(rows)}, {counts}, {len(speaker_ids)}")

    previous = json.loads((ROOT / ".tools/compact-dereverb/cap60-conditioned-2026-10-10/training-data-manifest.json").read_text())
    reserved = {str(x) for x in previous.get("future_holdout_speakers", [])}
    if speaker_ids & reserved:
        raise RuntimeError("fixed training schedule overlaps reserved final holdout speakers")

    tail_rows = [row for row in rows if row["kind"] == "rir_only"]
    eligible_count = 0
    impossible_bins = {"clean": 0, "hybrid": 0}
    total_bins = 0
    for row in rows:
        loaded = {}
        for key, digest_key in (("x", "x_sha256"), ("c", "clean_target_sha256"),
                ("a", "cap_target_sha256"), ("hybrid", "hybrid_target_sha256"),
                ("tail_ref", "tail_ref_sha256"), ("mask", "mask_sha256")):
            path = DATA / row[key]
            if sha256(path) != row[digest_key]:
                raise RuntimeError(f"array SHA-256 mismatch at row {row['index']}:{key}")
            arr = np.load(path, allow_pickle=False)
            if arr.shape != (CROP_SAMPLES,) or arr.dtype != np.float32 or not np.isfinite(arr).all():
                raise RuntimeError(f"invalid array at row {row['index']}:{key}")
            loaded[key] = arr
        if row["tail_ref_sha256"] != row["cap_target_sha256"] or not np.array_equal(loaded["tail_ref"], loaded["a"]):
            raise RuntimeError(f"tail_ref differs from independent a_q at row {row['index']}")
        hybrid = loaded["mask"] * loaded["c"] + (1 - loaded["mask"]) * loaded["a"]
        if float(np.max(np.abs(hybrid - loaded["hybrid"]))) > 1e-7:
            raise RuntimeError(f"hybrid target differs from frozen mask at row {row['index']}")
        if row["kind"] != "rir_only":
            if row.get("input_only_tail_eligible"):
                raise RuntimeError("non-RIR row cannot enable early-tail loss")
            continue
        if row.get("input_only_tail_eligible"):
            _, _, sx = stft(loaded["x"], fs=16_000, window="hann",
                nperseg=512, noverlap=384, boundary=None)
            for target_name, target in (("clean", loaded["c"]),
                    ("hybrid", loaded["hybrid"])):
                _, _, st = stft(target, fs=16_000, window="hann",
                    nperseg=512, noverlap=384, boundary=None)
                impossible_bins[target_name] += int(np.count_nonzero(
                    np.abs(st) > np.abs(sx) + 1e-12))
            total_bins += int(sx.size)
        recipe = row["rir_recipe"]
        history = recipe.get("candidate_history", [])
        if not history or [int(x["candidate_index"]) for x in history] != list(range(len(history))):
            raise RuntimeError(f"invalid candidate prefix at row {row['index']}")
        passing = []
        for candidate in history:
            w1 = candidate["windows"]["150_300"]
            valid = bool(candidate["reference_valid"] and w1["lx_db"] is not None and
                float(w1["lx_db"]) > max(-60.0, float(w1["ldry_db"]) + 6.0))
            if valid != bool(candidate["eligible_from_input_only"]):
                raise RuntimeError(f"W1 eligibility formula mismatch at row {row['index']}")
            if valid:
                passing.append(int(candidate["candidate_index"]))
        selected = int(recipe["candidate_index"])
        enabled = bool(row["input_only_tail_eligible"])
        if enabled:
            if not passing or selected != passing[0] or len(history) != selected + 1:
                raise RuntimeError(f"not first eligible W1 candidate at row {row['index']}")
            if recipe.get("classification") != "TAIL_ELIGIBLE" or not recipe.get("tail_loss_enabled"):
                raise RuntimeError(f"eligible recipe incorrectly classified at row {row['index']}")
            eligible_count += 1
        elif (len(history) != 32 or selected != 0 or passing or
              recipe.get("classification") != "BASE_ONLY_CONTROL" or recipe.get("tail_loss_enabled")):
            raise RuntimeError(f"BASE_ONLY_CONTROL rule mismatch at row {row['index']}")
    if (eligible_count != coverage.get("eligible_tail_slots") or eligible_count < 36 or
            not coverage.get("executable")):
        raise RuntimeError("G-early training coverage gate failed")
    result = {"name": "G-early-training-data-integrity-audit-v1",
        "pair_count": len(rows), "kind_counts": counts,
        "train_speaker_count": len(speaker_ids), "tail_slots": len(tail_rows),
        "eligible_tail_slots": eligible_count,
        "base_only_controls": len(tail_rows)-eligible_count,
        "training_gate_pass": True,
        "candidate_formula_audit": "PASS",
        "array_hash_and_target_audit": "PASS",
        "attenuation_only_target_excess_bins": {
            "eligible_rir_time_frequency_bins": total_bins,
            "clean_target_fraction_mag_above_x": (impossible_bins["clean"] / total_bins
                if total_bins else None),
            "hybrid_target_fraction_mag_above_x": (impossible_bins["hybrid"] / total_bins
                if total_bins else None),
            "interpretation": "diagnostic only; E-style attenuation-only model cannot increase these bins",
            "gate": False},
        "reserved_final_holdout_disjoint": True,
        "test_wav_accessed": False, "training_started": False,
        "pair_manifest_sha256": sha256(pair_path),
        "training_manifest_sha256": sha256(manifest_path)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    audit()
