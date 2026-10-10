#!/usr/bin/env python3
"""Independently audit the frozen v2 speaker roster before Cap60 inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
EXPERIMENT = ROOT / ".tools/compact-dereverb/g-early-v2-2026-10-10"
ROSTER = EXPERIMENT / "source-roster.json"


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def prior_speakers() -> set[str]:
    used: set[str] = set()
    root = ROOT / ".tools/compact-dereverb"
    for path in root.rglob("*"):
        if EXPERIMENT in path.parents or path == EXPERIMENT:
            continue
        if not path.is_file() or path.suffix.lower() not in (".json", ".jsonl", ".txt"):
            continue
        try:
            text = path.read_text()
            if path.suffix.lower() == ".txt":
                used.update(line.strip() for line in text.splitlines()
                            if line.strip().isdigit())
                continue
            docs = ([json.loads(line) for line in text.splitlines() if line.strip()]
                    if path.suffix.lower() == ".jsonl" else [json.loads(text)])
        except (OSError, json.JSONDecodeError):
            continue
        stack = docs[:]
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                for key, value in item.items():
                    if key in ("speaker", "speaker_id") and isinstance(value, (str, int)):
                        used.add(str(value))
                    elif key in ("speakers", "train_speakers", "dev_speakers",
                                 "holdout_speakers", "future_holdout_speakers") and isinstance(value, list):
                        for entry in value:
                            if isinstance(entry, dict):
                                sid = entry.get("speaker", entry.get("speaker_id"))
                                if sid is not None:
                                    used.add(str(sid))
                            elif isinstance(entry, (str, int)):
                                used.add(str(entry))
                    elif isinstance(value, (dict, list)):
                        stack.append(value)
            elif isinstance(item, list):
                stack.extend(item)
    return used


def historical_sealed_speakers() -> set[str]:
    """Collect speaker IDs from prior sealed/holdout manifests explicitly."""
    ids: set[str] = set()
    root = ROOT / ".tools/compact-dereverb"
    for path in root.rglob("*"):
        if EXPERIMENT in path.parents or not path.is_file():
            continue
        if not any(part.lower() in ("sealed", "holdout") for part in path.parts):
            continue
        if path.suffix.lower() not in (".json", ".jsonl"):
            continue
        try:
            text = path.read_text()
            docs = ([json.loads(line) for line in text.splitlines() if line.strip()]
                    if path.suffix.lower() == ".jsonl" else [json.loads(text)])
        except (OSError, json.JSONDecodeError):
            continue
        stack = docs[:]
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                for key, value in item.items():
                    if key in ("speaker", "speaker_id") and isinstance(value, (str, int)):
                        ids.add(str(value))
                    elif key in ("speakers", "holdout_speakers") and isinstance(value, list):
                        for entry in value:
                            if isinstance(entry, dict):
                                sid = entry.get("speaker", entry.get("speaker_id"))
                                if sid is not None:
                                    ids.add(str(sid))
                            elif isinstance(entry, (str, int)):
                                ids.add(str(entry))
                    elif isinstance(value, (dict, list)):
                        stack.append(value)
            elif isinstance(item, list):
                stack.extend(item)
    return ids


def audit() -> dict:
    if not ROSTER.is_file():
        raise FileNotFoundError(ROSTER)
    roster = json.loads(ROSTER.read_text())
    if (roster.get("name") != "G-early-v2-real-2026-10-10" or
            roster.get("frozen_before_cap60_inference") is not True or
            roster.get("test_wav_accessed") is not False or
            roster.get("dev_opened") is not False or
            roster.get("holdout_opened") is not False):
        raise RuntimeError("v2 source roster violates pre-inference/sealed policy")
    rows = roster.get("sources", [])
    counts = {split: sum(row.get("split") == split for row in rows)
              for split in ("train", "dev", "holdout")}
    ids = {split: {str(row["speaker_id"]) for row in rows if row.get("split") == split}
           for split in counts}
    expected_counts = {"train": 128, "dev": 16, "holdout": 12}
    if counts != expected_counts or any(len(ids[s]) != counts[s] for s in counts):
        raise RuntimeError(f"v2 speaker counts/uniqueness mismatch: {counts}")
    if ids["train"] & ids["dev"] or ids["train"] & ids["holdout"] or ids["dev"] & ids["holdout"]:
        raise RuntimeError("TRAIN/DEV/HOLDOUT speakers overlap")
    all_prior = prior_speakers()
    sealed_prior = historical_sealed_speakers()
    if set.union(*ids.values()) & all_prior:
        raise RuntimeError("v2 source speakers overlap a previously consumed speaker")
    if not sealed_prior <= all_prior or set.union(*ids.values()) & sealed_prior:
        raise RuntimeError("historical sealed speakers were not excluded from v2")
    prior_hash = sha_bytes("\n".join(sorted(all_prior)).encode())
    if prior_hash != roster.get("excluded_prior_speaker_ids_sha256"):
        raise RuntimeError("prior-speaker exclusion inventory changed after roster freeze")
    hashes = {split: sha_bytes("\n".join(sorted(ids[split])).encode())
              for split in counts}
    train_rows = roster.get("train_rows", [])
    expected_kinds = {"identity": 32, "rir_only": 48, "fan20": 24, "fan10": 24}
    kind_counts = {kind: sum(row.get("kind") == kind for row in train_rows)
                   for kind in expected_kinds}
    if len(train_rows) != 128 or kind_counts != expected_kinds:
        raise RuntimeError(f"frozen TRAIN schedule mismatch: {kind_counts}")
    for source in rows:
        for utterance in source["utterances"]:
            if not Path(utterance["path"]).is_file() or sha_file(Path(utterance["path"])) != utterance["sha256"]:
                raise RuntimeError(f"frozen source hash mismatch for speaker={source['speaker_id']}")
    result = {"name": "G-early-v2-source-integrity", "roster_sha256": sha_file(ROSTER),
        "prior_speaker_count": len(all_prior),
        "prior_speaker_ids_sha256": prior_hash,
        "prior_speaker_ids_excluded": sorted(all_prior),
        "historical_sealed_speaker_count": len(sealed_prior),
        "historical_sealed_speaker_ids_sha256": sha_bytes("\n".join(sorted(sealed_prior)).encode()),
        "historical_sealed_speaker_ids_excluded": sorted(sealed_prior),
        "split_speaker_counts": counts, "split_speaker_ids_sha256": hashes,
        "split_speaker_disjoint": True, "prior_speakers_disjoint": True,
        "train_kind_counts": kind_counts, "all_train_source_hashes_pass": True,
        "cap60_inference_started": False, "model_outputs_accessed": False,
        "dev_opened": False, "holdout_opened": False, "test_wav_accessed": False}
    output = EXPERIMENT / "source-roster-integrity.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    return {k: v for k, v in result.items()
            if k not in ("prior_speaker_ids_excluded", "historical_sealed_speaker_ids_excluded")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(json.dumps(audit(), indent=2))


if __name__ == "__main__":
    main()
