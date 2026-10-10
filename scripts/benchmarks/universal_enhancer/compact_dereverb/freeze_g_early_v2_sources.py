#!/usr/bin/env python3
"""Freeze a new speaker-disjoint G-early-v2 source roster (metadata only)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
EXPERIMENT = ROOT / ".tools/compact-dereverb/g-early-v2-2026-10-10"
POOL = ROOT / ".tools/compact-dereverb/data/LibriSpeech/train-clean-360"
NAMESPACE = "G-early-v2-real-2026-10-10"
EXPECTED = {"identity": 32, "rir_only": 48, "fan20": 24, "fan10": 24}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def used_speakers(root: Path, own_path: Path) -> set[str]:
    """Read speaker IDs only from prior metadata; never opens audio or checkpoints."""
    used: set[str] = set()
    for path in root.rglob("*"):
        if not path.is_file() or path == own_path or own_path in path.parents:
            continue
        if path.suffix.lower() not in (".json", ".jsonl", ".txt"):
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


def order_key(label: str) -> str:
    return hashlib.sha256(f"{NAMESPACE}|{label}".encode()).hexdigest()


def freeze(output: Path) -> dict:
    roster = output / "source-roster.json"
    if roster.exists():
        raise FileExistsError(f"immutable source roster already exists: {roster}")
    if not POOL.is_dir():
        raise FileNotFoundError(POOL)
    excluded = used_speakers(ROOT / ".tools/compact-dereverb", roster)
    eligible = []
    for folder in POOL.iterdir():
        if not folder.is_dir() or not folder.name.isdigit() or folder.name in excluded:
            continue
        files = sorted(folder.glob("*/*.flac"))
        if len(files) >= 8:
            eligible.append((folder.name, files[:8]))
    eligible.sort(key=lambda pair: order_key(pair[0]))
    if len(eligible) < 156:
        raise RuntimeError(f"need 156 fresh speakers; found {len(eligible)}")

    # Freeze each split before any Cap60 inference. Keep assignments disjoint.
    chosen = eligible[:156]
    train, dev, holdout = chosen[:128], chosen[128:144], chosen[144:]
    kinds = [kind for kind, count in EXPECTED.items() for _ in range(count)]
    kinds.sort(key=lambda kind_index: order_key(f"kind|{kind_index}"))
    train_rows = []
    for index, ((speaker, paths), kind) in enumerate(zip(train, kinds)):
        hashes = [sha256(path) for path in paths[:2]]
        seed = int(order_key(f"train|{speaker}")[:8], 16)
        import numpy as np
        rng = np.random.default_rng(seed)
        gain = float(10 ** (rng.uniform(-15.0, -3.0) / 20.0))
        t60 = float(rng.uniform(.25, 1.2))
        drr = float(rng.uniform(-6.0, 18.0))
        train_rows.append({
            "index": index, "speaker": speaker, "kind": kind,
            "source": {"source_pair": [paths[0].name, paths[1].name],
                       "pause_start_in_crop": None},
            "source_sha256": hashes, "crop_start_sample": 0,
            "post_cap_gain": gain, "shared_branch_scale": 1.0,
            "noise_seed": int(order_key(f"noise|{speaker}")[:8], 16),
            "rir_kind": "procedural",
            "rir_recipe": {"seed": int(order_key(f"rir|{speaker}")[:8], 16),
                           "t60_s": t60, "direct_to_reverb_db": drr},
        })
    train_rows.sort(key=lambda row: row["index"])
    source_rows = []
    for split, pairs in (("train", train), ("dev", dev), ("holdout", holdout)):
        for speaker, paths in pairs:
            source_rows.append({"split": split, "speaker_id": speaker,
                "utterances": [{"path": str(path), "sha256": sha256(path)}
                               for path in paths[:8]]})
    payload = {
        "name": NAMESPACE, "source_corpus": "LibriSpeech train-clean-360",
        "speaker_count": 156, "train_speaker_count": 128,
        "dev_speaker_count": 16, "holdout_speaker_count": 12,
        "train_kind_counts": EXPECTED,
        "excluded_prior_speaker_count": len(excluded),
        "excluded_prior_speaker_ids_sha256": hashlib.sha256(
            "\n".join(sorted(excluded)).encode()).hexdigest(),
        "split_speaker_disjoint": True, "all_prior_speaker_ids_excluded": True,
        "frozen_before_cap60_inference": True, "training_started": False,
        "dev_opened": False, "holdout_opened": False,
        "test_wav_accessed": False, "sources": source_rows,
        "train_rows": train_rows,
    }
    output.mkdir(parents=True, exist_ok=True)
    tmp = roster.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    tmp.replace(roster)
    return {"roster": str(roster), "sha256": sha256(roster),
            "excluded_speaker_count": len(excluded), "speaker_count": 156,
            "train": 128, "dev": 16, "holdout": 12,
            "train_kind_counts": EXPECTED,
            "speaker_disjoint": True, "holdout_opened": False,
            "test_wav_accessed": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT)
    args = parser.parse_args()
    print(json.dumps(freeze(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
