#!/usr/bin/env python3
"""Train the compact dereverberator on dry-proxy VCTK speech and measured RIRs.

This is an isolated research path. It reuses the existing model, loss, and
measured-RIR augmentation from train_measured_mix.py; product defaults are not
changed. VCTK mic1 is captured in a hemi-anechoic chamber, so it is a close,
low-reverberation target, not a mathematically anechoic recording.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.utils.data import DataLoader

from data import SAMPLE_RATE
from model import CompactDereverb16k, parameter_count
from screen_measured_rirs import select_rirs
from train import loss_components, protection_masks, total_loss
from train_measured_mix import (
    HOLDOUT_ROOMS,
    TRAIN_ROOMS,
    MixedRIRSpeech,
    measured_pair,
    measured_rir_dev,
    validation_loss_mixed,
)

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SPEECH_MANIFEST = ROOT / "corpus/samples/fullband-student-vctk-01/source-manifest.json"
DEFAULT_RIR_ROOT = ROOT / ".tools/compact-dereverb/data/but-reverbdb/extracted"
DEFAULT_OUTPUT = ROOT / ".tools/compact-dereverb/vctk-measured-room-pilot-2026-10-11"
VCTK_RATE = 48_000
SPEECH_RATE = SAMPLE_RATE


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_vctk_16k(manifest_path: Path, cache_dir: Path) -> tuple[list[dict], dict]:
    """Verify source hashes and materialize deterministic 48→16 kHz FLACs."""
    manifest_path = manifest_path.resolve()
    source_manifest = json.loads(manifest_path.read_text())
    if source_manifest.get("dataset") != "CSTR VCTK Corpus 0.92 mic1 only":
        raise ValueError("expected the documented VCTK 0.92 mic1-only manifest")
    if "CC BY 4.0" not in source_manifest.get("license", ""):
        raise ValueError("source manifest does not record CC BY 4.0")
    if "hemi-anechoic" not in source_manifest.get("recording_setup", ""):
        raise ValueError("source manifest lacks the hemi-anechoic recording note")
    split = source_manifest.get("speaker_split", {})
    train_speakers, validation_speakers = set(split.get("train", [])), set(split.get("validation", []))
    if not train_speakers or not validation_speakers or train_speakers & validation_speakers:
        raise ValueError("VCTK train/validation speaker split is empty or overlapping")

    root = ROOT
    records = []
    for row in source_manifest.get("files", []):
        split_name = row.get("split")
        speaker = row.get("speaker_id")
        expected = train_speakers if split_name == "train" else validation_speakers if split_name == "validation" else None
        if expected is None or speaker not in expected:
            raise ValueError(f"manifest row has invalid split/speaker: {row}")
        source = root / row["path"]
        if not source.is_file() or sha256(source) != row["sha256"]:
            raise ValueError(f"missing or changed VCTK source audio: {source}")
        audio, rate = sf.read(source, dtype="float32", always_2d=True)
        if rate != VCTK_RATE or audio.shape[1] != 1 or not np.isfinite(audio).all():
            raise ValueError(f"unexpected VCTK source format: {source}")
        converted = resample_poly(audio[:, 0], SPEECH_RATE, VCTK_RATE).astype(np.float32)
        out = cache_dir / split_name / f"{source.stem}.flac"
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out, converted, SPEECH_RATE, subtype="PCM_16", format="FLAC")
        records.append({
            "split": split_name,
            "speaker_id": speaker,
            "source_path": row["path"],
            "source_sha256": row["sha256"],
            "path": str(out),
            "sha256": sha256(out),
            "source_sample_rate_hz": VCTK_RATE,
            "sample_rate_hz": SPEECH_RATE,
            "resampler": "scipy.signal.resample_poly(up=1, down=3)",
            "frames": len(converted),
        })
    if not records:
        raise ValueError("VCTK manifest contains no audio rows")
    return records, source_manifest


def balanced_validation_rows(records: list[dict], cache_dir: Path,
                             per_speaker: int = 2) -> list[tuple[Path, str]]:
    """Interleave speakers so the fixed validation batch is not one-voice heavy."""
    by_speaker: dict[str, deque[Path]] = defaultdict(deque)
    for row in records:
        if row["split"] == "validation":
            by_speaker[row["speaker_id"]].append(Path(row["path"]))
    selected = []
    for _ in range(per_speaker):
        for speaker in sorted(by_speaker):
            if by_speaker[speaker]:
                selected.append((by_speaker[speaker].popleft(), speaker))
    if len(selected) < 6:
        raise ValueError("need at least six balanced VCTK validation clips")
    return selected


def rows_for_split(records: list[dict], split: str) -> list[tuple[Path, str]]:
    return [(Path(row["path"]), row["speaker_id"]) for row in records if row["split"] == split]


def train(args: argparse.Namespace) -> Path:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
    torch.set_num_threads(args.cpu_threads)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any((args.output_dir / name).exists() for name in ("training.jsonl", "checkpoints")):
        raise FileExistsError(f"refusing to reuse a nonempty run directory: {args.output_dir}")
    cache_dir = args.output_dir / "speech16k-cache"
    records, source_manifest = prepare_vctk_16k(args.speech_manifest, cache_dir)
    train_rows = rows_for_split(records, "train")
    dev_rows = balanced_validation_rows(records, cache_dir, per_speaker=2)
    train_speakers = {speaker for _, speaker in train_rows}
    all_dev_speakers = {row["speaker_id"] for row in records if row["split"] == "validation"}
    dev_speakers = {speaker for _, speaker in dev_rows}
    if not train_rows or train_speakers & all_dev_speakers:
        raise ValueError("VCTK speaker leakage or empty training split")

    selected = select_rirs(args.rir_root)
    but_train_rirs = [item for item in selected if item["room"] in TRAIN_ROOMS]
    if len(but_train_rirs) != 12:
        raise RuntimeError(f"expected 6 BUT rooms × 2 measured training RIRs, got {len(but_train_rirs)}")
    extra_rirs = []
    for extra_root in args.extra_rir_root:
        extra_path = str(extra_root.resolve()).lower()
        if "dechorate" in extra_path:
            source = {
                "source_url": "https://sofacoustics.org/data/database/dechorate/",
                "dataset_version": "dEchorate subset v2",
                "license_text": "MIT (embedded in each selected source SOFA; verified in local source manifest)",
                "attribution": "dEchorate dataset, Diego Di Carlo et al. (2019)",
            }
        elif "/air/" in extra_path:
            source = {
                "source_url": "https://www.iks.rwth-aachen.de/en/research/tools-downloads/databases/aachen-impulse-response-database",
                "dataset_version": "Aachen Impulse Response Database 1.4",
                "license_text": "MIT (included readme/license.txt; recorded in local AIR source manifest)",
                "attribution": "Marco Jeub et al., RWTH Aachen University; Jeub et al., DSP 2009 and ICA 2010",
            }
        else:
            raise ValueError(f"unrecognized extra RIR source root: {extra_root}")
        extra_items = select_rirs(extra_root, **source)
        if not extra_items:
            raise RuntimeError(f"no selected RIRs in extra root: {extra_root}")
        extra_rirs.extend(extra_items)
    train_rirs = but_train_rirs + extra_rirs
    train_room_ids = {item["room"] for item in train_rirs}
    if len(train_room_ids) != len({item["room"] for item in but_train_rirs}) + len({item["room"] for item in extra_rirs}):
        raise RuntimeError("RIR room IDs collide across source datasets")
    if train_room_ids & set(HOLDOUT_ROOMS):
        raise RuntimeError("measured training RIR rooms overlap the frozen BUT holdout rooms")
    dev_room_ids = HOLDOUT_ROOMS if args.room_heldout_validation else TRAIN_ROOMS
    dev_rirs = measured_rir_dev(args.rir_root, but_train_rirs, candidate_rooms=dev_room_ids)

    ckpt_dir = args.output_dir / "checkpoints"
    ckpt_dir.mkdir()
    loader = DataLoader(MixedRIRSpeech(train_rows, train_rirs, args.seed,
                                       dry_probability=args.dry_probability,
                                       target_early=args.target_early),
                        batch_size=args.batch_size, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda")
    batches = iter(loader)
    model = CompactDereverb16k(base_channels=args.base_channels).to(device)
    count = parameter_count(model)
    if not 500_000 <= count <= 2_000_000:
        raise RuntimeError(f"model parameter count outside pilot budget: {count}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    manifest = {
        "experiment": "VCTK mic1 close-speech × measured BUT ReverbDB RIR pilot",
        "seed": args.seed,
        "device": str(device),
        "torch": torch.__version__,
        "sample_rate_hz": SPEECH_RATE,
        "source_manifest": str(args.speech_manifest),
        "source_manifest_sha256": sha256(args.speech_manifest),
        "source_citation": source_manifest["citation"],
        "source_license": source_manifest["license"],
        "recording_setup": source_manifest["recording_setup"],
        "train_speakers": sorted(train_speakers),
        "validation_speakers": sorted(all_dev_speakers),
        "speaker_overlap": sorted(train_speakers & all_dev_speakers),
        "train_clip_count": len(train_rows),
        "validation_clip_count": len(dev_rows),
        "parameter_count": count,
        "rir_source": "BUT ReverbDB and optional licensed measured RIR roots, selected through screen_measured_rirs.select_rirs",
        "train_rooms": sorted(train_room_ids),
        "validation_rooms": sorted(set(dev_room_ids)),
        "final_evaluation_room_ids": ["room009", "room014"],
        "validation_rirs_config_disjoint": dev_rirs,
        "heldout_but_rooms": sorted(HOLDOUT_ROOMS),
        "train_rir_configurations": train_rirs,
        "measured_rir_source_counts": {
            source: sum(item.get("dataset_version") == source for item in train_rirs)
            for source in sorted({item.get("dataset_version", "unknown") for item in train_rirs})
        },
        "rir_mix": {"dry": args.dry_probability,
                    "procedural": (1.0 - args.dry_probability) / 2,
                    "measured": (1.0 - args.dry_probability) / 2},
        "room_sampling": "uniform train room, then uniform measured RIR within room",
        "weak_over_weight": args.weak_over_weight,
        "target_policy": "retain measured/procedural early path (first 50 ms) on reverberant pairs; dry target on clean controls"
                         if args.target_early else "dry target for all pairs",
        "steps_requested": args.steps,
        "source_data_note": "VCTK mic1 was recorded in a hemi-anechoic chamber; it is a close/dry proxy, not a perfect anechoic target.",
    }
    (args.output_dir / "training-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log_path = args.output_dir / "training.jsonl"
    best_dev = float("inf")
    started = time.perf_counter()
    model.train()
    for step in range(1, args.steps + 1):
        batch = next(batches)
        mix, clean, late = (batch[key].to(device, non_blocking=True)
                            for key in ("mixture", "clean", "late"))
        optimizer.zero_grad(set_to_none=True)
        parts = loss_components(model(mix), clean, *protection_masks(clean, late))
        loss = total_loss(parts, weak_over_weight=args.weak_over_weight)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite gradient at step {step}")
        optimizer.step()
        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(device) / 2**20
            if peak > 2048:
                raise MemoryError(f"training exceeded 2 GiB VRAM gate: {peak:.1f} MiB")
        if step == 1 or step % args.log_every == 0:
            row = {"step": step, "loss": float(loss.detach()), "grad_norm": float(grad_norm),
                   "elapsed_s": time.perf_counter() - started,
                   "components": {key: float(value.detach()) for key, value in parts.items()},
                   "rir_kind": batch["rir_kind"][0],
                   "peak_allocated_vram_mib": torch.cuda.max_memory_allocated(device) / 2**20
                   if device.type == "cuda" else None}
            with log_path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        if step % args.validate_every == 0 or step == args.steps:
            dev_loss = validation_loss_mixed(
                model, dev_rows, dev_rirs, args.seed + 41_009, device, args.weak_over_weight,
                target_early=args.target_early)
            state = {"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "dev_loss": dev_loss, "base_channels": args.base_channels,
                     "seed": args.seed, "variant": "C-vctk-measured-rir",
                     "target_early": args.target_early,
                     "training_manifest": str(args.output_dir / "training-manifest.json")}
            torch.save(state, ckpt_dir / f"step-{step:06d}.pt")
            if dev_loss < best_dev:
                best_dev = dev_loss
                torch.save(state, ckpt_dir / "best-dev.pt")
            print(json.dumps({"step": step, "dev_loss": dev_loss,
                              "best_dev": best_dev,
                              "peak_allocated_vram_mib": torch.cuda.max_memory_allocated(device) / 2**20
                              if device.type == "cuda" else None}), flush=True)
    result = {"checkpoint": str(ckpt_dir / "best-dev.pt"),
              "steps": args.steps, "best_validation_loss": best_dev,
              "elapsed_s": time.perf_counter() - started,
              "peak_allocated_vram_mib": torch.cuda.max_memory_allocated(device) / 2**20
              if device.type == "cuda" else None}
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return ckpt_dir / "best-dev.pt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speech-manifest", type=Path, default=DEFAULT_SPEECH_MANIFEST)
    parser.add_argument("--rir-root", type=Path, default=DEFAULT_RIR_ROOT)
    parser.add_argument("--extra-rir-root", type=Path, action="append", default=[])
    parser.add_argument("--room-heldout-validation", action="store_true",
                        help="select validation RIRs from BUT rooms entirely absent from the training split")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weak-over-weight", type=float, default=0.5)
    parser.add_argument("--dry-probability", type=float, default=0.20)
    parser.add_argument("--target-early", action="store_true",
                        help="use measured/procedural 0-50 ms early path as target for reverberant draws")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
