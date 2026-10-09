"""Domain-adaptation A/B: fixed synthetic/measured RIR mixture, same model/loss."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from data import (CROP_SAMPLES, SAMPLE_RATE, crop_audio, make_validation_pair,
                  read_librispeech)
from model import CompactDereverb16k, parameter_count
from screen_measured_rirs import prepare_rir, select_rirs
from train import loss_components, protection_masks, total_loss, _read_crop


HOLDOUT_ROOMS = ("Hotel_SkalskyDvur_Room112", "VUT_FIT_L212", "VUT_FIT_L227")
TRAIN_ROOMS = ("Hotel_SkalskyDvur_ConferenceRoom2", "VUT_FIT_C236", "VUT_FIT_D105",
               "VUT_FIT_E112", "VUT_FIT_L207", "VUT_FIT_Q301")


def measured_rir_dev(root: Path, training_items: list[dict]) -> list[dict]:
    """Choose one distance-median, training-config-disjoint RIR per train room."""
    train_configs = {item["configuration"] for item in training_items}
    candidates: dict[str, dict[str, dict]] = {}
    for wav in sorted(root.rglob("*.wav")):
        if wav.parent.name != "RIR":
            continue
        rel = wav.relative_to(root)
        mic_idx = next((i for i, part in enumerate(rel.parts) if part.startswith("MicID")), None)
        if mic_idx is None or mic_idx < 1:
            continue
        room = rel.parts[mic_idx - 1]
        if room not in TRAIN_ROOMS:
            continue
        config_dir = wav.parent.parent
        config = str(config_dir.relative_to(root))
        if config in train_configs:
            continue
        distance_path = config_dir / "mic_meta.txt"
        match = re.search(r"^\$EnvMic\d+RelDistance\s+([-+0-9.eE]+)",
                          distance_path.read_text(errors="replace"), re.MULTILINE) if distance_path.exists() else None
        if not match:
            continue
        candidates.setdefault(room, {}).setdefault(config, {
            "path": str(wav), "room": room, "configuration": config,
            "distance_m": float(match.group(1))})
    result = []
    for room in TRAIN_ROOMS:
        choices = sorted(candidates.get(room, {}).values(),
                          key=lambda x: (x["distance_m"], x["configuration"]))
        if not choices:
            raise RuntimeError(f"no config-disjoint measured validation RIR for {room}")
        item = dict(choices[len(choices) // 2])
        item["sha256"] = hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest()
        result.append(item)
    return result


def measured_pair(clean: np.ndarray, rir: np.ndarray) -> dict[str, np.ndarray]:
    """Same-length target/input pair with a fixed common peak scale."""
    from scipy.signal import fftconvolve

    edge = int(.05 * SAMPLE_RATE)
    early_rir = rir[:edge]
    late_rir = np.pad(rir[edge:], (edge, 0))
    wet_full = fftconvolve(clean, rir, mode="full")
    early = fftconvolve(clean, early_rir, mode="full")[:clean.size]
    late = fftconvolve(clean, late_rir, mode="full")[:clean.size]
    wet = wet_full[:clean.size]
    scale = .80 / max(float(np.max(np.abs(clean))), float(np.max(np.abs(wet))), 1e-8)
    return {"clean": (clean * scale).astype(np.float32),
            "reverberant": (wet * scale).astype(np.float32),
            "late": (late * scale).astype(np.float32)}


class MixedRIRSpeech(IterableDataset):
    """Same speech crops and procedural generator, with 50% measured RIR draws."""

    def __init__(self, rows, measured, seed: int):
        super().__init__()
        self.rows, self.measured, self.seed = list(rows), list(measured), int(seed)
        if not self.rows or not self.measured:
            raise ValueError("speech rows and measured RIRs must be nonempty")

    def __iter__(self):
        worker = get_worker_info()
        wid = 0 if worker is None else worker.id
        rng = np.random.default_rng(self.seed + wid * 1_000_003)
        py_rng = random.Random(self.seed + wid * 1_000_003)
        by_room: dict[str, list[dict]] = {}
        for item in self.measured:
            by_room.setdefault(item["room"], []).append(item)
        while True:
            path, speaker = self.rows[int(rng.integers(0, len(self.rows)))]
            audio, sr = sf.read(path, dtype="float32", always_2d=False)
            if sr != SAMPLE_RATE:
                raise ValueError(f"expected 16 kHz: {path}")
            if audio.ndim == 2:
                audio = audio.mean(axis=1, dtype=np.float32)
            clean = crop_audio(audio, CROP_SAMPLES, rng)
            level = float(10.0 ** (rng.uniform(-18.0, 0.0) / 20.0))
            dry_control = bool(rng.random() < .20)
            kind = "dry"
            if dry_control:
                pair = {"clean": clean, "reverberant": clean, "late": np.zeros_like(clean)}
            elif rng.random() < .5:
                kind = "procedural"
                pair = make_validation_pair(clean, {
                    "seed": py_rng.randrange(0, 2**32),
                    "t60_s": float(rng.uniform(.25, 1.20)),
                    "direct_to_reverb_db": float(rng.uniform(-6, 18)),
                })
            else:
                # Room then RIR are drawn uniformly so inventory-rich rooms do
                # not receive more training probability.
                kind = "but_measured"
                room = str(rng.choice(tuple(sorted(by_room))))
                item = by_room[room][int(rng.integers(0, len(by_room[room])))]
                rir, _meta = prepare_rir(item)
                pair = measured_pair(clean, rir)
            yield {"mixture": torch.from_numpy(pair["reverberant"] * level),
                   "clean": torch.from_numpy(pair["clean"] * level),
                   "late": torch.from_numpy(pair["late"] * level),
                   "dry_control": dry_control, "rir_kind": kind, "speaker_id": speaker}


def validation_loss_mixed(model, rows, measured_dev, seed, device, weak_over_weight):
    """Equal weight: fixed procedural/dev pairs + config-held-out measured rooms + dry."""
    rng = np.random.default_rng(seed)
    model.eval()
    values = []
    with torch.inference_mode():
        for index, (path, _speaker) in enumerate(rows[:12]):
            clean_np = _read_crop(path, rng)
            clean = torch.from_numpy(clean_np).unsqueeze(0).to(device)
            dry_parts = loss_components(model(clean), clean,
                *protection_masks(clean, torch.zeros_like(clean)))
            values.append(float(total_loss(dry_parts, weak_over_weight).item()))
            proc = make_validation_pair(clean_np, {
                "seed": int(rng.integers(0, 2**32)),
                "t60_s": float(rng.uniform(.25, 1.20)),
                "direct_to_reverb_db": float(rng.uniform(-6, 18)),
            })
            target = torch.from_numpy(proc["clean"]).unsqueeze(0).to(device)
            mix = torch.from_numpy(proc["reverberant"]).unsqueeze(0).to(device)
            late = torch.from_numpy(proc["late"]).unsqueeze(0).to(device)
            parts = loss_components(model(mix), target, *protection_masks(target, late))
            values.append(float(total_loss(parts, weak_over_weight).item()))
            item = measured_dev[index % len(measured_dev)]
            rir, _ = prepare_rir(item)
            pair = measured_pair(clean_np, rir)
            target = torch.from_numpy(pair["clean"]).unsqueeze(0).to(device)
            mix = torch.from_numpy(pair["reverberant"]).unsqueeze(0).to(device)
            late = torch.from_numpy(pair["late"]).unsqueeze(0).to(device)
            parts = loss_components(model(mix), target, *protection_masks(target, late))
            values.append(float(total_loss(parts, weak_over_weight).item()))
    model.train()
    return float(np.mean(values))


def train(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    rows = read_librispeech(args.data_root, "train-clean-100")
    dev_rows = read_librispeech(args.data_root, "dev-clean")
    overlap = {s for _, s in rows} & {s for _, s in dev_rows}
    if overlap:
        raise RuntimeError(f"train/dev speaker overlap: {sorted(overlap)[:5]}")
    selected = select_rirs(args.rir_root)
    train_rirs = [x for x in selected if x["room"] in TRAIN_ROOMS]
    if len(train_rirs) != 12:
        raise RuntimeError("expected 6 rooms × 2 selected train RIRs")
    dev_rirs = measured_rir_dev(args.rir_root, train_rirs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.output_dir / "checkpoints"; ckpt_dir.mkdir(exist_ok=True)
    loader = DataLoader(MixedRIRSpeech(rows, train_rirs, args.seed), batch_size=args.batch_size,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    batches = iter(loader)
    model = CompactDereverb16k(base_channels=args.base_channels).to(device)
    count = parameter_count(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    args.output_dir.mkdir(exist_ok=True, parents=True)
    manifest = {"seed": args.seed, "device": str(device), "torch": torch.__version__,
        "sample_rate": SAMPLE_RATE, "crop_samples": CROP_SAMPLES, "parameter_count": count,
        "reference_checkpoint": str(args.reference_checkpoint),
        "reference_checkpoint_sha256": hashlib.sha256(args.reference_checkpoint.read_bytes()).hexdigest(),
        "initialization": "fresh model initialized with same deterministic seed as reference training",
        "train_examples": len(rows), "train_speakers": len({s for _, s in rows}),
        "dev_examples": len(dev_rows), "dev_speakers": len({s for _, s in dev_rows}),
        "dry_control_probability": .20, "rir_mix": {"procedural": .50, "BUT_measured": .50},
        "room_sampling": "uniform room, then uniform selected RIR within room",
        "train_rooms": list(TRAIN_ROOMS), "holdout_rooms": list(HOLDOUT_ROOMS),
        "measured_train_rirs": train_rirs, "measured_dev_config_disjoint_rirs": dev_rirs,
        "steps_requested": args.steps, "weak_over_weight": args.weak_over_weight,
        "batch_size": args.batch_size,
        "only_changed_factor": "RIR training distribution; fresh initialization seed, speech, architecture, loss, optimizer, update budget match reference"}
    (args.output_dir / "training-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log_path = args.output_dir / "training.jsonl"
    started = time.perf_counter(); best_dev = float("inf"); model.train()
    for step in range(1, args.steps + 1):
        batch = next(batches)
        mix, clean, late = (batch[k].to(device, non_blocking=True)
                            for k in ("mixture", "clean", "late"))
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
        if step % args.log_every == 0 or step == 1:
            row = {"step": step, "loss": float(loss.detach()), "grad_norm": float(grad_norm),
                   "elapsed_s": time.perf_counter() - started,
                   "components": {k: float(v.detach()) for k, v in parts.items()},
                   "rir_kind": batch["rir_kind"][0]}
            if device.type == "cuda":
                row["peak_allocated_vram_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
            with log_path.open("a") as f: f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        if step % args.validate_every == 0 or step == args.steps:
            dev_loss = validation_loss_mixed(model, dev_rows, dev_rirs, args.seed + 41_009,
                                             device, args.weak_over_weight)
            state = {"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "dev_loss": dev_loss, "base_channels": args.base_channels,
                     "seed": args.seed, "training_manifest": str(args.output_dir / "training-manifest.json")}
            torch.save(state, ckpt_dir / f"step-{step:06d}.pt")
            if dev_loss < best_dev:
                best_dev = dev_loss
                torch.save(state, ckpt_dir / "best-dev.pt")
            print(json.dumps({"step": step, "dev_loss": dev_loss, "best_dev": best_dev}), flush=True)
    return ckpt_dir / "best-dev.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--rir-root", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weak-over-weight", type=float, default=.5)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    print(train(parser.parse_args()))


if __name__ == "__main__":
    main()
