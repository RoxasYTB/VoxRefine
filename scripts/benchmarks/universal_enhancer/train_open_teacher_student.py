#!/usr/bin/env python3
"""Distill the locally installed open DeepFilterNet teacher into a compact student."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from fullband_student import FullbandStudent, parameter_count, stft_l1

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "corpus/samples/open-teacher-student-01"
OUT = ROOT / "results/open-teacher-student-2026-10-10"
RATE = 48_000


def load_rows(split: str) -> list[dict]:
    manifest = json.loads((DATA / "manifest.json").read_text())
    rows = [row for row in manifest["records"] if row["split"] == split]
    for row in rows:
        for key in ("input_path", "teacher_path", "clean_path"):
            path = ROOT / row[key]
            if not path.is_file():
                raise FileNotFoundError(path)
    if not rows:
        raise RuntimeError(f"No {split} rows in {DATA / 'manifest.json'}")
    return rows


def read_audio(rel: str) -> torch.Tensor:
    audio, sr = sf.read(ROOT / rel, dtype="float32", always_2d=True)
    if sr != RATE or audio.shape[1] != 1:
        raise ValueError(f"Expected mono {RATE} Hz audio: {rel} ({sr}, {audio.shape})")
    return torch.from_numpy(audio[:, 0].copy())


def crop(rows: list[dict], seconds: float, device: torch.device,
         rng: random.Random, clean_identity: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    row = rng.choice(rows)
    x = read_audio(row["clean_path"] if clean_identity else row["input_path"])
    y = read_audio(row["clean_path"] if clean_identity else row["teacher_path"])
    length = round(seconds * RATE)
    if x.numel() != y.numel() or x.numel() < length:
        raise ValueError(f"Pair length mismatch or short crop: {row}")
    start = rng.randint(0, x.numel() - length)
    x, y = x[start:start + length], y[start:start + length]
    # Use one common gain for the pair; don't loudness-normalize teacher and input separately.
    scale = x.square().mean().sqrt().clamp_min(1e-4)
    return (x / scale).to(device)[None], (y / scale).to(device)[None]


@torch.no_grad()
def evaluate(model: FullbandStudent, rows: list[dict], seconds: float,
             device: torch.device, seed: int, count: int) -> dict[str, float]:
    model.eval()
    rng = random.Random(seed)
    teacher_l1, teacher_spec, identity_l1, identity_sdr = [], [], [], []
    for _ in range(count):
        x, y = crop(rows, seconds, device, rng)
        pred = model(x)
        teacher_l1.append(float((pred - y).abs().mean()))
        teacher_spec.append(float(stft_l1(pred, y, 1024, 256,
                                           torch.hann_window(1024, device=device))))
        clean_x, clean_y = crop(rows, seconds, device, rng, clean_identity=True)
        clean_out = model(clean_x)
        identity_l1.append(float((clean_out - clean_y).abs().mean()))
        noise = clean_out - clean_y
        identity_sdr.append(float(10 * torch.log10(clean_y.square().mean().clamp_min(1e-10) /
                                                    noise.square().mean().clamp_min(1e-10))))
    model.train()
    return {"teacher_wave_l1": float(np.mean(teacher_l1)),
            "teacher_stft_l1": float(np.mean(teacher_spec)),
            "clean_identity_l1": float(np.mean(identity_l1)),
            "clean_identity_sdr_db": float(np.mean(identity_sdr))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--validation-crops", type=int, default=12)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--identity-weight", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or args.seconds < 0.5 or args.width < 8:
        parser.error("steps >= 1, seconds >= 0.5, and width >= 8 are required")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda": torch.cuda.manual_seed_all(args.seed)
    train_rows, val_rows = load_rows("train"), load_rows("validation")
    model = FullbandStudent(base_channels=args.width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "latest.pt"
    start = 0
    if args.resume and checkpoint.exists():
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        start = int(state["step"])
    window = torch.hann_window(1024, device=device)
    baseline = evaluate(model, val_rows, args.seconds, device, args.seed + 1, args.validation_crops)
    print(json.dumps({"event": "start", "teacher": "DeepFilterNet 0.5.6; atten-lim 30 dB",
                      "target": "open teacher outputs only; no Adobe audio or metrics",
                      "device": str(device), "parameters": parameter_count(model),
                      "train_pairs": len(train_rows), "validation_pairs": len(val_rows),
                      "baseline": baseline,
                      "cuda_free_bytes": torch.cuda.mem_get_info(device)[0] if device.type == "cuda" else None},
                     indent=2), flush=True)
    metrics_path = args.output / "metrics.jsonl"
    best = float("inf")
    rng = random.Random(args.seed)
    for step in range(start + 1, args.steps + 1):
        model.train()
        identity = (step % 4 == 0)
        x, y = crop(train_rows, args.seconds, device, rng, clean_identity=identity)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            pred = model(x)
            wave_loss = F.l1_loss(pred, y)
            spectral_loss = stft_l1(pred, y, 1024, 256, window)
            # Open-teacher imitation is the main objective; clean identity examples receive a stronger gate.
            loss = (wave_loss + 0.5 * spectral_loss) * (args.identity_weight if identity else 1.0)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update()
        if step % args.validate_every == 0 or step == args.steps:
            metrics = evaluate(model, val_rows, args.seconds, device, args.seed + 1, args.validation_crops)
            score = metrics["teacher_wave_l1"] + metrics["teacher_stft_l1"] + 2.0 * metrics["clean_identity_l1"]
            row = {"step": step, "train_loss": float(loss.detach()), "validation": metrics,
                   "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                   "monotonic_seconds": time.monotonic()}
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            state = {"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "metrics": metrics, "config": vars(args),
                     "target_teacher": "DeepFilterNet open-source only; not Adobe"}
            torch.save(state, checkpoint)
            if score < best:
                best = score
                torch.save(state, args.output / "best.pt")
            print(json.dumps({"event": "validation", **row, "best_score": best}), flush=True)
    print("Open-teacher pretraining complete", flush=True)


if __name__ == "__main__":
    main()
