"""Train the isolated 16 kHz compact dereverberation proof."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data import CROP_SAMPLES, RandomPairedSpeech, make_validation_pair, read_librispeech
from model import CompactDereverb16k, parameter_count


def frame_rms(audio: torch.Tensor, frame_length: int = 320, hop: int = 160) -> torch.Tensor:
    return torch.sqrt(audio.unfold(-1, frame_length, hop).square().mean(-1) + 1e-12)


def protection_masks(clean: torch.Tensor, late: torch.Tensor):
    target, tail = frame_rms(clean), frame_rms(late)
    peak = target.amax(-1, keepdim=True).clamp_min(1e-6)
    active = target > peak * 0.025
    weak = active & (target < peak * 0.35)
    previous = F.pad(target[:, :-1], (1, 0))
    onset = active & (target > previous * 1.5)
    late_peak = tail.amax(-1, keepdim=True).clamp_min(1e-6)
    late_quiet = (target < peak * 0.18) & (tail > late_peak * 0.04)
    return weak.to(clean.dtype), onset.to(clean.dtype), late_quiet.to(clean.dtype)


def _mrstft_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    terms = []
    for n_fft, hop in ((256, 64), (512, 128), (1024, 256)):
        window = torch.hann_window(n_fft, device=pred.device, dtype=pred.dtype)
        p = torch.stft(pred, n_fft, hop, n_fft, window, center=True, return_complex=True).abs()
        t = torch.stft(target, n_fft, hop, n_fft, window, center=True, return_complex=True).abs()
        terms.append(F.l1_loss(torch.log1p(20 * p), torch.log1p(20 * t)))
    return torch.stack(terms).mean()


def loss_components(pred, clean, weak_mask, early_mask, late_mask):
    if pred.shape != clean.shape:
        raise ValueError("pred and clean must have identical shapes")
    p_rms, c_rms = frame_rms(pred), frame_rms(clean)
    if any(mask.shape != c_rms.shape for mask in (weak_mask, early_mask, late_mask)):
        raise ValueError("frame masks must match clean-speech RMS frames")
    deficit = F.relu(c_rms * 0.89 - p_rms)
    excess = F.relu(p_rms - c_rms * 1.12)

    def masked_mean(value, mask):
        return (value * mask).sum() / mask.sum().clamp_min(1.0)

    return {
        "wave": F.l1_loss(pred, clean),
        "mrstft": _mrstft_loss(pred, clean),
        "weak_under": masked_mean(deficit, weak_mask),
        # One-sided weak-frame over-gain penalty. It does not push quiet
        # estimates down; only energy above the clean-target band is penalized.
        "weak_over": masked_mean(excess, weak_mask),
        "early_under": masked_mean(deficit, early_mask),
        "late_over": masked_mean(excess, late_mask),
    }


def total_loss(parts, weak_over_weight: float = 0.0):
    return (parts["wave"] + .05 * parts["mrstft"] + .50 * parts["weak_under"]
            + .50 * parts["early_under"] + .35 * parts["late_over"]
            + weak_over_weight * parts["weak_over"])


def _read_crop(path: Path, rng: np.random.Generator) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if sr != 16_000:
        raise ValueError(f"expected 16 kHz, got {sr}: {path}")
    if audio.ndim == 2:
        audio = audio.mean(axis=1, dtype=np.float32)
    if len(audio) < CROP_SAMPLES:
        audio = np.tile(audio, int(np.ceil(CROP_SAMPLES / len(audio))))
    start = int(rng.integers(0, len(audio) - CROP_SAMPLES + 1))
    crop = audio[start:start + CROP_SAMPLES].copy()
    fade = int(0.005 * 16_000)
    ramp = np.linspace(0.0, 1.0, fade, endpoint=False, dtype=np.float32)
    crop[:fade] *= ramp
    crop[-fade:] *= ramp[::-1]
    return crop


def validation_loss(model: nn.Module, rows, seed: int, device: torch.device,
                    weak_over_weight: float = 0.0) -> float:
    rng = np.random.default_rng(seed)
    scores = []
    model.eval()
    with torch.inference_mode():
        for path, _speaker in rows[:12]:
            clean_np = _read_crop(path, rng)
            pair = make_validation_pair(clean_np, {
                "seed": int(rng.integers(0, 2**32)),
                "t60_s": float(rng.uniform(.25, 1.20)),
                "direct_to_reverb_db": float(rng.uniform(-6, 18)),
            })
            clean = torch.from_numpy(pair["clean"]).unsqueeze(0).to(device)
            mix = torch.from_numpy(pair["reverberant"]).unsqueeze(0).to(device)
            late = torch.from_numpy(pair["late"]).unsqueeze(0).to(device)
            parts = loss_components(model(mix), clean, *protection_masks(clean, late))
            scores.append(float(total_loss(parts, weak_over_weight).item()))
            # Matched dry control: the model must learn not to invent or suppress
            # content when the input is already dry.
            dry_parts = loss_components(model(clean), clean, *protection_masks(clean, torch.zeros_like(clean)))
            scores.append(float(total_loss(dry_parts, weak_over_weight).item()))
    model.train()
    return float(np.mean(scores))


def train(args: argparse.Namespace) -> Path:
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    train_rows = read_librispeech(args.data_root, "train-clean-100")
    dev_rows = read_librispeech(args.data_root, "dev-clean")
    train_speakers = {speaker for _, speaker in train_rows}
    dev_speakers = {speaker for _, speaker in dev_rows}
    overlap = train_speakers & dev_speakers
    if overlap: raise RuntimeError(f"train/dev speaker leakage: {sorted(overlap)[:5]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.output_dir / "checkpoints"; ckpt_dir.mkdir(exist_ok=True)
    loader = DataLoader(RandomPairedSpeech(train_rows, args.seed), batch_size=args.batch_size,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    batches = iter(loader)
    model = CompactDereverb16k(base_channels=args.base_channels).to(device)
    resume_state = None
    if args.resume is not None:
        resume_state = torch.load(args.resume, map_location="cpu", weights_only=False)
        if int(resume_state["base_channels"]) != args.base_channels:
            raise ValueError("resume checkpoint base_channels does not match requested model")
        model.load_state_dict(resume_state["model"])
    count = parameter_count(model)
    if not 500_000 <= count <= 2_000_000:
        raise RuntimeError(f"parameter count outside proof budget: {count}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer"])
    manifest = {"seed": args.seed, "device": str(device), "torch": torch.__version__,
        "sample_rate": 16000, "crop_samples": CROP_SAMPLES, "parameter_count": count,
        "train_examples": len(train_rows), "train_speakers": len(train_speakers),
        "dev_examples": len(dev_rows), "dev_speakers": len(dev_speakers),
        "train_dev_overlap": sorted(overlap), "steps_requested": args.steps,
        "dry_control_probability": 0.20, "weak_over_weight": args.weak_over_weight}
    (args.output_dir / "training-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log_path = args.output_dir / "training.jsonl"; best_dev = float("inf")
    started = time.perf_counter(); model.train()
    for step in range(1, args.steps + 1):
        batch = next(batches)
        mix = batch["mixture"].to(device, non_blocking=True)
        clean = batch["clean"].to(device, non_blocking=True)
        late = batch["late"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        parts = loss_components(model(mix), clean, *protection_masks(clean, late))
        loss = total_loss(parts, weak_over_weight=args.weak_over_weight)
        if not torch.isfinite(loss): raise FloatingPointError(f"non-finite loss at {step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        if not torch.isfinite(grad_norm): raise FloatingPointError(f"non-finite gradient at {step}")
        optimizer.step()
        if step % args.log_every == 0 or step == 1:
            row = {"step": step, "loss": float(loss.detach()), "grad_norm": float(grad_norm),
                   "elapsed_s": time.perf_counter() - started,
                   "components": {k: float(v.detach()) for k, v in parts.items()}}
            if device.type == "cuda":
                row["peak_allocated_vram_mib"] = torch.cuda.max_memory_allocated(device) / 1024**2
                row["peak_reserved_vram_mib"] = torch.cuda.max_memory_reserved(device) / 1024**2
            with log_path.open("a") as f: f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        if step % args.validate_every == 0 or step == args.steps:
            dev_loss = validation_loss(model, dev_rows, args.seed + 41_009, device,
                                       weak_over_weight=args.weak_over_weight)
            state = {"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "dev_loss": dev_loss, "base_channels": args.base_channels, "seed": args.seed}
            torch.save(state, ckpt_dir / f"step-{step:06d}.pt")
            if dev_loss < best_dev:
                best_dev = dev_loss
                torch.save(state, ckpt_dir / "best-dev.pt")
            print(json.dumps({"step": step, "dev_loss": dev_loss, "best_dev": best_dev}), flush=True)
    return ckpt_dir / "best-dev.pt"


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path(".tools/compact-dereverb/training"))
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=20261009)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--base-channels", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--weak-over-weight", type=float, default=0.0,
                   help="Weight for the one-sided weak-frame over-gain penalty.")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--validate-every", type=int, default=200)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--resume", type=Path, default=None)
    return p


if __name__ == "__main__":
    train(build_parser().parse_args())
