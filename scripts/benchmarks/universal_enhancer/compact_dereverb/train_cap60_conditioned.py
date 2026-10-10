#!/usr/bin/env python3
"""Train the frozen-size Cap60-conditioned dereverberator C."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from data import read_librispeech
from model import CompactDereverb16k, parameter_count

SR = 16_000
TRAIN_ROOMS = ("Hotel_SkalskyDvur_ConferenceRoom2", "VUT_FIT_C236", "VUT_FIT_D105",
               "VUT_FIT_E112", "VUT_FIT_L207", "VUT_FIT_Q301")
DEV_ROOMS = ("Hotel_SkalskyDvur_Room112", "VUT_FIT_L212", "VUT_FIT_L227")
DEFAULT_DATA = Path(".tools/compact-dereverb/cap60-conditioned-2026-10-10")
DEFAULT_OUTPUT = DEFAULT_DATA / "training"


def frame_rms(x: torch.Tensor, frame: int = 320, hop: int = 160) -> torch.Tensor:
    if x.shape[-1] < frame:
        return torch.sqrt(x.square().mean(-1, keepdim=True) + 1e-20)
    return torch.sqrt(x.unfold(-1, frame, hop).square().mean(-1) + 1e-20)


def activity_masks(clean: torch.Tensor, target: torch.Tensor):
    """Derive masks from original clean; weak/onset only within active speech."""
    c, t = frame_rms(clean), frame_rms(target)
    peak = c.amax(-1, keepdim=True).clamp_min(1e-8)
    active = c >= torch.maximum(peak * .02, torch.full_like(peak, 1e-5))
    weak = active & (c < peak * .35)
    previous = F.pad(c[..., :-1], (1, 0))
    onset = active & (c > previous * 1.5)
    silent = ~active
    return active, weak, onset, silent


def _stft(x: torch.Tensor, n_fft: int) -> torch.Tensor:
    window = torch.hann_window(n_fft, dtype=x.dtype, device=x.device)
    return torch.stft(x, n_fft=n_fft, hop_length=n_fft // 4, win_length=n_fft,
                      window=window, center=False, return_complex=True)


def cap60_conditioned_loss(pred: torch.Tensor, target: torch.Tensor,
                           clean: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if pred.shape != target.shape or target.shape != clean.shape or pred.ndim != 2:
        raise ValueError("pred, target, clean must have the same [batch, samples] shape")
    active, weak, onset, silent = activity_masks(clean, target)
    # Normalize waveform error by target RMS over clean-active samples.
    active_sample = F.interpolate(active.float().unsqueeze(1), size=clean.shape[-1], mode="nearest").squeeze(1)
    scale = torch.sqrt((target.square() * active_sample).sum(-1) /
                       active_sample.sum(-1).clamp_min(1.0) + 1e-20).mean()
    wave = (pred - target).abs().mean() / (scale + 1e-8)

    complex_terms, log_terms = [], []
    resolutions = (256, 512, 1024, 2048)
    for n_fft in resolutions:
        y_spec, t_spec = _stft(pred, n_fft), _stft(target, n_fft)
        # STFT frames are center=False and hop 1/4 window. Align masks using
        # nearest clean 20/10ms mask; active STFT frames are target-relative.
        active_stft = F.interpolate(active.float().unsqueeze(1), size=t_spec.shape[-1],
                                    mode="nearest").squeeze(1) > .5
        magnitude_target = t_spec.abs()
        denom = magnitude_target.permute(0, 2, 1)[active_stft].mean() if active_stft.any() else magnitude_target.mean()
        complex_delta = (y_spec - t_spec).abs()
        complex_terms.append(complex_delta.mean() / (denom + 1e-8))
        # A Cap60 file may be exactly silent for a low-level or rejected
        # utterance. Keep tau positive so zero/zero bins are excluded instead
        # of producing log(0/0); the floor term handles this region.
        tau = (10 ** -2.5) * denom.detach().clamp_min(1e-8)
        visible = magnitude_target >= tau
        log_delta = 20.0 * torch.log10((y_spec.abs() + tau) /
                                       (magnitude_target + tau)).abs()
        if visible.any():
            log_terms.append(log_delta[visible].mean())
        else:
            log_terms.append(log_delta.mean() * 0.0)
    cstft = torch.stack(complex_terms).mean()
    logmag = torch.stack(log_terms).mean()

    y_rms, t_rms = frame_rms(pred), frame_rms(target)
    eps_r = scale.detach().clamp_min(1e-8) * 1e-5
    gain_db = 20.0 * torch.log10((y_rms + eps_r) / (t_rms + eps_r))
    protect = weak | onset
    drop = ((F.relu((-1.0 - gain_db) / 3.0).square() * protect).sum() /
            protect.sum().clamp_min(1))
    floor_ref = torch.maximum(t_rms, scale.detach().clamp_min(1e-8) * 1e-4)
    floor_db = 20.0 * torch.log10((y_rms + eps_r) / (floor_ref + eps_r))
    floor = (F.relu((floor_db - 3.0) / 6.0).square() * silent).sum() / silent.sum().clamp_min(1)
    parts = {"wave": wave, "complex_stft": cstft, "log_magnitude": logmag,
             "drop": drop, "floor": floor}
    total = wave + .5 * cstft + .25 * logmag + .5 * drop + .5 * floor
    return total, parts


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class PairDataset(Dataset):
    def __init__(self, manifest: Path, enforce_train_mix: bool = True):
        self.root = manifest.parent
        self.rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        if not self.rows:
            raise ValueError(f"empty pair manifest: {manifest}")
        kinds = {k: sum(row["kind"] == k for row in self.rows)
                 for k in ("identity", "rir_only", "fan20", "fan10")}
        total = len(self.rows)
        expected = {"identity": .25, "rir_only": .375, "fan20": .1875, "fan10": .1875}
        if enforce_train_mix:
            for name, ratio in expected.items():
                if abs(kinds[name] / total - ratio) > 1 / total + 1e-9:
                    raise ValueError(f"sampling proportions invalid: {kinds}")
        self.kind_counts = kinds

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        x = torch.from_numpy(np.load(self.root / row["x"], allow_pickle=False).copy())
        t = torch.from_numpy(np.load(self.root / row["t"], allow_pickle=False).copy())
        clean = torch.from_numpy(np.load(self.root / row["c"], allow_pickle=False).copy())
        return x, t, clean, row["kind"]


def make_dev_pairs(rows: list, manifest: dict, device: torch.device, seed: int) -> list:
    """Use deterministic precomputed dev records; fail if absent rather than leak train."""
    # Dev pair precompute is part of the preparation CLI's dedicated dev pass.
    dev_manifest = Path(manifest["dev_pair_manifest"])
    if not dev_manifest.exists():
        raise FileNotFoundError(f"missing speaker/room-disjoint dev cache: {dev_manifest}")
    return PairDataset(dev_manifest)


def validate(model, dev_loader, device) -> float:
    model.eval()
    losses = []
    with torch.inference_mode():
        for x, t, clean, _kind in dev_loader:
            x, t, clean = x.to(device), t.to(device), clean.to(device)
            loss, _ = cap60_conditioned_loss(model(x), t, clean)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite development loss")
            losses.append(float(loss))
    model.train()
    if not losses:
        raise RuntimeError("development manifest has no valid pairs")
    return float(np.mean(losses))


def train(args) -> Path:
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    manifest_path = args.data_dir / "pairs.jsonl"
    dataset = PairDataset(manifest_path)
    metadata = json.loads((args.data_dir / "training-data-manifest.json").read_text())
    if metadata.get("test_wav_accessed") is not False:
        raise RuntimeError("sealed test.wav access invariant missing")
    train_rows = read_librispeech(args.librispeech_root, "train-clean-100")
    dev_rows = read_librispeech(args.librispeech_root, "dev-clean")
    if {s for _, s in train_rows} & {s for _, s in dev_rows}:
        raise RuntimeError("LibriSpeech train/dev speaker overlap")
    train_ids, dev_ids = {s for _, s in train_rows}, {s for _, s in dev_rows}
    pair_train_ids = {str(row["speaker"]) for row in dataset.rows}
    if not pair_train_ids <= train_ids:
        raise RuntimeError("training pair manifest contains a non-train speaker")
    if len(set(metadata["train_rir_rooms"]) & set(metadata["dev_rir_rooms"])):
        raise RuntimeError("BUT train/dev room leakage")
    if any(row["kind"] != "identity" and row.get("rir_room") in DEV_ROOMS
           for row in dataset.rows):
        raise RuntimeError("development BUT room found in training pairs")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda",
                        drop_last=True)
    dev_dataset = PairDataset(Path(metadata["dev_pair_manifest"]), enforce_train_mix=False)
    pair_dev_ids = {str(row["speaker"]) for row in dev_dataset.rows}
    if not pair_dev_ids <= dev_ids:
        raise RuntimeError("development pair manifest contains a non-dev speaker")
    if pair_train_ids & pair_dev_ids:
        raise RuntimeError("cached train/dev speakers overlap")
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=device.type == "cuda")
    model = CompactDereverb16k(base_channels=16).to(device)
    count = parameter_count(model)
    if count != 555_922:
        raise RuntimeError(f"expected frozen 555922 parameter architecture, got {count}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.output_dir / "checkpoints"; ckpt_dir.mkdir(exist_ok=True)
    manifest = {"seed": args.seed, "steps": args.steps, "sample_rate": SR,
                "parameter_count": count, "train_pair_manifest_sha256": sha256(manifest_path),
                "dev_pair_manifest_sha256": sha256(Path(metadata["dev_pair_manifest"])),
                "train_kind_counts": dataset.kind_counts, "train_rir_rooms": TRAIN_ROOMS,
                "dev_rir_rooms": DEV_ROOMS, "loss": "Cap60-conditioned GPT Web frozen recipe",
                "optimizer": "AdamW(lr=2e-4, weight_decay=1e-4)", "batch_size": args.batch_size,
                "device": str(device), "torch": torch.__version__}
    (args.output_dir / "training-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    batches = iter(loader)
    log_path = args.output_dir / "training.jsonl"
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        try:
            x, t, clean, kinds = next(batches)
        except StopIteration:
            batches = iter(loader)
            x, t, clean, kinds = next(batches)
        x, t, clean = (v.to(device, non_blocking=True) for v in (x, t, clean))
        optimizer.zero_grad(set_to_none=True)
        loss, parts = cap60_conditioned_loss(model(x), t, clean)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite gradient at step {step}")
        optimizer.step()
        if step == 1 or step % args.log_every == 0:
            row = {"step": step, "loss": float(loss.detach()), "grad_norm": float(grad_norm),
                   "components": {k: float(v.detach()) for k, v in parts.items()},
                   "kind": list(kinds), "elapsed_s": time.perf_counter() - started,
                   "peak_vram_mib": torch.cuda.max_memory_allocated(device) / 2**20
                       if device.type == "cuda" else None}
            with log_path.open("a") as f: f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        if step % args.validate_every == 0 or step == args.steps:
            dev_loss = validate(model, dev_loader, device)
            state = {"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "dev_loss": dev_loss, "base_channels": 16, "seed": args.seed,
                     "data_manifest_sha256": sha256(manifest_path)}
            torch.save(state, ckpt_dir / f"step-{step:06d}.pt")
            print(json.dumps({"step": step, "dev_loss": dev_loss,
                              "checkpoint_policy": "fixed final step; dev is monitoring only"}), flush=True)
    return ckpt_dir / f"step-{args.steps:06d}.pt"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    p.add_argument("--librispeech-root", type=Path,
                   default=Path(".tools/compact-dereverb/data/LibriSpeech"))
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--seed", type=int, default=20261010)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--validate-every", type=int, default=500)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    print(train(p.parse_args()))


if __name__ == "__main__":
    main()
