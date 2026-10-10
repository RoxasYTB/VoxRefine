#!/usr/bin/env python3
"""Deterministic synthetic-only waveform preflight for H2 direct guard."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
import torch.nn.functional as F

from model_h2_direct_guard import CompactDereverbH2DirectGuard
from train_g_early import tail_loss

HERE = Path(__file__).resolve().parent
SR = 16_000
SEED = 20261011
STEPS = 400


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(index: int, device: torch.device):
    f0 = (109, 131, 157, 179)[index]
    t = np.arange(32_000, dtype=np.float64) / SR
    envelope = np.clip((t - .18) / .05, 0, 1) * np.clip((1.42 - t) / .035, 0, 1)
    modulation = .68 + .20 * np.sin(2 * np.pi * (2.5 + .23 * index) * t)
    voice = np.zeros_like(t)
    for h in range(1, 13):
        voice += np.exp(-((h * f0 - 1050) / 820) ** 2) * np.sin(
            2 * np.pi * h * f0 * t + .13 * h * index) / h
    clean_np = (envelope * modulation * voice * .35).astype(np.float32)
    wet_np = clean_np.copy()
    for k in range(1, 25):
        delay = round((.017 * k + .0015 * ((k + index) % 4)) * SR)
        wet_np[delay:] += np.float32(.32 * .79 ** k) * clean_np[:-delay]
    clean = torch.from_numpy(clean_np)[None].to(device)
    wet = torch.from_numpy(wet_np)[None].to(device)
    pause = 23_200
    frames = clean.unfold(-1, 320, 160)
    rms = frames.square().mean(-1).sqrt()[0]
    peak = rms.max().clamp_min(1e-8)
    active = rms > peak * .02
    weak = active & (rms < peak * .35)
    onset = active & (rms > F.pad(rms[:-1], (1, 0)) * 1.5)
    protect = active | weak | onset
    protect = F.max_pool1d(protect.float()[None, None], 5, stride=1,
                           padding=2)[0, 0] > .5
    centers = torch.arange(rms.numel(), device=device) * 160 + 160
    w1 = (centers >= pause + 150 * 16) & (centers < pause + 300 * 16)
    if not protect.any() or not (w1 & ~protect).any():
        raise RuntimeError("fixture lacks guarded speech or eligible W1")
    return clean, wet, pause, protect, active, weak, onset, w1


def quantile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        raise RuntimeError("empty metric region")
    return float(torch.quantile(values.float(), q))


def run(device: torch.device) -> dict:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    fixtures = [fixture(i, device) for i in range(4)]
    model = CompactDereverbH2DirectGuard(base_channels=16).to(device)
    zero = model(torch.zeros(1, 32_000, device=device))
    if float(zero.abs().max()) >= 1e-7:
        raise RuntimeError("zero-preserving invariant failed")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4,
                                 weight_decay=1e-4)
    initial_gradient_norms = None
    for step in range(STEPS):
        clean, wet, pause, protect, *_ = fixtures[step % len(fixtures)]
        optimizer.zero_grad(set_to_none=True)
        pred, gain = model.forward_with_gain(wet)
        early, count = tail_loss(pred, wet, clean, clean,
            torch.tensor([pause], device=device), ("rir_only",),
            torch.tensor([True], device=device))
        if count != 1:
            raise RuntimeError("synthetic W1 fixture ineligible")
        frame_count = gain.shape[-1]
        protect_tf = F.interpolate(protect.float()[None, None],
            size=frame_count, mode="nearest") > .5
        guard = (((1.0 - gain) / .5).square() * protect_tf).sum() / protect_tf.sum().clamp_min(1)
        total = early + guard
        if not torch.isfinite(total):
            raise FloatingPointError("non-finite objective")
        if step == 0:
            grads = []
            for term in (early, guard):
                terms = torch.autograd.grad(term, tuple(model.parameters()),
                    retain_graph=True, allow_unused=True)
                squares = [g.detach().double().square().sum() for g in terms
                           if g is not None]
                grads.append(float(torch.stack(squares).sum().sqrt()) if squares else 0.0)
            initial_gradient_norms = grads
            if any(not math.isfinite(v) or v <= 0 for v in grads):
                raise RuntimeError("early/guard gradient is zero or non-finite")
        total.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        if not torch.isfinite(norm):
            raise FloatingPointError("non-finite gradient")
        optimizer.step()

    fixture_metrics = []
    with torch.inference_mode():
        for clean, wet, pause, protect, active, weak, onset, w1 in fixtures:
            pred = model(wet)
            a, b = pause + 150 * 16, pause + 300 * 16
            ein = wet[0, a:b].square().mean()
            eout = pred[0, a:b].square().mean()
            delta_w1 = float(10 * torch.log10((ein + 1e-20) / (eout + 1e-20)))
            # Compare reconstructed waveform against the identical wet input in
            # fixed 20 ms frames, then score clean-derived speech regions.
            xframes = wet[0].unfold(-1, 320, 160)
            yframes = pred[0].unfold(-1, 320, 160)
            gain_db = 20 * torch.log10((yframes.square().mean(-1).sqrt() + 1e-8) /
                                       (xframes.square().mean(-1).sqrt() + 1e-8))
            m = min(gain_db.numel(), active.numel())
            gain_db = gain_db[:m]
            active, weak, onset = active[:m], weak[:m], onset[:m]
            active_metrics = {"p50_db": quantile(gain_db[active], .5),
                              "p10_db": quantile(gain_db[active], .1)}
            weak_metrics = {"p10_db": quantile(gain_db[weak], .1),
                           "fraction_below_minus3_db": float((gain_db[weak] < -3).float().mean())}
            onset_metrics = {"p10_db": quantile(gain_db[onset], .1)}
            fixture_metrics.append({"w1_reduction_db": delta_w1,
                "active": active_metrics, "weak": weak_metrics,
                "onset": onset_metrics})
    w1 = [m["w1_reduction_db"] for m in fixture_metrics]
    active_p50 = [m["active"]["p50_db"] for m in fixture_metrics]
    active_p10 = [m["active"]["p10_db"] for m in fixture_metrics]
    weak_p10 = [m["weak"]["p10_db"] for m in fixture_metrics]
    weak_drop = [m["weak"]["fraction_below_minus3_db"] for m in fixture_metrics]
    onset_p10 = [m["onset"]["p10_db"] for m in fixture_metrics]
    passed = (all(v >= 2.5 for v in w1) and float(np.median(w1)) >= 3.0 and
        all(-.25 <= v <= .25 for v in active_p50) and all(v > -.75 for v in active_p10) and
        all(v > -1.0 for v in weak_p10) and all(v <= .03 for v in weak_drop) and
        all(v > -.75 for v in onset_p10) and float(zero.abs().max()) == 0.0)
    return {"name": "H2-direct-guard-synthetic-waveform-v1",
        "overall": "PASS" if passed else "FAIL", "steps": STEPS, "seed": SEED,
        "device": str(device), "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "fixture_metrics": fixture_metrics,
        "initial_component_gradient_norms_early_guard": initial_gradient_norms,
        "zero_output_max_abs": float(zero.abs().max()),
        "thresholds": {"each_w1_db_min": 2.5, "median_w1_db_min": 3.0,
            "active_p50_db_range": [-.25, .25], "active_p10_db_gt": -.75,
            "weak_p10_db_gt": -1.0, "weak_fraction_below_minus3_max": .03,
            "onset_p10_db_gt": -.75},
        "source_sha256": {"model": sha(HERE / "model_h2_direct_guard.py"),
                           "audit": sha(Path(__file__))},
        "training_corpus_read": False, "dev_or_holdout_read": False,
        "test_wav_accessed": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    result = run(device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    if result["overall"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
