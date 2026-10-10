#!/usr/bin/env python3
"""Synthetic-only deterministic feasibility audit for H-shield."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model_h_shield import CompactDereverbHShield  # noqa: E402
from train_cap60_conditioned import cap60_conditioned_loss  # noqa: E402
from train_g_early import tail_loss  # noqa: E402

SR = 16_000
FIXTURE_SEED = 20261010
STEPS = 400


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(index: int, device: torch.device):
    f0 = (103, 127, 149, 181)[index]
    t = np.arange(32_000, dtype=np.float64) / SR
    envelope = np.clip((t - .18) / .05, 0, 1) * np.clip((1.42 - t) / .035, 0, 1)
    modulation = .68 + .20 * np.sin(2 * np.pi * (2.7 + .2 * index) * t)
    voice = np.zeros_like(t)
    for h in range(1, 13):
        voice += np.exp(-((h * f0 - 1100) / 850) ** 2) * np.sin(
            2 * np.pi * h * f0 * t + .17 * h * index) / h
    clean_np = (envelope * modulation * voice * .35).astype(np.float32)
    wet_np = clean_np.copy()
    for k in range(1, 24):
        delay = round((.018 * k + .002 * ((k + index) % 3)) * SR)
        wet_np[delay:] += np.float32(.34 * .78 ** k) * clean_np[:-delay]
    clean = torch.from_numpy(clean_np)[None].to(device)
    wet = torch.from_numpy(wet_np)[None].to(device)
    pause = 23_200  # synthetic dry source is quiet after its early speech burst
    active_rms = clean.unfold(-1, 320, 160).square().mean(-1).sqrt()[0]
    peak = active_rms.max().clamp_min(1e-8)
    active = active_rms > peak * .02
    weak = active & (active_rms < peak * .35)
    previous = F.pad(active_rms[:-1], (1, 0))
    onset = active & (active_rms > previous * 1.5)
    protected_10ms = active | weak | onset
    protected_10ms = F.max_pool1d(protected_10ms.float()[None, None], 3,
        stride=1, padding=1)[0, 0] > 0
    # Model STFT hop is 8 ms; nearest mapping is deterministic and shared by
    # the confidence and guard terms.
    frame_count = (clean.shape[-1] + 64) // 128 + 1
    protected = F.interpolate(protected_10ms.float()[None, None], size=frame_count,
        mode="nearest")[0, 0] > .5
    w1 = torch.zeros(frame_count, dtype=torch.bool, device=device)
    centers = torch.arange(frame_count, device=device) * 128
    w1 = (centers >= pause + 150 * 16) & (centers < pause + 300 * 16)
    w1 &= ~protected
    return clean, wet, pause, protected, w1


def run(device: torch.device) -> dict:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(FIXTURE_SEED)
    np.random.seed(FIXTURE_SEED)
    fixtures = [fixture(i, device) for i in range(4)]
    model = CompactDereverbHShield(base_channels=16).to(device)
    if any(float(p.detach().abs().max()) == float("inf") for p in model.parameters()):
        raise RuntimeError("non-finite initialization")
    zeros = model(torch.zeros(1, 32_000, device=device))
    if float(zeros.abs().max()) >= 1e-7:
        raise RuntimeError("zero-preserving invariant failed")
    # Check the architecture's phase-free semantics and hard gain bounds.
    with torch.inference_mode():
        _, gain, confidence = model.forward_with_mask(fixtures[0][1])
    if not torch.all((gain > .5) & (gain <= 1.0)):
        raise RuntimeError("H-shield gain left its bounded interval")
    if not torch.isfinite(confidence).all():
        raise RuntimeError("confidence contains non-finite values")

    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    observations = []
    for step in range(STEPS):
        clean, wet, pause, protected, w1 = fixtures[step % len(fixtures)]
        optimizer.zero_grad(set_to_none=True)
        pred, gain, conf = model.forward_with_mask(wet)
        base, _ = cap60_conditioned_loss(pred, clean, clean)
        early, count = tail_loss(pred, wet, clean, clean,
            torch.tensor([pause], device=device), ("rir_only",),
            torch.tensor([True], device=device))
        if count != 1:
            raise RuntimeError("synthetic W1 fixture is not eligible")
        # Align clean-derived masks to mask-head frames. Masks are labels only;
        # no validation corpus or model output is used to create them.
        protected_tf = protected[:gain.shape[-1]][None, None, :].expand_as(gain)
        w1_tf = w1[:gain.shape[-1]][None, None, :].expand_as(gain)
        labelled = protected_tf | w1_tf
        confidence_target = w1_tf.to(conf.dtype)
        conf_loss = ((conf - confidence_target).square() * labelled).sum() / labelled.sum().clamp_min(1)
        guard_loss = (((1.0 - gain) / .5).square() * protected_tf).sum() / protected_tf.sum().clamp_min(1)
        total = base + early + conf_loss + guard_loss
        if not torch.isfinite(total):
            raise FloatingPointError("non-finite H-shield synthetic objective")
        if step == 0:
            params = tuple(model.parameters())
            term_norms = []
            for term in (base, early, conf_loss, guard_loss):
                term_grad = torch.autograd.grad(term, params, retain_graph=True,
                                                allow_unused=True)
                squares = [g.detach().double().square().sum() for g in term_grad if g is not None]
                term_norms.append(float(torch.stack(squares).sum().sqrt()) if squares else 0.0)
            observations.append({"initial_component_gradient_norms": term_norms})
            if any(not math.isfinite(n) or n <= 0 for n in term_norms):
                raise RuntimeError("a synthetic component gradient is zero or non-finite")
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("non-finite H-shield gradient")
        optimizer.step()
    # The same four fixtures each receive exactly 100 updates.
    reductions, protect_medians, protect_p10, conf_protect, conf_w1 = [], [], [], [], []
    with torch.inference_mode():
        for clean, wet, pause, protected, w1 in fixtures:
            pred, gain, conf = model.forward_with_mask(wet)
            a, b = pause + 150 * 16, pause + 300 * 16
            ein, eout = wet[0, a:b].square().mean(), pred[0, a:b].square().mean()
            reductions.append(float(10 * torch.log10((ein + 1e-20) / (eout + 1e-20))))
            pm = protected[:gain.shape[-1]][None, None, :].expand_as(gain)
            wm = w1[:gain.shape[-1]][None, None, :].expand_as(gain)
            if not pm.any() or not wm.any():
                raise RuntimeError("synthetic fixture lacks protected speech or W1 labels")
            protect_medians.append(float(gain[pm].median()))
            protect_p10.append(float(torch.quantile(gain[pm], .10)))
            conf_protect.append(float(conf[pm].median()))
            conf_w1.append(float(conf[wm].median()))
    metrics = {"w1_reduction_db_per_fixture": reductions,
        "protected_gain_median_per_fixture": protect_medians,
        "protected_gain_p10_per_fixture": protect_p10,
        "protected_confidence_median_per_fixture": conf_protect,
        "w1_confidence_median_per_fixture": conf_w1,
        "zero_output_max_abs": float(zeros.abs().max()),
        "initial_component_gradient_norms": observations[0]["initial_component_gradient_norms"],
        "fit_steps": STEPS, "seed": FIXTURE_SEED}
    passed = (all(v >= 2.5 for v in reductions) and
        all(v >= .995 for v in protect_medians) and
        all(v >= .98 for v in protect_p10) and
        all(v <= .05 for v in conf_protect) and
        all(v >= .8 for v in conf_w1) and
        metrics["zero_output_max_abs"] < 1e-7)
    return {"name": "H-shield-synthetic-mechanics-v1", "overall": "PASS" if passed else "FAIL",
        "metrics": metrics, "source_sha256": {
            "model": sha(HERE / "model_h_shield.py"),
            "audit": sha(Path(__file__)),
            "g2_model": sha(HERE / "model_g_early_v2.py")},
        "device": str(device), "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(0)
            if device.type == "cuda" else None,
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
