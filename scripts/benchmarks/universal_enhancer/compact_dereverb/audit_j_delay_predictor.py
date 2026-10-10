#!/usr/bin/env python3
"""Frozen synthetic-only train/eval preflight for J-delay-predictor."""
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

from model_j_delay_predictor import JDelayPredictor16k

HERE = Path(__file__).resolve().parent
SR, N, PAUSE = 16_000, 32_000, 23_200
TRAIN_SEED, EVAL_SEED, RUN_SEED, STEPS = 510_2026, 520_2026, 530_2026, 800


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(seed: int, index: int, device: torch.device):
    rng = np.random.default_rng(seed + index * 7919)
    t = np.arange(N, dtype=np.float64) / SR
    f0 = float(rng.uniform(88, 218))
    vibrato = rng.uniform(1.0, 4.0) * np.sin(2 * np.pi * rng.uniform(3.0, 6.0) * t)
    phase = 2 * np.pi * np.cumsum(f0 + vibrato) / SR
    # Four distinct syllabic regions, with silence gaps and soft word endings.
    starts = np.array([.15, .47, .81, 1.13]) + rng.uniform(-.035, .035, 4)
    widths = rng.uniform(.18, .27, 4)
    env = np.zeros_like(t)
    for start, width in zip(starts, widths):
        u = (t - start) / width
        pulse = np.where((u >= 0) & (u < 1),
                         np.sin(np.pi * np.clip(u, 0, 1)) ** .62, 0.0)
        env = np.maximum(env, pulse)
    env *= .75 + .25 * np.sin(2 * np.pi * rng.uniform(3.5, 7.0) * t + rng.uniform(0, 6))
    voice = np.zeros_like(t)
    for harmonic in range(1, 20):
        hz = f0 * harmonic
        formants = (np.exp(-((hz - rng.uniform(650, 850)) / 500) ** 2) +
                    .65 * np.exp(-((hz - rng.uniform(1_150, 1_650)) / 700) ** 2) +
                    .3 * np.exp(-((hz - rng.uniform(2_400, 3_300)) / 950) ** 2))
        voice += formants / harmonic * np.sin(harmonic * phase + rng.uniform(-.2, .2))
    clean_np = (env * voice * (.15 / max(np.max(np.abs(voice)), 1e-6))).astype(np.float32)

    # Direct path, a few early reflections, then a diffuse decaying tail.
    h = np.zeros(round(.48 * SR), np.float64)
    h[0] = 1.0
    for _ in range(5):
        delay = int(rng.uniform(.035, .125) * SR)
        h[delay] += rng.choice([-1.0, 1.0]) * rng.uniform(.05, .18)
    decay = rng.uniform(.12, .24)
    for delay in range(round(.12 * SR), h.size, 96):
        h[delay] += rng.normal() * .12 * np.exp(-(delay / SR - .12) / decay)
    wet_np = np.convolve(clean_np, h, mode="full")[:N].astype(np.float32)
    clean = torch.from_numpy(clean_np)[None].to(device)
    wet = torch.from_numpy(wet_np)[None].to(device)

    frames = clean[0].unfold(-1, 320, 160)
    rms = frames.square().mean(-1).sqrt()
    peak = rms.max().clamp_min(1e-8)
    active = rms > torch.maximum(peak * .02, rms.new_tensor(1e-5))
    weak = active & (rms < peak * .35)
    prev = F.pad(rms[:-1], (1, 0))
    onset = active & (rms > prev * 1.5)
    if min(int(active.sum()), int(weak.sum()), int(onset.sum())) == 0:
        raise RuntimeError("synthetic source lacks active/weak/onset coverage")
    return {"clean": clean, "wet": wet, "active": active, "weak": weak,
        "onset": onset, "pause": PAUSE, "id": f"{seed}-{index}"}


def fixtures(device):
    train = [_fixture(TRAIN_SEED, i, device) for i in range(8)]
    ev = [_fixture(EVAL_SEED, i, device) for i in range(4)]
    if {f["id"] for f in train} & {f["id"] for f in ev}:
        raise RuntimeError("train/eval source leakage")
    return train, ev


def _sample_mask(frame_mask: torch.Tensor, length: int) -> torch.Tensor:
    # Nearest centre assignment, matching the frozen 20 ms / 10 ms grid.
    return F.interpolate(frame_mask.float()[None, None], size=length,
                         mode="nearest")[0, 0] > .5


def _identity_loss(y, x, frame_mask):
    mask = _sample_mask(frame_mask, x.shape[-1])[None]
    if not bool(mask.any()):
        raise RuntimeError("empty identity region")
    numerator = ((y - x).square() * mask).sum() / mask.sum().clamp_min(1)
    denominator = (x.square() * mask).sum() / mask.sum().clamp_min(1)
    return numerator / (denominator + 1e-8)


def _w1_loss(y, x, pause):
    start, end = pause + 150 * 16, pause + 300 * 16
    ex = x[..., start:end].square().mean()
    ey = y[..., start:end].square().mean()
    delta = 10 * torch.log10((ex + 1e-20) / (ey + 1e-20))
    return (F.relu(3.0 - delta) / 3.0).square()


def train_model(train, device):
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(RUN_SEED)
    np.random.seed(RUN_SEED)
    torch.cuda.manual_seed_all(RUN_SEED) if device.type == "cuda" else None
    model = JDelayPredictor16k().to(device)
    zero = model(torch.zeros(1, N, device=device))
    if not torch.equal(zero, torch.zeros_like(zero)):
        raise RuntimeError("zero invariant not exact")
    # The zero residual head makes the initial map exactly identity.
    first, _ = model.forward_with_residual(train[0]["wet"])
    if not torch.equal(first, train[0]["wet"]):
        raise RuntimeError("initial model is not exact identity")
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    generator = torch.Generator(device="cpu").manual_seed(RUN_SEED + 1)
    order = []
    while len(order) < STEPS:
        order.extend(torch.randperm(len(train), generator=generator).tolist())
    initial_gradients = None
    for step, idx in enumerate(order[:STEPS]):
        f = train[idx]
        opt.zero_grad(set_to_none=True)
        y, _ = model.forward_with_residual(f["wet"])
        l_w1 = _w1_loss(y, f["wet"], f["pause"])
        l_active = _identity_loss(y, f["wet"], f["active"])
        l_weak = _identity_loss(y, f["wet"], f["weak"])
        l_onset = _identity_loss(y, f["wet"], f["onset"])
        loss = l_w1 + l_active + l_weak + l_onset
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite objective")
        if step == 0:
            component_norms = []
            for component in (l_w1, l_active, l_weak, l_onset):
                grads = torch.autograd.grad(component, tuple(model.parameters()),
                    retain_graph=True, allow_unused=True)
                vals = [g.detach().double().square().sum() for g in grads if g is not None]
                component_norms.append(float(torch.stack(vals).sum().sqrt()) if vals else 0.)
            initial_gradients = component_norms
            if any(not math.isfinite(g) for g in component_norms):
                raise RuntimeError(f"loss component has nonfinite gradient: {component_norms}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        if not torch.isfinite(norm):
            raise FloatingPointError("non-finite gradient")
        opt.step()
    return model, initial_gradients


def _q(t, q):
    if t.numel() == 0:
        raise RuntimeError("empty metric region")
    return float(torch.quantile(t.float(), q))


def evaluate(model, ev):
    metrics = []
    with torch.inference_mode():
        for f in ev:
            y, r = model.forward_with_residual(f["wet"])
            x = f["wet"]
            pause = f["pause"]
            w1 = slice(pause + 150 * 16, pause + 300 * 16)
            w2 = slice(pause + 300 * 16, pause + 450 * 16)
            def db(a, b):
                return float(20 * torch.log10((a.square().mean().sqrt() + 1e-10) /
                                               (b.square().mean().sqrt() + 1e-10)))
            xframes, yframes = x[0].unfold(-1, 320, 160), y[0].unfold(-1, 320, 160)
            frame_db = 20 * torch.log10((yframes.square().mean(-1).sqrt() + 1e-8) /
                                         (xframes.square().mean(-1).sqrt() + 1e-8))
            frame_db = frame_db[:f["active"].numel()]
            active, weak, onset = (f[k] for k in ("active", "weak", "onset"))
            residual_frames = r[0].unfold(-1, 320, 160)
            wet_frames = x[0].unfold(-1, 320, 160)
            residual_rms = residual_frames.square().mean(-1).sqrt()
            wet_rms = wet_frames.square().mean(-1).sqrt()
            w1_frames = (torch.arange(frame_db.numel(), device=x.device) * 160 + 160)
            w1_frames = (w1_frames >= w1.start) & (w1_frames < w1.stop)
            def res_stats(mask):
                if not bool(mask.any()): return {"rms": 0., "correlation": 0.}
                rr, xx = residual_rms[:mask.numel()][mask], wet_rms[:mask.numel()][mask]
                corr = torch.corrcoef(torch.stack((rr, xx)))[0, 1] if rr.numel() > 1 else rr.new_zeros(())
                return {"rms": float(rr.square().mean().sqrt()),
                        "correlation": float(torch.nan_to_num(corr))}
            metrics.append({
                "w1_reduction_db": -db(y[..., w1], x[..., w1]),
                "w2_reduction_db": -db(y[..., w2], x[..., w2]),
                "active_p50_db": _q(frame_db[active], .5),
                "active_p10_db": _q(frame_db[active], .1),
                "weak_p10_db": _q(frame_db[weak], .1),
                "weak_below_minus3_fraction": float((frame_db[weak] < -3).float().mean()),
                "weak_below_minus6_fraction": float((frame_db[weak] < -6).float().mean()),
                "onset_p10_db": _q(frame_db[onset], .1),
                "residual_diagnostics": {
                    "active": res_stats(active), "weak": res_stats(weak),
                    "onset": res_stats(onset), "w1": res_stats(w1_frames)}})
    return metrics


def run(device):
    train, ev = fixtures(device)
    model, grad_norms = train_model(train, device)
    metrics = evaluate(model, ev)
    w1 = [m["w1_reduction_db"] for m in metrics]
    w2 = [m["w2_reduction_db"] for m in metrics]
    passed = (all(x >= 2.5 for x in w1) and float(np.median(w1)) >= 3.0 and
        all(-.25 <= m["active_p50_db"] <= .25 for m in metrics) and
        all(m["active_p10_db"] > -.75 for m in metrics) and
        all(m["weak_p10_db"] > -1.0 for m in metrics) and
        all(m["weak_below_minus3_fraction"] <= .03 for m in metrics) and
        all(m["weak_below_minus6_fraction"] <= .01 for m in metrics) and
        all(m["onset_p10_db"] > -.75 for m in metrics) and
        all(x >= -.5 for x in w2) and float(np.median(w2)) >= 0.)
    zero = model(torch.zeros(1, N, device=device))
    finite = all(math.isfinite(v) for m in metrics for v in (
        m["w1_reduction_db"], m["w2_reduction_db"], m["active_p50_db"],
        m["active_p10_db"], m["weak_p10_db"], m["onset_p10_db"]))
    return {"experiment": "J-delay-predictor-synthetic-preflight-v1",
        "overall": "PASS" if passed and finite and torch.count_nonzero(zero).item() == 0 else "FAIL",
        "device": str(device), "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "seed": RUN_SEED, "steps": STEPS, "train_fixture_count": len(train),
        "eval_fixture_count": len(ev), "train_eval_source_ids_disjoint": True,
        "architecture": {"n_fft": 512, "hop_length": 128,
            "delay_frames": 4, "delay_ms": 1000*4*128/SR,
            "history_context_ms": 240, "gru_layers": 1, "hidden_size": 128,
            "output_head_zero_initialized": True},
        "loss_weights": {"w1": 1., "active": 1., "weak": 1., "onset": 1.},
        "initial_gradient_norms": grad_norms, "fixture_metrics": metrics,
        "thresholds": {"each_w1_db_min": 2.5, "median_w1_db_min": 3.,
            "active_p50_db_range": [-.25, .25], "active_p10_db_gt": -.75,
            "weak_p10_db_gt": -1., "weak_below_minus3_max": .03,
            "weak_below_minus6_max": .01, "onset_p10_db_gt": -.75,
            "each_w2_db_min": -.5, "median_w2_db_min": 0.},
        "zero_output_exact": torch.count_nonzero(zero).item() == 0,
        "nonfinite_metric_count": 0 if finite else 1,
        "corpus_read": False, "dev_or_holdout_read": False,
        "test_wav_accessed": False,
        "source_sha256": {"model": sha(HERE / "model_j_delay_predictor.py"),
                          "audit": sha(Path(__file__))}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
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
