#!/usr/bin/env python3
"""Synthetic-only gradient and overfit gate for G-early-v2.

This script deliberately reads no corpus, split, checkpoint, or G-early-v1
artifact. Passing this mechanism gate is required before preparing real v2 data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
from model_g_early_v2 import (  # noqa: E402
    CompactAttenuationOnlyDereverbG2, initial_gain_diagnostics,
)
from train_cap60_attenuation_only import quiet_loss  # noqa: E402
from train_cap60_conditioned import cap60_conditioned_loss  # noqa: E402

DEFAULT_OUTPUT = ROOT / ".tools/compact-dereverb/g-early-v2-mechanics-2026-10-10"
SAMPLE_RATE = 16_000
PAUSE_SAMPLE = 22_400  # 1.4 s
W1_START = PAUSE_SAMPLE + 150 * 16
W1_END = PAUSE_SAMPLE + 300 * 16
FIXTURE_COUNT = 4
OVERFIT_STEPS = 400
SEED = 20261010


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_fixtures() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create four deterministic harmonic speech surrogates and room tails."""
    length = 48_000
    time_axis = np.arange(length, dtype=np.float64) / SAMPLE_RATE
    window = np.clip((time_axis - 0.18) / 0.05, 0.0, 1.0)
    window *= np.clip((1.42 - time_axis) / 0.035, 0.0, 1.0)
    dry_rows, wet_rows = [], []
    for index in range(FIXTURE_COUNT):
        f0 = (103.0, 127.0, 149.0, 181.0)[index]
        modulation = 0.68 + 0.20 * np.sin(2 * np.pi * (2.7 + 0.2 * index) * time_axis)
        voiced = np.zeros_like(time_axis)
        for harmonic in range(1, 13):
            formant_weight = math.exp(-((harmonic * f0 - 1_100) / 850) ** 2)
            voiced += (formant_weight / harmonic) * np.sin(
                2 * np.pi * harmonic * f0 * time_axis + 0.17 * harmonic * index)
        dry = (0.12 * window * modulation * voiced).astype(np.float32)
        # A sparse, exponentially decaying echo train approximates early room
        # reflections while keeping this fixture deterministic and self-contained.
        tail = np.zeros_like(dry)
        for tap in range(1, 24):
            delay = int((0.018 * tap + 0.002 * ((tap + index) % 3)) * SAMPLE_RATE)
            if delay < length:
                tail[delay:] += (0.34 * (0.78 ** tap) * np.roll(dry, delay)[:-delay])
        wet = dry + tail.astype(np.float32)
        dry_rows.append(dry)
        wet_rows.append(wet.astype(np.float32))
    return (torch.from_numpy(np.stack(wet_rows)),
            torch.from_numpy(np.stack(dry_rows)),
            torch.from_numpy(np.stack(dry_rows.copy())))


def early_loss(pred: torch.Tensor, x: torch.Tensor, clean: torch.Tensor,
               tail_reference: torch.Tensor, pause: int = PAUSE_SAMPLE) -> torch.Tensor:
    frames = clean.unfold(-1, 320, 160)
    rms = (frames.square().mean(-1) + 1e-24).sqrt()
    active = rms >= torch.maximum(rms.max() * 0.02, rms.new_tensor(1e-5))
    centers = torch.arange(active.numel(), device=clean.device) * 160 + 160
    reference_frames = active & (centers >= pause - 8_000) & (centers < pause - 800)
    if int(reference_frames.sum()) * 160 < 3_200:
        raise RuntimeError("synthetic fixture has insufficient active reference duration")
    reference_samples = F.interpolate(reference_frames.float()[None, None, :],
        size=clean.numel(), mode="nearest")[0, 0] > 0.5
    eref = tail_reference[reference_samples].square().mean()
    start, end = W1_START, W1_END
    ex = x[start:end].square().mean()
    ey = pred[start:end].square().mean()
    delta_db = 10.0 * torch.log10((ex + eref * 1e-12) / (ey + eref * 1e-12))
    return (F.relu(3.0 - delta_db) / 3.0).square()


def base_loss(pred: torch.Tensor, target: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    speech, parts = cap60_conditioned_loss(pred, target, clean)
    return speech - 0.5 * parts["floor"] + 0.5 * quiet_loss(pred, target, clean)


def grad_vector(grads, parameters: list[torch.nn.Parameter], group: str) -> torch.Tensor:
    values = []
    for parameter, grad in zip(parameters, grads):
        if grad is None:
            continue
        belongs_to_head = id(parameter) in _HEAD_PARAMETER_IDS
        if (group == "mask_head") != belongs_to_head:
            continue
        values.append(grad.detach().reshape(-1))
    if not values:
        return torch.zeros(1)
    return torch.cat(values)


_HEAD_PARAMETER_IDS: set[int] = set()


def gradient_snapshot(model, x: torch.Tensor, target: torch.Tensor,
                      clean: torch.Tensor, tail_ref: torch.Tensor) -> dict:
    parameters = [p for p in model.parameters() if p.requires_grad]
    _HEAD_PARAMETER_IDS.clear()
    _HEAD_PARAMETER_IDS.update(id(p) for p in model.mask_head.parameters())
    prediction = model(x)
    loss_base = base_loss(prediction, target, clean)
    loss_early = early_loss(prediction, x, clean, tail_ref)
    grad_base = torch.autograd.grad(loss_base, parameters, retain_graph=True,
                                    allow_unused=True)
    grad_early = torch.autograd.grad(loss_early, parameters, retain_graph=True,
                                     allow_unused=True)
    total = [None if a is None and b is None else
             (torch.zeros_like(b) if a is None else a) +
             (torch.zeros_like(a) if b is None else b)
             for a, b in zip(grad_base, grad_early)]
    result = {"loss_base": float(loss_base.detach()),
              "loss_early": float(loss_early.detach())}
    for group in ("mask_head", "trunk", "all"):
        a = grad_vector(grad_base, parameters, group)
        b = grad_vector(grad_early, parameters, group)
        t = grad_vector(total, parameters, group)
        norm_a, norm_b, norm_t = (float(torch.linalg.vector_norm(v)) for v in (a, b, t))
        cosine = float(torch.dot(a, b) / (torch.linalg.vector_norm(a) *
            torch.linalg.vector_norm(b)).clamp_min(1e-20))
        result[f"grad_base_norm_{group}"] = norm_a
        result[f"grad_early_norm_{group}"] = norm_b
        result[f"grad_cosine_{group}"] = cosine
        result[f"grad_total_norm_{group}"] = norm_t
    result["clip_factor_at_3"] = min(1.0, 3.0 / max(result["grad_total_norm_all"], 1e-20))
    return result


def w1_reduction_db(model, wet: torch.Tensor, dry: torch.Tensor,
                    device: torch.device) -> list[float]:
    model.eval()
    values = []
    with torch.inference_mode():
        for index in range(FIXTURE_COUNT):
            x = wet[index:index + 1].to(device)
            prediction = model(x)[0]
            ex = x[0, W1_START:W1_END].square().mean()
            ey = prediction[W1_START:W1_END].square().mean()
            values.append(float(10.0 * torch.log10((ex + 1e-20) / (ey + 1e-20))))
    model.train()
    return values


def overfit(kind: str, initial_state: dict, wet: torch.Tensor,
            dry: torch.Tensor, tail_ref: torch.Tensor,
            device: torch.device) -> dict:
    torch.manual_seed(SEED)
    model = CompactAttenuationOnlyDereverbG2(base_channels=16).to(device)
    model.load_state_dict(initial_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    before = w1_reduction_db(model, wet, dry, device)
    started = time.perf_counter()
    for step in range(OVERFIT_STEPS):
        index = step % FIXTURE_COUNT
        x = wet[index:index + 1].to(device)
        target = dry[index:index + 1].to(device)
        clean = target
        reference = tail_ref[index].to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x)
        loss_early = early_loss(prediction[0], x[0], clean[0], reference)
        loss = loss_early if kind == "tail_only" else loss_early + base_loss(
            prediction, target, clean)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite {kind} loss at step {step + 1}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        optimizer.step()
    after = w1_reduction_db(model, wet, dry, device)
    active_start, active_end = 3_200, 22_400
    with torch.inference_mode():
        active_delta = []
        for index in range(FIXTURE_COUNT):
            prediction = model(wet[index:index + 1].to(device))[0]
            reference = dry[index].to(device)
            before_energy = reference[active_start:active_end].square().mean()
            after_energy = prediction[active_start:active_end].square().mean()
            active_delta.append(float(10 * torch.log10((after_energy + 1e-20) /
                                                       (before_energy + 1e-20))))
    return {"kind": kind, "steps": OVERFIT_STEPS, "before_w1_db": before,
        "after_w1_db": after,
        "median_w1_improvement_db": float(np.median(np.asarray(after) - np.asarray(before))),
        "active_speech_output_minus_target_db": active_delta,
        "elapsed_seconds": time.perf_counter() - started}


def audit(args) -> dict:
    if args.output_dir.resolve() == DEFAULT_OUTPUT.parent.resolve() or \
            "g-early-2026-10-10" in str(args.output_dir.resolve()):
        raise RuntimeError("v2 mechanics output must never overlap G-early-v1 artifacts")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    model = CompactAttenuationOnlyDereverbG2(base_channels=16).to(device)
    model.train()
    wet, dry, tail_ref = make_fixtures()
    initial_state = {name: tensor.detach().cpu().clone()
                     for name, tensor in model.state_dict().items()}
    gradients = []
    for index in range(FIXTURE_COUNT):
        gradients.append(gradient_snapshot(model, wet[index:index + 1].to(device),
            dry[index:index + 1].to(device), dry[index:index + 1].to(device),
            tail_ref[index].to(device)))
    tail = overfit("tail_only", initial_state, wet, dry, tail_ref, device)
    full = overfit("full_loss", initial_state, wet, dry, tail_ref, device)
    derivative = initial_gain_diagnostics()
    finite_gradient = all(math.isfinite(value) for row in gradients for key, value in row.items()
                          if key.startswith("grad_") or key == "clip_factor_at_3")
    nonzero_groups = all(row["grad_early_norm_mask_head"] > 0 and
                         row["grad_early_norm_trunk"] > 0 for row in gradients)
    tail_gate = tail["median_w1_improvement_db"] >= 3.0
    full_gate = full["median_w1_improvement_db"] >= 0.25
    zero_input = torch.zeros(1, 32_000, device=device)
    with torch.inference_mode():
        zero_max = float(model(zero_input).abs().max())
    zero_gate = zero_max < 1e-7
    return {
        "protocol": "G-early-v2-synthetic-mechanism-v1",
        "seed": SEED,
        "device": str(device),
        "torch_version": torch.__version__,
        "source_sha256": {"model": sha256(HERE / "model_g_early_v2.py"),
            "audit": sha256(Path(__file__))},
        "fixture": {"kind": "deterministic voiced-like harmonic signals plus sparse exponential echo train",
            "count": FIXTURE_COUNT, "samples_per_fixture": 48_000,
            "sample_rate": SAMPLE_RATE, "w1_ms_after_pause": [150, 300],
            "corpus_files_read": False, "g_early_v1_artifacts_read": False},
        "gain_analytic_checks": derivative,
        "gradient_initialization_by_fixture": gradients,
        "overfit": {"tail_only": tail, "full_loss": full},
        "zero_input_max_abs": zero_max,
        "gates": {"finite_component_gradients": finite_gradient,
            "early_gradient_reaches_head_and_trunk": nonzero_groups,
            "tail_only_median_w1_improvement_ge_3db": tail_gate,
            "full_loss_median_w1_improvement_ge_0_25db": full_gate,
            "zero_output_invariant_lt_1e-7": zero_gate},
        "overall": "PASS" if finite_gradient and nonzero_groups and tail_gate and
            full_gate and zero_gate else "FAIL",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    result = audit(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "mechanism-audit.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"overall": result["overall"], "output": str(output),
        "gates": result["gates"]}, indent=2))
    if result["overall"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
