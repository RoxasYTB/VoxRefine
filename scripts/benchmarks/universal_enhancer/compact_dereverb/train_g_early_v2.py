#!/usr/bin/env python3
"""Fit the frozen G-early-v2 model once on the fresh 128-speaker TRAIN set."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
from model_e import parameter_count  # noqa: E402
from model_g_early_v2 import CompactAttenuationOnlyDereverbG2  # noqa: E402
from train_cap60_attenuation_only import (  # noqa: E402
    AUX_DIR, assert_zero_invariant, noise_loss, quiet_loss,
)
from train_cap60_conditioned import cap60_conditioned_loss  # noqa: E402
from train_g_early import freeze_batch_schedule, tail_loss  # noqa: E402

EXPERIMENT = ROOT / ".tools/compact-dereverb/g-early-v2-2026-10-10"
DATA = EXPERIMENT / "data"
FIT = EXPERIMENT / "G-early-v2-clean"
SR = 16_000
STEPS = 3_000
SEED = 20261010


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def state_sha(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class TrainSet(Dataset):
    def __init__(self, root: Path):
        self.root = root
        path = root / "pairs.jsonl"
        self.rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        self.kinds = {kind: sum(row["kind"] == kind for row in self.rows)
                      for kind in ("identity", "rir_only", "fan20", "fan10")}
        if len(self.rows) != 128 or len({str(row["speaker"]) for row in self.rows}) != 128:
            raise RuntimeError("G2 requires exactly 128 unique frozen TRAIN speakers")
        if self.kinds != {"identity": 32, "rir_only": 48, "fan20": 24, "fan10": 24}:
            raise RuntimeError(f"G2 TRAIN kind mixture changed: {self.kinds}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        def load(key: str):
            return torch.from_numpy(np.load(self.root / row[key], allow_pickle=False).copy())
        return (load("x"), load("c"), load("c"), load("tail_ref"),
            row["kind"], bool(row.get("input_only_tail_eligible")),
            int(row["pause_start_in_crop"]), index)


def flat_gradient_stats(left, right) -> tuple[float, float, float | None]:
    dot = left_norm = right_norm = 0.0
    for a, b in zip(left, right):
        if a is None and b is None:
            continue
        if a is None:
            a = torch.zeros_like(b)
        if b is None:
            b = torch.zeros_like(a)
        dot += float(torch.sum(a.detach().double() * b.detach().double()))
        left_norm += float(torch.sum(a.detach().double() ** 2))
        right_norm += float(torch.sum(b.detach().double() ** 2))
    ln, rn = left_norm ** .5, right_norm ** .5
    cosine = dot / (ln * rn) if ln > 0 and rn > 0 else None
    return ln, rn, cosine


def audit_training_inputs(dataset: TrainSet, args) -> dict:
    manifest_path = args.data_dir / "training-data-manifest.json"
    coverage_path = args.data_dir / "coverage-audit.json"
    manifest, coverage = json.loads(manifest_path.read_text()), json.loads(coverage_path.read_text())
    if (manifest.get("test_wav_accessed") is not False or
            manifest.get("training_started") is not False or
            manifest.get("model_outputs_accessed") is not False or
            coverage.get("test_wav_accessed") is not False or
            manifest.get("pair_manifest_sha256") != sha(args.data_dir / "pairs.jsonl") or
            coverage.get("pair_manifest_sha256") != sha(args.data_dir / "pairs.jsonl")):
        raise RuntimeError("G2 data manifest/hash/pre-fit guard failed")
    eligible = 0
    for row in dataset.rows:
        for field, digest_key in (("x", "x_sha256"), ("c", "clean_target_sha256"),
                ("a", "cap_target_sha256"), ("hybrid", "hybrid_target_sha256"),
                ("tail_ref", "tail_ref_sha256"), ("mask", "mask_sha256")):
            if sha(args.data_dir / row[field]) != row[digest_key]:
                raise RuntimeError(f"G2 array hash mismatch: {row['index']}:{field}")
        tail = np.load(args.data_dir / row["tail_ref"], allow_pickle=False)
        cap = np.load(args.data_dir / row["a"], allow_pickle=False)
        if row["tail_ref_sha256"] != row["cap_target_sha256"] or not np.array_equal(tail, cap):
            raise RuntimeError("G2 tail reference must equal the exact shared Cap60 target")
        mask = np.load(args.data_dir / row["mask"], allow_pickle=False)
        clean = np.load(args.data_dir / row["c"], allow_pickle=False)
        hybrid = np.load(args.data_dir / row["hybrid"], allow_pickle=False)
        if mask.shape != clean.shape or np.max(np.abs(mask * clean + (1-mask) * cap - hybrid)) > 1e-7:
            raise RuntimeError("G2 frozen weak/onset hybrid does not match its arrays")
        eligible += int(row["kind"] == "rir_only" and row.get("input_only_tail_eligible"))
        if row["kind"] in ("fan20", "fan10"):
            if not row.get("noise_only") or sha(args.data_dir / row["noise_only"]) != row.get("noise_only_sha256"):
                raise RuntimeError("G2 synthetic fan-noise controls are missing or corrupted")
    if (coverage.get("resolved_tail_slots") != 48 or eligible < 36 or
            eligible != coverage.get("eligible_tail_slots") or not coverage.get("executable")):
        raise RuntimeError("G2 TRAIN W1 coverage gate did not pass")
    return {"data_manifest_sha256": sha(manifest_path),
        "pair_manifest_sha256": sha(args.data_dir / "pairs.jsonl"),
        "coverage_sha256": sha(coverage_path), "eligible_tail_slots": eligible}


def train(args) -> dict:
    if args.steps != STEPS or args.seed != SEED:
        raise RuntimeError("G2 fit update budget and seed are frozen")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    dataset = TrainSet(args.data_dir)
    data_audit = audit_training_inputs(dataset, args)
    split_audit_path = EXPERIMENT / "split-integrity-audit.json"
    split_index = EXPERIMENT / "splits/manifest.json"
    for path in (split_audit_path, split_index):
        if not path.is_file():
            raise FileNotFoundError(f"v2 TRAIN/DEV/HOLDOUT must be frozen and audited: {path}")
    split_audit = json.loads(split_audit_path.read_text())
    index = json.loads(split_index.read_text())
    if (split_audit.get("holdout_opened") is not False or
            split_audit.get("opened_for_metrics") is not False or
            index.get("holdout_opened") is not False or
            index.get("test_wav_accessed") is not False or
            split_audit.get("split_index_sha256") != sha(split_index)):
        raise RuntimeError("G2 split audit is missing, stale, or already opened")
    if FIT.exists() and any(FIT.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing fit: {FIT}")
    FIT.mkdir(parents=True, exist_ok=True)

    model = CompactAttenuationOnlyDereverbG2(base_channels=16)
    count = parameter_count(model)
    if count != 555_922:
        raise RuntimeError(f"G2 architecture parameter count changed: {count}")
    init_state = {name: tensor.detach().cpu().clone()
                  for name, tensor in model.state_dict().items()}
    init_sha = state_sha(model)
    torch.save({"model": init_state, "seed": SEED, "initialization_sha256": init_sha,
                "base_channels": 16, "test_wav_accessed": False},
               EXPERIMENT / "initial-state-v2.pt")
    model = model.to(device)
    assert_zero_invariant(model, device)

    schedule = freeze_batch_schedule(SEED, STEPS, len(dataset))
    schedule_path = EXPERIMENT / "batch-schedule-v2.npy"
    np.save(schedule_path, schedule, allow_pickle=False)
    schedule_sha = hashlib.sha256(schedule.tobytes()).hexdigest()
    noise_rows = {kind: [row for row in dataset.rows if row["kind"] == kind]
                  for kind in ("fan20", "fan10")}
    noise_controls = {kind: [torch.from_numpy(np.load(args.data_dir / row["noise_only"],
        allow_pickle=False).copy()) for row in rows] for kind, rows in noise_rows.items()}
    noise_index = {"fan20": 0, "fan10": 0}
    loader = DataLoader(dataset, batch_size=1, sampler=schedule.tolist(),
        num_workers=args.num_workers, pin_memory=device.type == "cuda", drop_last=True,
        generator=torch.Generator().manual_seed(SEED + 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    model.train()
    order, logs = [], FIT / "training.jsonl"
    params = tuple(model.parameters())
    started = time.perf_counter()
    iterator = iter(loader)
    for step in range(1, STEPS + 1):
        try:
            x, clean, target, tail_ref, kinds, tail_enabled, pauses, indices = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, clean, target, tail_ref, kinds, tail_enabled, pauses, indices = next(iterator)
        order.append(int(indices[0]))
        x, clean, target, tail_ref = (v.to(device, non_blocking=True)
                                       for v in (x, clean, target, tail_ref))
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        speech, components = cap60_conditioned_loss(pred, target, clean)
        quiet = quiet_loss(pred, target, clean)
        base = speech - .5 * components["floor"] + .5 * quiet
        early, eligible_count = tail_loss(pred, x, clean, tail_ref, pauses,
            tuple(kinds), tail_enabled)
        total = base + early
        noise_value = total.new_zeros(())
        if step % 4 == 0:
            noise_batch = []
            for kind in ("fan20", "fan10"):
                rows = noise_controls[kind]
                noise_batch.append(rows[noise_index[kind] % len(rows)])
                noise_index[kind] += 1
            noise_input = torch.stack(noise_batch).to(device, non_blocking=True)
            noise_value = noise_loss(model, noise_input)
            total = total + .5 * noise_value
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite G2 loss at step {step}")

        grad_diag = {}
        if step == 1 or step % 100 == 0 or step == STEPS:
            base_grads = torch.autograd.grad(base, params, retain_graph=True, allow_unused=True)
            if early.requires_grad:
                early_grads = torch.autograd.grad(early, params, retain_graph=True, allow_unused=True)
            else:
                early_grads = tuple(None for _ in params)
            base_norm, early_norm, cosine = flat_gradient_stats(base_grads, early_grads)
            grad_diag = {"grad_base_norm": base_norm, "grad_early_norm": early_norm,
                "grad_base_early_cosine": cosine}
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite G2 gradient at step {step}")
        clip_factor = min(1.0, 3.0 / max(float(grad_norm), 1e-12))
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == STEPS:
            log = {"step": step, "total_loss": float(total.detach()),
                "speech_loss": float(speech.detach()),
                "speech_floor_component": float(components["floor"].detach()),
                "quiet_loss": float(quiet.detach()), "noise_loss": float(noise_value.detach()),
                "early_tail_loss": float(early.detach()),
                "eligible_tail_examples": int(eligible_count),
                "grad_total_preclip_norm": float(grad_norm),
                "clip_factor_at_3": clip_factor, **grad_diag,
                "row_index": int(indices[0]), "kind": list(kinds),
                "elapsed_s": time.perf_counter() - started,
                "peak_vram_mib": torch.cuda.max_memory_allocated(device) / 2**20
                    if device.type == "cuda" else None}
            with logs.open("a") as stream:
                stream.write(json.dumps(log, allow_nan=False) + "\n")
            print(json.dumps(log, allow_nan=False), flush=True)
        if step % 500 == 0:
            assert_zero_invariant(model, device)

    order_array = np.asarray(order, dtype=np.int64)
    if not np.array_equal(order_array, schedule):
        raise RuntimeError("executed G2 batch order differs from the frozen schedule")
    np.save(FIT / "batch-order.npy", order_array, allow_pickle=False)
    order_sha = hashlib.sha256(order_array.tobytes()).hexdigest()
    final_state_sha = state_sha(model)
    checkpoint = FIT / "checkpoints/step-003000.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"step": STEPS, "model": model.state_dict(), "base_channels": 16,
        "seed": SEED, "variant": "G-early-v2-clean", "initialization_sha256": init_sha,
        "final_state_sha256": final_state_sha,
        "data_manifest_sha256": data_audit["pair_manifest_sha256"],
        "batch_order_sha256": order_sha, "test_wav_accessed": False}, checkpoint)
    config = {"variant": "G-early-v2-clean", "protocol_version": "G-early-v2-real-2026-10-10",
        "sample_rate": SR, "parameter_count": count, "seed": SEED, "steps": STEPS,
        "batch_size": 1, "optimizer": "AdamW(lr=2e-4, weight_decay=1e-4)",
        "objective": "Cap60 conditioned speech - 0.5 floor + 0.5 quiet + W1 tail + every-fourth-step 0.5 fan-noise loss",
        "tail_loss_weight": 1.0, "gradient_clip_norm": 3.0,
        "initialization_sha256": init_sha, "final_state_sha256": final_state_sha,
        "checkpoint_sha256": sha(checkpoint), "training_source_sha256": sha(Path(__file__)),
        "model_source_sha256": sha(HERE / "model_g_early_v2.py"),
        "evaluator_source_sha256": sha(HERE / "evaluate_g_early_v2.py"),
        "split_freeze_source_sha256": sha(HERE / "freeze_g_early_v2_splits.py"),
        "split_audit_source_sha256": sha(HERE / "audit_g_early_v2_splits.py"),
        "data_manifest_sha256": data_audit["data_manifest_sha256"],
        "pair_manifest_sha256": data_audit["pair_manifest_sha256"],
        "coverage_sha256": data_audit["coverage_sha256"],
        "split_index_sha256": sha(split_index),
        "split_audit_sha256": sha(split_audit_path),
        "batch_order_sha256": order_sha, "device": str(device),
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "training_elapsed_s": time.perf_counter() - started,
        "peak_vram_mib": torch.cuda.max_memory_allocated(device) / 2**20
            if device.type == "cuda" else None, "test_wav_accessed": False}
    (FIT / "model-config.json").write_text(json.dumps(config, indent=2) + "\n")
    (FIT / "training-manifest.json").write_text(json.dumps({**config,
        "train_kind_counts": dataset.kinds,
        "component_gradient_diagnostic_interval": 100,
        "batch_order_policy": "frozen 3000-index deterministic schedule",
        "holdout_opened": False}, indent=2) + "\n")
    return {"checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
        "steps": STEPS, "training_elapsed_s": config["training_elapsed_s"],
        "peak_vram_mib": config["peak_vram_mib"],
        "initialization_sha256": init_sha, "final_state_sha256": final_state_sha,
        "test_wav_accessed": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    print(json.dumps(train(args), indent=2))


if __name__ == "__main__":
    main()
