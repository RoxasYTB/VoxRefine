#!/usr/bin/env python3
"""Run the preregistered matched G-early-clean and G-early-hybrid fits."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
from model_e import CompactAttenuationOnlyDereverb16k, parameter_count  # noqa: E402
from train_cap60_attenuation_only import (  # noqa: E402
    AUX_DIR, assert_zero_invariant, noise_loss, quiet_loss,
)
from train_cap60_conditioned import cap60_conditioned_loss  # noqa: E402

EXPERIMENT = ROOT / ".tools/compact-dereverb/g-early-2026-10-10"
DATA = EXPERIMENT / "data"
SR = 16_000


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_dict_sha(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class GDataset(Dataset):
    def __init__(self, pair_manifest: Path, target_name: str):
        if target_name not in ("c", "hybrid"):
            raise ValueError("target_name must be c or hybrid")
        self.root = pair_manifest.parent
        self.rows = [json.loads(line) for line in pair_manifest.read_text().splitlines()
                     if line.strip()]
        self.target_name = target_name
        if len(self.rows) != 128:
            raise RuntimeError("G training set must have exactly 128 frozen rows")
        self.kind_counts = {kind: sum(row["kind"] == kind for row in self.rows)
            for kind in ("identity", "rir_only", "fan20", "fan10")}
        if self.kind_counts != {"identity": 32, "rir_only": 48,
                                "fan20": 24, "fan10": 24}:
            raise RuntimeError(f"unexpected training mix: {self.kind_counts}")
        if len({str(row["speaker"]) for row in self.rows}) != 128:
            raise RuntimeError("training rows must contain 128 unique speakers")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        def load(name):
            return torch.from_numpy(np.load(self.root / row[name], allow_pickle=False).copy())
        x, clean, target, tail_ref = (load(name) for name in
            ("x", "c", self.target_name, "tail_ref"))
        return (x, target, clean, tail_ref, row["kind"],
            bool(row.get("input_only_tail_eligible", False)),
            int(row["pause_start_in_crop"] or -1), index)


def tail_loss(pred: torch.Tensor, x: torch.Tensor, clean: torch.Tensor,
              tail_reference: torch.Tensor, pause_samples: torch.Tensor,
              kinds: tuple[str, ...], eligible_rows: torch.Tensor) -> tuple[torch.Tensor, int]:
    """W1-only hinge for input-eligible RIR rows; W2 has no training loss."""
    terms = []
    for i, kind in enumerate(kinds):
        if kind != "rir_only" or not bool(eligible_rows[i]):
            continue
        pause = int(pause_samples[i])
        if pause < 0:
            continue
        c = clean[i]
        frames = c.unfold(-1, 320, 160)
        rms = (frames.square().mean(-1) + 1e-24).sqrt()
        active = rms >= torch.maximum(rms.max() * .02, rms.new_tensor(1e-5))
        centers = torch.arange(active.numel(), device=c.device) * 160 + 160
        ref_frames = active & (centers >= pause - 8_000) & (centers < pause - 800)
        if int(ref_frames.sum()) * 160 < 3_200:
            continue
        ref_samples = torch.nn.functional.interpolate(ref_frames.float()[None, None, :],
            size=c.numel(), mode="nearest")[0, 0] > .5
        if not ref_samples.any():
            continue
        eref = tail_reference[i][ref_samples].square().mean()
        eps = eref * 1e-12
        start, end = pause + 150 * 16, pause + 300 * 16
        if start < 0 or end > x.shape[-1]:
            continue
        ex = x[i, start:end].square().mean()
        ey = pred[i, start:end].square().mean()
        delta1 = 10 * torch.log10((ex + eps) / (ey + eps))
        terms.append((torch.nn.functional.relu(3.0 - delta1) / 3.0).square())
    if not terms:
        # Controls skip W1 entirely; return a finite scalar without reading
        # the prediction graph (0 * NaN would still be NaN).
        return pred.new_zeros(()), 0
    return torch.stack(terms).mean(), len(terms)


def validate_tail_loss_reference() -> None:
    x = torch.zeros(1, 32_000)
    clean = torch.zeros_like(x)
    clean[:, 4_000:11_200] = .2
    tail_ref = clean.clone()
    x[:, 14_400:21_600] = .02
    pred = x.clone().requires_grad_(True)
    value, eligible = tail_loss(pred, x, clean, tail_ref,
        torch.tensor([12_000]), ("rir_only",), torch.tensor([True]))
    expected = 1.0
    if eligible != 1 or abs(float(value.detach()) - expected) > 1e-6:
        raise RuntimeError("G-early W1 loss must equal 1 when prediction retains input tail")
    grad, = torch.autograd.grad(value, pred)
    if not torch.isfinite(grad).all() or float(grad[0, 14_400:21_600].mean()) <= 0:
        raise RuntimeError("G tail gradient does not penalize residual tail energy")
    half_energy = pred.detach().clone()
    half_energy[:, 14_400:16_800] = .02 / math.sqrt(2.0)
    half_value, half_count = tail_loss(half_energy, x, clean, tail_ref,
        torch.tensor([12_000]), ("rir_only",), torch.tensor([True]))
    if half_count != 1 or abs(float(half_value) - 0.0) > 1e-6:
        raise RuntimeError("G-early W1 loss sign/energy convention failed at 3.01dB reduction")


def freeze_batch_schedule(seed: int, steps: int, rows: int = 128) -> np.ndarray:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order: list[int] = []
    while len(order) < steps:
        order.extend(torch.randperm(rows, generator=generator).tolist())
    return np.asarray(order[:steps], dtype=np.int64)


def train_one(variant: str, args, init_state: dict, init_sha: str,
              data_manifest: dict, rows: list[dict], schedule: np.ndarray,
              schedule_sha: str) -> Path:
    target_name = "c" if variant == "G-early-clean" else "hybrid"
    output = args.experiment_dir / variant
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to reuse nonempty fit directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    model = CompactAttenuationOnlyDereverb16k(base_channels=16).to(device)
    model.load_state_dict(init_state)
    count = parameter_count(model)
    if count != 555_922:
        raise RuntimeError(f"E architecture changed: {count} parameters")
    assert_zero_invariant(model, device)
    dataset = GDataset(args.data_dir / "pairs.jsonl", target_name)
    worker_generator = torch.Generator().manual_seed(args.seed + 1)
    loader = DataLoader(dataset, batch_size=1, sampler=schedule.tolist(), num_workers=args.num_workers,
        pin_memory=device.type == "cuda", drop_last=True, generator=worker_generator)
    aux_manifest = json.loads((AUX_DIR / "manifest.json").read_text())
    if aux_manifest.get("test_wav_accessed") is not False or aux_manifest.get("counts") != {
            "fan20": 24, "fan10": 24}:
        raise RuntimeError("frozen fan-only controls are missing or invalid")
    aux_rows = {kind: [row for row in aux_manifest["rows"] if row["kind"] == kind]
                for kind in ("fan20", "fan10")}
    aux = {kind: [torch.from_numpy(np.load(AUX_DIR / row["output"],
        allow_pickle=False).copy()) for row in group] for kind, group in aux_rows.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    config = {"variant": variant, "sample_rate": SR, "parameter_count": count,
        "protocol_version": "G-early-v1",
        "trainer_source_sha256": sha(HERE / "train_g_early.py"),
        "evaluator_source_sha256": sha(HERE / "evaluate_g_early.py"),
        "split_freeze_source_sha256": sha(HERE / "freeze_g_early_splits.py"),
        "training_audit_source_sha256": sha(HERE / "audit_g_early_training.py"),
        "split_audit_source_sha256": sha(HERE / "audit_g_early_splits.py"),
        "model_source_sha256": hashlib.sha256((HERE / "model_e.py").read_bytes()).hexdigest(),
        "initialization_sha256": init_sha,
        "training_data_manifest_sha256": sha(args.data_dir / "training-data-manifest.json"),
        "training_pair_manifest_sha256": sha(args.data_dir / "pairs.jsonl"),
        "coverage_audit_sha256": sha(args.data_dir / "coverage-audit.json"),
        "split_manifest_sha256": sha(args.experiment_dir / "splits/manifest.json"),
        "seed": args.seed, "optimizer": "AdamW(lr=2e-4, weight_decay=1e-4)",
        "steps": args.steps, "batch_size": 1, "tail_loss_weight": 1.0,
        "batch_schedule_sha256": schedule_sha,
        "target": "c_q dry clean" if variant == "G-early-clean" else "frozen weak/onset hybrid",
        "tail_ref": "a_q = Cap60(clean_scaled)*common_gain, independent of target",
        "shared_input_hashes": [row["x_sha256"] for row in rows],
        "shared_clean_truth_hashes": [row["clean_target_sha256"] for row in rows],
        "shared_cap60_reference_hashes": [row["cap_target_sha256"] for row in rows],
        "tail_ref_hashes": [row["tail_ref_sha256"] for row in rows],
        "test_wav_accessed": False}
    (output / "model-config.json").write_text(json.dumps(config, indent=2) + "\n")
    training_manifest = {**config, "device": str(device), "torch": torch.__version__,
        "train_kind_counts": dataset.kind_counts,
        "batch_order_policy": "same 3000-index schedule frozen and hashed before either fit; independent worker generator",
        "quiet_loss_weight": .5, "noise_aux_weight": .5,
        "noise_aux_every_n_steps": 4,
        "tail_eligibility": "input-only W1 first candidate with Lx>max(-60dB,Ldry+6dB); controls have lambda_early=0; W2 excluded from training loss",
        "checkpoint_policy": "fixed final step; both fits complete before DEV is opened"}
    (output / "training-manifest.json").write_text(json.dumps(training_manifest, indent=2) + "\n")

    iterator = iter(loader)
    aux_i = {"fan20": 0, "fan10": 0}
    order: list[int] = []
    log_path = output / "training.jsonl"
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        try:
            x, target, clean, tail_ref, kinds, tail_enabled, pauses, indices = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, target, clean, tail_ref, kinds, tail_enabled, pauses, indices = next(iterator)
        order.append(int(indices[0]))
        x, target, clean, tail_ref = (v.to(device, non_blocking=True)
                                      for v in (x, target, clean, tail_ref))
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        speech, components = cap60_conditioned_loss(pred, target, clean)
        total = speech - .5 * components["floor"] + .5 * quiet_loss(pred, target, clean)
        auxiliary_value = 0.0
        if step % 4 == 0:
            selected = []
            for kind in ("fan20", "fan10"):
                selected.append(aux[kind][aux_i[kind] % len(aux[kind])])
                aux_i[kind] += 1
            noise_x = torch.stack(selected).to(device, non_blocking=True)
            auxiliary = noise_loss(model, noise_x)
            total = total + .5 * auxiliary
            auxiliary_value = float(auxiliary.detach())
            with torch.no_grad():
                zeros = model(torch.zeros(2, 32_000, device=device))
                if float(zeros.abs().max()) >= 1e-7:
                    raise RuntimeError(f"zero invariant failed at step {step}")
        tail_value, eligible_count = tail_loss(pred, x, clean, tail_ref, pauses,
                                               kinds, tail_enabled)
        total = total + tail_value
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite loss at step {step}")
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite gradient at step {step}")
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            log_row = {"step": step, "loss": float(total.detach()),
                "speech_loss_without_c_floor": float((speech - .5 * components["floor"]).detach()),
                "quiet_loss": float(quiet_loss(pred.detach(), target, clean).detach()),
                "noise_aux_loss": auxiliary_value,
                "base_loss": float((total - tail_value).detach()),
                "early_tail_loss": float(tail_value.detach()),
                "tail_loss": float(tail_value.detach()),
                "eligible_tail_examples_in_batch": eligible_count,
                "grad_norm": float(grad_norm), "kind": list(kinds),
                "row_index": int(indices[0]), "elapsed_s": time.perf_counter() - started,
                "peak_vram_mib": torch.cuda.max_memory_allocated(device) / 2**20
                    if device.type == "cuda" else None}
            with log_path.open("a") as file:
                file.write(json.dumps(log_row) + "\n")
            print(json.dumps({"variant": variant, **log_row}), flush=True)
        if step % 500 == 0:
            assert_zero_invariant(model, device)

    order_array = np.asarray(order, dtype=np.int64)
    if not np.array_equal(order_array, schedule):
        raise RuntimeError("executed batch order differs from frozen prefit schedule")
    order_hash = hashlib.sha256(order_array.tobytes()).hexdigest()
    np.save(output / "batch-order.npy", order_array, allow_pickle=False)
    checkpoint = output / "checkpoints/step-003000.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"step": args.steps, "model": model.state_dict(),
        "base_channels": 16, "seed": args.seed, "variant": variant,
        "data_manifest_sha256": sha(args.data_dir / "pairs.jsonl"),
        "batch_order_sha256": order_hash}, checkpoint)
    config.update({"checkpoint_sha256": sha(checkpoint), "batch_order_sha256": order_hash,
        "training_elapsed_s": time.perf_counter() - started,
        "peak_vram_mib": torch.cuda.max_memory_allocated(device) / 2**20
            if device.type == "cuda" else None})
    (output / "model-config.json").write_text(json.dumps(config, indent=2) + "\n")
    training_manifest.update({"batch_order_sha256": order_hash,
        "batch_order_path": "batch-order.npy"})
    (output / "training-manifest.json").write_text(json.dumps(training_manifest, indent=2) + "\n")
    return checkpoint


def train(args) -> list[str]:
    validate_tail_loss_reference()
    data_manifest_path = args.data_dir / "training-data-manifest.json"
    pair_path = args.data_dir / "pairs.jsonl"
    coverage_path = args.data_dir / "coverage-audit.json"
    split_path = args.experiment_dir / "splits/manifest.json"
    for path in (data_manifest_path, pair_path, coverage_path, split_path):
        if not path.is_file():
            raise FileNotFoundError(f"freeze all G inputs before training: {path}")
    data_manifest = json.loads(data_manifest_path.read_text())
    coverage = json.loads(coverage_path.read_text())
    splits = json.loads(split_path.read_text())
    if data_manifest.get("test_wav_accessed") is not False or splits.get("test_wav_accessed") is not False:
        raise RuntimeError("G manifests do not preserve sealed test.wav policy")
    if data_manifest.get("pair_manifest_sha256") != sha(pair_path):
        raise RuntimeError("G training data manifest does not match the pair manifest")
    if (coverage.get("pair_manifest_sha256") != sha(pair_path) or
            not coverage.get("executable") or coverage.get("resolved_tail_slots") != 48 or
            int(coverage.get("eligible_tail_slots", 0)) < 36):
        raise RuntimeError("G-early train needs 48 resolved slots and >=36 W1 eligible")
    if (not splits.get("frozen_before_training") or splits.get("opened_for_metrics") or
            splits.get("holdout_opened") is not False or
            splits.get("dev_coverage_gate_pass") is not True or
            splits.get("holdout_coverage_gate_pass") is not True or
            splits.get("dev_holdout_disjoint") is not True or
            splits.get("disjoint_from_training") is not True or
            splits.get("dev_speaker_count") != 16 or splits.get("holdout_speaker_count") != 12):
        raise RuntimeError("fresh frozen 16/12 G speaker splits are required")
    rows = [json.loads(line) for line in pair_path.read_text().splitlines() if line.strip()]
    actual_train_eligible = sum(bool(row.get("input_only_tail_eligible"))
                                for row in rows if row["kind"] == "rir_only")
    if actual_train_eligible != int(coverage.get("eligible_tail_slots", -1)):
        raise RuntimeError("G-early train coverage count differs from frozen rows")
    train_ids = {str(row["speaker"]) for row in rows}
    split_ids = {}
    for name, expected_speakers, expected_pairs in (("dev", 16, 64), ("sealed", 12, 48)):
        manifest_path = args.experiment_dir / f"splits/{name}/manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing frozen G {name} speaker manifest")
        split_doc = json.loads(manifest_path.read_text())
        expected_hash = splits["dev_manifest_sha256" if name == "dev" else "holdout_manifest_sha256"]
        ids = {str(item["speaker_id"]) for item in split_doc.get("speakers", [])}
        if (sha(manifest_path) != expected_hash or len(ids) != expected_speakers or
                split_doc.get("pair_count") != expected_pairs or
                not split_doc.get("coverage_gate_pass") or
                not split_doc.get("frozen_before_training") or
                split_doc.get("opened_for_metrics") or
                split_doc.get("test_wav_accessed") is not False or ids & train_ids):
            raise RuntimeError(f"invalid or overlapping frozen G {name} speaker manifest")
        counts = [sum(bool(pair.get("tail_loss_enabled")) for pair in sp["pairs"])
                  for sp in split_doc["speakers"]]
        eligible_count = sum(counts)
        if (len(counts) != expected_speakers or any(len(sp["pairs"]) != 4 for sp in split_doc["speakers"]) or
                eligible_count != int(split_doc.get("eligible_pair_count", -1)) or
                sum(c >= 3 for c in counts) != int(split_doc.get("speakers_with_at_least_3_of_4", -1)) or
                any(c == 0 for c in counts)):
            raise RuntimeError(f"{name} slot coverage aggregates do not match frozen rows")
        if name == "dev" and (eligible_count < 48 or sum(c >= 3 for c in counts) < 12):
            raise RuntimeError("DEV16 needs >=48/64 eligible and >=12 speakers at >=3/4")
        if name == "sealed" and (eligible_count < 36 or sum(c >= 3 for c in counts) < 9):
            raise RuntimeError("HOLDOUT12 needs >=36/48 eligible and >=9 speakers at >=3/4")
        split_ids[name] = ids
    if split_ids["dev"] & split_ids["sealed"]:
        raise RuntimeError("G DEV and HOLDOUT-G speaker manifests overlap")
    tail_rows = [row for row in rows if row["kind"] == "rir_only"]
    tail_slots = sorted(int(row["rir_recipe"].get("tail_slot", -1)) for row in tail_rows)
    if tail_slots != list(range(48)):
        raise RuntimeError("G-early tailbank slots must be the exact frozen 0..47 set")

    def audit_selection(recipe: dict, history: list[dict], eligible: bool, label: str) -> None:
        if not recipe.get("input_only_selection") or not history:
            raise RuntimeError(f"missing input-only candidate trace: {label}")
        if [int(item["candidate_index"]) for item in history] != list(range(len(history))):
            raise RuntimeError(f"candidate history is not a j=0 prefix: {label}")
        passing = []
        for item in history:
            w1 = item.get("windows", {}).get("150_300", {})
            good = bool(item.get("reference_valid") and w1.get("lx_db") is not None and
                float(w1["lx_db"]) > max(-60.0, float(w1["ldry_db"]) + 6.0))
            recorded = item.get("eligible_from_input_only", item.get("tail_eligible"))
            if good != bool(recorded):
                raise RuntimeError(f"W1 eligibility formula mismatch: {label}")
            if good:
                passing.append(int(item["candidate_index"]))
        chosen = int(recipe.get("candidate_index", -1))
        if eligible:
            if not passing or chosen != passing[0] or len(history) != chosen + 1:
                raise RuntimeError(f"tail slot did not select first W1 success: {label}")
        elif (len(history) != 32 or chosen != 0 or passing or
                recipe.get("classification") != "BASE_ONLY_CONTROL"):
            raise RuntimeError(f"base-only control is not deterministic j=0: {label}")
        if (not 0 <= chosen < 32 or not 1.65 <= float(recipe.get("t60_s", 0)) <= 1.85 or
                not -14.5 <= float(recipe.get("direct_to_reverb_db", 99)) <= -12.5):
            raise RuntimeError(f"candidate parameters are outside frozen range: {label}")
    for row in tail_rows:
        recipe = row["rir_recipe"]
        audit_selection(recipe, recipe.get("candidate_history", []),
                        bool(row.get("input_only_tail_eligible")),
                        f"train row {row['index']}")
    for name, expected_pairs in (("dev", 64), ("sealed", 48)):
        split_doc = json.loads((args.experiment_dir / f"splits/{name}/manifest.json").read_text())
        if sum(len(speaker["pairs"]) for speaker in split_doc["speakers"]) != expected_pairs:
            raise RuntimeError(f"incomplete frozen G {name} pair coverage")
        for speaker in split_doc["speakers"]:
            if len(speaker["pairs"]) != 4:
                raise RuntimeError(f"frozen G {name} requires exactly four RIRs per speaker")
            for pair in speaker["pairs"]:
                recipe = pair["procedural_rir"]
                audit_selection(recipe, pair.get("input_tail_selection_history", []),
                    bool(pair.get("tail_loss_enabled")), f"{name}:{speaker['speaker_id']}:{pair['pair_index']}")
    for row in rows:
        for field, hash_field in (("x", "x_sha256"), ("c", "clean_target_sha256"),
                ("a", "cap_target_sha256"), ("hybrid", "hybrid_target_sha256"),
                ("tail_ref", "tail_ref_sha256"), ("mask", "mask_sha256")):
            if sha(args.data_dir / row[field]) != row[hash_field]:
                raise RuntimeError(f"G array hash mismatch at row {row['index']}:{field}")
        if row["tail_ref_sha256"] != row["cap_target_sha256"]:
            raise RuntimeError(f"tail reference is not a_q at row {row['index']}")
        clean = np.load(args.data_dir / row["c"], allow_pickle=False)
        cap_target = np.load(args.data_dir / row["a"], allow_pickle=False)
        hybrid = np.load(args.data_dir / row["hybrid"], allow_pickle=False)
        mask = np.load(args.data_dir / row["mask"], allow_pickle=False)
        reconstructed = mask * clean + (1.0 - mask) * cap_target
        if mask.shape != clean.shape or np.max(np.abs(reconstructed - hybrid)) > 1e-7:
            raise RuntimeError(f"G-hybrid target does not match its frozen mask at row {row['index']}")
        frames = np.lib.stride_tricks.sliding_window_view(clean, 320)[::160]
        frame_rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-24)
        peak = max(float(frame_rms.max()), 1e-12)
        active = frame_rms > .02 * peak
        weak = active & (frame_rms < .35 * peak)
        previous = np.r_[0.0, frame_rms[:-1]]
        onset = active & (frame_rms > 1.5 * previous)
        protected = weak | onset
        dilated = protected.copy()
        for shift in (1, 2):
            dilated[shift:] |= protected[:-shift]
            dilated[:-shift] |= protected[shift:]
        for frame_i in np.flatnonzero(dilated):
            start = frame_i * 160
            end = min(start + 320, len(mask))
            if np.any(mask[start:end] < 1.0 - 1e-7):
                raise RuntimeError(f"hybrid mask attenuates weak/onset frame at row {row['index']}")
    if args.steps != 3000:
        raise RuntimeError("G protocol is fixed at exactly 3,000 optimizer updates per fit")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(args.cpu_threads)
    schedule = freeze_batch_schedule(args.seed, args.steps, len(rows))
    schedule_path = args.experiment_dir / "batch-schedule.npy"
    schedule_meta_path = args.experiment_dir / "batch-schedule.json"
    schedule_sha = hashlib.sha256(schedule.tobytes()).hexdigest()
    if schedule_path.exists() or schedule_meta_path.exists():
        if not (schedule_path.is_file() and schedule_meta_path.is_file()):
            raise RuntimeError("partial frozen batch schedule artifact")
        prior = np.load(schedule_path, allow_pickle=False)
        meta = json.loads(schedule_meta_path.read_text())
        if (not np.array_equal(prior, schedule) or meta.get("schedule_sha256") != schedule_sha or
                meta.get("steps") != args.steps or meta.get("seed") != args.seed):
            raise RuntimeError("existing prefit batch schedule differs from current fixed config")
    else:
        np.save(schedule_path, schedule, allow_pickle=False)
        schedule_meta_path.write_text(json.dumps({"name": "G-early-batch-schedule-v1",
            "schedule_sha256": schedule_sha, "steps": args.steps, "seed": args.seed,
            "row_count": len(rows), "frozen_before_training": True,
            "test_wav_accessed": False}, indent=2) + "\n")
    initial_model = CompactAttenuationOnlyDereverb16k(base_channels=16)
    initial_sha = state_dict_sha(initial_model)
    init_path = args.experiment_dir / "initial-state.pt"
    if init_path.exists():
        payload = torch.load(init_path, map_location="cpu", weights_only=False)
        if payload.get("initialization_sha256") != initial_sha:
            raise RuntimeError("frozen G initialization differs from the current seed/model")
        init_state = payload["model"]
    else:
        args.experiment_dir.mkdir(parents=True, exist_ok=True)
        init_state = {name: tensor.detach().cpu().clone()
                      for name, tensor in initial_model.state_dict().items()}
        torch.save({"model": init_state, "seed": args.seed,
            "base_channels": 16, "initialization_sha256": initial_sha}, init_path)
    paths = []
    for variant in ("G-early-clean", "G-early-hybrid"):
        paths.append(str(train_one(variant, args, init_state, initial_sha,
                                   data_manifest, rows, schedule, schedule_sha)))
    configs = [json.loads((args.experiment_dir / variant / "model-config.json").read_text())
               for variant in ("G-early-clean", "G-early-hybrid")]
    orders = [np.load(args.experiment_dir / variant / "batch-order.npy", allow_pickle=False)
              for variant in ("G-early-clean", "G-early-hybrid")]
    if configs[0]["initialization_sha256"] != configs[1]["initialization_sha256"] or not np.array_equal(*orders):
        raise RuntimeError("matched G-early fits differ in initialization or batch order")
    for field in ("training_data_manifest_sha256", "training_pair_manifest_sha256",
            "shared_input_hashes", "shared_clean_truth_hashes",
            "shared_cap60_reference_hashes", "tail_ref_hashes",
            "batch_schedule_sha256"):
        if configs[0][field] != configs[1][field]:
            raise RuntimeError(f"G-early-clean/G-early-hybrid shared field differs: {field}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, default=EXPERIMENT)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.data_dir is None:
        args.data_dir = args.experiment_dir / "data"
    print(json.dumps(train(args), indent=2))


if __name__ == "__main__":
    main()
