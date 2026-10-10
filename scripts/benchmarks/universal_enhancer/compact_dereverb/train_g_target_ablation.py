#!/usr/bin/env python3
"""Run the preregistered matched G-clean and G-hybrid training fits."""
from __future__ import annotations

import argparse
import hashlib
import json
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

EXPERIMENT = ROOT / ".tools/compact-dereverb/g-target-ablation-2026-10-10"
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
        return x, target, clean, tail_ref, row["kind"], int(row["pause_start_in_crop"] or -1), index


def tail_loss(pred: torch.Tensor, x: torch.Tensor, clean: torch.Tensor,
              tail_reference: torch.Tensor, pause_samples: torch.Tensor,
              kinds: tuple[str, ...]) -> tuple[torch.Tensor, int]:
    """Input-eligible one-sided tail hinge, with Eref independent of the fit target."""
    terms = []
    for i, kind in enumerate(kinds):
        if kind != "rir_only":
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
        deltas = []
        eligible = True
        for start_ms, end_ms in ((150, 300), (300, 600)):
            start = pause + start_ms * 16
            end = pause + end_ms * 16
            if start < 0 or end > x.shape[-1]:
                eligible = False
                break
            ex = x[i, start:end].square().mean()
            ey = pred[i, start:end].square().mean()
            input_db = 10 * torch.log10((ex + eps) / (eref + eps))
            if not bool(input_db.detach() > -50.0):
                eligible = False
                break
            deltas.append(10 * torch.log10((ex + eps) / (ey + eps)))
        if eligible and len(deltas) == 2:
            terms.append((torch.nn.functional.relu(3.0 - deltas[0]) / 3.0).square() +
                         (torch.nn.functional.relu(2.0 - deltas[1]) / 3.0).square())
    if not terms:
        return pred.sum() * 0.0, 0
    return torch.stack(terms).mean(), len(terms)


def validate_tail_loss_reference() -> None:
    x = torch.zeros(1, 32_000)
    clean = torch.zeros_like(x)
    clean[:, 4_000:11_200] = .2
    tail_ref = clean.clone()
    x[:, 14_400:21_600] = .02
    pred = x.clone().requires_grad_(True)
    value, eligible = tail_loss(pred, x, clean, tail_ref,
        torch.tensor([12_000]), ("rir_only",))
    expected = 1 + 4 / 9
    if eligible != 1 or abs(float(value.detach()) - expected) > 1e-6:
        raise RuntimeError("G tail loss must preserve the frozen 13/9 y=x sanity value")
    grad, = torch.autograd.grad(value, pred)
    if not torch.isfinite(grad).all() or float(grad[0, 14_400:21_600].mean()) <= 0:
        raise RuntimeError("G tail gradient does not penalize residual tail energy")


def train_one(variant: str, args, init_state: dict, init_sha: str,
              data_manifest: dict, rows: list[dict]) -> Path:
    target_name = "c" if variant == "G-clean" else "hybrid"
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
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=args.num_workers,
        pin_memory=device.type == "cuda", drop_last=True, generator=generator)
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
        "model_source_sha256": hashlib.sha256((HERE / "model_e.py").read_bytes()).hexdigest(),
        "initialization_sha256": init_sha,
        "training_data_manifest_sha256": sha(args.data_dir / "training-data-manifest.json"),
        "training_pair_manifest_sha256": sha(args.data_dir / "pairs.jsonl"),
        "coverage_audit_sha256": sha(args.data_dir / "coverage-audit.json"),
        "split_manifest_sha256": sha(args.experiment_dir / "splits/manifest.json"),
        "seed": args.seed, "optimizer": "AdamW(lr=2e-4, weight_decay=1e-4)",
        "steps": args.steps, "batch_size": 1, "tail_loss_weight": 1.0,
        "target": "c_q dry clean" if variant == "G-clean" else "frozen weak/onset hybrid",
        "tail_ref": "a_q = Cap60(clean_scaled)*common_gain, independent of target",
        "shared_input_hashes": [row["x_sha256"] for row in rows],
        "shared_clean_truth_hashes": [row["clean_target_sha256"] for row in rows],
        "shared_cap60_reference_hashes": [row["cap_target_sha256"] for row in rows],
        "tail_ref_hashes": [row["tail_ref_sha256"] for row in rows],
        "test_wav_accessed": False}
    (output / "model-config.json").write_text(json.dumps(config, indent=2) + "\n")
    training_manifest = {**config, "device": str(device), "torch": torch.__version__,
        "train_kind_counts": dataset.kind_counts,
        "batch_order_policy": "same DataLoader RandomSampler, isolated torch.Generator, identical seed for both fits",
        "quiet_loss_weight": .5, "noise_aux_weight": .5,
        "noise_aux_every_n_steps": 4,
        "tail_eligibility": "both Cap60 input tail windows strictly > -50 dB relative to independent a_q reference; raw-clean activity mask",
        "checkpoint_policy": "fixed final step; both fits complete before DEV is opened"}
    (output / "training-manifest.json").write_text(json.dumps(training_manifest, indent=2) + "\n")

    iterator = iter(loader)
    aux_i = {"fan20": 0, "fan10": 0}
    order: list[int] = []
    log_path = output / "training.jsonl"
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        try:
            x, target, clean, tail_ref, kinds, pauses, indices = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, target, clean, tail_ref, kinds, pauses, indices = next(iterator)
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
        tail_value, eligible_count = tail_loss(pred, x, clean, tail_ref, pauses, kinds)
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
                "noise_aux_loss": auxiliary_value, "tail_loss": float(tail_value.detach()),
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
    if coverage.get("pair_manifest_sha256") != sha(pair_path) or not coverage.get("executable") or coverage.get("resolved_tail_slots") != 48:
        raise RuntimeError("G data coverage gate is not complete and hash-matched")
    if (not splits.get("frozen_before_training") or splits.get("opened_for_metrics") or
            splits.get("dev_speaker_count") != 16 or splits.get("holdout_speaker_count") != 12):
        raise RuntimeError("fresh frozen 16/12 G speaker splits are required")
    rows = [json.loads(line) for line in pair_path.read_text().splitlines() if line.strip()]
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
                not split_doc.get("frozen_before_training") or
                split_doc.get("opened_for_metrics") or
                split_doc.get("test_wav_accessed") is not False or ids & train_ids):
            raise RuntimeError(f"invalid or overlapping frozen G {name} speaker manifest")
        split_ids[name] = ids
    if split_ids["dev"] & split_ids["sealed"]:
        raise RuntimeError("G DEV and HOLDOUT-G speaker manifests overlap")
    tail_rows = [row for row in rows if row["kind"] == "rir_only"]
    tail_slots = sorted(int(row["rir_recipe"].get("tail_slot", -1)) for row in tail_rows)
    if tail_slots != list(range(48)):
        raise RuntimeError("G training tailbank slots must be the exact frozen 0..47 set")
    for row in tail_rows:
        recipe = row["rir_recipe"]
        levels = recipe.get("input_tail_db", {})
        history = recipe.get("candidate_history", [])
        first_valid = bool(history and history[-1].get("eligible_from_input_only") and
            [int(item["candidate_index"]) for item in history] ==
            list(range(int(recipe.get("candidate_index", -1)) + 1)) and
            not any(item.get("eligible_from_input_only") for item in history[:-1]))
        if (not row.get("input_only_tail_eligible") or
                not recipe.get("input_only_selection") or
                not first_valid or
                not 0 <= int(recipe.get("candidate_index", -1)) < 32 or
                not 1.65 <= float(recipe.get("t60_s", 0)) <= 1.85 or
                not -14.5 <= float(recipe.get("direct_to_reverb_db", 99)) <= -12.5 or
                not all(float(levels.get(key, -100)) > -50
                        for key in ("150_300", "300_600"))):
            raise RuntimeError(f"invalid input-only strong-RIR gate at training row {row['index']}")
    for name, expected_pairs in (("dev", 64), ("sealed", 48)):
        split_doc = json.loads((args.experiment_dir / f"splits/{name}/manifest.json").read_text())
        if sum(len(speaker["pairs"]) for speaker in split_doc["speakers"]) != expected_pairs:
            raise RuntimeError(f"incomplete frozen G {name} pair coverage")
        for speaker in split_doc["speakers"]:
            if len(speaker["pairs"]) != 4:
                raise RuntimeError(f"frozen G {name} requires exactly four RIRs per speaker")
            for pair in speaker["pairs"]:
                recipe = pair["procedural_rir"]
                levels = recipe.get("input_tail_db", {})
                history = pair.get("input_tail_selection_history", [])
                first_valid = bool(history and history[-1].get("input_only_eligible") and
                    [int(item["candidate_index"]) for item in history] ==
                    list(range(int(recipe.get("candidate_index", -1)) + 1)) and
                    not any(item.get("input_only_eligible") for item in history[:-1]))
                if (not recipe.get("input_only_selection") or
                        not first_valid or
                        not 0 <= int(recipe.get("candidate_index", -1)) < 32 or
                        not 1.65 <= float(recipe.get("t60_s", 0)) <= 1.85 or
                        not -14.5 <= float(recipe.get("direct_to_reverb_db", 99)) <= -12.5 or
                        not all(float(levels.get(key, -100)) > -50
                                for key in ("150_300", "300_600"))):
                    raise RuntimeError(f"invalid full-exact input-only {name} RIR recipe")
    for row in rows:
        for field, hash_field in (("x", "x_sha256"), ("c", "clean_target_sha256"),
                ("a", "cap_target_sha256"), ("hybrid", "hybrid_target_sha256"),
                ("tail_ref", "tail_ref_sha256")):
            if sha(args.data_dir / row[field]) != row[hash_field]:
                raise RuntimeError(f"G array hash mismatch at row {row['index']}:{field}")
        if row["tail_ref_sha256"] != row["cap_target_sha256"]:
            raise RuntimeError(f"tail reference is not a_q at row {row['index']}")
    if args.steps != 3000:
        raise RuntimeError("G protocol is fixed at exactly 3,000 optimizer updates per fit")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(args.cpu_threads)
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
    for variant in ("G-clean", "G-hybrid"):
        paths.append(str(train_one(variant, args, init_state, initial_sha,
                                   data_manifest, rows)))
    configs = [json.loads((args.experiment_dir / variant / "model-config.json").read_text())
               for variant in ("G-clean", "G-hybrid")]
    orders = [np.load(args.experiment_dir / variant / "batch-order.npy", allow_pickle=False)
              for variant in ("G-clean", "G-hybrid")]
    if configs[0]["initialization_sha256"] != configs[1]["initialization_sha256"] or not np.array_equal(*orders):
        raise RuntimeError("matched G fits differ in initialization or batch order")
    for field in ("training_data_manifest_sha256", "training_pair_manifest_sha256",
            "shared_input_hashes", "shared_clean_truth_hashes",
            "shared_cap60_reference_hashes", "tail_ref_hashes"):
        if configs[0][field] != configs[1][field]:
            raise RuntimeError(f"G-clean/G-hybrid shared field differs: {field}")
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
