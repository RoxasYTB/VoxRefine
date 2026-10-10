#!/usr/bin/env python3
"""Train/evaluate a compact FLEURS-only 16 kHz enhancement student."""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from multilingual_student import (
    DATA, RATE, MultilingualStudent, corrupt, group_by_locale, load_manifest,
    make_noise_bank, normalize_pair, parameter_count, read_crop, records_for_split,
)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = ROOT / "results/multilingual-student-fleurs-16k-01"


def mrstft_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    values = []
    for n_fft, hop in ((256, 64), (512, 128), (1024, 256)):
        window = torch.hann_window(n_fft, dtype=pred.dtype, device=pred.device)
        p = torch.stft(pred, n_fft, hop, n_fft, window, center=True, return_complex=True).abs()
        t = torch.stft(target, n_fft, hop, n_fft, window, center=True, return_complex=True).abs()
        values.append(F.l1_loss(torch.log1p(10 * p), torch.log1p(10 * t)))
    return torch.stack(values).mean()


def si_sdr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target - target.mean(dim=-1, keepdim=True)
    pred = pred - pred.mean(dim=-1, keepdim=True)
    alpha = (pred * target).sum(dim=-1, keepdim=True) / target.square().sum(dim=-1, keepdim=True).clamp_min(1e-8)
    projection = alpha * target
    noise = pred - projection
    return 10 * torch.log10(projection.square().sum(dim=-1).clamp_min(1e-8) /
                            noise.square().sum(dim=-1).clamp_min(1e-8))


def balanced_sample(groups: dict[str, list[dict]], rng: random.Random) -> dict:
    locale = rng.choice(sorted(groups))
    return rng.choice(groups[locale])


def crop_example(row: dict, seconds: float, rng: random.Random,
                 noise_bank: list[torch.Tensor], device: torch.device,
                 mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    clean = read_crop(row, seconds, rng)
    if mode == "clean":
        noisy = clean.to(device)
    else:
        noisy = corrupt(clean, noise_bank, rng, device, mode=mode)
    noisy, target = normalize_pair(clean.to(device), noisy)
    return noisy[None], target[None]


@torch.no_grad()
def validate(model: MultilingualStudent, rows_by_locale: dict[str, list[dict]],
             noise_bank: list[torch.Tensor], device: torch.device, seconds: float,
             seed: int, clips_per_locale: int) -> dict:
    rng = random.Random(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model.eval()
    measures: dict[str, list[dict[str, float]]] = {m: [] for m in ("clean", "noise", "room", "mixed")}
    for locale_index, locale in enumerate(sorted(rows_by_locale)):
        rows = rows_by_locale[locale]
        locale_rng = random.Random(seed + locale_index * 1_000_003)
        selected = rows if len(rows) <= clips_per_locale else locale_rng.sample(rows, clips_per_locale)
        for row in selected:
            try:
                clean = read_crop(row, seconds, locale_rng).to(device)
            except (OSError, RuntimeError, ValueError):
                continue
            clean = clean / clean.square().mean().sqrt().clamp_min(1e-4)
            for mode_index, mode in enumerate(measures):
                mode_rng = random.Random(seed + locale_index * 1_000_003 + mode_index * 97_409)
                inp = clean if mode == "clean" else corrupt(clean, noise_bank, mode_rng, device, mode=mode)
                with torch.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=device.type == "cuda"):
                    pred = model(inp[None])[0]
                if not torch.isfinite(pred).all():
                    raise FloatingPointError(f"Non-finite output in validation: {locale}/{mode}")
                measures[mode].append({
                    "si_sdr_db": float(si_sdr(pred[None], clean[None]).item()),
                    "wave_l1": float((pred - clean).abs().mean().item()),
                    "mrstft_log_l1": float(mrstft_loss(pred[None], clean[None]).item()),
                    "peak": float(pred.abs().max().item()),
                })
    model.train()
    summaries = {}
    for mode, rows in measures.items():
        if not rows:
            raise RuntimeError(f"No FLEURS validation clips passed for condition {mode}")
        summaries[mode] = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
        summaries[mode]["clips"] = len(rows)
    summaries["locales"] = len(rows_by_locale)
    # Heavily protect clean speech: the product must remain close to identity when
    # the input is already usable. This is a model-selection score, not a MOS.
    summaries["selection_score"] = (
        (summaries["noise"]["mrstft_log_l1"] + summaries["room"]["mrstft_log_l1"] +
         summaries["mixed"]["mrstft_log_l1"]) / 3.0 +
        2.0 * summaries["clean"]["mrstft_log_l1"] +
        0.5 * summaries["clean"]["wave_l1"] +
        0.03 * max(0.0, summaries["mixed"]["peak"] - 1.0)
    )
    return summaries


def write_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--validation-clips-per-locale", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=1.5)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--max-vram-gib", type=float, default=2.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or args.validate_every < 1 or args.seconds < 0.75 or args.width < 4:
        parser.error("steps/validate-every must be positive; seconds >= 0.75 and width >= 4")
    args.output = args.output.resolve()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    manifest = load_manifest()
    train_rows = records_for_split(manifest, "train")
    dev_rows = records_for_split(manifest, "dev")
    train_groups = group_by_locale(train_rows)
    dev_groups = group_by_locale(dev_rows)
    absent_dev = sorted(set(train_groups) - set(dev_groups))
    if absent_dev:
        raise SystemExit(f"Missing speaker-disjoint FLEURS dev locales: {', '.join(absent_dev)}")
    summary = {
        "manifest_revision": manifest.get("revision"),
        "train_locales": len(train_groups), "train_clips": len(train_rows),
        "train_hours": round(sum(r["duration_seconds"] for r in train_rows) / 3600, 2),
        "dev_locales": len(dev_groups), "dev_clips": len(dev_rows),
        "dev_hours": round(sum(r["duration_seconds"] for r in dev_rows) / 3600, 2),
        "sample_rate_hz": RATE, "device": str(device),
    }
    print(json.dumps({"event": "data_ready", **summary}, indent=2), flush=True)
    if args.dry_run:
        return
    if args.output.exists() and not args.resume:
        raise SystemExit(f"Refusing to overwrite {args.output}; pass --resume")
    args.output.mkdir(parents=True, exist_ok=True)
    noise_bank = make_noise_bank()
    if not noise_bank:
        raise SystemExit("No CC0 noise sources decoded; refusing to train without real noise examples")
    model = MultilingualStudent(args.width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    latest = args.output / "latest.pt"
    start = 0
    if args.resume and latest.exists():
        state = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        start = int(state["step"])
    score_path = args.output / "metrics.jsonl"
    best_path = args.output / "best.pt"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    initial = validate(model, dev_groups, noise_bank, device, args.seconds,
                       args.seed + 1000, args.validation_clips_per_locale)
    print(json.dumps({"event": "baseline", "validation": initial,
                      "parameters": parameter_count(model),
                      "cuda_free_bytes": torch.cuda.mem_get_info(device)[0] if device.type == "cuda" else None}, indent=2), flush=True)
    best_score = float("inf")
    if best_path.exists():
        saved = torch.load(best_path, map_location="cpu", weights_only=False)
        best_score = float(saved.get("selection_score", best_score))
    rng = random.Random(args.seed)
    if start:
        rng.setstate(torch.load(latest, map_location="cpu", weights_only=False).get("python_rng_state", rng.getstate()))
    start_time = time.perf_counter()
    for step in range(start + 1, args.steps + 1):
        model.train()
        row = balanced_sample(train_groups, rng)
        try:
            clean = read_crop(row, args.seconds, rng)
        except (OSError, RuntimeError, ValueError):
            continue
        identity = (step % 5 == 0)
        if identity:
            noisy, target = normalize_pair(clean.to(device), clean.to(device))
        else:
            noisy = corrupt(clean, noise_bank, rng, device, mode="mixed")
            noisy, target = normalize_pair(clean.to(device), noisy)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == "cuda"):
            pred = model(noisy[None])
            wave_loss = F.l1_loss(pred, target[None])
            spectral_loss = mrstft_loss(pred, target[None])
            peak_loss = F.relu(pred.abs() - 0.98).square().mean()
            loss = wave_loss + 0.5 * spectral_loss + 2.0 * peak_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if step % args.validate_every == 0 or step == args.steps:
            validation = validate(model, dev_groups, noise_bank, device, args.seconds,
                                  args.seed + 1000, args.validation_clips_per_locale)
            peak_vram = (torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0)
            peak_gib = peak_vram / 1024**3
            row_log = {"step": step, "train_loss": float(loss.detach()),
                       "locale": row["locale"], "identity_batch": identity,
                       "validation": validation, "peak_allocated_vram_gib": peak_gib,
                       "elapsed_seconds": time.perf_counter() - start_time}
            with score_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row_log) + "\n")
            state = {"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "scaler": scaler.state_dict(), "selection_score": validation["selection_score"],
                     "validation": validation, "python_rng_state": rng.getstate(),
                     "config": {**vars(args), "output": str(args.output)},
                     "dataset": "FLEURS CC BY 4.0 train/dev only", "sample_rate_hz": RATE}
            torch.save(state, latest)
            if validation["selection_score"] < best_score:
                best_score = validation["selection_score"]
                torch.save(state, best_path)
            print(json.dumps({"event": "validation", **row_log, "best_score": best_score}), flush=True)
            if peak_gib > args.max_vram_gib:
                raise RuntimeError(f"Peak allocated VRAM {peak_gib:.3f} GiB exceeds --max-vram-gib={args.max_vram_gib}")
    card = {
        "name": "VoxRefine multilingual 16 kHz student (research checkpoint)",
        "architecture": f"Complex STFT residual U-Net, width={args.width}",
        "parameters": parameter_count(model), "sample_rate_hz": RATE, "languages": sorted(train_groups),
        "training_data": "FLEURS official train split only; dev is validation; test remains unused",
        "data_license": manifest.get("license"), "data_license_url": manifest.get("license_url"),
        "citation": manifest.get("citation"), "dataset_revision": manifest.get("revision"),
        "training_procedure": "Synthetic CC0 noise, synthetic room responses, moderate low-band capture and clipping; clean identity batches included.",
        "limitations": ["16 kHz output cannot restore true content above 8 kHz.",
                        "FLEURS is read speech and does not represent all conversational accents/noise/rooms.",
                        "A validation score is not a perceptual or Adobe-equivalence claim."],
        "checkpoint_selection_score": best_score,
    }
    write_json(args.output / "model-card.json", card)
    print(json.dumps({"event": "complete", "best_checkpoint": str(best_path),
                      "model_card": str(args.output / "model-card.json"),
                      "peak_vram_gib": torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None}, indent=2), flush=True)


if __name__ == "__main__":
    main()
