#!/usr/bin/env python3
"""Prepare matched G-early-clean/G-early-hybrid pairs with exact Cap60 inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts/benchmarks/universal_enhancer"))

from data import make_procedural_rir  # noqa: E402
from prepare_cap60_conditioned_pairs import (  # noqa: E402
    BUT_ROOT, CROP_SAMPLES, DEEP_FILTER, LIBRISPEECH, SR, cap60_one,
    make_fan_mix, sha256,
)
from prepare_f2_tailbank import (  # noqa: E402
    cap60_cached, crop, index_cap60_cache, resolve_source, tail_input_levels,
)
from pilot_g2_feasibility import active_dry_levels, hash_seed  # noqa: E402
from pilot_g_early_feasibility import RT60_RANGE, DRR_RANGE  # noqa: E402
from screen_measured_rirs import prepare_rir, select_rirs  # noqa: E402
from train_measured_mix import measured_pair  # noqa: E402

SOURCE_DATA = ROOT / ".tools/compact-dereverb/f-pause-2026-10-10/data"
DEFAULT_OUT = ROOT / ".tools/compact-dereverb/g-early-2026-10-10/data"
TAIL_T60 = (1.65, 1.85)
TAIL_DRR = (-14.5, -12.5)
TAIL_LIMIT = 32
TAIL_ABSOLUTE_FLOOR_DB = -60.0
TAIL_RELATIVE_MARGIN_DB = 6.0
FRAME = 320
HOP = 160
FADE_SAMPLES = 160


def seed_for(slot: int, candidate: int) -> int:
    return hash_seed(f"G-early-train-v1|{slot}|{candidate}")


def candidate_parameters(seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    return (float(rng.uniform(*RT60_RANGE)), float(rng.uniform(*DRR_RANGE)))


def source_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def hybrid_mask(clean: np.ndarray) -> tuple[np.ndarray, dict]:
    """Sample mask: weak/onset frames stay at one, with 10ms cosine fades outside."""
    clean = np.asarray(clean, np.float32)
    if clean.ndim != 1 or clean.size < FRAME:
        raise ValueError("clean must be mono and at least 20ms long")
    frames = np.lib.stride_tricks.sliding_window_view(clean, FRAME)[::HOP]
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-24)
    peak = max(float(rms.max()), 1e-12)
    active = rms > .02 * peak
    weak = active & (rms < .35 * peak)
    previous = np.r_[0.0, rms[:-1]]
    onset = active & (rms > 1.5 * previous)
    protected = weak | onset
    dilated = protected.copy()
    for shift in (1, 2):
        dilated[shift:] |= protected[:-shift]
        dilated[:-shift] |= protected[shift:]
    sample_regions = np.zeros(clean.size, dtype=bool)
    frame_ids = np.minimum(np.arange(clean.size) // HOP, len(dilated) - 1)
    sample_regions[:] = dilated[frame_ids]
    # Include all samples in each selected analysis frame, not only the hop cell.
    for frame_idx in np.flatnonzero(dilated):
        start = frame_idx * HOP
        sample_regions[start:min(start + FRAME, clean.size)] = True
    indices = np.flatnonzero(sample_regions)
    mask = np.zeros(clean.size, dtype=np.float64)
    mask[sample_regions] = 1.0
    if indices.size:
        # Cosine fades are applied outside the protected region; no selected
        # weak/onset frame is ever blended away from the dry target.
        left, right = int(indices[0]), int(indices[-1]) + 1
        for distance in range(1, FADE_SAMPLES + 1):
            weight = .5 * (1.0 + np.cos(np.pi * distance / (FADE_SAMPLES + 1)))
            li, ri = left - distance, right - 1 + distance
            if li >= 0 and not sample_regions[li]:
                mask[li] = max(mask[li], weight)
            if ri < clean.size and not sample_regions[ri]:
                mask[ri] = max(mask[ri], weight)
        # Separate protected runs need their own outside transitions.
        starts = np.flatnonzero(sample_regions & ~np.r_[False, sample_regions[:-1]])
        ends = np.flatnonzero(sample_regions & ~np.r_[sample_regions[1:], False]) + 1
        for start, end in zip(starts, ends):
            for distance in range(1, FADE_SAMPLES + 1):
                weight = .5 * (1.0 + np.cos(np.pi * distance / (FADE_SAMPLES + 1)))
                li, ri = int(start) - distance, int(end) - 1 + distance
                if li >= 0 and not sample_regions[li]:
                    mask[li] = max(mask[li], weight)
                if ri < clean.size and not sample_regions[ri]:
                    mask[ri] = max(mask[ri], weight)
    mask[sample_regions] = 1.0
    if not np.isfinite(mask).all() or mask.min() < 0 or mask.max() > 1:
        raise RuntimeError("hybrid mask is outside [0,1]")
    return mask.astype(np.float32), {
        "frame_ms": 20, "hop_ms": 10, "active_fraction_threshold": .02,
        "weak_fraction_threshold": .35, "onset_ratio": 1.5,
        "dilation_ms": 20, "fade_ms": 10,
        "active_frames": int(active.sum()), "weak_frames": int(weak.sum()),
        "onset_frames": int(onset.sum()),
        "protected_sample_fraction": float(sample_regions.mean()),
    }


def load_fixed_rows() -> list[dict]:
    path = SOURCE_DATA / "pairs.jsonl"
    manifest = json.loads((SOURCE_DATA / "training-data-manifest.json").read_text())
    if manifest.get("test_wav_accessed") is not False:
        raise RuntimeError("source F-pause dataset violates sealed test.wav policy")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    counts = {kind: sum(row["kind"] == kind for row in rows)
              for kind in ("identity", "rir_only", "fan20", "fan10")}
    if len(rows) != 128 or counts != {"identity": 32, "rir_only": 48,
                                     "fan20": 24, "fan10": 24}:
        raise RuntimeError(f"fixed training schedule mismatch: {counts}")
    if len({row["speaker"] for row in rows}) != 128:
        raise RuntimeError("fixed training schedule must have 128 unique speakers")
    return rows


def save_audio(path: Path, value: np.ndarray) -> str:
    value = np.asarray(value, dtype=np.float32)
    if value.shape != (CROP_SAMPLES,) or not np.isfinite(value).all():
        raise RuntimeError(f"invalid training audio array: {path}")
    np.save(path, value, allow_pickle=False)
    return sha256(path)


def prepare(args) -> dict:
    if not args.deep_filter.is_file():
        raise FileNotFoundError(args.deep_filter)
    rows = load_fixed_rows()
    c_rows = [json.loads(line) for line in
              (ROOT / ".tools/compact-dereverb/cap60-conditioned-2026-10-10/pairs.jsonl")
              .read_text().splitlines() if line.strip()]
    if [(r["speaker"], r["kind"], r["source_sha256"]) for r in rows] != [
            (r["speaker"], r["kind"], r["source_sha256"]) for r in c_rows]:
        raise RuntimeError("pause training schedule no longer matches frozen C source schedule")
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {args.output_dir}; use --resume")
    data_dir = args.output_dir
    arrays = data_dir / "arrays"
    cache = data_dir / "cap60-cache"
    work = data_dir / "cap60-work"
    for path in (arrays, cache, work):
        path.mkdir(parents=True, exist_ok=True)
    manifest_path = data_dir / "pairs.jsonl"
    if manifest_path.exists():
        raise FileExistsError("completed G data manifest already exists; immutable output")
    progress_path = data_dir / "progress.json"
    progress = json.loads(progress_path.read_text()) if args.resume and progress_path.exists() else {
        "name": "G-target-ablation-data-progress-v1", "rows": [],
        "source_schedule_sha256": sha256(SOURCE_DATA / "pairs.jsonl"),
        "deep_filter_sha256": sha256(args.deep_filter), "test_wav_accessed": False}
    if (progress.get("source_schedule_sha256") != sha256(SOURCE_DATA / "pairs.jsonl") or
            progress.get("deep_filter_sha256") != sha256(args.deep_filter) or
            progress.get("test_wav_accessed") is not False):
        raise RuntimeError("partial G cache belongs to a changed source schedule/binary")
    out_rows = progress.get("rows", [])
    if [int(row["index"]) for row in out_rows] != list(range(len(out_rows))):
        raise RuntimeError("partial G data cache is not a contiguous schedule prefix")
    if len(out_rows) > len(rows):
        raise RuntimeError("partial G cache includes unexpected rows")
    rir_index = {item["configuration"]: item for item in select_rirs(BUT_ROOT)}
    reuse = index_cap60_cache(SOURCE_DATA / "cap60-cache")
    started = time.perf_counter()
    tail_stats: list[dict] = list(progress.get("resolved_tail_slots", []))
    if len(tail_stats) != sum(row["kind"] == "rir_only" for row in out_rows):
        raise RuntimeError("partial G cache tailbank audit does not match completed rows")

    for row in rows[len(out_rows):]:
        index = int(row["index"])
        clean, pause_meta, source_paths = resolve_source(row, LIBRISPEECH / "train-clean-100")
        kind = row["kind"]
        pause = row["source"].get("pause_start_in_crop")
        crop_start = int(row["crop_start_sample"])
        if kind == "rir_only":
            crop_start = int(pause_meta["clip_relative_start_sample"]) - int(.75 * SR)
            pause = int(pause_meta["clip_relative_start_sample"]) - crop_start
        if crop_start < 0 or crop_start + CROP_SAMPLES > len(clean):
            raise RuntimeError(f"fixed crop for G row {index} falls outside its source")
        clean_crop = crop(clean, crop_start)
        gain = float(row["post_cap_gain"])
        branch_scale = float(row.get("shared_branch_scale", 1.0))
        selected = None
        base_control = None
        history: list[dict] = []

        if kind == "identity":
            clean_branch = clean
            wet = clean
            cap_input, cap_in_meta = cap60_cached(args.deep_filter, wet, work,
                f"g-input-{index:03d}", cache, reuse)
            cap_target, cap_target_meta = cap_input, cap_in_meta
            chosen_recipe = {"kind": "identity"}
        elif kind == "rir_only":
            tail_slot = len(tail_stats)
            for candidate in range(TAIL_LIMIT):
                seed = hash_seed(
                    f"G-early-train-v1|{row['speaker']}|{row['index']}|{candidate}")
                t60_s, drr_db = candidate_parameters(seed)
                rir, _early, _late = make_procedural_rir(SR, seed, t60_s, drr_db,
                    max_t60_s=2.0, min_direct_to_reverb_db=-20.0)
                pair = measured_pair(clean, rir)
                in_key = f"g-tail-input-{index:03d}-candidate-{candidate:02d}"
                target_key = f"g-tail-clean-{index:03d}-candidate-{candidate:02d}"
                cap_input, cap_in_meta = cap60_cached(args.deep_filter,
                    pair["reverberant"], work, in_key, cache, reuse)
                cap_target, cap_target_meta = cap60_cached(args.deep_filter,
                    pair["clean"], work, target_key, cache, reuse)
                pause_start = int(pause_meta["clip_relative_start_sample"])
                measured = tail_input_levels(
                    crop(cap_target, crop_start) * gain,
                    crop(cap_input, crop_start) * gain,
                    int(pause) if pause is not None else pause_start - crop_start,
                    activity_clean_crop=clean_crop)
                valid_ref = bool(measured.get("reference_valid"))
                windows = {}
                if valid_ref:
                    dry_levels = active_dry_levels(
                        crop(cap_target, crop_start) * gain,
                        int(pause) if pause is not None else pause_start - crop_start,
                        float(measured["reference_energy"]))
                    for band in ("150_300", "300_600"):
                        lx, ldry = float(measured["window_db"][band]), float(dry_levels[band])
                        threshold = max(TAIL_ABSOLUTE_FLOOR_DB,
                                        ldry + TAIL_RELATIVE_MARGIN_DB)
                        windows[band] = {"lx_db": lx, "ldry_db": ldry,
                            "threshold_db": threshold, "passes": lx > threshold}
                else:
                    windows = {band: {"lx_db": None, "ldry_db": None,
                        "threshold_db": None, "passes": False}
                        for band in ("150_300", "300_600")}
                eligible = bool(valid_ref and windows["150_300"]["passes"])
                history.append({"candidate_index": candidate, "seed": seed,
                    "t60_s": t60_s, "drr_db": drr_db,
                    "reference_valid": valid_ref, "windows": windows,
                    "eligible_from_input_only": eligible})
                candidate_tuple = (pair, cap_input, cap_in_meta, cap_target,
                    cap_target_meta, seed, t60_s, drr_db, candidate, measured, windows)
                if candidate == 0:
                    base_control = candidate_tuple
                if eligible:
                    selected = candidate_tuple
                    break
            if selected is None:
                if len(history) != TAIL_LIMIT or base_control is None:
                    raise RuntimeError("BASE_ONLY_CONTROL requires 32 candidates and saved j=0")
                selected = base_control
                tail_eligible, classification = False, "BASE_ONLY_CONTROL"
            else:
                tail_eligible, classification = True, "TAIL_ELIGIBLE"
            (pair, cap_input, cap_in_meta, cap_target, cap_target_meta,
             seed, t60_s, drr_db, candidate, measured, windows) = selected
            clean_branch = pair["clean"]
            wet = pair["reverberant"]
            chosen_recipe = {"tail_slot": tail_slot, "seed": seed,
                "t60_s": t60_s, "direct_to_reverb_db": drr_db,
                "candidate_index": candidate, "candidates_tested": len(history),
                "windows": windows, "classification": classification,
                "tail_loss_enabled": tail_eligible, "input_only_selection": True,
                "candidate_history": history}
            tail_stats.append({"index": index, "resolved": True,
                "candidate_index": candidate, "classification": classification,
                "tail_loss_enabled": tail_eligible, "windows": windows})
        else:
            if row["rir_kind"] == "procedural":
                recipe = row["rir_recipe"]
                rir, _early, _late = make_procedural_rir(SR,
                    int(recipe["seed"]), float(recipe["t60_s"]),
                    float(recipe["direct_to_reverb_db"]))
            elif row["rir_kind"] == "but_measured":
                recipe = row["rir_recipe"]
                item = rir_index.get(recipe["configuration"])
                if item is None or sha256(Path(item["path"])) != recipe["rir_sha256"]:
                    raise RuntimeError(f"frozen measured RIR hash mismatch at row {index}")
                rir, _ = prepare_rir(item)
            else:
                raise RuntimeError(f"unexpected fixed schedule RIR kind: {row['rir_kind']}")
            pair = measured_pair(clean, rir)
            clean_branch = (pair["clean"] * np.float32(branch_scale)).astype(np.float32)
            wet = (make_fan_mix(clean, pair["reverberant"], int(row["noise_seed"]),
                20 if kind == "fan20" else 10) * np.float32(branch_scale)).astype(np.float32)
            cap_input, cap_in_meta = cap60_cached(args.deep_filter, wet, work,
                f"g-input-{index:03d}", cache, reuse)
            cap_target, cap_target_meta = cap60_cached(args.deep_filter,
                clean_branch, work, f"g-clean-cap60-{index:03d}", cache, reuse)
            chosen_recipe = row["rir_recipe"]

        x_crop = (crop(cap_input, crop_start) * gain).astype(np.float32)
        clean_q = (crop(clean_branch, crop_start) * gain).astype(np.float32)
        cap_q = (crop(cap_target, crop_start) * gain).astype(np.float32)
        mask, mask_meta = hybrid_mask(clean_crop)
        hybrid_q = (mask * clean_q + (1.0 - mask) * cap_q).astype(np.float32)
        stem = f"{index:03d}"
        files = {}
        for key, audio in (("x", x_crop), ("c", clean_q), ("a", cap_q),
                           ("hybrid", hybrid_q), ("mask", mask), ("tail_ref", cap_q)):
            rel = f"arrays/{key}-{stem}.npy"
            files[key] = rel
            save_audio(arrays / f"{key}-{stem}.npy", audio)
        out = {"index": index, "kind": kind, "speaker": str(row["speaker"]),
            "source_pair": row["source"]["source_pair"],
            "source_sha256": row["source_sha256"], "crop_start_sample": crop_start,
            "pause_start_in_crop": pause, "post_cap_gain": gain,
            "shared_branch_scale": branch_scale, "rir_kind": row.get("rir_kind"),
            "rir_recipe": chosen_recipe, **files,
            "x_sha256": sha256(arrays / f"x-{stem}.npy"),
            "clean_target_sha256": sha256(arrays / f"c-{stem}.npy"),
            "cap_target_sha256": sha256(arrays / f"a-{stem}.npy"),
            "hybrid_target_sha256": sha256(arrays / f"hybrid-{stem}.npy"),
            "tail_ref_sha256": sha256(arrays / f"tail_ref-{stem}.npy"),
            "mask_sha256": sha256(arrays / f"mask-{stem}.npy"),
            "mask_definition": mask_meta, "cap60_input": cap_in_meta,
            "cap60_clean_reference": cap_target_meta,
            "input_only_tail_eligible": (kind == "rir_only" and
                chosen_recipe.get("tail_loss_enabled", False))}
        out_rows.append(out)
        progress.update({"rows": out_rows, "resolved_tail_slots": tail_stats,
            "test_wav_accessed": False})
        progress_tmp = progress_path.with_suffix(".json.tmp")
        progress_tmp.write_text(json.dumps(progress, indent=2, allow_nan=False) + "\n")
        progress_tmp.replace(progress_path)
        print(json.dumps({"row": index, "kind": kind,
            "resolved_tail_slots": len(tail_stats),
            "candidate_index": chosen_recipe.get("candidate_index"),
            "elapsed_s": time.perf_counter() - started}), flush=True)

    tail_rows = [row for row in out_rows if row["kind"] == "rir_only"]
    eligible_tail_rows = [row for row in tail_rows if row["input_only_tail_eligible"]]
    if len(out_rows) != 128 or len(tail_rows) != 48:
        raise RuntimeError("G training data coverage must be exactly 128 rows and 48 tail rows")
    eligible_speakers = {str(row["speaker"]) for row in eligible_tail_rows}
    train_coverage_pass = len(eligible_tail_rows) >= 36
    pair_path = data_dir / "pairs.jsonl"
    tmp = pair_path.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in out_rows))
    tmp.replace(pair_path)
    manifest = {"name": "G-early-shared-training-data-v1",
        "pair_manifest_sha256": sha256(pair_path),
        "source_schedule_sha256": sha256(SOURCE_DATA / "pairs.jsonl"),
        "source_manifest_sha256": sha256(SOURCE_DATA / "training-data-manifest.json"),
        "deep_filter_sha256": sha256(args.deep_filter), "sample_rate": SR,
        "crop_samples": CROP_SAMPLES, "row_count": 128,
        "kind_counts": {kind: sum(row["kind"] == kind for row in out_rows)
                         for kind in ("identity", "rir_only", "fan20", "fan10")},
        "tailbank": {"slots": len(tail_rows), "eligible_slots": len(eligible_tail_rows),
            "base_only_control_slots": len(tail_rows) - len(eligible_tail_rows),
            "required_slots": 48, "minimum_eligible_slots": 36,
            "candidate_limit": TAIL_LIMIT, "namespace": "G-early-train-v1|speaker|row|candidate",
            "rt60_s_uniform": list(RT60_RANGE), "drr_db_uniform": list(DRR_RANGE),
            "selection": "first exact Cap60 candidate j=0..31 with Lx(W1)>max(-60dB,Ldry(W1)+6dB); if none, use j=0 with lambda_early=0",
            "input_only": True},
        "targets": {"G-clean": "c_q = shared_peak_scaled_clean * post_cap_gain",
            "G-hybrid": "mask*c_q + (1-mask)*a_q; masks derived only from raw clean",
            "tail_ref": "a_q = Cap60(shared_peak_scaled_clean)*post_cap_gain, identical in both fits"},
        "train_eligible_speakers": len(eligible_speakers),
        "train_coverage_gate": {"minimum_eligible_slots": 36,
            "passed": train_coverage_pass},
        "test_wav_accessed": False}
    manifest_path = data_dir / "training-data-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    coverage = {"pair_manifest_sha256": sha256(pair_path),
        "resolved_tail_slots": len(tail_rows), "required_tail_slots": 48,
        "eligible_tail_slots": len(eligible_tail_rows), "minimum_eligible_tail_slots": 36,
        "executable": train_coverage_pass, "test_wav_accessed": False}
    (args.output_dir / "coverage-audit.json").write_text(
        json.dumps(coverage, indent=2, allow_nan=False) + "\n")
    progress_path.unlink(missing_ok=True)
    return {"rows": len(out_rows), "tail_slots": len(tail_rows),
        "eligible_tail_slots": len(eligible_tail_rows), "executable": train_coverage_pass,
        "pair_manifest_sha256": manifest["pair_manifest_sha256"],
        "test_wav_accessed": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deep-filter", type=Path, default=DEEP_FILTER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args), indent=2))


if __name__ == "__main__":
    main()
