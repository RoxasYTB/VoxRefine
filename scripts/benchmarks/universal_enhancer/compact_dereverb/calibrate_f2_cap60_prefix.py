#!/usr/bin/env python3
"""Compare full-utterance Cap60 against its causal prefix for F2 selection."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts/benchmarks/universal_enhancer"))

from data import make_procedural_rir  # noqa: E402
from prepare_cap60_conditioned_pairs import (  # noqa: E402
    DEEP_FILTER, GUARD_SAMPLES, LIBRISPEECH, SR, cap60_one, load_audio, to_16k,
)
from prepare_f2_tailbank import (  # noqa: E402
    C_DATA, candidate_parameters, crop, load_schedule, resolve_source, seed_for,
    tail_input_levels,
)
from train_measured_mix import measured_pair  # noqa: E402

DEFAULT_OUT = ROOT / ".tools/compact-dereverb/e2-f2-tailbank-2026-10-10/prefix-calibration"
DEFAULT_CANDIDATES = (0, 1, 3, 7, 15, 31, 63, 127)


def hash_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def cap60_many(executable: Path, arrays: dict[str, np.ndarray], work: Path,
               key: str, exact_cache_dir: Path | None = None) -> tuple[dict[str, np.ndarray], float]:
    """Run same Cap60 CLI once for multiple independent files."""
    work.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    outputs: dict[str, np.ndarray] = {}
    pending: dict[str, np.ndarray] = {}
    pending_hashes: dict[str, str] = {}
    if exact_cache_dir is not None:
        exact_cache_dir.mkdir(parents=True, exist_ok=True)
    for name, audio in arrays.items():
        canon = to_16k(np.asarray(audio, dtype=np.float32), SR)
        source_hash = hash_array(canon)
        audio_path = (exact_cache_dir / f"{source_hash}.npy"
                      if exact_cache_dir is not None else None)
        meta_path = (exact_cache_dir / f"{source_hash}.json"
                     if exact_cache_dir is not None else None)
        if audio_path is not None and audio_path.is_file() and meta_path.is_file():
            meta = json.loads(meta_path.read_text())
            if (meta.get("source_sha256") == source_hash and
                    meta.get("deep_filter_sha256") == hashlib.sha256(executable.read_bytes()).hexdigest() and
                    meta.get("sample_rate") == SR and meta.get("sample_count") == len(canon)):
                outputs[name] = np.load(audio_path, allow_pickle=False).astype(np.float32)
                continue
        pending[name] = canon
        pending_hashes[name] = source_hash
    if not pending:
        return outputs, 0.0
    with tempfile.TemporaryDirectory(prefix="cap60-prefix-cal-") as td:
        td_path = Path(td)
        out_dir = td_path / "out"
        inputs: dict[str, np.ndarray] = {}
        input_paths = []
        for name, canon in pending.items():
            if not len(canon):
                raise ValueError(f"empty signal for {name}")
            guarded = np.pad(canon, (0, GUARD_SAMPLES)).astype(np.float32)
            in_path = td_path / f"{name}.wav"
            sf.write(in_path, guarded, SR, subtype="PCM_16")
            inputs[name] = canon
            input_paths.append(in_path)
        proc = subprocess.run([str(executable), "--atten-lim-db", "60",
            "--compensate-delay", "-o", str(out_dir), *map(str, input_paths)],
            check=True, capture_output=True, text=True)
        for name, canon in inputs.items():
            out_path = out_dir / f"{name}.wav"
            if not out_path.exists():
                wavs = sorted(out_dir.glob(f"{name}*.wav"))
                if len(wavs) != 1:
                    raise RuntimeError(f"Cap60 output missing for {name}: {proc.stdout[-500:]}")
                out_path = wavs[0]
            audio, sr = load_audio(out_path)
            if sr != SR or len(audio) < len(canon):
                raise RuntimeError(f"invalid Cap60 output for {name}: {sr} Hz, {len(audio)} samples")
            audio = audio[:len(canon)].astype(np.float32)
            if not np.isfinite(audio).all() or np.max(np.abs(audio)) >= 1.0:
                raise RuntimeError(f"non-finite or clipped Cap60 output for {name}")
            outputs[name] = audio
            if exact_cache_dir is not None:
                source_hash = pending_hashes[name]
                np.save(exact_cache_dir / f"{source_hash}.npy", audio, allow_pickle=False)
                atomic_json(exact_cache_dir / f"{source_hash}.json", {
                    "source_sha256": source_hash,
                    "deep_filter_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
                    "sample_rate": SR, "sample_count": len(canon),
                    "guard_samples": GUARD_SAMPLES})
    return outputs, time.perf_counter() - started


def cap60_single_cached(executable: Path, audio: np.ndarray, work: Path,
                        cache_dir: Path, reuse_dirs: tuple[Path, ...] = ()) -> tuple[np.ndarray, float]:
    canon = to_16k(np.asarray(audio, dtype=np.float32), SR)
    source_hash = hash_array(canon)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dst_audio = cache_dir / f"{source_hash}.npy"
    dst_meta = cache_dir / f"{source_hash}.json"
    exe_hash = hashlib.sha256(executable.read_bytes()).hexdigest()
    if dst_audio.is_file() and dst_meta.is_file():
        meta = json.loads(dst_meta.read_text())
        if meta.get("source_sha256") == source_hash and meta.get("deep_filter_sha256") == exe_hash:
            return np.load(dst_audio, allow_pickle=False).astype(np.float32), 0.0
    # Reuse the solo outputs already computed by the batching audit when their
    # source bytes and Cap60 binary hash are an exact match.
    for reuse_dir in reuse_dirs:
        for meta_path in reuse_dir.glob("*.json"):
            try:
                meta = json.loads(meta_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            audio_path = meta_path.with_suffix(".npy")
            if (meta.get("source_sha256") == hashlib.sha256(canon.tobytes()).hexdigest() and
                    meta.get("sample_rate") == SR and audio_path.is_file()):
                shutil.copy2(audio_path, dst_audio)
                meta["deep_filter_sha256"] = exe_hash
                meta["sample_count"] = len(canon)
                atomic_json(dst_meta, meta)
                return np.load(dst_audio, allow_pickle=False).astype(np.float32), 0.0
    y, meta = cap60_one(executable, canon, SR, work, source_hash, cache_dir)
    meta["deep_filter_sha256"] = exe_hash
    meta["sample_count"] = len(canon)
    atomic_json(dst_meta, meta)
    return y, float(meta.get("elapsed_s", 0.0))


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def build_calibration_inputs(args, schedule_rows: list[dict], tail_slot: int,
                             candidate: int) -> dict[str, np.ndarray]:
    row = schedule_rows[tail_slot]
    clean, pause_meta, _ = resolve_source(row, args.data_root / "train-clean-100")
    global_pause = int(pause_meta["clip_relative_start_sample"])
    crop_start = global_pause - int(.75 * SR)
    prefix_end = crop_start + int(.75 * SR) + 9_600
    if prefix_end > len(clean):
        raise RuntimeError(f"prefix exceeds source at slot {tail_slot}")
    seed = seed_for(tail_slot, candidate)
    t60, drr = candidate_parameters(seed)
    rir, _, _ = make_procedural_rir(SR, seed, t60, drr)
    pair = measured_pair(clean, rir)
    return {"full_wet": pair["reverberant"], "full_clean": pair["clean"],
        "prefix_wet": pair["reverberant"][:prefix_end],
        "prefix_clean": pair["clean"][:prefix_end]}


def validate_batch_cli(args, schedule_rows: list[dict], out_dir: Path) -> dict:
    """Check 8 varied files plus first/last-order duplicate against solo runs."""
    selected = {}
    for slot, candidate in ((0, 0), (7, 127)):
        for name, signal in build_calibration_inputs(args, schedule_rows, slot, candidate).items():
            selected[f"s{slot}-j{candidate}-{name}"] = signal
    first_name = next(iter(selected))
    # Put an identical signal at both ends to catch file-order/state leakage.
    files = {"order_first": selected[first_name], **selected,
             "order_last": selected[first_name]}
    batch, _ = cap60_many(args.deep_filter, files, out_dir, "batch-order-validation")
    solo_dir = out_dir / "batch-validation-solo-cache"
    solo: dict[str, np.ndarray] = {}
    checks = []
    for name, signal in selected.items():
        solo[name], _ = cap60_one(args.deep_filter, signal, SR, out_dir,
            f"solo-{name}", solo_dir)
        error = float(np.max(np.abs(batch[name] - solo[name])))
        checks.append({"name": name, "max_abs_error": error,
            "identical_sha256": hash_array(batch[name]) == hash_array(solo[name])})
    ref = solo[first_name]
    for name in ("order_first", "order_last"):
        error = float(np.max(np.abs(batch[name] - ref)))
        checks.append({"name": name, "max_abs_error": error,
            "identical_sha256": hash_array(batch[name]) == hash_array(ref)})
    maximum = max(item["max_abs_error"] for item in checks)
    report = {"file_count_in_group": len(files), "unique_solo_files": len(selected),
        "order_probe_same_audio_first_and_last": True,
        "per_file": checks, "max_abs_error": maximum,
        "all_below_1e-7": maximum < 1e-7,
        "test_wav_accessed": False}
    atomic_json(out_dir / "batch-cli-validation.json", report)
    if maximum >= 1e-7:
        raise RuntimeError(f"multi-file Cap60 disagrees with solo outputs: {maximum:.3g}")
    return report


def run(args) -> dict:
    schedule_path = C_DATA / "pairs.jsonl"
    schedule = load_schedule()
    tail_rows = [row for row in schedule if row["kind"] == "rir_only"]
    if len(tail_rows) < args.slots:
        raise RuntimeError("fixed C schedule does not have enough RIR-only slots")
    if args.slots <= 0:
        raise ValueError("slots must be positive")
    if not args.deep_filter.is_file():
        raise FileNotFoundError(args.deep_filter)
    source_manifest = json.loads((C_DATA / "training-data-manifest.json").read_text())
    exe_hash = hashlib.sha256(args.deep_filter.read_bytes()).hexdigest()
    if source_manifest.get("deep_filter_sha256") != exe_hash:
        raise RuntimeError("Cap60 binary differs from the frozen C schedule")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.jsonl"
    progress_path = args.output_dir / "progress.json"
    schedule_hash = hashlib.sha256(schedule_path.read_bytes()).hexdigest()
    candidate_indices = tuple(args.candidate_indices)
    if len(set(candidate_indices)) != len(candidate_indices) or any(x < 0 or x >= 256 for x in candidate_indices):
        raise ValueError("candidate indices must be unique integers in [0, 255]")
    if args.resume and progress_path.is_file():
        progress = json.loads(progress_path.read_text())
        if (progress.get("source_schedule_sha256") != schedule_hash or
                progress.get("deep_filter_sha256") != exe_hash or
                progress.get("slots") != args.slots or
                progress.get("candidate_indices") != list(candidate_indices)):
            raise RuntimeError("prefix calibration resume manifest mismatch")
        results = [json.loads(line) for line in results_path.read_text().splitlines() if line]
    else:
        if results_path.exists() or progress_path.exists():
            raise FileExistsError("calibration output exists; use --resume for its own partial run")
        results = []
        batch_validation_path = args.output_dir / "batch-cli-validation.json"
        if batch_validation_path.is_file():
            batch_validation = json.loads(batch_validation_path.read_text())
        else:
            try:
                batch_validation = validate_batch_cli(args, tail_rows, args.output_dir)
            except RuntimeError:
                # The validator persists its measured mismatch before raising.
                # Fall back to serialized CLI invocations for this calibration.
                batch_validation = json.loads(batch_validation_path.read_text())
        progress = {"name": "F2-Cap60-prefix-calibration-v1",
            "source_schedule_sha256": schedule_hash,
            "deep_filter_sha256": exe_hash, "slots": args.slots,
            "candidate_indices": list(candidate_indices),
            "batch_cli_validation": batch_validation,
            "cap60_execution_mode": ("multi-file" if batch_validation.get("all_below_1e-7")
                                     else "single-file"),
            "completed_candidates": 0, "test_wav_accessed": False}
        atomic_json(progress_path, progress)

    completed = {(int(item["tail_slot"]), int(item["candidate_index"])) for item in results}
    cache_dir = args.output_dir / "single-call-equivalence-cache"
    total = args.slots * len(candidate_indices)
    for tail_slot, row in enumerate(tail_rows[:args.slots]):
        clean, pause_meta, _ = resolve_source(row, args.data_root / "train-clean-100")
        global_pause = int(pause_meta["clip_relative_start_sample"])
        crop_start = global_pause - int(.75 * SR)
        pause_in_crop = global_pause - crop_start
        gain = float(row["post_cap_gain"])
        prefix_end = min(len(clean), crop_start + pause_in_crop + 9_600)
        if prefix_end <= crop_start + pause_in_crop + 9_600 - 1:
            raise RuntimeError("prefix does not contain both complete tail windows")
        clean_crop = clean[crop_start:prefix_end]

        for candidate in candidate_indices:
            key = (tail_slot, candidate)
            if key in completed:
                continue
            candidate_seed = seed_for(tail_slot, candidate)
            t60, drr = candidate_parameters(candidate_seed)
            rir, _, _ = make_procedural_rir(SR, candidate_seed, t60, drr)
            pair = measured_pair(clean, rir)
            prefix = {"full_wet": pair["reverberant"], "full_clean": pair["clean"],
                "prefix_wet": pair["reverberant"][:prefix_end],
                "prefix_clean": pair["clean"][:prefix_end]}
            exact_cache = args.output_dir / "exact-hash-cache"
            if progress["cap60_execution_mode"] == "multi-file":
                outputs, elapsed = cap60_many(args.deep_filter, prefix, args.output_dir,
                    f"slot-{tail_slot:02d}-candidate-{candidate:03d}", exact_cache_dir=exact_cache)
            else:
                reuse_dirs = (args.output_dir / "batch-validation-solo-cache",)
                outputs = {}
                elapsed = 0.0
                for signal_name, signal in prefix.items():
                    outputs[signal_name], elapsed_one = cap60_single_cached(
                        args.deep_filter, signal, args.output_dir, exact_cache, reuse_dirs)
                    elapsed += elapsed_one

            full_x = outputs["full_wet"][crop_start:prefix_end] * gain
            full_t = outputs["full_clean"][crop_start:prefix_end] * gain
            prefix_x = outputs["prefix_wet"][crop_start:prefix_end] * gain
            prefix_t = outputs["prefix_clean"][crop_start:prefix_end] * gain
            activity = clean_crop
            full_levels = tail_input_levels(full_t, full_x, pause_in_crop,
                activity_clean_crop=activity)
            prefix_levels = tail_input_levels(prefix_t, prefix_x, pause_in_crop,
                activity_clean_crop=activity)
            compare_start = max(crop_start, global_pause - 8_000)
            compare_end = global_pause + 9_600
            diffs = []
            for full_name, prefix_name in (("full_wet", "prefix_wet"),
                                           ("full_clean", "prefix_clean")):
                delta = (outputs[prefix_name][compare_start:compare_end].astype(np.float64) -
                         outputs[full_name][compare_start:compare_end].astype(np.float64))
                diffs.append({"max_abs": float(np.max(np.abs(delta))),
                    "rms": float(np.sqrt(np.mean(delta * delta)))})
            level_deltas = {window: (float(prefix_levels["window_db"][window] -
                full_levels["window_db"][window]) if prefix_levels.get("window_db", {}).get(window) is not None
                and full_levels.get("window_db", {}).get(window) is not None else None)
                for window in ("150_300", "300_600")}
            result = {"tail_slot": tail_slot, "candidate_index": candidate,
                "seed": candidate_seed, "t60_s": t60, "direct_to_reverb_db": drr,
                "shared_scale": float(np.max(np.abs(pair["clean"]))) /
                    max(float(np.max(np.abs(clean))), 1e-8),
                "target_source_sha256": hash_array(pair["clean"]),
                "full_window_db": full_levels.get("window_db"),
                "prefix_window_db": prefix_levels.get("window_db"),
                "full_eligible": bool(full_levels.get("eligible", False)),
                "prefix_eligible": bool(prefix_levels.get("eligible", False)),
                "window_level_delta_db": level_deltas,
                "waveform_error_reference_plus_windows": diffs,
                "cap60_batch_elapsed_s": elapsed,
                "near_threshold_windows": [key for key, value in (full_levels.get("window_db") or {}).items()
                    if value is not None and -55.0 <= float(value) <= -45.0]}
            with results_path.open("a") as stream:
                stream.write(json.dumps(result, allow_nan=False) + "\n")
            results.append(result)
            completed.add(key)
            progress["completed_candidates"] = len(results)
            progress["last_candidate"] = {"slot": tail_slot, "candidate": candidate}
            atomic_json(progress_path, progress)
            print(json.dumps({"completed": len(results), "total": total,
                "slot": tail_slot, "candidate": candidate,
                "full_db": result["full_window_db"],
                "prefix_delta": level_deltas, "max_wave_error": max(
                    item["max_abs"] for item in diffs)}, allow_nan=False), flush=True)

    report = {**progress, "completed_candidates": len(results),
        "slot_count": len({item["tail_slot"] for item in results}),
        "all_batch_cli_equivalent": bool(progress.get("batch_cli_validation", {}).get("all_below_1e-7")),
        "max_waveform_error": max((max(x["max_abs"] for x in r["waveform_error_reference_plus_windows"])
            for r in results), default=None),
        "max_waveform_rms_error": max((max(x["rms"] for x in r["waveform_error_reference_plus_windows"])
            for r in results), default=None),
        "max_abs_window_level_delta_db": max((abs(v) for r in results
            for v in r["window_level_delta_db"].values() if v is not None), default=None),
        "eligibility_disagreements": sum(r["full_eligible"] != r["prefix_eligible"] for r in results),
        "near_threshold_count": sum(bool(r["near_threshold_windows"]) for r in results),
        "empirical_prefix_gate_pass": (len(results) == total and
            len({r["tail_slot"] for r in results}) == args.slots and
            report_value_max_wave(results) < 1e-7 and
            report_value_max_level(results) < .01 and
            sum(r["full_eligible"] != r["prefix_eligible"] for r in results) == 0),
        "test_wav_accessed": False}
    atomic_json(args.output_dir / "report.json", report)
    atomic_json(progress_path, report)
    return report


def report_value_max_wave(results: list[dict]) -> float:
    return max((entry["max_abs"] for row in results
        for entry in row["waveform_error_reference_plus_windows"]), default=float("inf"))


def report_value_max_level(results: list[dict]) -> float:
    return max((abs(value) for row in results
        for value in row["window_level_delta_db"].values() if value is not None),
        default=float("inf"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=LIBRISPEECH)
    parser.add_argument("--deep-filter", type=Path, default=DEEP_FILTER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument("--candidate-indices", type=int, nargs="+", default=list(DEFAULT_CANDIDATES))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
