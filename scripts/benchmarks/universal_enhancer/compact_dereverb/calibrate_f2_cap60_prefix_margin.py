#!/usr/bin/env python3
"""Calibrate a longer Cap60 screening prefix against the frozen full runs."""
from __future__ import annotations

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
    CROP_SAMPLES, DEEP_FILTER, LIBRISPEECH, SR,
)
from prepare_f2_tailbank import (  # noqa: E402
    C_DATA, candidate_parameters, load_schedule, resolve_source, seed_for,
    tail_input_levels,
)
from calibrate_f2_cap60_prefix import (  # noqa: E402
    cap60_single_cached, hash_array,
)
from train_measured_mix import measured_pair  # noqa: E402

BASE = ROOT / ".tools/compact-dereverb/e2-f2-tailbank-2026-10-10/prefix-calibration"
OUT = BASE / "prefix-margin-1600ms-common-mask-release"
PREVIOUS_OUT = BASE / "prefix-margin-1600ms-fixed2s-activity"
RESCORED_BASE = BASE / "rescored-fixed2s-activity"
CANDIDATES = (0, 1, 3, 7, 15, 31, 63, 127)
WINDOW_END = 9_600  # 600 ms after pause start
PREFIX_END_AFTER_PAUSE = 25_600  # 1,600 ms after pause start


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def run() -> dict:
    if not (BASE / "report.json").is_file():
        raise RuntimeError("the initial 64-case full/prefix calibration is missing")
    base_report = json.loads((BASE / "report.json").read_text())
    if base_report.get("completed_candidates") != 64:
        raise RuntimeError("the initial calibration is incomplete")
    base_rows = [json.loads(line) for line in (BASE / "results.jsonl").read_text().splitlines() if line]
    base_by_key = {(int(x["tail_slot"]), int(x["candidate_index"])): x for x in base_rows}
    rescored_rows = [json.loads(line) for line in (RESCORED_BASE / "results.jsonl").read_text().splitlines() if line]
    rescored_by_key = {(int(x["tail_slot"]), int(x["candidate_index"])): x for x in rescored_rows}
    if len(rescored_by_key) != 64:
        raise RuntimeError("common-mask rescore for the +600ms prefix is incomplete")
    schedule = [r for r in load_schedule() if r["kind"] == "rir_only"]
    exe_hash = hashlib.sha256(DEEP_FILTER.read_bytes()).hexdigest()
    if exe_hash != base_report.get("deep_filter_sha256"):
        raise RuntimeError("Cap60 binary differs from the initial calibration")
    OUT.mkdir(parents=True, exist_ok=True)
    results_path = OUT / "results.jsonl"
    progress_path = OUT / "progress.json"
    if results_path.exists() or progress_path.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    exact_full_cache = BASE / "exact-hash-cache"
    output_cache = OUT / "exact-hash-cache"
    done = []
    started = time.perf_counter()

    for slot in range(8):
        row = schedule[slot]
        clean, pause_meta, _ = resolve_source(row, LIBRISPEECH / "train-clean-100")
        global_pause = int(pause_meta["clip_relative_start_sample"])
        crop_start = global_pause - int(.75 * SR)
        pause_in_crop = global_pause - crop_start
        prefix_end = global_pause + PREFIX_END_AFTER_PAUSE
        if prefix_end > len(clean):
            raise RuntimeError(f"extended prefix exceeds source at slot {slot}")
        # Keep the speech activity mask identical to the frozen 2s training
        # crop and coverage audit. Prefix future context never changes it.
        measure_end = global_pause + WINDOW_END
        raw_clean_crop = clean[crop_start:crop_start + CROP_SAMPLES]
        gain = float(row["post_cap_gain"])

        for candidate in CANDIDATES:
            key = (slot, candidate)
            if key not in base_by_key:
                raise RuntimeError(f"initial calibration missing slot/candidate {key}")
            base = base_by_key[key]
            rescored_base = rescored_by_key[key]
            seed = seed_for(slot, candidate)
            t60, drr = candidate_parameters(seed)
            rir, _, _ = make_procedural_rir(SR, seed, t60, drr)
            pair = measured_pair(clean, rir)
            signals = {"wet": pair["reverberant"][:prefix_end],
                       "clean": pair["clean"][:prefix_end]}
            out = {}
            elapsed = 0.0
            for name, signal in signals.items():
                out[name], seconds = cap60_single_cached(DEEP_FILTER, signal, OUT,
                    output_cache, reuse_dirs=(PREVIOUS_OUT / "exact-hash-cache",))
                elapsed += seconds

            # Score only the common reference+tail interval. The extra future
            # samples are context for Cap60, not part of the activity mask.
            px = out["wet"][crop_start:measure_end] * gain
            pt = out["clean"][crop_start:measure_end] * gain
            levels = tail_input_levels(pt, px, pause_in_crop,
                                       activity_clean_crop=raw_clean_crop)
            full_window = rescored_base["full_window_db"]
            delta = {window: (float(levels["window_db"][window] - full_window[window])
                     if levels["window_db"].get(window) is not None and full_window.get(window) is not None
                     else None) for window in ("150_300", "300_600")}

            # The previous full outputs were cached by exact input hash. Compare
            # the same post-Cap60 samples around the target windows.
            full_outputs = {}
            for name, signal in (("wet", pair["reverberant"]), ("clean", pair["clean"])):
                source_hash = hash_array(np.asarray(signal, dtype=np.float32))
                npy = exact_full_cache / f"{source_hash}.npy"
                meta_path = exact_full_cache / f"{source_hash}.json"
                if not npy.is_file() or not meta_path.is_file():
                    raise RuntimeError(f"missing full output cache for {key}/{name}")
                meta = json.loads(meta_path.read_text())
                if meta.get("source_sha256") != source_hash or meta.get("deep_filter_sha256") != exe_hash:
                    raise RuntimeError(f"invalid full output cache for {key}/{name}")
                full_outputs[name] = np.load(npy, allow_pickle=False)
            compare_start = max(crop_start, global_pause - 8_000)
            compare_end = global_pause + WINDOW_END
            wave = {}
            for name in ("wet", "clean"):
                diff = (out[name][compare_start:compare_end].astype(np.float64) -
                        full_outputs[name][compare_start:compare_end].astype(np.float64))
                wave[name] = {"max_abs": float(np.max(np.abs(diff))),
                              "rms": float(np.sqrt(np.mean(diff * diff)))}

            result = {"tail_slot": slot, "candidate_index": candidate,
                "full_input_window_db": full_window,
                "prefix_margin_window_db": levels["window_db"],
                "window_delta_db": delta,
                "full_eligible": bool(rescored_base["full_eligible"]),
                "prefix_margin_eligible": bool(levels.get("eligible", False)),
                "waveform_error_reference_plus_windows": wave,
                "wet_prefix_source_sha256": hash_array(np.asarray(signals["wet"], np.float32)),
                "clean_prefix_source_sha256": hash_array(np.asarray(signals["clean"], np.float32)),
                "cap60_elapsed_s": elapsed}
            with results_path.open("a") as stream:
                stream.write(json.dumps(result, allow_nan=False) + "\n")
            done.append(result)
            atomic_json(progress_path, {"completed_candidates": len(done), "total": 64,
                "last_candidate": {"slot": slot, "candidate": candidate},
                "test_wav_accessed": False})
            print(json.dumps({"completed": len(done), "total": 64, "slot": slot,
                "candidate": candidate, "full": full_window, "prefix_delta": delta,
                "max_wave_error": max(x["max_abs"] for x in wave.values())},
                allow_nan=False), flush=True)

    report = {"name": "F2-Cap60-prefix-margin-calibration-v1",
        "source_schedule_sha256": base_report["source_schedule_sha256"],
        "deep_filter_sha256": exe_hash, "slots": 8,
        "candidate_indices": list(CANDIDATES),
        "prefix_end_after_pause_ms": 1600,
        "guard_policy": "Cap60 wrapper appends the same 100ms EOF guard",
        "measurement_end_after_pause_ms": 600,
        "activity_mask_source": "raw clean on the fixed 2s tailbank crop",
        "reference_energy_window": "active Cap60-clean samples from pause-500ms to pause-50ms",
        "tail_windows": {"150_300": [150, 300], "300_600": [300, 600]},
        "frame_mapping": "10ms sample-time bins",
        "completed_candidates": len(done),
        "max_waveform_error": max(x["max_abs"] for r in done for x in r["waveform_error_reference_plus_windows"].values()),
        "max_waveform_rms_error": max(x["rms"] for r in done for x in r["waveform_error_reference_plus_windows"].values()),
        "max_abs_window_level_delta_db": max(abs(v) for r in done for v in r["window_delta_db"].values() if v is not None),
        "eligibility_disagreements": sum(r["full_eligible"] != r["prefix_margin_eligible"] for r in done),
        "near_threshold_count": sum(any(v is not None and -55 <= v <= -45 for v in r["full_input_window_db"].values()) for r in done),
        "empirical_prefix_gate_pass": (len(done) == 64 and
            max(x["max_abs"] for r in done for x in r["waveform_error_reference_plus_windows"].values()) < 1e-7 and
            max(abs(v) for r in done for v in r["window_delta_db"].values() if v is not None) < .01 and
            sum(r["full_eligible"] != r["prefix_margin_eligible"] for r in done) == 0),
        "per_window_screening_margin_db": {"150_300": 1.0, "300_600": 1.0},
        "screening_reject_db": {"150_300": -51.0, "300_600": -51.0},
        "elapsed_s": time.perf_counter() - started,
        "test_wav_accessed": False}
    atomic_json(OUT / "report.json", report)
    atomic_json(progress_path, report)
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, allow_nan=False))
