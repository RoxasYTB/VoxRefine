#!/usr/bin/env python3
"""Audit speech preservation and amplitude equivariance for F2 strong RIRs."""
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
from data import convolve_same_length, make_procedural_rir  # noqa: E402
from prepare_cap60_conditioned_pairs import (  # noqa: E402
    CROP_SAMPLES, DEEP_FILTER, LIBRISPEECH, SR, cap60_one, sha256,
)
from prepare_f2_tailbank import crop, load_schedule, resolve_source  # noqa: E402
from train_measured_mix import measured_pair  # noqa: E402

DEFAULT_PREVIOUS = ROOT / ".tools/compact-dereverb/f2-strong-rir-preflight-2026-10-10/report.json"
DEFAULT_OUT = ROOT / ".tools/compact-dereverb/f2-strong-rir-integrity-v2-2026-10-10"
SCHEDULE_INDICES = (7, 10, 15, 18, 20, 23, 27, 31)
EPS = 1e-8


def frame_rms(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    frame, hop = 320, 160
    if len(x) < frame:
        return np.asarray([np.sqrt(np.mean(x * x) + 1e-24)])
    frames = np.lib.stride_tricks.sliding_window_view(x, frame)[::hop]
    return np.sqrt(np.mean(frames * frames, axis=1) + 1e-24)


def masks_from_raw_clean(raw_clean: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rms = frame_rms(raw_clean)
    peak = float(np.max(rms))
    active = rms > .02 * peak
    weak = active & (rms < .35 * peak)
    previous = np.concatenate(([0.0], rms[:-1]))
    onset = active & (rms > 1.5 * previous)
    return rms, active, weak, onset


def levels_db(output_rms: np.ndarray, input_rms: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10((output_rms + EPS) / (input_rms + EPS))


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize_case_metrics(case_metrics: list[dict]) -> dict:
    names = ("active_median_db", "active_p10_db", "weak_p10_db", "onset_p10_db")
    return {name: {"median_across_64_cases": float(np.median([x[name] for x in case_metrics])),
                   "min_across_cases": float(np.min([x[name] for x in case_metrics])),
                   "max_across_cases": float(np.max([x[name] for x in case_metrics]))}
            for name in names}


def run(args: argparse.Namespace) -> dict:
    if not args.deep_filter.is_file():
        raise FileNotFoundError(args.deep_filter)
    if not args.previous_report.is_file():
        raise FileNotFoundError(args.previous_report)
    previous = json.loads(args.previous_report.read_text())
    if (previous.get("name") != "F2-strong-RIR-distribution-preflight-v1" or
            previous.get("full_exact_cases") != 64 or
            previous.get("schedule_indices") != list(SCHEDULE_INDICES) or
            previous.get("surrogate_used") is not False or
            previous.get("no_candidate_selection") is not True):
        raise RuntimeError("previous full-exact preflight does not match the frozen 64-case set")
    if previous.get("deep_filter_sha256") != sha256(args.deep_filter):
        raise RuntimeError("Cap60 binary changed since the original preflight")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"refusing to overwrite nonempty {args.output_dir}; pass --resume")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = args.output_dir / "cap60-cache"
    cache.mkdir(exist_ok=True)

    schedule = load_schedule()
    rows = [schedule[index] for index in SCHEDULE_INDICES]
    records_by_key = {(int(r["schedule_index"]), int(r["candidate_index"])): r
                      for r in previous["records"]}
    if len(records_by_key) != 64:
        raise RuntimeError("preflight report does not contain 64 unique cases")
    case_metrics: list[dict] = []
    equivariance: list[dict] = []
    onset_peaks: list[float] = []
    realized_drr_errors: list[float] = []
    clipped_cases = 0
    source_hashes_verified = 0
    started = time.perf_counter()

    for slot_order, row in enumerate(rows):
        schedule_index = int(row["index"])
        joined, pause_meta, _ = resolve_source(row, args.data_root / "train-clean-100")
        global_pause = int(pause_meta["clip_relative_start_sample"])
        crop_start = global_pause - int(.75 * SR)
        raw_clean_crop = crop(joined, crop_start)
        raw_rms, active, weak, onset = masks_from_raw_clean(raw_clean_crop)
        unscaled_cap = None

        for candidate_index in range(8):
            rec = records_by_key[(schedule_index, candidate_index)]
            rir, early, late = make_procedural_rir(
                SR, int(rec["geometry_seed"]), float(rec["t60_s"]),
                float(rec["direct_to_reverb_db"]), max_t60_s=1.85,
                min_direct_to_reverb_db=-14.5)
            early_peak = float(np.max(np.abs(rir[:int(.010 * SR)])))
            actual_drr = float(10 * np.log10(
                (float(np.sum(early.astype(np.float64) ** 2)) + 1e-24) /
                (float(np.sum(late.astype(np.float64) ** 2)) + 1e-24)))
            realized_drr_error = actual_drr - float(rec["direct_to_reverb_db"])
            onset_peaks.append(early_peak)
            realized_drr_errors.append(realized_drr_error)

            pair = measured_pair(joined, rir)
            key = f"row-{schedule_index:03d}-candidate-{candidate_index:02d}"
            original_cache = (ROOT / ".tools/compact-dereverb/f2-strong-rir-preflight-2026-10-10"
                              / "cap60-cache" / f"{key}-clean.json")
            if not original_cache.is_file():
                raise FileNotFoundError(original_cache)
            original_meta = json.loads(original_cache.read_text())
            input_hash = hashlib.sha256(np.asarray(pair["clean"]).tobytes()).hexdigest()
            if original_meta.get("source_sha256") != input_hash:
                raise RuntimeError(f"reconstructed scaled-clean hash mismatch: {key}")
            source_hashes_verified += 1
            scaled_output = np.load(original_cache.with_suffix(".npy"), allow_pickle=False).astype(np.float32)

            input_crop = crop(pair["clean"], crop_start)
            output_crop = crop(scaled_output, crop_start)
            input_rms = frame_rms(input_crop)
            output_rms = frame_rms(output_crop)
            if not (len(input_rms) == len(raw_rms) == len(output_rms)):
                raise RuntimeError(f"frame alignment mismatch for {key}")
            deltas = levels_db(output_rms, input_rms)
            active_delta = deltas[active]
            weak_delta = deltas[weak]
            onset_delta = deltas[onset]
            if not active.any() or not weak.any() or not onset.any():
                raise RuntimeError(f"frozen clean masks are empty for {key}")
            retain = float(np.sqrt(np.mean(output_rms[active] ** 2) + 1e-24) /
                           max(np.sqrt(np.mean(input_rms[active] ** 2) + 1e-24), EPS))
            wet_cache = (ROOT / ".tools/compact-dereverb/f2-strong-rir-preflight-2026-10-10"
                         / "cap60-cache" / f"{key}-wet.npy")
            if not wet_cache.is_file():
                raise FileNotFoundError(wet_cache)
            wet_output = np.load(wet_cache, allow_pickle=False).astype(np.float32)
            if len(wet_output) != len(pair["reverberant"]):
                raise RuntimeError(f"cached wet output length mismatch: {key}")
            input_peak = max(float(np.max(np.abs(pair["clean"]))),
                             float(np.max(np.abs(pair["reverberant"]))))
            output_peak = max(float(np.max(np.abs(scaled_output))),
                              float(np.max(np.abs(wet_output))))
            if abs(float(rec["shared_scale"]) -
                   float(np.max(np.abs(pair["clean"]))) / max(float(np.max(np.abs(joined))), EPS)) > 1e-7:
                raise RuntimeError(f"shared scale mismatch while rebuilding {key}")
            clipped = input_peak >= 1.0 or output_peak >= 1.0
            clipped_cases += int(clipped)
            peak_expansion_db = float(20 * np.log10(
                max(float(np.max(np.abs(pair["reverberant"]))), EPS) /
                max(float(np.max(np.abs(pair["clean"]))), EPS)))
            metric = {"schedule_index": schedule_index, "speaker": str(row["speaker"]),
                "candidate_index": candidate_index, "shared_scale": float(rec["shared_scale"]),
                "shared_scale_db": float(rec["shared_scale_db"]),
                "early_path_peak_first_10ms": early_peak,
                "requested_drr_db": float(rec["direct_to_reverb_db"]),
                "realized_drr_db": actual_drr, "realized_drr_error_db": realized_drr_error,
                "active_median_db": float(np.median(active_delta)),
                "active_p10_db": percentile(active_delta.tolist(), 10),
                "weak_p10_db": percentile(weak_delta.tolist(), 10),
                "onset_p10_db": percentile(onset_delta.tolist(), 10),
                "active_retention_rms": retain,
                "peak_expansion_before_scaling_db": peak_expansion_db,
                "scaled_pair_input_peak": input_peak, "scaled_pair_cap60_output_peak": output_peak,
                "clipped": clipped, "cached_cap60_source_sha256": input_hash}
            case_metrics.append(metric)

            if candidate_index == 0:
                unscaled_cap, unscaled_meta = cap60_one(
                    args.deep_filter, joined, SR, args.output_dir,
                    f"row-{schedule_index:03d}-candidate-00-clean-unscaled", cache)
                if len(unscaled_cap) != len(joined):
                    raise RuntimeError("unscaled Cap60 output length mismatch")
                u_crop = crop(unscaled_cap, crop_start)
                v_crop = output_crop / max(float(rec["shared_scale"]), EPS)
                u_rms, v_rms = frame_rms(u_crop), frame_rms(v_crop)
                if not (len(u_rms) == len(v_rms) == len(raw_rms)):
                    raise RuntimeError(f"equivariance frame alignment mismatch for {key}")
                d = levels_db(v_rms, u_rms)
                corr = float(np.corrcoef(u_rms[active], v_rms[active])[0, 1])
                equivariance.append({
                    "schedule_index": schedule_index, "speaker": str(row["speaker"]),
                    "candidate_index": candidate_index, "shared_scale": float(rec["shared_scale"]),
                    "active_median_delta_db": float(np.median(d[active])),
                    "active_median_abs_delta_db": float(abs(np.median(d[active]))),
                    "active_p90_abs_delta_db": percentile(np.abs(d[active]).tolist(), 90),
                    "active_envelope_correlation": corr,
                    "weak_delta_p10_db": percentile(d[weak].tolist(), 10),
                    "onset_delta_p10_db": percentile(d[onset].tolist(), 10),
                    "unscaled_cap60_output_peak": float(np.max(np.abs(u_crop))),
                    "source_sha256": unscaled_meta["source_sha256"]})

    case_summary = summarize_case_metrics(case_metrics)
    eq_summary = {
        "active_median_abs_delta_db_median_across_8": float(np.median([x["active_median_abs_delta_db"] for x in equivariance])),
        "active_p90_abs_delta_db_median_across_8": float(np.median([x["active_p90_abs_delta_db"] for x in equivariance])),
        "active_envelope_correlation_median_across_8": float(np.median([x["active_envelope_correlation"] for x in equivariance])),
        "weak_delta_p10_db_median_across_8": float(np.median([x["weak_delta_p10_db"] for x in equivariance])),
        "onset_delta_p10_db_median_across_8": float(np.median([x["onset_delta_p10_db"] for x in equivariance])),
    }
    gates = {
        "active_median_within_plus_minus_0_5_db": -.5 <= case_summary["active_median_db"]["median_across_64_cases"] <= .5,
        "active_p10_above_minus_1_db": case_summary["active_p10_db"]["median_across_64_cases"] > -1.0,
        "weak_p10_above_minus_1_5_db": case_summary["weak_p10_db"]["median_across_64_cases"] > -1.5,
        "onset_p10_above_minus_1_db": case_summary["onset_p10_db"]["median_across_64_cases"] > -1.0,
        "minimum_active_retention_at_least_0_90": min(x["active_retention_rms"] for x in case_metrics) >= .90,
        "no_clipping_after_scaling": clipped_cases == 0,
        "early_path_peak_unit_for_all_64": all(abs(x - 1.0) <= 1e-7 for x in onset_peaks),
        "realized_drr_within_0_5_db_for_all_64": all(abs(x) <= .5 for x in realized_drr_errors),
        "equivariance_active_median_abs_delta_at_most_0_5_db": eq_summary["active_median_abs_delta_db_median_across_8"] <= .5,
        "equivariance_active_p90_abs_delta_at_most_1_db": eq_summary["active_p90_abs_delta_db_median_across_8"] <= 1.0,
        "equivariance_active_envelope_correlation_at_least_0_98": eq_summary["active_envelope_correlation_median_across_8"] >= .98,
        "equivariance_weak_p10_above_minus_1_db": eq_summary["weak_delta_p10_db_median_across_8"] > -1.0,
        "equivariance_onset_p10_above_minus_1_db": eq_summary["onset_delta_p10_db_median_across_8"] > -1.0,
    }
    scales = np.asarray([x["shared_scale"] for x in case_metrics], dtype=np.float64)
    expansions = np.asarray([x["peak_expansion_before_scaling_db"] for x in case_metrics], dtype=np.float64)
    eligible = int(previous["eligible_count"])
    slot_counts = previous["eligible_by_schedule_index"]
    tail_gates = {
        "tail_pass_rate_at_least_48_of_64": eligible >= 48,
        "six_slots_at_least_six_valid": sum(int(v) >= 6 for v in slot_counts.values()) >= 6,
        "no_slot_below_four_valid": min(int(v) for v in slot_counts.values()) >= 4,
    }
    report = {
        "name": "F2-strong-RIR-integrity-v2",
        "previous_preflight": str(args.previous_report.relative_to(ROOT)),
        "previous_preflight_result_preserved": previous.get("preflight_pass") is False,
        "schedule_indices": list(SCHEDULE_INDICES),
        "speaker_ids": [str(row["speaker"]) for row in rows],
        "cases": 64, "new_full_exact_cap60_runs": len(equivariance),
        "reused_scaled_clean_cap60_outputs": source_hashes_verified == 64,
        "surrogate_used": False, "candidate_selection_used": False,
        "shared_scale_is_descriptive_only": True,
        "shared_scale": {"min": float(scales.min()), "p10": float(np.percentile(scales, 10)),
            "median": float(np.median(scales)), "p90": float(np.percentile(scales, 90)),
            "max": float(scales.max()), "log_db_min": float(20*np.log10(scales.min())),
            "log_db_p10": float(20*np.log10(np.percentile(scales, 10))),
            "log_db_median": float(np.median(20*np.log10(scales))),
            "log_db_p90": float(20*np.log10(np.percentile(scales, 90))),
            "log_db_max": float(20*np.log10(scales.max()))},
        "peak_expansion_before_scaling_db": {"min": float(expansions.min()),
            "p10": float(np.percentile(expansions, 10)), "median": float(np.median(expansions)),
            "p90": float(np.percentile(expansions, 90)), "max": float(expansions.max())},
        "target_preservation_frame_db_by_case": case_summary,
        "active_retention_rms": {"min": float(min(x["active_retention_rms"] for x in case_metrics)),
            "median": float(np.median([x["active_retention_rms"] for x in case_metrics])),
            "max": float(max(x["active_retention_rms"] for x in case_metrics))},
        "early_path_peak_first_10ms": {"min": float(np.min(onset_peaks)),
            "median": float(np.median(onset_peaks)), "max": float(np.max(onset_peaks))},
        "realized_drr_error_db": {"min": float(np.min(realized_drr_errors)),
            "median": float(np.median(realized_drr_errors)), "max": float(np.max(realized_drr_errors))},
        "equivariance_subset": "candidate 0 on each of the 8 fixed schedule rows",
        "equivariance_summary": eq_summary, "equivariance_cases": equivariance,
        "clipped_case_count": int(clipped_cases), "gates": gates,
        "integrity_v2_pass": all(gates.values()),
        "tail_eligibility_previous_report": {
            "eligible_count": previous["eligible_count"],
            "eligible_rate": previous["eligible_rate"],
            "eligible_by_schedule_index": previous["eligible_by_schedule_index"],
            "strict_threshold_db": previous["selection_criterion_db"],
            "gates": tail_gates},
        "records": case_metrics,
        "elapsed_s": time.perf_counter() - started,
        "selection_or_training_data_modified": False,
        "test_wav_accessed": False,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return {k: v for k, v in report.items() if k not in ("records", "equivariance_cases")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=LIBRISPEECH)
    parser.add_argument("--deep-filter", type=Path, default=DEEP_FILTER)
    parser.add_argument("--previous-report", type=Path, default=DEFAULT_PREVIOUS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--resume", action="store_true", help="reuse exact-hash unscaled clean Cap60 outputs")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
