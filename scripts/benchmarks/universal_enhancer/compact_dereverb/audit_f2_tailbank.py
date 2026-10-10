#!/usr/bin/env python3
"""Verify the frozen E″/F″ tailbank cache before either matched fit."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
from prepare_f2_tailbank import (  # noqa: E402
    DEEP_FILTER, C_DATA, SCALE_SURROGATE_OBSERVED_ERROR_DB,
    SCALE_SURROGATE_OPERATIONAL_BOUND_DB, candidate_parameters,
    load_prefix_screening_policy, seed_for, tail_input_levels,
)

EXPERIMENT = ROOT / ".tools/compact-dereverb/e2-f2-tailbank-2026-10-10"
DATA = EXPERIMENT / "data"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(data_dir: Path, output: Path) -> dict:
    pair_path = data_dir / "pairs.jsonl"
    manifest_path = data_dir / "training-data-manifest.json"
    rows = [json.loads(line) for line in pair_path.read_text().splitlines() if line.strip()]
    meta = json.loads(manifest_path.read_text())
    expected_counts = {"identity": 32, "rir_only": 48, "fan20": 24, "fan10": 24}
    counts = {kind: sum(row.get("kind") == kind for row in rows) for kind in expected_counts}
    unique_speakers = len({str(row["speaker"]) for row in rows})
    issues: list[str] = []
    if len(rows) != 128 or counts != expected_counts or unique_speakers != 128:
        issues.append("fixed 128-row schedule does not match expected counts/unique speakers")
    if meta.get("test_wav_accessed") is not False or meta.get("input_only_selection") is not True:
        issues.append("data manifest missing sealed-source/input-only declarations")
    if meta.get("pair_manifest_sha256") != sha(pair_path):
        issues.append("pair manifest hash mismatch")
    if meta.get("resolved_tail_slots") != 48:
        issues.append("tailbank did not resolve exactly 48 slots")
    calibration_dir = (ROOT / "docs/benchmarking/assets/compact-dereverb/"
        "f2-prefix-calibration-2026-10-10")
    calibration_paths = {"prefix": calibration_dir / "prefix-1600ms-common-activity-report.json",
        "scale": calibration_dir / "scale-equivariance-report.json",
        "positive": calibration_dir / "positive-scale-surrogate-check.json"}
    if any(not path.is_file() for path in calibration_paths.values()):
        issues.append("one or more frozen prefix/surrogate calibration artifacts are missing")
    else:
        if (meta.get("prefix_calibration_report_sha256") != sha(calibration_paths["prefix"]) or
                meta.get("scale_calibration_report_sha256") != sha(calibration_paths["scale"]) or
                meta.get("scale_positive_control_sha256") != sha(calibration_paths["positive"]) or
                meta.get("prefix_calibration_gate_pass") is not True or
                abs(float(meta.get("scale_surrogate_operational_eref_error_bound_db", -1)) -
                    SCALE_SURROGATE_OPERATIONAL_BOUND_DB) > 1e-12):
            issues.append("screening policy is not tied to the passing frozen calibration artifacts")
    progress_path = data_dir / "progress.json"
    if progress_path.is_file():
        progress_meta = json.loads(progress_path.read_text())
        if progress_meta.get("screen_surrogate_invalidated"):
            issues.append("surrogate was invalidated; generated tailbank rows require regeneration/audit")
    try:
        _, policy_hash = load_prefix_screening_policy(C_DATA / "pairs.jsonl", DEEP_FILTER)
        if meta.get("selection_policy_sha256") != policy_hash:
            issues.append("manifest screening policy hash does not match recalculated calibration policy")
    except Exception as exc:
        issues.append(f"could not validate frozen screening policy: {type(exc).__name__}: {exc}")
    tail_rows = [row for row in rows if row.get("kind") == "rir_only"]
    row_checks = []
    for tail_slot, row in enumerate(tail_rows):
        x_path, c_path, t_path = (data_dir / row[key] for key in ("x", "c", "t"))
        if (sha(x_path) != row.get("input_sha256") or
                sha(c_path) != row.get("clean_sha256") or
                sha(t_path) != row.get("target_sha256")):
            issues.append(f"row {row.get('index')} array hash mismatch")
            continue
        x = np.load(x_path, allow_pickle=False).astype(np.float32)
        c = np.load(c_path, allow_pickle=False).astype(np.float32)
        t = np.load(t_path, allow_pickle=False).astype(np.float32)
        pause = int(row.get("source", {}).get("pause_start_in_crop", -1))
        measured = tail_input_levels(t, x, pause, activity_clean_crop=c)
        eligible = bool(measured.get("reference_valid") and measured.get("eligible"))
        recipe = row.get("rir_recipe", {})
        history = recipe.get("candidate_history", [])
        chosen_index = recipe.get("candidate_index")
        recorded = recipe.get("input_tail_db", {})
        if any(item.get("screening_reference_mode") == "scale_surrogate" for item in history):
            if (recipe.get("scale_calibration_sha256") != sha(calibration_paths["scale"]) or
                    recipe.get("positive_control_sha256") != sha(calibration_paths["positive"]) or
                    recipe.get("screening_policy_sha256") != meta.get("selection_policy_sha256")):
                issues.append(f"row {row.get('index')} surrogate history has stale calibration hashes")
        candidate_specs_valid = True
        for candidate_index, candidate_row in enumerate(history):
            candidate_seed = seed_for(tail_slot, candidate_index)
            t60, drr = candidate_parameters(candidate_seed)
            levels = candidate_row.get("input_tail_db") or {}
            prefix_levels = candidate_row.get("prefix_input_tail_db") or {}
            thresholds = candidate_row.get("prefix_reject_threshold_db") or {}
            mode = candidate_row.get("screening_reference_mode", "exact_prefix_target")
            expected_threshold = ({"150_300": -51.0, "300_600": -51.0}
                if mode in ("exact_prefix_target", "exact_full_target") else
                {"150_300": -50.0 - SCALE_SURROGATE_OPERATIONAL_BOUND_DB,
                 "300_600": -50.0 - SCALE_SURROGATE_OPERATIONAL_BOUND_DB})
            expected_screen_reject = any(prefix_levels.get(key) is not None and
                float(prefix_levels[key]) <= expected_threshold[key]
                for key in ("150_300", "300_600"))
            thresholds_valid = all(abs(float(thresholds.get(key, 0)) -
                expected_threshold[key]) < 1e-9 for key in ("150_300", "300_600"))
            screened = bool(candidate_row.get("prefix_screened_out"))
            reason = candidate_row.get("reason")
            if reason == "prefix_screen_reject":
                expected_eligible = False
                score_policy_valid = (expected_screen_reject and screened and
                    candidate_row.get("input_tail_db") is None)
            elif reason in ("full_reject", "accept_full"):
                expected_eligible = all(levels.get(key) is not None and
                    float(levels[key]) > -50.0 for key in ("150_300", "300_600"))
                score_policy_valid = (not screened and not expected_screen_reject and
                    bool(candidate_row.get("input_tail_db")) and
                    ((reason == "accept_full") == expected_eligible))
            else:
                expected_eligible = False
                score_policy_valid = False
            if mode == "scale_surrogate":
                error = candidate_row.get("eref_surrogate_error_db")
                deltas = candidate_row.get("prefix_vs_full_exact_target_window_delta_db")
                if error is not None:
                    score_policy_valid &= (abs(float(error)) <= SCALE_SURROGATE_OPERATIONAL_BOUND_DB and
                        isinstance(deltas, dict) and all(abs(float(deltas.get(key, 999))) <= .01
                        for key in ("150_300", "300_600")))
                if candidate_row.get("scale_in_calibration_range") is not True:
                    score_policy_valid = False
            elif mode not in ("exact_prefix_target", "exact_full_target"):
                score_policy_valid = False
            candidate_specs_valid &= (
                int(candidate_row.get("seed", -1)) == candidate_seed and
                int(candidate_row.get("candidate_index", -1)) == candidate_index and
                abs(float(candidate_row.get("t60_s", -1)) - t60) < 1e-12 and
                abs(float(candidate_row.get("direct_to_reverb_db", -99)) - drr) < 1e-12 and
                bool(candidate_row.get("eligible")) == expected_eligible and
                all((prefix_levels.get(key) is None or
                     np.isfinite(float(prefix_levels[key])))
                    for key in ("150_300", "300_600")) and
                thresholds_valid and score_policy_valid)
        level_error = max((abs(float(measured["window_db"][key]) - float(recorded[key]))
                           for key in ("150_300", "300_600")
                           if measured.get("window_db", {}).get(key) is not None
                           and recorded.get(key) is not None), default=float("inf"))
        history_valid = (len(history) == int(chosen_index) + 1 and
            [item.get("candidate_index") for item in history] == list(range(len(history))) and
            all(not item.get("eligible") for item in history[:-1]) and
            bool(history[-1].get("eligible")) and
            history[-1].get("reason") == "accept_full" and
            candidate_specs_valid and
            int(recipe.get("tail_slot", -1)) == tail_slot and
            int(recipe.get("seed", -1)) == seed_for(tail_slot, int(chosen_index)) and
            .45 <= float(recipe.get("t60_s", -1)) <= 1.10 and
            -6.0 <= float(recipe.get("direct_to_reverb_db", -99)) <= 18.0)
        if recipe.get("input_only_selection") is not True or not eligible:
            issues.append(f"row {row.get('index')} does not satisfy input-only tail eligibility")
        if not history_valid:
            issues.append(f"row {row.get('index')} does not preserve first-valid candidate history")
        if level_error > 1e-4:
            issues.append(f"row {row.get('index')} tail levels disagree with arrays")
        row_checks.append({"index": int(row["index"]), "reference_valid": bool(
            measured.get("reference_valid")), "window_db": measured.get("window_db"),
            "eligible": eligible, "candidate_index": chosen_index,
            "candidates_tested": recipe.get("candidates_tested"),
            "candidate_history_valid": history_valid,
            "max_abs_level_error_db": level_error})
    if len(row_checks) != 48:
        issues.append("did not audit all 48 tailbank arrays")
    result = {"name": "E2-F2-tailbank-coverage-audit-v2",
        "pair_manifest_sha256": sha(pair_path),
        "training_data_manifest_sha256": sha(manifest_path),
        "row_count": len(rows), "kind_counts": counts,
        "unique_training_speakers": unique_speakers,
        "rir_only_rows_audited": len(row_checks),
        "resolved_tail_slots": sum(int(row["eligible"]) for row in row_checks),
        "required_tail_slots": 48, "threshold_db": -50.0,
        "eligibility_uses_cap60_input_only": True,
        "per_row_measurements": row_checks,
        "executable": not issues and len(row_checks) == 48 and all(
            row["eligible"] for row in row_checks),
        "issues": issues, "speaker_ids_in_report": False,
        "test_wav_accessed": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return {key: value for key, value in result.items() if key != "per_row_measurements"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=EXPERIMENT / "coverage-audit.json")
    args = parser.parse_args()
    result = audit(args.data_dir, args.output)
    print(json.dumps(result, indent=2))
    if not result["executable"]:
        raise SystemExit("E2/F2 non-executable: tailbank coverage failed 48/48 validation")


if __name__ == "__main__":
    main()
