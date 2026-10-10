#!/usr/bin/env python3
"""Evaluate the frozen G-early-v2 fit on DEV, then conditionally open HOLDOUT."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from scipy.stats import theilslopes

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
from model_g_early_v2 import CompactAttenuationOnlyDereverbG2  # noqa: E402
from data import make_procedural_rir  # noqa: E402
from prepare_cap60_conditioned_pairs import (  # noqa: E402
    DEEP_FILTER, LIBRISPEECH, SR, cap60_one, make_fan_components,
)
from prepare_f2_tailbank import (  # noqa: E402
    cap60_cached, crop, index_cap60_cache, tail_input_levels,
)
from pilot_g2_feasibility import active_dry_levels  # noqa: E402
from screen_measured_rirs import infer, make_inserted_pause_pair, rms_frames  # noqa: E402
from train_measured_mix import measured_pair  # noqa: E402
from evaluate_e_synthetic_sealed import noise_stats  # noqa: E402

EXPERIMENT = ROOT / ".tools/compact-dereverb/g-early-v2-2026-10-10"
SR, FRAME, HOP = 16_000, 320, 160
VARIANTS = ("G-early-v2-clean",)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(values: list[float]) -> dict:
    arr = np.asarray(values, np.float64)
    arr = arr[np.isfinite(arr)]
    if not arr.size:
        return {"n": 0, "median": None, "p10": None, "p90": None}
    return {"n": int(arr.size), "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)), "p90": float(np.percentile(arr, 90))}


def json_finite(value):
    if isinstance(value, dict):
        return {key: json_finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_finite(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return "+inf" if value > 0 else "-inf" if value < 0 else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def speech_truth_stats(clean: np.ndarray, output: np.ndarray,
                       pause: dict) -> tuple[dict, dict]:
    """Compare model output with pre-Cap60 dry clean truth."""
    c, y = rms_frames(clean), rms_frames(output)
    n = min(c.size, y.size)
    c, y = c[:n], y[:n]
    peak = max(float(c.max()), 1e-12)
    active = c > peak * .02
    start = int(pause["clip_relative_start_sample"])
    end = int(pause["second_speech_start_sample"])
    lo = max(0, (start - FRAME) // HOP)
    hi = min(active.size, (end + HOP - 1) // HOP)
    active[lo:hi] = False
    ids = np.flatnonzero(active)
    if not ids.size:
        return ({"active_p50_db": None, "active_p10_db": None,
            "onset_p10_db": None, "weak_p10_db": None,
            "weak_p50_db": None, "active_frames": 0},
            {"weak_fraction_below_minus3_db": None,
             "weak_fraction_below_minus6_db": None})
    gain = 20 * np.log10(np.maximum(y, 1e-10) / np.maximum(c, 1e-10))
    rising = np.zeros_like(active)
    rising[1:] = active[1:] & (c[1:] > c[:-1] * 1.5)
    weak = ids[c[ids] < .35 * peak]
    bottom20 = ids[np.argsort(c[ids])[:max(1, int(np.ceil(.2 * ids.size)))]]
    stats = {"active_p50_db": float(np.median(gain[ids])),
        "active_p10_db": float(np.percentile(gain[ids], 10)),
        "onset_p10_db": float(np.percentile(gain[rising], 10)) if rising.any() else None,
        "weak_p10_db": float(np.percentile(gain[weak], 10)) if weak.size else None,
        "weak_p50_db": float(np.median(gain[weak])) if weak.size else None,
        "bottom20_active_p10_db": float(np.percentile(gain[bottom20], 10)),
        "active_frames": int(ids.size), "weak_frames": int(weak.size)}
    fractions = {"weak_fraction_below_minus3_db": float(np.mean(gain[weak] < -3.0)),
        "weak_fraction_below_minus6_db": float(np.mean(gain[weak] < -6.0))} if weak.size else {
            "weak_fraction_below_minus3_db": None,
            "weak_fraction_below_minus6_db": None}
    return stats, fractions


def validate_evaluator_binding(config: dict) -> str | None:
    """Require the evaluator that was frozen into the fit configuration."""
    current_sha = sha(HERE / "evaluate_g_early_v2.py")
    if config.get("evaluator_source_sha256") != current_sha:
        raise RuntimeError("evaluator differs from the prefit frozen source hash")
    return None


def load_g_model(variant: str, device: torch.device) -> tuple[torch.nn.Module, Path, dict]:
    output = EXPERIMENT / variant
    checkpoint = output / "checkpoints/step-003000.pt"
    config = json.loads((output / "model-config.json").read_text())
    if (not checkpoint.is_file() or config.get("variant") != variant or
            config.get("checkpoint_sha256") != sha(checkpoint) or config.get("steps") != 3000 or
            config.get("protocol_version") != "G-early-v2-real-2026-10-10" or
            config.get("training_source_sha256") != sha(HERE / "train_g_early_v2.py") or
            config.get("split_freeze_source_sha256") != sha(HERE / "freeze_g_early_v2_splits.py") or
            config.get("split_audit_source_sha256") != sha(HERE / "audit_g_early_v2_splits.py") or
            config.get("split_audit_sha256") != sha(EXPERIMENT / "split-integrity-audit.json") or
            config.get("gradient_diagnostic_inertness_sha256") != sha(
                EXPERIMENT / "gradient-diagnostic-inertness-v2.json")):
        raise RuntimeError(f"{variant} is missing a frozen step-3000 checkpoint/config")
    validate_evaluator_binding(config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state.get("step") != 3000 or state.get("variant") != variant:
        raise RuntimeError(f"invalid G checkpoint for {variant}")
    model = CompactAttenuationOnlyDereverbG2(base_channels=int(state["base_channels"])).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, checkpoint, config


def assert_zero_100(model: torch.nn.Module, device: torch.device) -> int:
    count = 0
    with torch.inference_mode():
        for index in range(100):
            length = 32_000 if index % 2 == 0 else 48_000
            out = model(torch.zeros(1, length, device=device))
            if not torch.isfinite(out).all() or float(out.abs().max()) >= 1e-7:
                raise RuntimeError("zero-preserving invariant failed during 100/100 audit")
            count += 1
    return count


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float | None:
    values, weights = np.asarray(values).ravel(), np.asarray(weights, np.float64).ravel()
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return None
    order = np.argsort(values[valid])
    v, w = values[valid][order], weights[valid][order]
    cumulative = np.cumsum(w)
    return float(v[min(int(np.searchsorted(cumulative, q * cumulative[-1], side="left")), len(v)-1)])


def infer_with_mechanism(model: torch.nn.Module, audio: np.ndarray,
                         pause: dict, device: torch.device) -> tuple[np.ndarray, dict, float]:
    """Return actual output plus gain-only/phase-only W1 diagnostics (no gates)."""
    x = torch.as_tensor(np.asarray(audio, np.float32), device=device)[None, :]
    started = time.perf_counter()
    with torch.inference_mode():
        window = model.window.to(dtype=x.dtype, device=x.device)
        spec = torch.stft(x, n_fft=model.n_fft, hop_length=model.hop_length,
            win_length=model.n_fft, window=window, center=True, return_complex=True)
        features = torch.stack((spec.real, spec.imag), dim=1)
        logits = model._predict_mask(features)
        gain = 1.0 - model.MAX_ATTENUATION * torch.sigmoid(logits[:, 0])
        phase = torch.pi * torch.tanh(logits[:, 1])
        def synth(g, p):
            transformed = spec * torch.polar(g, p)
            return torch.istft(transformed, n_fft=model.n_fft,
                hop_length=model.hop_length, win_length=model.n_fft,
                window=window, center=True, length=x.shape[-1])
        actual = synth(gain, phase)[0]
        gain_only = synth(gain, torch.zeros_like(phase))[0]
        phase_only = synth(torch.ones_like(gain), phase)[0]
    x_np = x[0].detach().cpu().numpy()
    actual_np = actual.detach().cpu().numpy()
    gain_np = gain[0].detach().cpu().numpy()
    spec_energy = spec[0].abs().square().detach().cpu().numpy()
    centers = np.arange(spec_energy.shape[-1]) * model.hop_length
    pause_start = int(pause["clip_relative_start_sample"])
    frame_mask = (centers >= pause_start + 150 * SR // 1000) & (centers < pause_start + 300 * SR // 1000)
    weights = spec_energy[:, frame_mask]
    gains = gain_np[:, frame_mask]
    total_weight = float(weights.sum())
    weighted_mean = float((weights * gains).sum() / total_weight) if total_weight > 0 else None
    a = pause_start + 150 * SR // 1000
    b = pause_start + 300 * SR // 1000
    def reduction(output: torch.Tensor) -> float | None:
        candidate = output.detach().cpu().numpy()
        if b > min(len(x_np), len(candidate)):
            return None
        ein = float(np.mean(np.asarray(x_np[a:b], np.float64) ** 2))
        eout = float(np.mean(np.asarray(candidate[a:b], np.float64) ** 2))
        return float(10 * np.log10(max(ein, 1e-30) / max(eout, 1e-30)))
    mechanism = {"w1_actual_reduction_db": reduction(actual),
        "w1_gain_only_reduction_db": reduction(gain_only[0]),
        "w1_phase_only_reduction_db": reduction(phase_only[0]),
        "w1_input_energy_weighted_gain_mean": weighted_mean,
        "w1_input_energy_weighted_gain_p10": weighted_quantile(gains, weights, .10),
        "w1_input_energy_weighted_gain_p50": weighted_quantile(gains, weights, .50),
        "w1_input_energy_weighted_gain_p90": weighted_quantile(gains, weights, .90),
        "interpretation": "descriptive only; no candidate or pass gate uses this decomposition"}
    return actual_np, mechanism, time.perf_counter() - started


def audit_frozen_tail_references(split: str, split_dir: Path,
                                 split_manifest: dict) -> tuple[dict, dict]:
    """Recreate each frozen Cap60 a_q reference before loading any model."""
    cap_cache = split_dir / "cap60-cache"
    references: dict[tuple[str, int], dict] = {}
    valid_count, max_window_delta = 0, 0.0
    primary_w1_count, available_w2_count = 0, 0
    for speaker_index, speaker_doc in enumerate(split_manifest["speakers"]):
        speaker = str(speaker_doc["speaker_id"])
        files = {path.name: path for path in
            (LIBRISPEECH / "train-clean-360" / speaker).glob("*/*.flac")}
        for spec in speaker_doc["pairs"]:
            paths = [files[spec[key]] for key in ("first", "second")]
            if [sha(path) for path in paths] != [spec["first_sha256"], spec["second_sha256"]]:
                raise RuntimeError("frozen source hash mismatch in Cap60 reference audit")
            joined, pause = make_inserted_pause_pair(*paths)
            joined = np.asarray(joined, dtype=np.float32)
            joined *= .80 / max(float(np.max(np.abs(joined))), 1e-8)
            crop_start, pause_crop = int(spec["crop_start_sample"]), int(spec["pause_start_in_crop"])
            raw_clean_crop = crop(joined, crop_start)
            tag = (f"{split}-{speaker_index:02d}-{int(spec['pair_index']):02d}-"
                   f"{int(spec['candidate_index']):02d}")
            frozen_dry_path, frozen_wet_path = cap_cache / f"{tag}-dry.npy", cap_cache / f"{tag}-wet.npy"
            frozen_dry = np.load(frozen_dry_path, allow_pickle=False)
            frozen_wet = np.load(frozen_wet_path, allow_pickle=False)
            recipe = spec["procedural_rir"]
            if (hashlib.sha256(np.asarray(frozen_dry, np.float32).tobytes()).hexdigest() !=
                    recipe["dry_cap60_output_sha256"] or
                    hashlib.sha256(np.asarray(frozen_wet, np.float32).tobytes()).hexdigest() !=
                    recipe["wet_cap60_output_sha256"]):
                raise RuntimeError("frozen selected Cap60 array hash differs from split manifest")
            dry_crop, wet_crop = crop(frozen_dry, crop_start), crop(frozen_wet, crop_start)
            measured = tail_input_levels(dry_crop, wet_crop, pause_crop,
                activity_clean_crop=raw_clean_crop)
            expected_windows = recipe["windows"]
            if measured.get("reference_valid"):
                valid_count += 1
                dry_levels = active_dry_levels(dry_crop, pause_crop,
                    float(measured["reference_energy"]))
                for band in ("150_300", "300_600"):
                    expected = expected_windows[band]
                    actual_lx = measured["window_db"][band]
                    actual_ldry = dry_levels[band]
                    if (actual_lx is None or actual_ldry is None or
                            abs(float(actual_lx) - float(expected["lx_db"])) > .01 or
                            abs(float(actual_ldry) - float(expected["ldry_db"])) > .01):
                        raise RuntimeError("recomputed Cap60 Eref/window differs from frozen input-only record")
                    max_window_delta = max(max_window_delta,
                        abs(float(actual_lx) - float(expected["lx_db"])),
                        abs(float(actual_ldry) - float(expected["ldry_db"])))
                reference_energy = float(measured["reference_energy"])
            else:
                if spec["tail_loss_enabled"]:
                    raise RuntimeError("W1-eligible split row has no frozen Cap60 speech reference")
                reference_energy = None
            baseline_tails = g_tail_stats(frozen_dry, frozen_wet, frozen_wet,
                frozen_dry, [frozen_dry, frozen_dry], pause, reference_energy)
            available_bands = {item["band_ms"] for item in baseline_tails}
            if spec["tail_loss_enabled"] and "150_300" not in available_bands:
                raise RuntimeError("W1-eligible slot cannot produce its primary window with frozen Eref")
            primary_w1_count += int("150_300" in available_bands)
            available_w2_count += int("300_600" in available_bands)
            references[(speaker, int(spec["pair_index"]))] = {
                "reference_energy": reference_energy,
                "reference_energy_sha256": (hashlib.sha256(
                    np.asarray(reference_energy, dtype=np.float64).tobytes()).hexdigest()
                    if reference_energy is not None else None),
                "reference_valid": bool(measured.get("reference_valid")),
                "frozen_selected_dry_sha256": recipe["dry_cap60_output_sha256"]}
    audit = {"pair_count": len(references), "reference_valid_pair_count": valid_count,
        "max_abs_frozen_window_delta_db": max_window_delta,
        "primary_w1_window_count": primary_w1_count,
        "available_w2_window_count": available_w2_count,
        "w2_window_unavailable_count": len(references) - available_w2_count,
        "tolerance_db": .01, "all_eligible_W1_references_valid": True}
    return references, audit


def g_tail_stats(clean_floor: np.ndarray, wet: np.ndarray, output: np.ndarray,
                 dry_output: np.ndarray, noise_controls: list[np.ndarray],
                 pause: dict, reference_energy: float | None) -> list[dict]:
    """Tail levels normalized by the exact frozen input-only Cap60 a_q Eref."""
    if reference_energy is None or reference_energy <= 0:
        return []
    start = int(pause["clip_relative_start_sample"])
    result = []
    energy = lambda value, a, b: float(np.mean(np.asarray(value[a:b], np.float64) ** 2))
    for band, lo_ms, hi_ms in (("150_300", 150, 300), ("300_600", 300, 600)):
        a, b = start + int(lo_ms * SR / 1000), start + int(hi_ms * SR / 1000)
        if b > int(pause["second_speech_start_sample"]) - int(.20 * SR):
            continue
        if b > min(len(wet), len(output), len(dry_output), len(clean_floor),
                   *(len(value) for value in noise_controls)):
            continue
        in_db = 10 * np.log10(max(energy(wet, a, b), 1e-30) / reference_energy)
        out_db = 10 * np.log10(max(energy(output, a, b), 1e-30) / reference_energy)
        floor_energy = max(energy(clean_floor, a, b), energy(dry_output, a, b),
            *(energy(value, a, b) for value in noise_controls), 1e-30)
        floor_db = 10 * np.log10(floor_energy / reference_energy)
        result.append({"band_ms": band, "input_tail_db": float(in_db),
            "output_tail_db": float(out_db), "common_floor_db": float(floor_db),
            "input_censored": bool(in_db <= floor_db + 3),
            "output_censored": bool(out_db <= floor_db + 3),
            "tail_reduction_db": float(in_db - out_db),
            "common_floor_censored": bool(in_db <= floor_db + 3 or
                                           out_db <= floor_db + 3)})
    return result


def g_tail_slope(wet: np.ndarray, output: np.ndarray, pause: dict,
                 reference_energy: float | None, common_floor_db: float) -> float | None:
    if reference_energy is None or reference_energy <= 0:
        return None
    start = int(pause["clip_relative_start_sample"])
    out_levels = rms_frames(output)
    times = (np.arange(out_levels.size) * 160 + 160 - start) / SR
    relative = 20 * np.log10(np.maximum(out_levels, 1e-12) /
                             max(float(np.sqrt(reference_energy)), 1e-12))
    keep = ((times >= .08) & (times <= .50) & np.isfinite(relative) &
            (relative > common_floor_db + 3))
    return float(theilslopes(relative[keep], times[keep]).slope) if int(keep.sum()) >= 8 else None


def interval_for_tail(item: dict) -> tuple[float, float, str]:
    """Common-floor censor-aware interval for Cap60-to-G tail reduction."""
    common = float(item["common_floor_db"]) + 3.0
    cap_level = float(item["input_tail_db"])
    candidate_level = float(item["output_tail_db"])
    cap_exact, candidate_exact = cap_level > common, candidate_level > common
    if cap_exact and candidate_exact:
        value = cap_level - candidate_level
        return value, value, "both_exact"
    if cap_exact and not candidate_exact:
        return cap_level - common, float("inf"), "candidate_at_floor"
    if not cap_exact and candidate_exact:
        return float("-inf"), common - candidate_level, "reference_at_floor"
    return float("-inf"), float("inf"), "both_at_floor"


def w2_is_informative(item: dict) -> bool:
    """W2 is informative only when Cap60 input is above the frozen common floor."""
    return float(item["input_tail_db"]) > float(item["common_floor_db"]) + 3.0


def bootstrap_lcb_speaker(values_by_speaker: dict[str, list[float]], seed: int = 20261010,
                          iterations: int = 5000) -> float | None:
    speakers = sorted(values_by_speaker)
    if len(speakers) < 2:
        return None
    rng = np.random.default_rng(seed)
    medians = []
    for _ in range(iterations):
        sampled = rng.choice(speakers, len(speakers), replace=True)
        values = [value for speaker in sampled
                  for value in values_by_speaker[str(speaker)]]
        medians.append(float(np.median(values)))
    return float(np.percentile(np.asarray(medians), 2.5))


def summarize_variant(rows: list[dict], variant: str) -> dict:
    metrics = [row["models"][variant] for row in rows]
    speakers = sorted({row["speaker_hash"] for row in rows})
    grouped = {speaker: [row["models"][variant] for row in rows
                         if row["speaker_hash"] == speaker] for speaker in speakers}
    speech_keys = ("active_p50_db", "active_p10_db", "weak_p10_db",
                   "weak_p50_db", "onset_p10_db", "bottom20_active_p10_db")
    fraction_keys = ("weak_fraction_below_minus3_db", "weak_fraction_below_minus6_db")
    speech_summary, fraction_summary = {}, {}
    for key in speech_keys:
        vals = [float(np.median([m["speech"][key] for m in group
            if m["speech"][key] is not None])) for group in grouped.values()
            if any(m["speech"][key] is not None for m in group)]
        speech_summary[key] = summarize(vals)
    for key in fraction_keys:
        vals = [float(np.median([m["weak_fractions"][key] for m in group
            if m["weak_fractions"][key] is not None])) for group in grouped.values()
            if any(m["weak_fractions"][key] is not None for m in group)]
        fraction_summary[key] = summarize(vals)

    noise_summary = {}
    for condition in ("fan20", "fan10"):
        pair_rows = [m["noise_only"][condition] for m in metrics]
        per_speaker = []
        for group in grouped.values():
            vals = [m["noise_only"][condition] for m in group]
            per_speaker.append({key: float(np.median([v[key] for v in vals]))
                for key in ("ac_delta_db", "output_dc_dbfs")})
        noise_summary[condition] = {
            "ac_delta_db_per_speaker": summarize([v["ac_delta_db"] for v in per_speaker]),
            "dc_dbfs_per_speaker": summarize([v["output_dc_dbfs"] for v in per_speaker]),
            "pair_ac_delta_db": summarize([v["ac_delta_db"] for v in pair_rows]),
            "pair_dc_dbfs": summarize([v["output_dc_dbfs"] for v in pair_rows])}

    bands = ("150_300", "300_600")
    intervals = {band: [] for band in bands}
    lower_by_speaker = {band: {} for band in bands}
    counts = {band: {status: 0 for status in
        ("both_exact", "candidate_at_floor", "reference_at_floor", "both_at_floor")}
        for band in bands}
    for row in rows:
        speaker = row["speaker_hash"]
        model_metrics = row["models"][variant]
        for item in model_metrics["tails"]:
            band = item["band_ms"]
            # W1 is efficacy only on input-eligible slots; W2 safety is
            # descriptive over every frozen RIR slot, including controls.
            if band == "150_300" and not row["tail_eligible"]:
                continue
            if band == "300_600" and not w2_is_informative(item):
                continue
            lower, upper, status = interval_for_tail(item)
            counts[band][status] += 1
            intervals[band].append((lower, upper, status))
            lower_by_speaker[band].setdefault(speaker, []).append(lower)
    tail_summary, lcb = {}, {}
    for band, values in intervals.items():
        lows = np.asarray([item[0] for item in values], np.float64)
        highs = np.asarray([item[1] for item in values], np.float64)
        exact = [item[0] for item in values if item[2] == "both_exact"]
        tail_summary[band] = {"n": len(values), "statuses": counts[band],
            "Mlow_db": float(np.median(lows)) if lows.size else None,
            "Mhigh_db": float(np.median(highs)) if highs.size else None,
            "exact_only_median_secondary_db": float(np.median(exact)) if exact else None}
        lcb[band] = bootstrap_lcb_speaker(lower_by_speaker[band])

    slope_summary = {}
    for band in bands:
        per_speaker = []
        for speaker in speakers:
            vals = []
            for row in rows:
                if row["speaker_hash"] != speaker:
                    continue
                if band == "150_300" and not row["tail_eligible"]:
                    continue
                v = row["models"][variant]["slopes"].get(band, {}).get("delta_db_s")
                if v is not None and np.isfinite(v):
                    vals.append(float(v))
            if vals:
                per_speaker.append(float(np.median(vals)))
        slope_summary[band] = summarize(per_speaker)
    w2_all = [item for row in rows for item in row["models"][variant]["tails"]
              if item["band_ms"] == "300_600"]
    informative = [item for item in w2_all if w2_is_informative(item)]
    certain_regressions = 0
    for item in informative:
        _lower, upper, _status = interval_for_tail(item)
        if upper < -1.0:
            certain_regressions += 1
    # The denominator is every frozen RIR pair. A missing W2 interval is
    # uninformative, not a failed evaluation or an eligible W2 measurement.
    w2_safety = {"pair_count": len(rows),
        "informative_pair_count": len(informative),
        "informative_fraction": len(informative) / len(rows) if rows else None,
        "informative_rule": "input_tail_db > common_floor_db + 3 dB; W2 informative rows include BASE_ONLY_CONTROL",
        "tail_summary_denominator": "informative W2 rows only",
        "certain_regression_below_minus1_count": certain_regressions,
        "certain_regression_denominator": len(informative),
        "certain_regression_fraction": (certain_regressions / len(informative)
                                         if informative else None)}
    return {"speaker_count": len(speakers), "pair_count": len(rows),
        "dry_speech_db_per_speaker": speech_summary,
        "weak_loss_fractions_per_speaker": fraction_summary,
        "noise_only": noise_summary, "tail_reduction_vs_cap60": tail_summary,
        "tail_reduction_speaker_cluster_bootstrap_lcb95_db": lcb,
        "tail_slope_delta_db_s_per_speaker": slope_summary,
        "w2_safety": w2_safety,
        "zero_invariant": {"passed": 100, "required": 100},
        "mean_forward_seconds_per_branch": float(np.mean([
            elapsed for metric in metrics for elapsed in metric["forward_seconds"].values()]))}


def gate_summary(summary: dict) -> dict:
    gates = {}
    value = summary["tail_reduction_vs_cap60"]["150_300"]["Mlow_db"]
    gates["tail_Mlow_150_300_ge_2db"] = (
        "NE" if value is None else "PASS" if value >= 2.0 else "FAIL")
    slope = summary["tail_slope_delta_db_s_per_speaker"]["150_300"]["median"]
    gates["tail_slope_delta_150_300_le_2db_s"] = (
        "NE" if slope is None else "PASS" if slope <= 2.0 else "FAIL")
    w2 = summary["w2_safety"]
    mlow = summary["tail_reduction_vs_cap60"]["300_600"]["Mlow_db"]
    info = w2["informative_fraction"]
    regressions = w2["certain_regression_fraction"]
    if info is None or info < .25:
        gates["w2_safety_Mlow_ge_minus0_5db"] = "NE"
        gates["w2_certain_regressions_below_minus1_le_10pct"] = "NE"
    else:
        gates["w2_safety_Mlow_ge_minus0_5db"] = (
            "NE" if mlow is None else "PASS" if mlow >= -0.5 else "FAIL")
        gates["w2_certain_regressions_below_minus1_le_10pct"] = (
            "NE" if regressions is None else "PASS" if regressions <= .10 else "FAIL")
    gates["w2_informative_coverage_ge_25pct"] = (
        "NE" if info is None else "PASS" if info >= .25 else "NE")
    speech = summary["dry_speech_db_per_speaker"]
    for key, fn in (("active_p50_db", lambda v: -.5 <= v <= .5),
            ("active_p10_db", lambda v: v > -1.5),
            ("weak_p10_db", lambda v: v > -1.5),
            ("weak_p50_db", lambda v: -1.0 <= v <= 1.0),
            ("onset_p10_db", lambda v: v > -1.0)):
        value = speech[key]["median"]
        gates[f"speech_{key}"] = "NE" if value is None else "PASS" if fn(value) else "FAIL"
    fractions = summary["weak_loss_fractions_per_speaker"]
    for key, bound in (("weak_fraction_below_minus3_db", .10),
                       ("weak_fraction_below_minus6_db", .03)):
        value = fractions[key]["median"]
        gates[f"{key}_le_{bound}"] = "NE" if value is None else "PASS" if value <= bound else "FAIL"
    for condition in ("fan20", "fan10"):
        noise = summary["noise_only"][condition]
        median = noise["ac_delta_db_per_speaker"]["median"]
        p90 = noise["ac_delta_db_per_speaker"]["p90"]
        dc_median = noise["dc_dbfs_per_speaker"]["median"]
        dc_p90 = noise["dc_dbfs_per_speaker"]["p90"]
        gates[f"{condition}_floor_median_le_3db"] = "NE" if median is None else "PASS" if median <= 3 else "FAIL"
        gates[f"{condition}_floor_p90_lt_6db"] = "NE" if p90 is None else "PASS" if p90 < 6 else "FAIL"
        gates[f"{condition}_dc_median_lt_minus80dbfs"] = "NE" if dc_median is None else "PASS" if dc_median < -80 else "FAIL"
        gates[f"{condition}_dc_p90_lt_minus70dbfs"] = "NE" if dc_p90 is None else "PASS" if dc_p90 < -70 else "FAIL"
    gates["zero_invariant_100_of_100"] = (
        "PASS" if summary["zero_invariant"]["passed"] == 100 else "FAIL")
    return gates


def run_split(split: str, requested_winner: str | None, device_name: str,
              deep_filter: Path) -> dict:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    split_root = EXPERIMENT / "splits"
    index_path = split_root / "manifest.json"
    if not index_path.is_file():
        raise FileNotFoundError("G splits must be frozen before evaluation")
    split_index = json.loads(index_path.read_text())
    split_audit_path = EXPERIMENT / "split-integrity-audit.json"
    if not split_audit_path.is_file():
        raise FileNotFoundError("independent v2 split audit must pass before any model forward")
    split_audit = json.loads(split_audit_path.read_text())
    if (not split_index.get("frozen_before_training") or
            split_index.get("test_wav_accessed") is not False or
            not split_index.get("dev_coverage_gate_pass") or
            not split_index.get("holdout_coverage_gate_pass") or
            split_audit.get("holdout_opened") is not False or
            split_audit.get("opened_for_metrics") is not False or
            split_audit.get("split_index_sha256") != sha(index_path)):
        raise RuntimeError("G-early split freeze record or coverage gates are invalid")
    if split_index.get("training_pair_manifest_sha256") != sha(
            EXPERIMENT / "data/pairs.jsonl"):
        raise RuntimeError("G-early split index points to a different training pair manifest")
    data_path = EXPERIMENT / "data/pairs.jsonl"
    data_manifest_path = EXPERIMENT / "data/training-data-manifest.json"
    data_meta = json.loads(data_manifest_path.read_text())
    if data_meta.get("test_wav_accessed") is not False:
        raise RuntimeError("G training data manifest violates sealed-data policy")
    if split not in ("dev", "sealed"):
        raise ValueError("split must be dev or sealed")
    if split == "sealed":
        dev_path = EXPERIMENT / "dev-evaluation/summary.json"
        if not dev_path.is_file():
            raise FileNotFoundError("DEV must be opened and a winner frozen before sealed HOLDOUT")
        dev = json.loads(dev_path.read_text())
        if requested_winner != dev.get("selected_winner") or requested_winner not in VARIANTS:
            raise RuntimeError("HOLDOUT-G accepts only the winner frozen by DEV")
        if dev["gates_by_variant"][requested_winner]["overall_state"] not in (
                "PASS", "PASS_EARLY_ONLY_PROVISIONAL"):
            raise RuntimeError("DEV model failed a primary gate; holdout stays sealed")
        variants = (requested_winner,)
        out_dir = EXPERIMENT / "sealed/evaluation"
        split_dir = split_root / "sealed"
    else:
        variants = VARIANTS
        out_dir = EXPERIMENT / "dev-evaluation"
        split_dir = split_root / "dev"
    if (out_dir / "summary.json").exists():
        raise FileExistsError(f"{split} evaluation is already complete and one-shot: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    split_manifest_path = split_dir / "manifest.json"
    split_manifest = json.loads(split_manifest_path.read_text())
    if split_manifest.get("opened_for_metrics") or not split_manifest.get("frozen_before_training"):
        raise RuntimeError(f"{split} manifest is already opened or was not frozen")

    if split == "sealed":
        decision_path = EXPERIMENT / "dev-evaluation/dev-decision.json"
        if not decision_path.is_file():
            raise RuntimeError("HOLDOUT-G requires a frozen DEV decision artifact before any model load")
        decision = json.loads(decision_path.read_text())
        summary_path = EXPERIMENT / "dev-evaluation/summary.json"
        dev_manifest_path = split_root / "dev/manifest.json"
        if (not decision.get("dev_pass") or decision.get("winner") != requested_winner or
                decision.get("protocol_version") != "G-early-v2-real-2026-10-10" or
                decision.get("dev_manifest_sha256") != sha(dev_manifest_path) or
                decision.get("dev_summary_sha256") != sha(summary_path) or
                decision.get("winner_checkpoint_sha256") != sha(
                    EXPERIMENT / requested_winner / "checkpoints/step-003000.pt") or
                decision.get("evaluator_source_sha256") != sha(HERE / "evaluate_g_early_v2.py") or
                decision.get("training_source_sha256") != sha(HERE / "train_g_early_v2.py") or
                decision.get("split_freeze_source_sha256") != sha(HERE / "freeze_g_early_v2_splits.py") or
                decision.get("training_manifest_sha256") != sha(
                    EXPERIMENT / "data/training-data-manifest.json")):
            raise RuntimeError("HOLDOUT-G DEV decision hashes, gate state, or winner do not match")
        if decision.get("gate_states") != dev["gates_by_variant"][requested_winner]:
            raise RuntimeError("HOLDOUT-G decision gate states do not match the frozen DEV summary")

    # Recreate and validate the exact frozen Cap60 a_q reference before model
    # load/forward. The split decision and score then share one Eref definition.
    frozen_references, reference_audit = audit_frozen_tail_references(
        split, split_dir, split_manifest)

    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available()
                          else "cpu" if device_name == "auto" else device_name)
    models, checkpoint_hashes = {}, {}
    for variant in variants:
        model, checkpoint, config = load_g_model(variant, device)
        if config.get("data_manifest_sha256") != sha(data_manifest_path):
            raise RuntimeError("G checkpoint points to different training data")
        models[variant], checkpoint_hashes[variant] = model, sha(checkpoint)
    zero_counts = {variant: assert_zero_100(model, device) for variant, model in models.items()}

    cache, work = out_dir / "cap60-cache", out_dir / "cap60-work"
    cache.mkdir(exist_ok=True)
    work.mkdir(exist_ok=True)
    reuse = index_cap60_cache(split_dir / "cap60-cache")
    rows, started, cache_seconds, pair_index = [], time.perf_counter(), 0.0, 0
    cap60_reused, cap60_new = 0, 0
    for speaker_doc in split_manifest["speakers"]:
        speaker = str(speaker_doc["speaker_id"])
        files = {path.name: path for path in (LIBRISPEECH / "train-clean-360" / speaker).glob("*/*.flac")}
        for spec in speaker_doc["pairs"]:
            paths = [files[spec[key]] for key in ("first", "second")]
            if [sha(path) for path in paths] != [spec["first_sha256"], spec["second_sha256"]]:
                raise RuntimeError("frozen G source utterance hash mismatch")
            joined, pause = make_inserted_pause_pair(*paths)
            joined = joined.astype(np.float32)
            joined *= .80 / max(float(np.max(np.abs(joined))), 1e-8)
            rir_spec = spec["procedural_rir"]
            rir, _early, _late = make_procedural_rir(SR, int(rir_spec["seed"]),
                float(rir_spec["t60_s"]), float(rir_spec["direct_to_reverb_db"]),
                max_t60_s=2.0, min_direct_to_reverb_db=-20.0)
            pair = measured_pair(joined, rir)
            clean, wet = pair["clean"], pair["reverberant"]
            mix20, noise20 = make_fan_components(clean, wet, int(spec["fan20_seed"]), 20)
            mix10, noise10 = make_fan_components(clean, wet, int(spec["fan10_seed"]), 10)
            frozen_reference = frozen_references[(speaker, int(spec["pair_index"]))]
            scale = min(1.0, .95 / max(float(np.max(np.abs(clean))),
                float(np.max(np.abs(wet))), float(np.max(np.abs(mix20))),
                float(np.max(np.abs(mix10))), 1e-8))
            branches = {"dry": clean, "wet": wet, "mix20": mix20,
                "mix10": mix10, "noise20": noise20, "noise10": noise10}
            caps, cap_meta = {}, {}
            for name, signal in branches.items():
                cap_input = np.asarray(signal * scale, np.float32)
                input_hash = hashlib.sha256(cap_input.tobytes()).hexdigest()
                reused_exact = input_hash in reuse
                caps[name], cap_meta[name] = cap60_cached(deep_filter,
                    cap_input, work,
                    f"{split}-{pair_index:03d}-{name}", cache, reuse)
                if reused_exact:
                    cap60_reused += 1
                else:
                    cap60_new += 1
                    cache_seconds += float(cap_meta[name]["elapsed_s"])
                caps[name] *= float(spec["common_post_cap_gain"])
            truth = (clean * scale * np.float32(spec["common_post_cap_gain"])).astype(np.float32)
            model_metrics = {}
            for variant, model in models.items():
                outputs, timings, mechanism = {}, {}, None
                for name in branches:
                    if name == "wet":
                        outputs[name], mechanism, timings[name] = infer_with_mechanism(
                            model, caps[name], pause, device)
                    else:
                        outputs[name], timings[name] = infer(model, caps[name], device)
                    if not np.isfinite(outputs[name]).all() or float(np.max(np.abs(outputs[name]))) >= 1.0:
                        raise RuntimeError(f"non-finite or clipped G output: {variant}/{name}")
                speech, weak_fractions = speech_truth_stats(truth, outputs["dry"], pause)
                noise = {"fan20": noise_stats(caps["noise20"], outputs["noise20"]),
                         "fan10": noise_stats(caps["noise10"], outputs["noise10"])}
                tails = g_tail_stats(caps["dry"], caps["wet"], outputs["wet"],
                    outputs["dry"], [caps["noise20"], outputs["noise20"],
                                     caps["noise10"], outputs["noise10"]], pause,
                    frozen_reference["reference_energy"])
                slopes = {}
                for item in tails:
                    band, common = item["band_ms"], item["common_floor_db"]
                    out_slope = g_tail_slope(caps["wet"], outputs["wet"], pause,
                        frozen_reference["reference_energy"], common)
                    in_slope = g_tail_slope(caps["wet"], caps["wet"], pause,
                        frozen_reference["reference_energy"], common)
                    slopes[band] = {"input_db_s": in_slope, "output_db_s": out_slope,
                        "delta_db_s": out_slope - in_slope
                            if out_slope is not None and in_slope is not None else None}
                model_metrics[variant] = {"speech": speech,
                    "weak_fractions": weak_fractions, "noise_only": noise,
                    "tails": tails, "slopes": slopes, "forward_seconds": timings,
                    "mechanism_diagnostics": mechanism}
            rows.append({"pair_index": pair_index,
                "speaker_hash": hashlib.sha256(speaker.encode()).hexdigest(),
                "rir_seed": int(rir_spec["seed"]), "shared_pre_cap_scale": scale,
                "tail_eligible": bool(spec["tail_loss_enabled"]),
                "classification": spec["classification"],
                "frozen_reference_energy": frozen_reference["reference_energy"],
                "frozen_reference_energy_sha256": frozen_reference["reference_energy_sha256"],
                "models": model_metrics})
            pair_index += 1
            if pair_index % 4 == 0:
                print(json.dumps({"split": split, "pairs": pair_index,
                    "total": split_manifest["pair_count"],
                    "elapsed_s": time.perf_counter() - started}), flush=True)

    if pair_index != int(split_manifest["pair_count"]):
        raise RuntimeError("G evaluation did not cover every frozen speaker pair")
    for row in rows:
        for variant in variants:
            bands = {item["band_ms"] for item in row["models"][variant]["tails"]}
            if row["tail_eligible"] and "150_300" not in bands:
                raise RuntimeError("W1-eligible G pair is missing its primary 150–300 ms interval")

    summary = {"protocol": split_manifest["name"],
        "protocol_version": "G-early-v2-real-2026-10-10",
        "evaluator_source_sha256": sha(HERE / "evaluate_g_early_v2.py"),
        "training_source_sha256": sha(HERE / "train_g_early_v2.py"),
        "split_freeze_source_sha256": sha(HERE / "freeze_g_early_v2_splits.py"),
        "split_index_sha256": sha(index_path),
        "training_manifest_sha256": sha(data_manifest_path),
        "evaluation_code_amendment_sha256": validate_evaluator_binding(
            json.loads((EXPERIMENT / variants[0] / "model-config.json").read_text())),
        "frozen_reference_audit": reference_audit,
        "split_manifest_sha256": sha(split_manifest_path),
        "training_pair_manifest_sha256": sha(data_path),
        "checkpoints": checkpoint_hashes, "pair_count": len(rows),
        "speaker_count": split_manifest["speaker_count"], "device": str(device),
        "torch": torch.__version__, "cap60_exact_inference_seconds": cache_seconds,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cap60_exact_reused_branches": cap60_reused,
        "cap60_exact_new_branches": cap60_new,
        "evaluation_elapsed_seconds": time.perf_counter() - started,
        "test_wav_accessed": False, "speaker_ids_in_report": False,
        "model_metrics": {}, "gates_by_variant": {},
        "limitations": ["Fresh speaker transfer in the frozen procedural strong-RIR domain only.",
            "Does not establish measured-room generalization or Adobe Podcast parity."]}
    for variant in variants:
        variant_summary = summarize_variant(rows, variant)
        mechanism_rows = [row["models"][variant]["mechanism_diagnostics"] for row in rows]
        mechanism_summary = {}
        for key in ("w1_actual_reduction_db", "w1_gain_only_reduction_db",
                    "w1_phase_only_reduction_db", "w1_input_energy_weighted_gain_mean",
                    "w1_input_energy_weighted_gain_p10", "w1_input_energy_weighted_gain_p50",
                    "w1_input_energy_weighted_gain_p90"):
            mechanism_summary[key] = summarize([float(item[key]) for item in mechanism_rows
                if item is not None and item.get(key) is not None])
        variant_summary["mechanism_diagnostics_descriptive_only"] = mechanism_summary
        variant_summary["zero_invariant"] = {"passed": zero_counts[variant], "required": 100}
        gates = gate_summary(variant_summary)
        non_w2 = [value for key, value in gates.items() if not key.startswith("w2_")]
        w2_states = [value for key, value in gates.items() if key.startswith("w2_")]
        if all(value == "PASS" for value in gates.values()):
            variant_summary["overall_state"] = "PASS"
            variant_summary["claim_level"] = "early W1 with W2 safety gate passed"
        elif (all(value == "PASS" for value in non_w2) and
              all(value in ("PASS", "NE") for value in w2_states)):
            variant_summary["overall_state"] = "PASS_EARLY_ONLY_PROVISIONAL"
            variant_summary["claim_level"] = "early-only provisional; W2 safety not established"
        else:
            variant_summary["overall_state"] = "FAIL"
            variant_summary["claim_level"] = "no positive claim"
        summary["model_metrics"][variant] = variant_summary
        summary["gates_by_variant"][variant] = {**gates,
            "overall_state": variant_summary["overall_state"]}

    if split == "dev":
        passing = [v for v in VARIANTS if summary["gates_by_variant"][v]["overall_state"] in
                   ("PASS", "PASS_EARLY_ONLY_PROVISIONAL")]
        if len(passing) == 1:
            summary["selected_winner"] = passing[0]
        elif len(passing) == 2:
            def tie_key(variant):
                m = summary["model_metrics"][variant]
                def finite_or_low(value):
                    return (float(value) if value is not None and np.isfinite(value)
                            else float("-inf"))
                def finite_or_high(value):
                    return (float(value) if value is not None and np.isfinite(value)
                            else float("inf"))
                return (finite_or_low(m["tail_reduction_speaker_cluster_bootstrap_lcb95_db"]["300_600"]),
                    finite_or_low(m["tail_reduction_speaker_cluster_bootstrap_lcb95_db"]["150_300"]),
                    finite_or_low(m["dry_speech_db_per_speaker"]["weak_p10_db"]["median"]),
                    -finite_or_high(m["weak_loss_fractions_per_speaker"]["weak_fraction_below_minus3_db"]["median"]))
            summary["selected_winner"] = max(passing, key=tie_key)
        else:
            summary["selected_winner"] = None
    else:
        summary["selected_winner"] = requested_winner
        summary["holdout_state"] = summary["model_metrics"][requested_winner]["overall_state"]
    (out_dir / "evaluation.jsonl").write_text("".join(
        json.dumps(json_finite(row), allow_nan=False) + "\n" for row in rows))
    printable_summary = json_finite(summary)
    (out_dir / "summary.json").write_text(json.dumps(printable_summary, indent=2, allow_nan=False) + "\n")
    if split == "dev" and summary.get("selected_winner") in VARIANTS:
        winner = summary["selected_winner"]
        winner_state = summary["gates_by_variant"][winner]["overall_state"]
        dev_pass = winner_state in ("PASS", "PASS_EARLY_ONLY_PROVISIONAL")
        passing = [v for v in VARIANTS if summary["gates_by_variant"][v]["overall_state"] in
                   ("PASS", "PASS_EARLY_ONLY_PROVISIONAL")]
        rule = ("sole_passing_variant" if len(passing) == 1 else
            "max_lexicographic_lcb300_600,lcb150_300,weak_p10,-weak_loss_below_minus3; missing/nonfinite metrics rank last")
        winner_checkpoint = EXPERIMENT / winner / "checkpoints/step-003000.pt"
        decision = {"name": "G-early-v2-dev-decision", "dev_pass": dev_pass,
            "protocol_version": "G-early-v2-real-2026-10-10",
            "winner": winner, "winner_state": winner_state, "rule": rule,
            "gate_states": summary["gates_by_variant"][winner],
            "evaluator_source_sha256": sha(HERE / "evaluate_g_early_v2.py"),
            "training_source_sha256": sha(HERE / "train_g_early_v2.py"),
            "split_freeze_source_sha256": sha(HERE / "freeze_g_early_v2_splits.py"),
            "training_manifest_sha256": sha(EXPERIMENT / "data/training-data-manifest.json"),
            "dev_manifest_sha256": sha(split_root / "dev/manifest.json"),
            "dev_summary_sha256": sha(out_dir / "summary.json"),
            "winner_checkpoint_sha256": sha(winner_checkpoint),
            "test_wav_accessed": False}
        (out_dir / "dev-decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    if split == "sealed":
        receipt = {"split_manifest_sha256": sha(split_manifest_path),
            "winner": requested_winner, "checkpoint_sha256": checkpoint_hashes[requested_winner],
            "opened_once": True, "test_wav_accessed": False}
        (out_dir / "opening-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(printable_summary, indent=2, allow_nan=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "sealed"), required=True)
    parser.add_argument("--winner", choices=VARIANTS)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--deep-filter", type=Path, default=DEEP_FILTER)
    args = parser.parse_args()
    if args.split == "sealed" and not args.winner:
        parser.error("--winner is required for the one-time HOLDOUT-G evaluation")
    run_split(args.split, args.winner, args.device, args.deep_filter)


if __name__ == "__main__":
    main()
