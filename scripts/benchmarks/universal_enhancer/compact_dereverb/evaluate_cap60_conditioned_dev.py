#!/usr/bin/env python3
"""Evaluate frozen Cap60→C against the paired Cap60 route on BUT dev only."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts/benchmarks/universal_enhancer"))
sys.path.insert(0, str(HERE))
from bench_air_cap60_c_factorial import bootstrap, pause_decay_slope_db_s  # noqa: E402
from bench_air_dpdfnet_factorial import steady_speech_metrics  # noqa: E402
from screen_measured_rirs import (  # noqa: E402
    infer, load_model, pause_metrics,
)

SR = 16_000
DEV_ROOMS = ("Hotel_SkalskyDvur_Room112", "VUT_FIT_L212", "VUT_FIT_L227")
DEFAULT_DATA = ROOT / ".tools/compact-dereverb/cap60-conditioned-2026-10-10"
DEFAULT_CHECKPOINT = DEFAULT_DATA / "training/checkpoints/step-003000.pt"
DEFAULT_OUTPUT = DEFAULT_DATA / "dev-evaluation"
BANDS = ("150_300", "300_600")
CONDITIONS = ("rir_only", "fan20", "fan10")


def load_array(root: Path, rel: str) -> np.ndarray:
    audio = np.load(root / rel, allow_pickle=False).astype(np.float32, copy=False)
    if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
        raise RuntimeError(f"invalid cached waveform: {rel}")
    return audio


def absolute_floor(signal: np.ndarray, pause_start: int, band: str) -> dict:
    lo, hi = (int(value) for value in band.split("_"))
    start = pause_start + int(lo * SR / 1000)
    end = pause_start + int(hi * SR / 1000)
    segment = np.asarray(signal[start:end], dtype=np.float64)
    if not segment.size:
        return {"sample_count": 0, "rms_dbfs": None,
                "demeaned_rms_dbfs": None, "mean": None, "exact_zero": False}
    mean = float(np.mean(segment))
    rms = float(np.sqrt(np.mean(segment ** 2)))
    demeaned_rms = float(np.sqrt(np.mean((segment - mean) ** 2)))
    to_dbfs = lambda value: (20 * np.log10(value) if value > 0 else None)
    return {"sample_count": int(segment.size), "rms_dbfs": to_dbfs(rms),
            "demeaned_rms_dbfs": to_dbfs(demeaned_rms), "mean": mean,
            "exact_zero": bool(rms == 0)}


def add_common_floor(rows: list[dict], noise_rows: list[dict] | None) -> list[dict]:
    noise_by_band = ({row["band_ms"]: row for row in noise_rows}
                     if noise_rows is not None else {})
    for row in rows:
        noise = noise_by_band.get(row["band_ms"])
        noise_floor = (noise["output_tail_vs_same_fixed_input_speech_db"]
                       if noise is not None else
                       row["dry_model_floor_vs_fixed_input_speech_db"])
        row["noise_only_floor_db"] = noise_floor
        row["common_output_floor_db"] = max(
            row["dry_model_floor_vs_fixed_input_speech_db"], noise_floor)
        row["common_floor_censored"] = bool(
            row["output_tail_vs_same_fixed_input_speech_db"] <=
            row["common_output_floor_db"] + 3.0)
    return rows


def summarise_tail(rows: list[dict], condition: str, band: str) -> dict:
    paired: dict[tuple[str, str], dict[str, dict]] = {}
    for row in rows:
        if row["condition"] != condition or row["band_ms"] != band:
            continue
        paired.setdefault((row["candidate_index"], row["speaker"], row["room"],
                           row["rir_configuration"]), {})[
            row["route"]] = row
    gains: list[tuple[str, float]] = []
    censored = 0
    by_room: dict[str, list[float]] = {}
    for (_candidate_index, speaker, room, _rir_configuration), routes in paired.items():
        if set(routes) != {"cap60", "cap60_then_c"}:
            continue
        base, candidate = routes["cap60"], routes["cap60_then_c"]
        if base["common_floor_censored"] or candidate["common_floor_censored"]:
            censored += 1
            continue
        gain = base["output_tail_vs_same_fixed_input_speech_db"] - candidate[
            "output_tail_vs_same_fixed_input_speech_db"]
        gains.append((speaker, gain))
        by_room.setdefault(room, []).append(gain)
    values = [value for _, value in gains]
    return {"valid_pair_count": len(values), "censored_pair_count": censored,
            "speaker_count": len({speaker for speaker, _ in gains}),
            "positive_pair_count": sum(value > 0 for value in values),
            "positive_pair_fraction": float(np.mean(np.asarray(values) > 0)) if values else None,
            "median_gain_db": float(np.median(values)) if values else None,
            "speaker_cluster_bootstrap_95ci_db": bootstrap(gains),
            "positive_room_count": sum(float(np.median(x)) > 0 for x in by_room.values()),
            "measurable_pair_count_by_room": {room: len(x) for room, x in by_room.items()},
            "room_medians_db": {room: float(np.median(x)) for room, x in by_room.items()}}


def summarize_delta(rows: list[tuple[str, str, float]]) -> dict:
    values = [float(value) for _, _, value in rows]
    clustered = [(speaker, value) for speaker, _room, value in rows]
    return {"valid_pair_count": len(values),
            "measurable_pair_count_by_room": {room: sum(item[1] == room for item in rows)
                                              for room in sorted({item[1] for item in rows})},
            "median_db": float(np.median(values)) if values else None,
            "absolute_median_db": float(np.median(np.abs(values))) if values else None,
            "p90_increase_db": float(np.percentile(values, 90)) if values else None,
            "speaker_cluster_bootstrap_95ci_db": bootstrap(clustered)}


def gate_state(raw_pass: bool | None, *, coverage_sufficient: bool) -> str:
    """Keep observed regressions visible, but don't promote under-covered wins."""
    if raw_pass is None:
        return "NE"
    if not raw_pass:
        return "FAIL"
    return "PASS" if coverage_sufficient else "NE"


def make_summary(case_rows: list[dict], speech_deltas: dict[str, list[tuple[str, float]]],
                 identity_floor_levels: dict[str, list[dict]],
                 slope_deltas: list[tuple[str, str, float]], manifest: dict) -> dict:
    tails = {f"{condition}/{band}": summarise_tail(case_rows, condition, band)
             for condition in CONDITIONS for band in BANDS}
    noise_floors = {}
    for condition in ("fan20", "fan10"):
        for band in BANDS:
            rows = [(row["speaker"], row["room"],
                     row["candidate_noise_floor_db"] - row["baseline_noise_floor_db"])
                    for row in case_rows
                    if row["condition"] == condition and row["band_ms"] == band and
                    row["route"] == "cap60_then_c"]
            noise_floors[f"{condition}/{band}"] = summarize_delta(rows)

    def summarize_absolute_levels(selected: list[dict]) -> dict:
        measurable = [item for item in selected if item["rms_dbfs"] is not None]
        ac = [item["demeaned_rms_dbfs"] for item in measurable
              if item["demeaned_rms_dbfs"] is not None]
        dc = [20 * np.log10(abs(item["mean"])) for item in measurable if item["mean"] != 0]
        return {
            "case_count": len(selected),
            "exact_zero_count": sum(item["exact_zero"] for item in selected),
            "median_rms_dbfs": (float(np.median([x["rms_dbfs"] for x in measurable]))
                                if measurable else None),
            "median_demeaned_rms_dbfs": float(np.median(ac)) if ac else None,
            "median_abs_dc_dbfs": float(np.median(dc)) if dc else None,
            "median_mean": (float(np.median([x["mean"] for x in measurable]))
                            if measurable else None),
        }

    absolute_floors = {}
    for condition in ("dry_control", "fan20", "fan10"):
        for branch in ("dry_control", "noise_only"):
            if branch == "noise_only" and condition == "dry_control":
                continue
            for route in ("cap60", "cap60_then_c"):
                for band in BANDS:
                    selected = [row[f"{branch}_absolute_floor"] for row in case_rows
                                if row["condition"] == ("rir_only" if branch == "dry_control"
                                                         else condition)
                                and row["route"] == route and row["band_ms"] == band]
                    key = f"{condition}/{branch}/{route}/{band}"
                    absolute_floors[key] = summarize_absolute_levels(selected)
                    absolute_floors[key]["by_room"] = {
                        room: summarize_absolute_levels([row[f"{branch}_absolute_floor"]
                            for row in case_rows if row["room"] == room and
                            row["condition"] == ("rir_only" if branch == "dry_control" else condition)
                            and row["route"] == route and row["band_ms"] == band])
                        for room in DEV_ROOMS}
    dry_floor = {}
    for band, rows in identity_floor_levels.items():
        base = [row["base"] for row in rows]
        candidate = [row["candidate"] for row in rows]
        base_measurable = [row for row in base if row["rms_dbfs"] is not None]
        candidate_measurable = [row for row in candidate if row["rms_dbfs"] is not None]
        candidate_dc_dbfs = [20 * np.log10(abs(row["mean"])) for row in candidate_measurable
                             if row["mean"] != 0]
        exact_zero_references = sum(row["exact_zero"] for row in base)
        valid_deltas = [row["candidate"]["rms_dbfs"] - row["base"]["rms_dbfs"]
                        for row in rows if row["base"]["rms_dbfs"] is not None and
                        row["candidate"]["rms_dbfs"] is not None]
        dry_floor[band] = {
            "valid_pair_count": len(rows),
            "exact_zero_reference_count": exact_zero_references,
            "finite_delta_pair_count": len(valid_deltas),
            "median_delta_db": float(np.median(valid_deltas)) if valid_deltas else None,
            "base_median_rms_dbfs": (float(np.median([row["rms_dbfs"] for row in base_measurable]))
                                      if base_measurable else None),
            "candidate_median_rms_dbfs": (
                float(np.median([row["rms_dbfs"] for row in candidate_measurable]))
                if candidate_measurable else None),
            "candidate_median_demeaned_rms_dbfs": (
                float(np.median([row["demeaned_rms_dbfs"] for row in candidate_measurable
                                 if row["demeaned_rms_dbfs"] is not None]))
                if any(row["demeaned_rms_dbfs"] is not None for row in candidate_measurable)
                else None),
            "candidate_median_abs_dc_dbfs": (float(np.median(candidate_dc_dbfs))
                                             if candidate_dc_dbfs else None),
            "candidate_median_mean": (float(np.median([row["mean"] for row in candidate_measurable]))
                                      if candidate_measurable else None),
        }
    speech = {}
    for name, values in speech_deltas.items():
        dbs = [value for _, value in values]
        speech[name] = {"valid_pair_count": len(dbs),
                        "median_db": float(np.median(dbs)) if dbs else None,
                        "speaker_cluster_bootstrap_95ci_db": bootstrap(values)}
    slope_delta = summarize_delta(slope_deltas)
    slope_summary = {"valid_pair_count": slope_delta["valid_pair_count"],
                     "measurable_pair_count_by_room": slope_delta["measurable_pair_count_by_room"],
                     "median_delta_db_per_second": slope_delta["median_db"],
                     "speaker_cluster_bootstrap_95ci_db_per_second":
                         slope_delta["speaker_cluster_bootstrap_95ci_db"]}
    room_counts = manifest["dev_room_case_counts"]
    identity_count = int(manifest["dev_identity_evaluation_cases"])

    def numeric(value, predicate, coverage: bool):
        return gate_state(None if value is None else bool(predicate(value)),
                          coverage_sufficient=coverage)

    dry_floor_states = {}
    for band, delta in dry_floor.items():
        if (delta["valid_pair_count"] > 0 and
                delta["exact_zero_reference_count"] == delta["valid_pair_count"] and
                delta["candidate_median_rms_dbfs"] is not None):
            # A finite +dB value against a digital zero has no physical meaning;
            # the predeclared relative gate still fails because C emitted audio.
            dry_floor_states[band] = "FAIL"
        elif delta["finite_delta_pair_count"] == 0:
            dry_floor_states[band] = "NE"
        else:
            dry_floor_states[band] = numeric(delta["median_delta_db"], lambda x: x <= 3.0, False)

    gates = {
        "dev_rir_150_300_median_gain_ge_2db": numeric(
            tails["rir_only/150_300"]["median_gain_db"], lambda x: x >= 2.0,
            all(tails["rir_only/150_300"]["measurable_pair_count_by_room"].get(room, 0) >= 4
                for room in DEV_ROOMS)),
        "dev_rir_300_600_median_gain_ge_1_5db": numeric(
            tails["rir_only/300_600"]["median_gain_db"], lambda x: x >= 1.5,
            all(tails["rir_only/300_600"]["measurable_pair_count_by_room"].get(room, 0) >= 4
                for room in DEV_ROOMS)),
        "dev_rir_positive_in_at_least_75pct_measurable_pairs": {
            band: numeric(tails[f"rir_only/{band}"]["positive_pair_fraction"],
                          lambda x: x >= .75,
                          all(tails[f"rir_only/{band}"]["measurable_pair_count_by_room"].get(room, 0) >= 4
                              for room in DEV_ROOMS)) for band in BANDS},
        "dev_rir_positive_in_all_three_rooms": {
            band: gate_state(
                None if tails[f"rir_only/{band}"]["valid_pair_count"] == 0 else
                tails[f"rir_only/{band}"]["positive_room_count"] == sum(
                    count > 0 for count in tails[f"rir_only/{band}"]["measurable_pair_count_by_room"].values()),
                coverage_sufficient=all(
                    tails[f"rir_only/{band}"]["measurable_pair_count_by_room"].get(room, 0) >= 4
                    for room in DEV_ROOMS)) for band in BANDS},
        "dev_decay_slope_delta_le_plus_2db_s": numeric(
            slope_summary["median_delta_db_per_second"], lambda x: x <= 2.0,
            all(slope_summary["measurable_pair_count_by_room"].get(room, 0) >= 4
                for room in DEV_ROOMS)),
        "dev_dry_active_median_within_half_db": numeric(
            speech["active_p50"]["median_db"], lambda x: abs(x) <= .5, False),
        "dev_dry_onset_p10_ge_minus_1db": numeric(
            speech["onset_p10"]["median_db"], lambda x: x >= -1.0, False),
        "dev_dry_weak_p10_ge_minus_1_5db": numeric(
            speech["weak_p10"]["median_db"], lambda x: x >= -1.5, False),
        "dev_dry_weak_p50_within_1db": numeric(
            speech["weak_p50"]["median_db"], lambda x: abs(x) <= 1.0, False),
        "dev_noise_floor_absolute_median_le_3db": {
            key: numeric(value["absolute_median_db"], lambda x: x <= 3.0,
                         all(value.get("measurable_pair_count_by_room", {}).get(room, 0) >= 4
                             for room in DEV_ROOMS))
            for key, value in noise_floors.items()},
        "dev_noise_floor_p90_increase_lt_6db": {
            key: numeric(value["p90_increase_db"], lambda x: x < 6.0,
                         all(value.get("measurable_pair_count_by_room", {}).get(room, 0) >= 4
                             for room in DEV_ROOMS))
            for key, value in noise_floors.items()},
        "dev_dry_floor_median_increase_le_3db": {
            band: dry_floor_states[band] for band in dry_floor},
    }
    flat_states = [state for value in gates.values() for state in
                   (value.values() if isinstance(value, dict) else [value])]
    overall_state = "FAIL" if "FAIL" in flat_states else (
        "NE" if "NE" in flat_states else "PASS")
    return {"protocol": "Frozen Cap60 vs Cap60→C BUT development gates",
            "interpretation": "Development-only GO/NO-GO monitoring. AIR is a separate one-shot diagnostic. Positive tail gain means C has lower tail energy than Cap60; censored cases are excluded when either route is within 3 dB of its matched common floor.",
            "tail_comparisons": tails, "noise_only_floor_delta": noise_floors,
            "absolute_pause_floors": absolute_floors,
            "dry_speech_incremental": speech, "dry_floor_incremental": dry_floor,
            "pause_decay_slope_rir_only": slope_summary, "predeclared_gates": gates,
            "gate_legend": {"PASS": "criterion met with sufficient predeclared coverage",
                            "FAIL": "criterion not met",
                            "NE": "not evaluable as a positive result: missing data or insufficient coverage"},
            "coverage": {"rir_candidate_cases_by_room": room_counts,
                         "rir_minimum_measurable_per_room": 4,
                         "identity_cases": identity_count,
                         "identity_positive_gates_require_more_evidence": True},
            "overall_state": overall_state, "overall_pass": overall_state == "PASS",
            "identity_speech_case_count": len(speech_deltas["active_p50"]),
            "case_rows": len(case_rows), "model_manifest": manifest}


def plot_summary(summary: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    tail = summary["tail_comparisons"]
    bands = BANDS
    positions = np.arange(2)
    vals = [tail[f"rir_only/{band}"]["median_gain_db"] for band in bands]
    axes[0, 0].bar(positions, [0 if value is None else value for value in vals],
                color=["#4b8f8c", "#6a9e98"])
    axes[0, 0].hlines(2.0, -.4, .4, color="#4b8f8c", linestyle=":", linewidth=1.5,
                      label="150–300 ms gate: +2 dB")
    axes[0, 0].hlines(1.5, .6, 1.4, color="#7353ba", linestyle="--", linewidth=1.5,
                      label="300–600 ms gate: +1.5 dB")
    axes[0, 0].set_xticks(positions, ["150–300 ms", "300–600 ms"])
    axes[0, 0].set_title("RIR-only tail gain")
    axes[0, 0].set_ylabel("Cap60 minus Cap60→C (dB)")
    axes[0, 0].legend(fontsize=7)

    floor = summary["noise_only_floor_delta"]
    categories = [(condition, band) for condition in ("fan20", "fan10") for band in bands]
    floor_vals = [floor[f"{condition}/{band}"]["median_db"] for condition, band in categories]
    axes[0, 1].bar(np.arange(4), [0 if value is None else value for value in floor_vals],
                color="#d27850")
    axes[0, 1].axhline(0, color="#555", linewidth=.8)
    axes[0, 1].axhline(3, color="#b74747", linestyle=":", linewidth=1,
                       label="absolute median gate: ±3 dB")
    axes[0, 1].axhline(-3, color="#b74747", linestyle=":", linewidth=1)
    axes[0, 1].set_xticks(np.arange(4), [f"{condition}\n{band.replace('_', '–')} ms"
                                    for condition, band in categories])
    axes[0, 1].set_title("Noise-only common-floor delta")
    axes[0, 1].set_ylabel("C minus Cap60 (dB)")
    axes[0, 1].legend(fontsize=7)

    speech = summary["dry_speech_incremental"]
    names = ("active_p50", "onset_p10", "weak_p10", "weak_p50")
    speech_vals = [speech[name]["median_db"] for name in names]
    axes[1, 0].bar(np.arange(4), [0 if value is None else value for value in speech_vals],
                color="#6387a8")
    axes[1, 0].axhline(0, color="#555", linewidth=.8)
    axes[1, 0].axhline(-1, color="#b74747", linestyle=":", linewidth=1)
    axes[1, 0].axhline(1, color="#b74747", linestyle=":", linewidth=1)
    axes[1, 0].set_xticks(np.arange(4), ["active p50", "onset p10", "weak p10", "weak p50"])
    axes[1, 0].set_title("Dry identity speech change")
    axes[1, 0].set_ylabel("Cap60→C minus Cap60 (dB)")

    slope = summary["pause_decay_slope_rir_only"]
    slope_value = slope["median_delta_db_per_second"]
    axes[1, 1].bar([0], [0 if slope_value is None else slope_value], color="#7353ba")
    axes[1, 1].axhline(0, color="#555", linewidth=.8)
    axes[1, 1].axhline(2, color="#b74747", linestyle=":", linewidth=1,
                       label="gate: ≤ +2 dB/s")
    axes[1, 1].set_xticks([0], ["Cap60→C minus Cap60"])
    axes[1, 1].set_title("80–500 ms pause decay slope")
    axes[1, 1].set_ylabel("More negative = faster decay (dB/s)")
    axes[1, 1].legend(fontsize=7)
    for ax in axes.flat:
        ax.grid(axis="y", alpha=.25)
    fig.suptitle("Predeclared BUT dev gates for Cap60-conditioned dereverberator C")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)

    plot_absolute_pause_floors(summary, path.with_name("absolute-pause-floors.png"))


def plot_absolute_pause_floors(summary: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    floors = summary["absolute_pause_floors"]
    scenarios = (
        ("dry_control", "dry_control", "Dry digital pause"),
        ("fan20", "noise_only", "Fan noise, 20 dB SNR"),
        ("fan10", "noise_only", "Fan noise, 10 dB SNR"),
    )
    measures = ("median_rms_dbfs", "median_demeaned_rms_dbfs", "median_abs_dc_dbfs")
    titles = ("Raw RMS", "AC RMS after mean removal", "Absolute DC mean")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
    x = np.arange(len(scenarios))
    width = .34
    for ax, measure, title in zip(axes, measures, titles):
        for offset, route, label, color in (
                (-width / 2, "cap60", "Cap60", "#6d91b8"),
                (width / 2, "cap60_then_c", "Cap60→C", "#d07850")):
            values = []
            for condition, branch, _ in scenarios:
                row = floors[f"{condition}/{branch}/{route}/150_300"]
                values.append(row[measure])
            plotted = [np.nan if value is None else value for value in values]
            ax.bar(x + offset, plotted, width, label=label, color=color)
            for index, value in enumerate(values):
                if value is None and route == "cap60" and scenarios[index][0] == "dry_control":
                    ax.text(x[index] + offset, -116, "exact zero", rotation=90,
                            ha="center", va="bottom", fontsize=8, color=color)
        ax.set_title(title)
        ax.set_xticks(x, [row[2] for row in scenarios], rotation=18, ha="right")
        ax.set_ylabel("dBFS (lower is quieter)")
        ax.grid(axis="y", alpha=.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("BUT dev pause floor, 150–300 ms (n=13; room profile 6/5/2)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def evaluate(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    data_manifest_path = args.data_dir / "training-data-manifest.json"
    dev_manifest_path = args.data_dir / "dev-evaluation.jsonl"
    for path in (data_manifest_path, dev_manifest_path, args.checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)
    data_manifest = json.loads(data_manifest_path.read_text())
    if Path(data_manifest.get("dev_evaluation_manifest", "")).resolve() != dev_manifest_path.resolve():
        raise RuntimeError("development evaluation manifest path differs from frozen data manifest")
    rows = [json.loads(line) for line in dev_manifest_path.read_text().splitlines() if line.strip()]
    rir_rows = [row for row in rows if row.get("kind") == "rir"]
    identity_rows = [row for row in rows if row.get("kind") == "identity"]
    if (len(rir_rows) != data_manifest["dev_evaluation_rir_cases"] or
            len(identity_rows) != data_manifest["dev_evaluation_identity_cases"]):
        raise RuntimeError("development evaluation row count differs from its frozen manifest")
    counts = {room: sum(row["room"] == room for row in rir_rows) for room in DEV_ROOMS}
    coverage_sufficient = set(counts) == set(DEV_ROOMS) and all(value >= 4 for value in counts.values())
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    model = load_model(args.checkpoint, device)
    if sum(parameter.numel() for parameter in model.parameters()) != 555_922:
        raise RuntimeError("dereverberator architecture does not match frozen parameter count")
    args.output_dir.mkdir(parents=True)
    case_rows = []
    speech_deltas = {name: [] for name in ("active_p50", "onset_p10", "weak_p10", "weak_p50")}
    identity_floor_levels = {band: [] for band in BANDS}
    slope_deltas: list[tuple[str, str, float]] = []
    c_forward_seconds = 0.0
    c_forward_calls = 0
    output_rows = []
    for row in rows:
        arrays = {name: load_array(args.data_dir, path) for name, path in row["arrays"].items()}
        lengths = {len(value) for value in arrays.values()}
        if len(lengths) != 1:
            raise RuntimeError(f"dev pair arrays have mismatched lengths for {row['speaker']}")
        clean, target, pause = arrays["clean"], arrays["target"], [row["pause"]]
        dry_c, elapsed = infer(model, target, device)
        c_forward_seconds += elapsed
        c_forward_calls += 1
        if row["kind"] == "identity":
            baseline_speech = steady_speech_metrics(clean, target, target, pause)
            candidate_speech = steady_speech_metrics(clean, target, dry_c, pause)
            base_weak = baseline_speech["weak_output_vs_clean_db_p50_p90_p99"][0]
            cand_weak = candidate_speech["weak_output_vs_clean_db_p50_p90_p99"][0]
            metrics = {
                "active_p50": (baseline_speech["active_output_vs_clean_db_p10_p50_p90"][1],
                               candidate_speech["active_output_vs_clean_db_p10_p50_p90"][1]),
                "onset_p10": (baseline_speech["rising_output_vs_clean_db_p10_p50_p90"][0],
                              candidate_speech["rising_output_vs_clean_db_p10_p50_p90"][0]),
                "weak_p10": (baseline_speech["weak_speech_gain_db_p10"],
                             candidate_speech["weak_speech_gain_db_p10"]),
                "weak_p50": (base_weak, cand_weak),
            }
            for name, (base, candidate) in metrics.items():
                if base is not None and candidate is not None:
                    speech_deltas[name].append((row["speaker"], float(candidate - base)))
            base_dry = pause_metrics(clean, clean, target, target, pause)
            candidate_dry = pause_metrics(clean, clean, dry_c, dry_c, pause)
            for band in BANDS:
                base_metric = next((x for x in base_dry if x["band_ms"] == band), None)
                candidate_metric = next((x for x in candidate_dry if x["band_ms"] == band), None)
                if base_metric and candidate_metric:
                    start_sample = row["pause"]["clip_relative_start_sample"]
                    identity_floor_levels[band].append({
                        "speaker": row["speaker"], "selection_room": row["selection_room"],
                        "base": absolute_floor(target, start_sample, band),
                        "candidate": absolute_floor(dry_c, start_sample, band)})
            output_rows.append({"kind": "identity", "speaker": row["speaker"],
                "selection_room": row["selection_room"], "candidate_index": row["candidate_index"],
                "speech_incremental_db": {name: None if metrics[name][0] is None or metrics[name][1] is None
                    else float(metrics[name][1] - metrics[name][0]) for name in metrics},
                "dry_floor_incremental_db": {band: (None if not next((
                    x["exact_zero"] for x in [absolute_floor(target,
                        row["pause"]["clip_relative_start_sample"], band)]), False)
                    else next((float(c["dry_model_floor_vs_fixed_dry_speech_db"] -
                        b["dry_model_floor_vs_fixed_dry_speech_db"]) for b in base_dry
                        for c in candidate_dry if b["band_ms"] == c["band_ms"] == band), None))
                    for band in BANDS}})
            continue
        if row["kind"] != "rir":
            raise RuntimeError(f"unknown development evaluation row kind: {row['kind']}")
        candidates = {"target": dry_c}
        for condition in CONDITIONS:
            candidates[condition], elapsed = infer(model, arrays[f"x_{condition}"], device)
            c_forward_seconds += elapsed
            c_forward_calls += 1
        for condition in ("fan20", "fan10"):
            candidates[f"noise_{condition}"], elapsed = infer(model, arrays[f"noise_{condition}"], device)
            c_forward_seconds += elapsed
            c_forward_calls += 1

        per_condition = {}
        for condition in CONDITIONS:
            wet = arrays[f"wet_{condition}"]
            noise_only_rows = None
            if condition in ("fan20", "fan10"):
                noise_only_rows = {
                    "cap60": pause_metrics(clean, wet, arrays[f"noise_{condition}"], target, pause),
                    "cap60_then_c": pause_metrics(clean, wet, candidates[f"noise_{condition}"], dry_c, pause),
                }
            condition_metrics = {}
            for route, output, dry_output in (
                    ("cap60", arrays[f"x_{condition}"], target),
                    ("cap60_then_c", candidates[condition], dry_c)):
                values = pause_metrics(clean, wet, output, dry_output, pause)
                noise_rows = None if noise_only_rows is None else noise_only_rows[route]
                condition_metrics[route] = add_common_floor(values, noise_rows)
            per_condition[condition] = condition_metrics
            dry_outputs = {"cap60": target, "cap60_then_c": dry_c}
            for route in ("cap60", "cap60_then_c"):
                dry_output = dry_outputs[route]
                for metric in condition_metrics[route]:
                    noise_floor = metric["noise_only_floor_db"]
                    case_rows.append({"candidate_index": row["candidate_index"],
                        "speaker": row["speaker"], "room": row["room"],
                        "rir_configuration": row["rir_configuration"], "rir_sha256": row["rir_sha256"],
                        "condition": condition, "route": route, "band_ms": metric["band_ms"],
                        "output_tail_vs_same_fixed_input_speech_db":
                            metric["output_tail_vs_same_fixed_input_speech_db"],
                        "common_output_floor_db": metric["common_output_floor_db"],
                        "common_floor_censored": metric["common_floor_censored"],
                        "dry_model_floor_db": metric["dry_model_floor_vs_fixed_input_speech_db"],
                        "noise_only_floor_db": noise_floor,
                        "dry_control_absolute_floor": absolute_floor(
                            dry_output, row["pause"]["clip_relative_start_sample"], metric["band_ms"]),
                        "noise_only_absolute_floor": (None if condition not in ("fan20", "fan10")
                            else absolute_floor(
                                candidates[f"noise_{condition}"] if route == "cap60_then_c"
                                else arrays[f"noise_{condition}"],
                                row["pause"]["clip_relative_start_sample"], metric["band_ms"])),
                        "baseline_noise_floor_db": None,
                        "candidate_noise_floor_db": None})
            if condition in ("fan20", "fan10"):
                for metric in noise_only_rows["cap60"]:
                    match = next(x for x in noise_only_rows["cap60_then_c"]
                                 if x["band_ms"] == metric["band_ms"])
                    baseline_common = max(metric["dry_model_floor_vs_fixed_input_speech_db"],
                        metric["output_tail_vs_same_fixed_input_speech_db"])
                    candidate_common = max(match["dry_model_floor_vs_fixed_input_speech_db"],
                        match["output_tail_vs_same_fixed_input_speech_db"])
                    for case_row in reversed(case_rows):
                        if (case_row["speaker"] == row["speaker"] and
                                case_row["candidate_index"] == row["candidate_index"] and
                                case_row["room"] == row["room"] and
                                case_row["condition"] == condition and
                                case_row["band_ms"] == metric["band_ms"]):
                            case_row["baseline_noise_floor_db"] = baseline_common
                            case_row["candidate_noise_floor_db"] = candidate_common
                            break
        rir_metrics = per_condition["rir_only"]
        pair_floor = max(metric["common_output_floor_db"]
            for route in ("cap60", "cap60_then_c")
            for metric in rir_metrics[route] if metric["band_ms"] in BANDS)
        slopes = {}
        for route, output in (("cap60", arrays["x_rir_only"]),
                              ("cap60_then_c", candidates["rir_only"])):
            slopes[route] = pause_decay_slope_db_s(clean, arrays["wet_rir_only"],
                                                    output, pause[0], pair_floor)
        if slopes["cap60"] is not None and slopes["cap60_then_c"] is not None:
            slope_deltas.append((row["speaker"], row["room"],
                                 slopes["cap60_then_c"] - slopes["cap60"]))
        output_rows.append({"speaker": row["speaker"], "room": row["room"],
                            "candidate_index": row["candidate_index"], "pause_decay_slope_db_s": slopes})

    manifest = {"data_manifest_sha256": __import__("hashlib").sha256(
        data_manifest_path.read_bytes()).hexdigest(),
        "dev_evaluation_manifest_sha256": __import__("hashlib").sha256(
            dev_manifest_path.read_bytes()).hexdigest(),
        "checkpoint_sha256": __import__("hashlib").sha256(args.checkpoint.read_bytes()).hexdigest(),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device), "torch": torch.__version__,
        "dev_room_case_counts": counts, "dev_candidate_cases": len(rows),
        "dev_room_coverage_sufficient_for_predeclared_gate": coverage_sufficient,
        "dev_rir_evaluation_cases": len(rir_rows),
        "dev_identity_evaluation_cases": len(identity_rows),
        "cap60_inference_seconds_from_prep_cache": sum(
            float(row["cap60"]["target"].get("elapsed_s", 0.0)) for row in rows) + sum(
            float(meta.get("elapsed_s", 0.0)) for row in rir_rows
            for bucket in ("inputs", "noise") for meta in row["cap60"][bucket].values()),
        "c_cuda_forward_seconds": c_forward_seconds,
        "c_forward_calls": c_forward_calls,
        "preprocessor_exclusion_summary": data_manifest.get("dev_exclusion_summary"),
        "fixed_checkpoint_step": 3000, "test_wav_accessed": False}
    summary = make_summary(case_rows, speech_deltas, identity_floor_levels, slope_deltas, manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output_dir / "cases.jsonl").write_text("".join(json.dumps(row, allow_nan=False) + "\n"
                                                                       for row in case_rows))
    (args.output_dir / "speech-and-slope.jsonl").write_text("".join(
        json.dumps(row, allow_nan=False) + "\n" for row in output_rows))
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2,
        ensure_ascii=False, allow_nan=False) + "\n")
    plot_summary(summary, args.output_dir / "but-dev-gates.png")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    print(json.dumps(evaluate(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
