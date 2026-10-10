#!/usr/bin/env python3
"""Frozen Cap60 vs Cap60→C cascade compatibility screen on AIR."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.stats import theilslopes

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts/benchmarks/universal_enhancer"))
sys.path.insert(0, str(HERE))
from bench_air_dpdfnet_factorial import (  # noqa: E402
    DEFAULT_OUTPUT as SOURCE_OUTPUT, DEFAULT_SPEECH, DEFAULT_RIR,
    NOISE_CONDITIONS, SR, file_sha256,
    mix_at_snr, rms, steady_speech_metrics,
)
from screen_measured_rirs import (  # noqa: E402
    FRAME, HOP, convolve_pair, infer, load_model, pause_metrics, prepare_rir, rms_frames,
)

ARMS = ("cap60", "cap60_then_c")
TAIL_BANDS = ("150_300", "300_600")
DEFAULT_DEEP_FILTER = ROOT / ".tools/deepfilternet/deep-filter"
DEFAULT_CHECKPOINT = ROOT / ".tools/compact-dereverb/cap60-conditioned-2026-10-10/training/checkpoints/step-003000.pt"
DEFAULT_OUTPUT = ROOT / ".tools/compact-dereverb/air-cap60-c-factorial-2026-10-10"


def make_cases(manifest_path: Path, speech_root: Path, rir_root: Path) -> tuple[dict, list[dict]]:
    """Regenerate the exact frozen speech/RIR cases and confirm saved identities."""
    from bench_air_dpdfnet_factorial import choose_speakers, load_air_rirs

    source_manifest = json.loads(manifest_path.read_text())
    frozen_rirs = source_manifest["rir_rirs"]
    frozen_speakers = source_manifest["speakers"]
    exclusion = SOURCE_OUTPUT.parent / "air-external-2026-10-09" / "excluded_prior-speakers.txt"
    selected = choose_speakers(speech_root, exclusion, len(frozen_speakers))
    rirs = load_air_rirs(rir_root)
    if [item[0] for item in selected] != [row["speaker"] for row in frozen_speakers]:
        raise RuntimeError("speaker list differs from the frozen DPDFNet2 factorial")
    if [item["sha256"] for item in rirs] != [row["sha256"] for row in frozen_rirs]:
        raise RuntimeError("AIR RIR hashes differ from the frozen DPDFNet2 factorial")
    for (_, paths, _, _), saved in zip(selected, frozen_speakers):
        if [file_sha256(path) for path in paths] != saved["sha256"]:
            raise RuntimeError(f"speech hashes differ for speaker {saved['speaker']}")
    cases = []
    for speaker, paths, raw_clean, pauses in selected:
        for rir_item in rirs:
            rir, rir_meta = prepare_rir(rir_item)
            pair = convolve_pair(raw_clean, rir)
            n = len(pair["clean"])
            clean = pair["clean"][:n]
            wet = pair["reverberant"][:n]
            if len(clean) != len(wet):
                raise RuntimeError("clean/wet source alignment mismatch")
            keybase = f"s{speaker}-{rir_item['room']}-{Path(rir_item['path']).stem}"
            entries = [("dry", clean, None, None), ("rir_only", wet, None, None)]
            for condition, target_snr in NOISE_CONDITIONS:
                seed_material = f"{speaker}:{rir_item['sha256']}:{condition}".encode()
                seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:4], "little")
                noisy, noise, measured = mix_at_snr(clean, wet, seed, target_snr)
                entries.append((condition, noisy, noise, measured))
            # The reverberant conditions share the same underlying wet signal.
            cases.append({"keybase": keybase, "speaker": speaker, "paths": paths,
                          "clean": clean, "wet": wet, "pauses": pauses,
                          "rir": rir_item, "rir_meta": rir_meta, "conditions": entries})
    return source_manifest, cases


def make_summary(rows: list[dict], floor_rows: list[dict], manifest: dict) -> dict:
    paired: dict[tuple, dict[str, dict]] = {}
    for row in rows:
        key = (row["speaker"], row["room"], row["rir"], row["condition"])
        paired.setdefault(key, {})[row["arm"]] = row
    tail_summary = {}
    for condition in ("rir_only", "fan20", "fan10"):
        for band in TAIL_BANDS:
            values, censored = [], 0
            by_room: dict[str, list[float]] = {}
            for (speaker, room, _rir, item_condition), arms in paired.items():
                if item_condition != condition or set(arms) != set(ARMS):
                    continue
                cap = next(x for x in arms["cap60"]["tail_metrics"] if x["band_ms"] == band)
                cas = next(x for x in arms["cap60_then_c"]["tail_metrics"] if x["band_ms"] == band)
                common_floor = max(cap["common_output_floor_db"], cas["common_output_floor_db"])
                if (cap["output_tail_db"] <= common_floor + 3 or
                        cas["output_tail_db"] <= common_floor + 3):
                    censored += 1
                    continue
                gain = cap["output_tail_db"] - cas["output_tail_db"]
                values.append((speaker, gain))
                by_room.setdefault(room, []).append(gain)
            tail_summary[f"{condition}/{band}"] = {
                "valid_pair_count": len(values), "censored_pair_count": censored,
                "speaker_count": len({speaker for speaker, _ in values}),
                "positive_pair_count": int(sum(value > 0 for _, value in values)),
                "positive_pair_fraction": (float(np.mean([value > 0 for _, value in values]))
                                           if values else None),
                "median_gain_db": float(np.median([v for _, v in values])) if values else None,
                "speaker_cluster_bootstrap_95ci_db": bootstrap(values),
                "positive_location_count_of_four": int(
                    sum(np.median(v) > 0 for v in by_room.values())),
                "location_medians_db": {room: float(np.median(v)) for room, v in by_room.items()},
            }

    floor_summary = {}
    for condition in ("fan20", "fan10"):
        for band in TAIL_BANDS:
            values = []
            for row in floor_rows:
                if row["condition"] == condition and row["band_ms"] == band:
                    values.append((row["speaker"], row["cap60_then_c_floor_db"] - row["cap60_floor_db"]))
            reduction_ci = bootstrap([(speaker, -increase) for speaker, increase in values])
            floor_summary[f"{condition}/{band}"] = {
                "valid_pair_count": len(values), "median_floor_increase_db":
                    float(np.median([v for _, v in values])) if values else None,
                "p90_floor_increase_db": (float(np.percentile([v for _, v in values], 90))
                                           if values else None),
                "speaker_cluster_bootstrap_95ci_db": ([-reduction_ci[1], -reduction_ci[0]]
                                                       if reduction_ci else None),
            }

    dry = [r for r in rows if r["condition"] == "dry_control"]
    speech = {}
    for name in ("active_p50", "onset_p10", "weak_p10", "weak_p50"):
        vals = []
        for row in dry:
            if row["arm"] != "cap60_then_c":
                continue
            other = paired[(row["speaker"], row["room"], row["rir"], "dry_control")]["cap60"]
            vals.append((row["speaker"], row["speech_metrics"][name] - other["speech_metrics"][name]))
        speech[name] = {"median_db": float(np.median([v for _, v in vals])) if vals else None,
                        "speaker_cluster_bootstrap_95ci_db": bootstrap(vals)}

    dry_floor = {}
    for band in TAIL_BANDS:
        vals = []
        for key, arms in paired.items():
            if key[3] != "dry_control" or set(arms) != set(ARMS):
                continue
            cap = next(x for x in arms["cap60"]["tail_metrics"] if x["band_ms"] == band)
            cascade = next(x for x in arms["cap60_then_c"]["tail_metrics"] if x["band_ms"] == band)
            vals.append((key[0], cascade["common_output_floor_db"] - cap["common_output_floor_db"]))
        dry_floor[band] = {"valid_pair_count": len(vals),
                           "median_increase_db": float(np.median([v for _, v in vals])) if vals else None,
                           "p90_increase_db": float(np.percentile([v for _, v in vals], 90)) if vals else None,
                           "speaker_cluster_bootstrap_95ci_db": bootstrap(vals)}

    slope_pairs: dict[str, list[tuple[str, float]]] = {
        "cap60": [], "cap60_then_c": [], "incremental_delta": []}
    for key, arms in paired.items():
        if key[3] != "rir_only" or set(arms) != set(ARMS):
            continue
        cap_slope = arms["cap60"]["pause_decay_slope_80_500_db_s"]
        c_slope = arms["cap60_then_c"]["pause_decay_slope_80_500_db_s"]
        if cap_slope is not None:
            slope_pairs["cap60"].append((key[0], cap_slope))
        if c_slope is not None:
            slope_pairs["cap60_then_c"].append((key[0], c_slope))
        if cap_slope is not None and c_slope is not None:
            slope_pairs["incremental_delta"].append((key[0], c_slope - cap_slope))
    slope_summary = {name: {"valid_pair_count": len(values),
                            "median_db_per_second": float(np.median([v for _, v in values])) if values else None,
                            "speaker_cluster_bootstrap_95ci_db_per_second": bootstrap(values)}
                     for name, values in slope_pairs.items()}

    gates = {
        "tail_150_300_ge_2db": (tail_summary["rir_only/150_300"]["median_gain_db"] is not None and
                                 tail_summary["rir_only/150_300"]["median_gain_db"] >= 2),
        "tail_300_600_ge_1_5db": (tail_summary["rir_only/300_600"]["median_gain_db"] is not None and
                                   tail_summary["rir_only/300_600"]["median_gain_db"] >= 1.5),
        "positive_in_at_least_75pct_of_measurable_pairs": all(
            tail_summary[f"rir_only/{band}"]["positive_pair_fraction"] is not None and
            tail_summary[f"rir_only/{band}"]["positive_pair_fraction"] >= .75 for band in TAIL_BANDS),
        "noise_floor_abs_median_le_3db": all(v["median_floor_increase_db"] is not None and
                                               abs(v["median_floor_increase_db"]) <= 3
                                               for v in floor_summary.values()),
        "noise_floor_p90_increase_lt_6db": all(v["p90_floor_increase_db"] is not None and
                                                 v["p90_floor_increase_db"] < 6
                                                 for v in floor_summary.values()),
        "active_speech_within_half_db": speech["active_p50"]["median_db"] is not None and
                                         abs(speech["active_p50"]["median_db"]) <= .5,
        "onset_p10_gt_minus_1db": speech["onset_p10"]["median_db"] is not None and
                                  speech["onset_p10"]["median_db"] > -1,
        "weak_p10_gt_minus_1_5db": speech["weak_p10"]["median_db"] is not None and
                                    speech["weak_p10"]["median_db"] > -1.5,
        "weak_p50_within_1db": speech["weak_p50"]["median_db"] is not None and
                                abs(speech["weak_p50"]["median_db"]) <= 1,
        "dry_floor_increase_le_3db": all(v["median_increase_db"] is not None and
                                          v["median_increase_db"] <= 3 for v in dry_floor.values()),
        "decay_slope_delta_le_2db_s": (slope_summary["incremental_delta"]["median_db_per_second"]
                                        is not None and
                                        slope_summary["incremental_delta"]["median_db_per_second"] <= 2),
    }
    return {"tail_comparisons": tail_summary, "noise_floor": floor_summary,
            "dry_speech_incremental": speech, "dry_floor": dry_floor,
            "pause_decay_slope_rir_only": slope_summary,
            "predeclared_gates": gates,
            "overall_pass": all(gates.values()),
            "interpretation": "Positive tail gain means Cap60→C has a lower tail than Cap60. A pair is censored when either route is within 3 dB of the maximum matched dry/noise-only floor. Bootstrap resamples speakers as clusters.",
            "case_rows": len(rows), "source_manifest_sha256": manifest["source_manifest_sha256"]}


def plot_summary(summary: dict, output: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    bands = ("150_300", "300_600")
    labels = ("150–300 ms", "300–600 ms")
    tail = summary["tail_comparisons"]
    x = np.arange(2)
    vals = [tail[f"rir_only/{band}"]["median_gain_db"] for band in bands]
    lows, highs = [], []
    for band, value in zip(bands, vals):
        ci = tail[f"rir_only/{band}"]["speaker_cluster_bootstrap_95ci_db"]
        lows.append(0 if value is None or ci is None else value - ci[0])
        highs.append(0 if value is None or ci is None else ci[1] - value)
    axes[0].errorbar(x, vals, yerr=np.asarray([lows, highs]), fmt="o", capsize=4,
                     color="#2a788e")
    axes[0].axhline(0, color="#555", linewidth=.8)
    axes[0].set_xticks(x, labels)
    axes[0].set_title("RIR-only tail gain")
    axes[0].set_ylabel("Cap60 vs Cap60→C (dB)\npositive = lower tail")
    for i, band in enumerate(bands):
        row = tail[f"rir_only/{band}"]
        if row["median_gain_db"] is None:
            axes[0].text(i, 0, f"not measurable\n{row['censored_pair_count']} censored",
                         ha="center", va="bottom", fontsize=8)
        else:
            axes[0].text(i, vals[i], f"n={row['valid_pair_count']}",
                         ha="center", va="bottom", fontsize=8)

    floor = summary["noise_floor"]
    positions = np.arange(4)
    categories = [(condition, band) for condition in ("fan20", "fan10") for band in bands]
    floor_vals, floor_low, floor_high = [], [], []
    for condition, band in categories:
        row = floor[f"{condition}/{band}"]
        value = row["median_floor_increase_db"]
        ci = row["speaker_cluster_bootstrap_95ci_db"]
        floor_vals.append(value)
        floor_low.append(0 if value is None or ci is None else value - ci[0])
        floor_high.append(0 if value is None or ci is None else ci[1] - value)
    axes[1].errorbar(positions, floor_vals, yerr=np.asarray([floor_low, floor_high]),
                     fmt="o", capsize=4, color="#d1495b")
    axes[1].axhline(3, color="#d1495b", linestyle=":", linewidth=1, label="+3 dB gate")
    axes[1].axhline(0, color="#555", linewidth=.8)
    axes[1].set_xticks(positions, [f"{c}\n{b.replace('_','–')} ms" for c, b in categories])
    axes[1].set_title("Noise-only floor increase")
    axes[1].set_ylabel("Cap60→C minus Cap60 (dB)")
    axes[1].legend(fontsize=8)

    slope = summary["pause_decay_slope_rir_only"]
    delta = slope["incremental_delta"]
    if delta["median_db_per_second"] is None:
        axes[2].text(.5, .5, "No shared above-floor\nframes to fit a slope",
                     ha="center", va="center", transform=axes[2].transAxes)
    else:
        val, ci = delta["median_db_per_second"], delta["speaker_cluster_bootstrap_95ci_db_per_second"]
        axes[2].errorbar([0], [val], yerr=[[val - ci[0]], [ci[1] - val]], fmt="o",
                         capsize=4, color="#7353ba")
        axes[2].set_xticks([0], ["Cap60→C − Cap60"])
    axes[2].axhline(0, color="#555", linewidth=.8)
    axes[2].set_title("80–500 ms decay-slope change")
    axes[2].set_ylabel("dB/s (more negative = faster decay)")
    for ax in axes:
        ax.grid(axis="y", alpha=.25)
    fig.suptitle("Frozen AIR compatibility screen: DeepFilterNet Cap60 and dereverb C")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def analyze_existing(output_dir: Path) -> dict:
    """Rebuild aggregates from a completed row file without rerunning models."""
    manifest_path = output_dir / "manifest.json"
    rows_path = output_dir / "cases.jsonl"
    manifest = json.loads(manifest_path.read_text())
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    expected = int(manifest["cases"]) * 4 * len(ARMS)
    keys = [(r["speaker"], r["room"], r["rir"], r["condition"], r["arm"]) for r in rows]
    if len(rows) != expected or len(keys) != len(set(keys)):
        raise RuntimeError(f"incomplete or duplicate rows: {len(rows)} rows; expected {expected}")
    pairs: dict[tuple, dict[str, dict]] = {}
    for row in rows:
        key = (row["speaker"], row["room"], row["rir"], row["condition"])
        pairs.setdefault(key, {})[row["arm"]] = row
    if any(set(arms) != set(ARMS) for arms in pairs.values()):
        raise RuntimeError("one or more paired cases are missing a processing route")
    floor_rows = []
    for (speaker, _room, _rir, condition), arms in pairs.items():
        if condition not in ("fan20", "fan10"):
            continue
        for band in TAIL_BANDS:
            cap = next(m for m in arms["cap60"]["tail_metrics"] if m["band_ms"] == band)
            cascade = next(m for m in arms["cap60_then_c"]["tail_metrics"] if m["band_ms"] == band)
            floor_rows.append({"speaker": speaker, "condition": condition, "band_ms": band,
                               "cap60_floor_db": cap["common_output_floor_db"],
                               "cap60_then_c_floor_db": cascade["common_output_floor_db"]})
    summary = make_summary(rows, floor_rows, manifest)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    plot_summary(summary, output_dir / "factorial-cap60-c.png")
    return {"case_rows": len(rows), "unique_cases": len(pairs),
            "summary": summary, "summary_rebuilt_from_saved_rows": True}


def bootstrap(values: list[tuple[str, float]], seed: int = 20261010, samples: int = 5000):
    by_speaker: dict[str, list[float]] = {}
    for speaker, value in values:
        by_speaker.setdefault(str(speaker), []).append(float(value))
    data = np.asarray([np.median(x) for x in by_speaker.values()], dtype=np.float64)
    if not data.size:
        return None
    rng = np.random.default_rng(seed)
    boot = np.median(rng.choice(data, size=(samples, len(data)), replace=True), axis=1)
    return [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


def pause_decay_slope_db_s(clean: np.ndarray, wet: np.ndarray, output: np.ndarray,
                           pause: dict, common_floor_db: float) -> float | None:
    """Robust 80–500 ms pause-decay slope, excluding frames near the common floor."""
    levels = rms_frames(clean)
    start = int(pause["clip_relative_start_sample"])
    frame_starts = np.arange(levels.size) * HOP
    active = levels >= max(float(levels.max()) * .02, 1e-5)
    ref_ids = np.flatnonzero((frame_starts >= max(0, start - int(.30 * SR))) &
                             (frame_starts + FRAME <= start) & active)
    if not ref_ids.size:
        return None
    ref_indices = np.unique(np.concatenate([
        np.arange(i * HOP, i * HOP + FRAME) for i in ref_ids]))
    ref_rms = rms(wet[ref_indices])
    out_levels = rms_frames(output)
    times = (np.arange(out_levels.size) * HOP + FRAME / 2 - start) / SR
    relative_db = 20 * np.log10(np.maximum(out_levels, 1e-12) / max(ref_rms, 1e-12))
    keep = ((times >= .08) & (times <= .50) & np.isfinite(relative_db) &
            (relative_db > common_floor_db + 3))
    if int(keep.sum()) < 8:
        return None
    return float(theilslopes(relative_db[keep], times[keep]).slope)


def run(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    for path in (args.deep_filter, args.checkpoint, args.speech_root, args.rir_root,
                 args.source_manifest):
        if not path.exists():
            raise FileNotFoundError(path)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    source_manifest, cases = make_cases(args.source_manifest, args.speech_root, args.rir_root)
    out = args.output_dir
    stage, cap_dir, render_dir = out / "pcm16-input", out / "cap60", out / "renders"
    for directory in (stage, cap_dir, render_dir):
        directory.mkdir(parents=True)

    # Materialize only the matched Cap60 inputs. Downstream inference and metrics
    # are reconstructed from these PCM16 inputs and the frozen clean/RIR manifest.
    input_records = {}
    for case in cases:
        for condition, audio, noise, measured_snr in case["conditions"]:
            name = f"{case['keybase']}-{condition}"
            path = stage / f"{name}.wav"
            sf.write(path, audio, SR, subtype="PCM_16")
            input_records[(case["keybase"], condition)] = {
                "path": path, "measured_snr_db": measured_snr, "samples": len(audio)}
            if noise is not None:
                noise_path = stage / f"{case['keybase']}-{condition}-noise-only.wav"
                sf.write(noise_path, noise, SR, subtype="PCM_16")
                input_records[(case["keybase"], condition + "_noise")] = {
                    "path": noise_path, "measured_snr_db": measured_snr, "samples": len(noise)}
    command = [str(args.deep_filter), "--atten-lim-db", "60", "--compensate-delay",
               "-o", str(cap_dir)] + [str(item["path"]) for item in input_records.values()]
    if args.cached_cap60_dir is None:
        started = time.perf_counter()
        proc = subprocess.run(command, check=True, capture_output=True, text=True)
        cap_wall = time.perf_counter() - started
        cap_log = proc.stdout + "\n--- STDERR ---\n" + proc.stderr
    else:
        cached_dir = args.cached_cap60_dir.resolve()
        cached_inputs = cached_dir.parent / "pcm16-input"
        cached_manifest_path = cached_dir.parent / "manifest.json"
        cached_manifest = (json.loads(cached_manifest_path.read_text())
                           if cached_manifest_path.is_file() else {})
        for rec in input_records.values():
            cached_input = cached_inputs / rec["path"].name
            cached_output = cached_dir / rec["path"].name
            if not cached_input.is_file() or not cached_output.is_file():
                raise FileNotFoundError(f"incomplete frozen Cap60 cache for {rec['path'].name}")
            if file_sha256(cached_input) != file_sha256(rec["path"]):
                raise RuntimeError(f"cached Cap60 input differs from regenerated input: {rec['path'].name}")
            shutil.copy2(cached_output, cap_dir / rec["path"].name)
        cap_wall = 0.0
        cap_log = f"reused byte-verified Cap60 cache: {cached_dir}"
        proc = None

    model = load_model(args.checkpoint, device)
    cases_rows, floor_rows, timing = [], [], {arm: [] for arm in ARMS}
    length_deltas = []
    for case in cases:
        dry_caps, dry_cs = None, None
        condition_outputs = {}
        for condition, audio, _noise, measured_snr in case["conditions"]:
            rec = input_records[(case["keybase"], condition)]
            cap_path = cap_dir / rec["path"].name
            cap, cap_sr = sf.read(cap_path, dtype="float32")
            source, source_sr = sf.read(rec["path"], dtype="float32")
            if cap_sr != SR or source_sr != SR or not np.isfinite(cap).all():
                raise RuntimeError(f"invalid Cap60 output: {cap_path}")
            length_deltas.append(len(source) - len(cap))
            c_out, c_elapsed = infer(model, cap, device)
            if len(c_out) != len(cap) or not np.isfinite(c_out).all():
                raise RuntimeError("C changed the Cap60 output duration or returned non-finite samples")
            sf.write(render_dir / f"{rec['path'].stem}-cap60.wav", cap, SR, subtype="PCM_24")
            sf.write(render_dir / f"{rec['path'].stem}-cap60-c.wav", c_out, SR, subtype="PCM_24")
            timing["cap60"].append((0.0, len(source) / SR))
            timing["cap60_then_c"].append((c_elapsed, len(source) / SR))
            condition_outputs[condition] = (cap, c_out)
            if condition == "dry":
                dry_caps, dry_cs = cap, c_out
        if dry_caps is None or dry_cs is None:
            raise RuntimeError("missing dry Cap60 control")

        # Render matched noise-only controls through the same Cap60 executable.
        noise_outputs = {}
        for condition, _snr in NOISE_CONDITIONS:
            rec = input_records[(case["keybase"], condition + "_noise")]
            cap, sr = sf.read(cap_dir / rec["path"].name, dtype="float32")
            if sr != SR or not np.isfinite(cap).all():
                raise RuntimeError(f"invalid Cap60 noise-only output: {rec['path'].name}")
            c_out, c_elapsed = infer(model, cap, device)
            if len(c_out) != len(cap) or not np.isfinite(c_out).all():
                raise RuntimeError("C changed the Cap60 noise output duration or returned non-finite samples")
            noise_outputs[condition] = (cap, c_out)
            timing["cap60_then_c"].append((c_elapsed, len(cap) / SR))

        for condition, audio, noise, measured_snr in case["conditions"]:
            label = "dry_control" if condition == "dry" else condition
            wet = case["clean"] if condition == "dry" else case["wet"]
            cap, c_out = condition_outputs[condition]
            n = min(len(case["clean"]), len(wet), len(audio), len(cap), len(c_out))
            if condition in ("fan20", "fan10"):
                noise_caps, noise_cs = noise_outputs[condition]
                n = min(n, len(noise_caps), len(noise_cs))
            else:
                noise_caps = noise_cs = None
            clean = case["clean"][:n]
            wet_eval = wet[:n]
            cap = cap[:n]
            c_out = c_out[:n]
            pauses = case["pauses"]
            speech_cap = steady_speech_metrics(clean, wet_eval, cap, pauses)
            speech_c = steady_speech_metrics(clean, wet_eval, c_out, pauses)
            if noise_caps is not None:
                noise_caps, noise_cs = noise_caps[:n], noise_cs[:n]
            route_rows = []
            for arm, output, dry_output, noise_output, speech in (
                    ("cap60", cap, dry_caps[:n], None if noise_caps is None else noise_caps, speech_cap),
                    ("cap60_then_c", c_out, dry_cs[:n], None if noise_cs is None else noise_cs, speech_c)):
                measured = pause_metrics(clean, wet_eval, output, dry_output, pauses)
                floors = (pause_metrics(clean, wet_eval, noise_output, dry_output, pauses)
                          if noise_output is not None else [])
                floor_by_band = {m["band_ms"]: m["output_tail_vs_same_fixed_input_speech_db"]
                                 for m in floors}
                for metric in measured:
                    noise_floor = floor_by_band.get(metric["band_ms"],
                                metric["dry_model_floor_vs_fixed_input_speech_db"])
                    common_floor = max(metric["dry_model_floor_vs_fixed_input_speech_db"], noise_floor)
                    metric["noise_only_floor_db"] = noise_floor
                    metric["common_output_floor_db"] = common_floor
                    metric["output_tail_db"] = metric["output_tail_vs_same_fixed_input_speech_db"]
                case_row = {"speaker": case["speaker"], "room": case["rir"]["room"],
                            "rir": Path(case["rir"]["path"]).name,
                            "distance_role": case["rir"]["distance_role"],
                            "condition": label, "arm": arm,
                            "measured_snr_db": measured_snr,
                            "duration_seconds": n / SR,
                            "speech_metrics": {"active_p50": speech["active_output_vs_clean_db_p10_p50_p90"][1],
                                               "onset_p10": speech["rising_output_vs_clean_db_p10_p50_p90"][0],
                                               "weak_p10": speech["weak_speech_gain_db_p10"],
                                               "weak_p50": speech["weak_output_vs_clean_db_p50_p90_p99"][0]},
                            "tail_metrics": measured}
                route_rows.append(case_row)
            pair_floor = max(
                max(metric["common_output_floor_db"] for metric in row["tail_metrics"]
                    if metric["band_ms"] in TAIL_BANDS)
                for row in route_rows)
            for row in route_rows:
                output = cap if row["arm"] == "cap60" else c_out
                row["pause_decay_slope_80_500_db_s"] = pause_decay_slope_db_s(
                    clean, wet_eval, output, pauses[0], pair_floor)
                cases_rows.append(row)
            if condition in ("fan20", "fan10"):
                for band in TAIL_BANDS:
                    cr = next(r for r in route_rows if r["arm"] == "cap60")
                    br = next(r for r in route_rows if r["arm"] == "cap60_then_c")
                    cap_m = next(x for x in cr["tail_metrics"] if x["band_ms"] == band)
                    b_m = next(x for x in br["tail_metrics"] if x["band_ms"] == band)
                    floor_rows.append({"speaker": case["speaker"], "condition": label, "band_ms": band,
                                       "cap60_floor_db": cap_m["common_output_floor_db"],
                                       "cap60_then_c_floor_db": b_m["common_output_floor_db"]})

    # Derive per route timing from total C CUDA forward time and one batched
    # Cap60 invocation; include both explicitly in the public manifest.
    measured_wall_by_route = {"cap60": cap_wall,
                              "cap60_then_c_forward_seconds": sum(v[0] for v in timing["cap60_then_c"])}
    manifest = {"protocol": "Frozen Cap60 vs Cap60→C compatibility factorial replaying the DPDFNet2 run's exact 12 speakers, eight AIR RIRs, controlled-offset clips, RIR-only/fan20/fan10 and dry/noise-only controls.",
                "source_manifest_sha256": file_sha256(args.source_manifest),
                "deep_filter_executable_sha256": file_sha256(args.deep_filter),
                "deep_filter_command": command, "deep_filter_attenuation_limit_db": 60,
                "cap60_cache_reused": args.cached_cap60_dir is not None,
                "cap60_cache_directory": (None if args.cached_cap60_dir is None else str(args.cached_cap60_dir)),
                "deep_filter_batch_wall_seconds": cap_wall,
                "cached_cap60_original_batch_wall_seconds": (cached_manifest.get("deep_filter_batch_wall_seconds")
                                                               if args.cached_cap60_dir is not None else None),
                "cap60_total_input_seconds": sum(item["samples"] for item in input_records.values()) / SR,
                "c_cuda_forward_seconds": measured_wall_by_route["cap60_then_c_forward_seconds"],
                "dereverb_checkpoint_sha256": file_sha256(args.checkpoint),
                "dereverb_parameter_count": sum(p.numel() for p in model.parameters()),
                "length_delta_samples_min_max": [int(min(length_deltas)), int(max(length_deltas))],
                "rows": len(cases_rows), "cases": len(cases),
                "limitation": "AIR and speakers are reused from the previous screen; this isolates denoiser compatibility and is not a new external holdout. The DeepFilterNet CLI trims about 30 ms at EOF on pilot; all routes are scored over their shared available prefix. Synthetic fan-like noise is not a real noise recording."}
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    (out / "cases.jsonl").write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in cases_rows))
    summary = make_summary(cases_rows, floor_rows, manifest)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    plot_summary(summary, out / "factorial-cap60-c.png")
    (out / "cap60-log.txt").write_text(cap_log)
    return {"output_dir": str(out), "rows": len(cases_rows), "summary": summary,
            "cap60_batch_wall_seconds": cap_wall,
            "c_forward_seconds": measured_wall_by_route["cap60_then_c_forward_seconds"],
            "length_delta_samples_min_max": manifest["length_delta_samples_min_max"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deep-filter", type=Path, default=DEFAULT_DEEP_FILTER)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--speech-root", type=Path, default=DEFAULT_SPEECH)
    parser.add_argument("--rir-root", type=Path, default=DEFAULT_RIR)
    parser.add_argument("--source-manifest", type=Path, default=SOURCE_OUTPUT / "manifest.json")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cached-cap60-dir", type=Path, default=None,
                        help="reuse outputs only when every regenerated PCM16 input hash matches")
    parser.add_argument("--analyze-existing", action="store_true",
                        help="rebuild summary and plot from a completed cases.jsonl without rerendering models")
    args = parser.parse_args()
    result = analyze_existing(args.output_dir) if args.analyze_existing else run(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
