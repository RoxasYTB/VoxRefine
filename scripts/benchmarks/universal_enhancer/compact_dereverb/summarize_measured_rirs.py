"""Balanced summaries and plot for the frozen measured-RIR screen."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def summarize(input_dir: Path, output_dir: Path) -> dict:
    rows = [json.loads(line) for line in (input_dir / "cases.jsonl").read_text().splitlines()]
    manifest = json.loads((input_dir / "manifest.json").read_text())
    model_names = list(manifest["models"])
    baseline_name, candidate_name = model_names[0], model_names[1]
    reverberant = [r for r in rows if r["condition"] == "measured_reverb"]
    by = defaultdict(lambda: defaultdict(list))
    censored = defaultdict(lambda: defaultdict(int))
    totals = defaultdict(lambda: defaultdict(int))
    for row in reverberant:
        for metric in row.get("inserted_pause_tail_metrics", row.get("natural_pause_tail_metrics", [])):
            band = metric["band_ms"]
            totals[row["model"]][band] += 1
            censored[row["model"]][band] += int(metric["output_floor_censored"])
            if not metric["output_floor_censored"]:
                by[row["model"]][band].append(metric["tail_reduction_input_to_output_db"])
    tails = {model: {band: {"median_input_to_output_reduction_db_uncensored": float(np.median(vals)) if vals else None,
                            "p10_db_uncensored": float(np.percentile(vals, 10)) if vals else None,
                            "uncensored_count": len(vals), "censored_count": censored[model][band],
                            "total_count": totals[model][band]}
                     for band, vals in bands.items()} for model, bands in by.items()}

    pair_values = defaultdict(list)
    by_room_model = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in reverberant:
        for metric in row.get("inserted_pause_tail_metrics", row.get("natural_pause_tail_metrics", [])):
            key = (row["speaker"], row["room"], metric["band_ms"])
            if not metric["output_floor_censored"]:
                pair_values[(row["model"], *key)].append(metric["tail_reduction_input_to_output_db"])
                by_room_model[row["room"]][row["model"]][metric["band_ms"]].append(
                    metric["tail_reduction_input_to_output_db"])
    room_censoring = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"censored": 0, "total": 0})))
    for row in reverberant:
        for metric in row.get("inserted_pause_tail_metrics", row.get("natural_pause_tail_metrics", [])):
            cell = room_censoring[row["room"]][row["model"]][metric["band_ms"]]
            cell["total"] += 1
            cell["censored"] += int(metric["output_floor_censored"])
    # One median per speaker × room × bin prevents repeated configurations
    # from acting as independent observations.
    paired = {}
    for band in ("150_300", "300_600"):
        deltas = []
        paired_rooms, candidate_floor_wins = set(), 0
        for speaker in sorted({r["speaker"] for r in reverberant}):
            for room in sorted({r["room"] for r in reverberant}):
                b = pair_values.get((baseline_name, speaker, room, band), [])
                c = pair_values.get((candidate_name, speaker, room, band), [])
                if b and c:
                    deltas.append(float(np.median(c) - np.median(b)))
                    paired_rooms.add(room)
                elif b and not c:
                    # Candidate at/below its separately measured dry floor;
                    # count this as a floor-reaching result, not an invented dB value.
                    candidate_floor_wins += 1
        paired[band] = {f"{candidate_name}_minus_{baseline_name}_median_db": float(np.median(deltas)),
                        "p10_db": float(np.percentile(deltas, 10)),
                        "paired_uncensored_speaker_room_count": len(deltas),
                        "paired_uncensored_room_count": len(paired_rooms),
                        "candidate_reaches_floor_when_baseline_measurable_count": candidate_floor_wins,
                        "room_count": len({r["room"] for r in reverberant})}

    # Voice preservation metrics over each complete reverberant input and dry
    # controls, with the inserted gap excluded from the weak-frame mask.
    speech = {}
    for model in model_names:
        items = [r["speech_metrics"] for r in reverberant if r["model"] == model]
        dry_items = [r["speech_metrics"] for r in rows if r["model"] == model and r["condition"] == "dry"]
        speech[model] = {
            "weak_gain_clipwise_p50_median_db": float(np.median([x["weak_output_vs_clean_db_p50_p90_p99"][0] for x in items])),
            "weak_gain_clipwise_p50_p90_db": float(np.percentile([x["weak_output_vs_clean_db_p50_p90_p99"][0] for x in items], 90)),
            "weak_fraction_over_6db_median": float(np.median([x["weak_fraction_over_6db"] for x in items])),
            "active_gain_clipwise_p50_median_db": float(np.median([x["active_output_vs_clean_db_p10_p50_p90"][1] for x in items])),
            "onset_p10_clipwise_median_db": float(np.median([x["rising_output_vs_clean_db_p10_p50_p90"][0] for x in items if x["rising_output_vs_clean_db_p10_p50_p90"][0] is not None])),
            "spectral_mae_clipwise_median_db": float(np.median([
                np.mean(list(x["output_vs_clean_multiband_log_spectral_mae_db"].values()))
                for x in items])),
            "dry_control_active_gain_median_db": float(np.median([
                x["active_output_vs_clean_db_p10_p50_p90"][1] for x in dry_items])),
            "dry_control_onset_p10_median_db": float(np.median([
                x["rising_output_vs_clean_db_p10_p50_p90"][0] for x in dry_items
                if x["rising_output_vs_clean_db_p10_p50_p90"][0] is not None])),
        }
    manifest = json.loads((input_dir / "manifest.json").read_text())
    rir_scales = {}
    for row in reverberant:
        meta = row["rir_processing"]
        peak = meta.get("early_path_peak_before_normalization_0_10ms",
                        meta.get("direct_early_peak_0_10ms"))
        rir_scales[row["rir_file"]] = 1.0 / peak
    # Reconstruct actual model-input durations: complete speech pair for dry;
    # pair plus complete aligned RIR for reverberant input.
    import soundfile as sf
    speaker_by_id = {s["speaker"]: s for s in manifest["speech_speakers"]}
    duration_cache = {}
    for row in rows:
        pair_key = (row["speaker"], tuple(row["utterance"]))
        if pair_key not in duration_cache:
            files = speaker_by_id[row["speaker"]]["paths"]
            duration_cache[pair_key] = sum(sf.info(path).frames for path in files) / 16_000 + .8
        duration = duration_cache[pair_key]
        if row["condition"] == "measured_reverb":
            info = row["rir_processing"]
            raw = sf.info(row["rir_file"])
            duration += (raw.frames - info["direct_path_onset_sample"]) / raw.samplerate
        row["model_input_duration_seconds"] = duration
    for model, info in manifest["models"].items():
        model_rows = [r for r in rows if r["model"] == model]
        info.pop("median_forward_s_6s", None)
        info["median_forward_seconds_per_case"] = float(np.median([r["runtime_s"] for r in model_rows]))
        info["median_input_duration_seconds"] = float(np.median([r["model_input_duration_seconds"] for r in model_rows]))
        info["median_rtf"] = float(np.median([r["runtime_s"] / r["model_input_duration_seconds"] for r in model_rows]))
    protocol = (f"12 speaker-disjoint train-clean-360 voices; full successive sorted same-speaker utterance-file pairs + inserted controlled 800 ms digital silence; dry-only 20 ms RMS/10 ms hop activity threshold = 2% of phrase peak; t0 is last active-frame end of phrase 1; score 50–150/150–300/300–600 ms after t0, retain 200 ms guard before phrase 2; full sequence and full RIR convolution tail; dataset={manifest.get('rir_dataset', 'measured RIR')}; selected rooms × preselected shortest/longest source-mic distances; mono 16 kHz; training-domain early-path peak normalization to unit gain (not SPL calibration); common dry/wet scale; no output normalization; tail reference is reverberant-input energy on phrase-1 clean-active mask. The dry-model floor is recorded relative to dry speech and also expressed against the same fixed wet-speech reference for censoring. Floors are never subtracted; paired A/B analysis uses the more conservative floor plus 3 dB.")
    result = {"protocol": protocol, "speaker_count": len(manifest["speech_speakers"]),
              "room_count": manifest["room_count"], "rir_count": manifest["rir_count"],
              "case_count": manifest["case_count"], "models": manifest["models"],
              "tail_reduction_input_to_output_db": tails,
              "paired_weak_over_minus_baseline_db": paired,
              "training_domain_rir_scalar_applied": {
                  "min": float(np.min(list(rir_scales.values()))),
                  "median": float(np.median(list(rir_scales.values()))),
                  "max": float(np.max(list(rir_scales.values()))),
                  "per_rir": rir_scales},
              "speech": speech, "room_censoring": room_censoring,
              "rooms": {room: {
                  model: {band: float(np.median(vals)) if vals else None for band, vals in bands.items()}
                  for model, bands in models.items()} for room, models in by_room_model.items()},
              "limitations": manifest["limitations"]}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    rooms = sorted(by_room_model)
    labels = [r.replace("Hotel_SkalskyDvur_", "Hotel ").replace("VUT_FIT_", "FIT ") for r in rooms]
    x = np.arange(len(rooms))
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    for ax, band in zip(axes, ("150_300", "300_600")):
        for offset, model, label, color in ((-.18, baseline_name, baseline_name, "#4776a8"),
                                            (.18, candidate_name, candidate_name, "#d67a32")):
            vals = [np.median(by_room_model[r][model][band]) if by_room_model[r][model][band] else np.nan for r in rooms]
            ax.bar(x + offset, vals, width=.34, label=label, color=color)
        ax.set_ylabel(f"Queue réduite (dB)\nfenêtre {band.replace('_','–')} ms")
        ax.grid(axis="y", alpha=.25)
        ax.legend(loc="upper right")
    axes[-1].set_xticks(x, labels, rotation=30, ha="right")
    fig.suptitle(f"{manifest.get('rir_dataset', 'RIR mesurées')} — réduction de queue entrée → sortie\n12 locuteurs, pause numérique insérée 800 ms; comparaison de checkpoints")
    fig.savefig(output_dir / "measured-rir-by-room.png", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.hist(list(rir_scales.values()), bins=8, color="#537f74", edgecolor="white")
    ax.set_xlabel("Scalaire appliqué pour mettre le pic early-path à 1 (échelle linéaire)")
    ax.set_ylabel("Nombre de RIR")
    ax.set_title(f"Répartition des adaptations de gain de {len(rir_scales)} RIR mesurées")
    ax.grid(axis="y", alpha=.25)
    fig.savefig(output_dir / "rir-training-domain-scales.png", dpi=170)
    plt.close(fig)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    print(json.dumps(summarize(p.parse_args().input_dir, p.parse_args().output_dir), indent=2))


if __name__ == "__main__":
    main()
