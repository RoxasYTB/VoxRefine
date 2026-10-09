"""Censor-aware paired comparison using one conservative common output floor."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


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


def analyze(input_dir: Path, output_path: Path) -> dict:
    rows = [json.loads(line) for line in (input_dir / "cases.jsonl").read_text().splitlines()]
    manifest = json.loads((input_dir / "manifest.json").read_text())
    names = list(manifest["models"])
    a_name, b_name = names
    paired = {}
    for row in rows:
        if row["condition"] != "measured_reverb":
            continue
        for metric in row.get("inserted_pause_tail_metrics", row.get("natural_pause_tail_metrics", [])):
            key = (row["speaker"], row["rir_file"], metric["band_ms"])
            paired.setdefault(key, {})[row["model"]] = {
                "room": row["room"],
                "distance_role": row["distance_role"],
                "tail_db": metric["output_tail_vs_same_fixed_input_speech_db"],
                "floor_db": metric["dry_model_floor_vs_fixed_input_speech_db"],
            }
    bands = {}
    for band in ("150_300", "300_600"):
        examples = []
        for (speaker, rir_file, row_band), models in paired.items():
            if row_band != band or a_name not in models or b_name not in models:
                continue
            a, b = models[a_name], models[b_name]
            common_censor_limit = max(a["floor_db"], b["floor_db"]) + 3.0
            a_exact = a["tail_db"] > common_censor_limit
            b_exact = b["tail_db"] > common_censor_limit
            if a_exact and b_exact:
                lower = upper = a["tail_db"] - b["tail_db"]
                status = "both_exact"
            elif a_exact and not b_exact:
                lower, upper = a["tail_db"] - common_censor_limit, float("inf")
                status = "candidate_at_floor"
            elif not a_exact and b_exact:
                lower, upper = float("-inf"), common_censor_limit - b["tail_db"]
                status = "reference_at_floor"
            else:
                lower, upper = float("-inf"), float("inf")
                status = "both_at_floor"
            examples.append({"speaker": speaker, "rir_file": rir_file, "room": a["room"],
                "distance_role": a["distance_role"],
                "a_tail_db": a["tail_db"], "b_tail_db": b["tail_db"],
                "a_floor_db": a["floor_db"], "b_floor_db": b["floor_db"],
                "common_floor_plus_3db": common_censor_limit, "status": status,
                "improvement_lower_bound_db": lower, "improvement_upper_bound_db": upper})
        status_counts = defaultdict(int)
        for item in examples:
            status_counts[item["status"]] += 1
        exact = [e["improvement_lower_bound_db"] for e in examples if e["status"] == "both_exact"]
        lower = np.asarray([e["improvement_lower_bound_db"] for e in examples], dtype=np.float64)
        upper = np.asarray([e["improvement_upper_bound_db"] for e in examples], dtype=np.float64)
        by_room = {}
        for room in sorted({e["room"] for e in examples}):
            room_rows = [e for e in examples if e["room"] == room]
            room_exact = [e["improvement_lower_bound_db"] for e in room_rows if e["status"] == "both_exact"]
            room_lower = np.asarray([e["improvement_lower_bound_db"] for e in room_rows], dtype=np.float64)
            room_upper = np.asarray([e["improvement_upper_bound_db"] for e in room_rows], dtype=np.float64)
            by_room[room] = {
                "pair_count": len(room_rows),
                "statuses": dict(sorted((k, sum(e["status"] == k for e in room_rows))
                                         for k in {e["status"] for e in room_rows})),
                "both_exact_median_improvement_db": float(np.median(room_exact)) if room_exact else None,
                "possible_room_median_improvement_interval_db": [
                    float(np.median(room_lower)), float(np.median(room_upper))],
                "guaranteed_candidate_wins_gt_0db": sum(e["improvement_lower_bound_db"] > 0 for e in room_rows),
                "guaranteed_candidate_wins_gt_3db": sum(e["improvement_lower_bound_db"] > 3 for e in room_rows),
                "guaranteed_reference_wins": sum(e["improvement_upper_bound_db"] < 0 for e in room_rows),
            }
        by_distance_role = {}
        for role in sorted({e["distance_role"] for e in examples}):
            role_rows = [e for e in examples if e["distance_role"] == role]
            role_lower = np.asarray([e["improvement_lower_bound_db"] for e in role_rows], dtype=np.float64)
            role_upper = np.asarray([e["improvement_upper_bound_db"] for e in role_rows], dtype=np.float64)
            by_distance_role[role] = {
                "pair_count": len(role_rows),
                "possible_role_median_improvement_interval_db": [
                    float(np.median(role_lower)), float(np.median(role_upper))],
                "guaranteed_candidate_wins_gt_0db": sum(e["improvement_lower_bound_db"] > 0 for e in role_rows),
                "guaranteed_candidate_wins_gt_3db": sum(e["improvement_lower_bound_db"] > 3 for e in role_rows),
                "guaranteed_reference_wins": sum(e["improvement_upper_bound_db"] < 0 for e in role_rows),
                "censor_status_counts": dict(sorted((status, sum(e["status"] == status for e in role_rows))
                    for status in {e["status"] for e in role_rows})),
            }
        bands[band] = {
            "pair_count": len(examples),
            "censor_status_counts": dict(sorted(status_counts.items())),
            "common_floor_plus_3db_rule": "C_i = max(F_A_i, F_B_i) + 3 dB; T <= C is censored",
            "exact_pair_median_improvement_db_secondary": float(np.median(exact)) if exact else None,
            "possible_median_improvement_interval_db": [float(np.median(lower)), float(np.median(upper))],
            "guaranteed_candidate_wins_gt_0db": sum(e["improvement_lower_bound_db"] > 0 for e in examples),
            "guaranteed_candidate_wins_gt_3db": sum(e["improvement_lower_bound_db"] > 3 for e in examples),
            "guaranteed_reference_wins": sum(e["improvement_upper_bound_db"] < 0 for e in examples),
            "both_censored_or_unresolved": sum(e["status"] == "both_at_floor" for e in examples),
            "by_room": by_room,
            "by_distance_role": by_distance_role,
            "paired_rir_observations": examples,
        }
    room_count = int(manifest["room_count"])
    dataset_name = manifest.get("rir_dataset", "measured-RIR corpus")
    limitations = [f"The benchmark covers {room_count} measured-RIR rooms/configurations from {dataset_name}; it does not establish universal room generalization.",
                   "Common-floor bounds depend on the fixed dry-control floor estimate and +3 dB threshold.",
                   "Speaker×RIR pairs are correlated; do not use naive p-values as if each were an independent room."]
    if "BUT" in dataset_name:
        limitations.append("The same BUT rooms were inspected in the prior nine-room screen; this is cross-room development, not a final external test.")
    result = {"comparison": {"reference_model": a_name, "candidate_model": b_name,
                             "room_count": manifest["room_count"], "speech_speaker_count": len(manifest["speech_speakers"]),
                             "rir_count": manifest["rir_count"],
                             "independent_generalization_units": f"rooms/configurations ({room_count}); no naive p-values over speaker×RIR rows"},
              "bands": bands,
              "interpretation": "Censored pairs are intervals, not discarded rows. Positive lower bounds are guaranteed candidate wins; both-floor rows remain unresolved.",
              "limitations": limitations}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(json_finite(result), indent=2, allow_nan=False) + "\n")
    rooms = sorted({room for item in bands.values() for room in item["by_room"]})
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    colors = {"150_300": "#4677a8", "300_600": "#d17b32"}
    for ax, band in zip(axes, ("150_300", "300_600")):
        y, low_err, high_err, x = [], [], [], []
        for index, room in enumerate(rooms):
            info = bands[band]["by_room"].get(room)
            if not info:
                continue
            low, high = info["possible_room_median_improvement_interval_db"]
            if not np.isfinite(low) or not np.isfinite(high):
                continue
            x.append(index); y.append((low + high) / 2)
            low_err.append((low + high) / 2 - low)
            high_err.append(high - (low + high) / 2)
        ax.errorbar(x, y, yerr=[low_err, high_err], fmt="o", capsize=5,
                    color=colors[band], label=f"{band.replace('_','–')} ms")
        ax.axhline(0, color="#444444", linewidth=1)
        ax.axhline(3, color="#777777", linewidth=1, linestyle="--", label="gate +3 dB")
        ax.set_ylabel("Gain médian possible (dB)")
        ax.grid(axis="y", alpha=.25); ax.legend(loc="best")
    axes[-1].set_xticks(range(len(rooms)), [r.replace("Hotel_SkalskyDvur_", "Hotel ").replace("VUT_FIT_", "FIT ") for r in rooms], rotation=20, ha="right")
    fig.suptitle(f"Comparaison censored-aware — {a_name} → {b_name}\n{dataset_name}; intervalles de médiane par salle")
    fig.savefig(output_path.with_suffix(".png"), dpi=170)
    plt.close(fig)
    for band, stats in bands.items():
        print(band, json.dumps(json_finite({k: v for k, v in stats.items()
                                            if k != "paired_rir_observations"}), allow_nan=False))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.input_dir, args.output)


if __name__ == "__main__":
    main()
