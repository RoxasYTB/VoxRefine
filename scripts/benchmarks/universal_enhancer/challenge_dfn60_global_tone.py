#!/usr/bin/env python3
"""Leave-one-voice-out tone/gain challenge on top of DeepFilterNet cap 60."""
from __future__ import annotations

import csv
import json
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compare_clear_noisy_to_adobe_v2 import (
    CORPUS, EXPORTS, RATE, correlation, db, read, rms_frames, speech_spectrum,
)

ROOT = Path(__file__).resolve().parents[3]
RENDERS = ROOT / "results/adobe-v2-dfn100-noisy-2026-10-09/renders/cap-60db"
OUT = ROOT / "results/adobe-v2-dfn60-global-tone-2026-10-09"
SAMPLES = ("emy_mixed_ambience18", "remi_crowd18", "stephanie_fan18")
FRAME = 960
GLOBAL_EQ = {"low_db": 1.5, "low_hz": 200.0, "high_db": -3.0, "high_hz": 3_500.0}


def load_sample(sample: str) -> dict:
    noise_name = {
        "emy_mixed_ambience18": "emy_mixed_ambience18-fan+crowd-stem.wav",
        "remi_crowd18": "remi_crowd18-crowd-stem.wav",
        "stephanie_fan18": "stephanie_fan18-fan-stem.wav",
    }[sample]
    audio = {
        "base": read(RENDERS / sample / f"{sample}-noisy.wav"),
        "adobe": read(EXPORTS / f"{sample}-adobe-v2.wav"),
        "clean": read(CORPUS / "clean" / f"{sample}-clean-reference.wav"),
        "noise": read(CORPUS / "noise" / noise_name),
    }
    n = min(map(len, audio.values()))
    audio = {key: value[:n] for key, value in audio.items()}
    clean_env = rms_frames(audio["clean"])
    threshold = np.percentile(clean_env, 95) * 10 ** (-30 / 20)
    speech = clean_env >= threshold
    pad = 6
    speech = np.convolve(
        np.pad(speech.astype(np.int8), pad), np.ones(2 * pad + 1, dtype=np.int32), mode="same"
    )[pad:-pad] > 0
    weak = speech & (clean_env <= np.percentile(clean_env[speech], 25))
    gap = ~speech
    clean_level = float(np.median(clean_env[speech]))
    adobe_env = rms_frames(audio["adobe"])
    adobe_gain = clean_level / max(float(np.median(adobe_env[speech])), 1e-12)
    freq, adobe_curve = speech_spectrum(audio["adobe"] * adobe_gain, speech)
    use = (freq >= 80) & (freq <= 12_000)
    return {**audio, "speech": speech, "weak": weak, "gap": gap,
            "clean_env": clean_env, "clean_level": clean_level,
            "adobe_curve": adobe_curve, "freq": freq, "use": use}


def score(x: np.ndarray, data: dict) -> dict:
    env = rms_frames(x)[:len(data["clean_env"])]
    delta = 20 * np.log10((env + 1e-12) / (data["clean_env"] + 1e-12))
    active = float(np.median(env[data["speech"]]))
    matched = x * (data["clean_level"] / max(active, 1e-12))
    _, curve = speech_spectrum(matched, data["speech"])
    return {
        "mae_db_vs_adobe": float(np.mean(np.abs(curve[data["use"]] - data["adobe_curve"][data["use"]]))),
        "envelope_corr_vs_clean": correlation(np.log(data["clean_env"][data["speech"]] + 1e-12),
                                               np.log(env[data["speech"]] + 1e-12)),
        "weak_p10_db_vs_clean": float(np.percentile(delta[data["weak"]], 10)),
        "active_level_db_vs_clean": float(np.median(delta[data["speech"]])),
        "adobe_level_db_vs_clean": float(np.median(20 * np.log10(
            (rms_frames(data["adobe"])[:len(data["clean_env"])] + 1e-12)
            / (data["clean_env"] + 1e-12))[data["speech"]])),
        "gap_dbfs": db(float(np.median(env[data["gap"]]))) if np.any(data["gap"]) else None,
    }


def tone(audio: np.ndarray, low_db: float, low_hz: float, high_db: float, high_hz: float) -> np.ndarray:
    from voxrefine.toneshape import apply_shelves
    return apply_shelves(audio, RATE, bass_db=low_db, bass_corner_hz=low_hz,
                         treble_db=high_db, treble_corner_hz=high_hz)


def summary_median(key: str, rows: dict[str, dict]) -> float | None:
    values = [rows[sample][key] for sample in SAMPLES if rows[sample][key] is not None]
    return statistics.median(values) if values else None


def rir_cross_domain() -> list[dict]:
    """Apply the fixed EQ to existing cap-60 RIR renders and exact Adobe pairs."""
    from sweep_deepfilternet_attenuation import (
        BENCH as RIR_BENCH, EVENTS as RIR_EVENTS, OUT as RIR_OUT,
        metrics as rir_metrics, read as rir_read,
    )
    adobe_root = ROOT / "results/noise-rir-truth-01"
    rows = []
    for event in RIR_EVENTS:
        render = RIR_OUT / "renders" / "atten-60db" / event / f"{event}-rir-only-dfn-input.wav"
        adobe_path = adobe_root / f"adobe-pair-{event}-rir-only" / "adobe-v2-output.wav"
        if not all(path.is_file() for path in (render, adobe_path)):
            return []
        source = rir_read(RIR_BENCH / "inputs" / f"{event}-rir-only.wav")
        dry = rir_read(RIR_BENCH / "inputs" / f"{event}-dry-control.wav")
        adobe = rir_read(adobe_path)
        baseline = rir_read(render)
        tuned = tone(baseline, **GLOBAL_EQ)
        before = rir_metrics(baseline, dry, adobe, source)
        after = rir_metrics(tuned, dry, adobe, source)
        rows.append({
            "event": event,
            "baseline_spectrum_mae_db_vs_adobe": before["speech_spectrum_mae_vs_adobe_db"],
            "fixed_eq_spectrum_mae_db_vs_adobe": after["speech_spectrum_mae_vs_adobe_db"],
            "spectrum_mae_change_db": after["speech_spectrum_mae_vs_adobe_db"] - before["speech_spectrum_mae_vs_adobe_db"],
            "baseline_tail_reduction_db": before["tail_reduction_vs_input_db"],
            "fixed_eq_tail_reduction_db": after["tail_reduction_vs_input_db"],
            "baseline_speech_envelope_corr": before["speech_env_corr_vs_dry"],
            "fixed_eq_speech_envelope_corr": after["speech_env_corr_vs_dry"],
        })
    return rows


def longform_adobe_pairs() -> tuple[list[dict], list[dict]]:
    """Render cap60 on three exact long-form Adobe pairs, then score fixed EQ."""
    names = ("01-pauses-nombreuses", "02-parole-continue", "03-variations-intensite")
    corpus = ROOT / "corpus/samples/librivox-fortune"
    adobe_dir = ROOT / "results/librivox-fortune/adobe-v2"
    output_dir = OUT / "librivox-fortune/cap60"
    input_dir = OUT / "librivox-fortune/input-pcm16"
    binary = ROOT / ".tools/deepfilternet/deep-filter"
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)
    from compare_clear_noisy_to_adobe_v2 import sha256
    renders, runtimes = [], []
    listen_dir = OUT / "listening-levelmatched"
    listen_dir.mkdir(exist_ok=True)
    for name in names:
        source_path = corpus / f"{name}.wav"
        input_path = input_dir / source_path.name
        render_path = output_dir / source_path.name
        adobe_path = adobe_dir / f"{name}-adobe-v2.wav"
        source, rate = sf.read(source_path, dtype="float32")
        if rate != RATE or source.ndim != 1:
            raise ValueError(f"Expected mono {RATE} Hz source: {source_path}")
        sf.write(input_path, source, rate, subtype="PCM_16")
        started = time.perf_counter()
        proc = subprocess.run([str(binary), "--atten-lim-db", "60", "--compensate-delay",
                               "-o", str(output_dir), str(input_path)], capture_output=True, text=True)
        elapsed = time.perf_counter() - started
        if proc.returncode or not render_path.is_file():
            raise RuntimeError(f"DFN cap60 failed on {name}: {(proc.stderr or proc.stdout)[-1500:]}")
        runtimes.append({"sample": name, "wall_seconds": elapsed,
                         "duration_seconds": len(source) / RATE, "rtf": elapsed / (len(source) / RATE),
                         "input_sha256": sha256(source_path), "adobe_sha256": sha256(adobe_path)})
        audio = {"source": read(source_path), "cap60": read(render_path), "adobe": read(adobe_path)}
        n = min(map(len, audio.values()))
        audio = {key: value[:n] for key, value in audio.items()}
        env = rms_frames(audio["source"])
        active = env >= np.percentile(env, 95) * 10 ** (-35 / 20)
        if active.sum() < 20:
            raise ValueError(f"Activity detector found too little speech in {name}")
        weak = active & (env <= np.percentile(env[active], 25))
        gap = ~active
        target = float(np.median(env[active]))
        freq, adobe_curve = speech_spectrum(audio["adobe"] * (target / max(float(np.median(rms_frames(audio["adobe"])[active])), 1e-12)), active)
        use = (freq >= 80) & (freq <= 12_000)
        variant_metrics = {}
        variants = {"cap60": audio["cap60"],
                    "cap60+global-EQ": tone(audio["cap60"], **GLOBAL_EQ)}
        for label, x in variants.items():
            out_env = rms_frames(x)
            matched = x * (target / max(float(np.median(out_env[active])), 1e-12))
            _, curve = speech_spectrum(matched, active)
            delta = 20 * np.log10((out_env + 1e-12) / (env + 1e-12))
            variant_metrics[label] = {
                "speech_spectrum_mae_db_vs_adobe": float(np.mean(np.abs(curve[use] - adobe_curve[use]))),
                "speech_envelope_corr_vs_input": correlation(np.log(env[active] + 1e-12), np.log(out_env[active] + 1e-12)),
                "weak_speech_p10_db_vs_input": float(np.percentile(delta[weak], 10)),
                "active_level_delta_db_vs_input": float(np.median(delta[active])),
                "gap_rms_dbfs": db(float(np.median(out_env[gap]))) if np.any(gap) else None,
            }
        adobe_level = float(np.median(rms_frames(audio["adobe"])[active]))
        for label, x in (("cap60", audio["cap60"]),
                         ("cap60-global-eq", variants["cap60+global-EQ"])):
            level = float(np.median(rms_frames(x)[active]))
            sf.write(listen_dir / f"fortune-{name}-{label}-at-adobe-level.wav",
                     x * (adobe_level / max(level, 1e-12)), RATE, subtype="PCM_24")
        sf.write(listen_dir / f"fortune-{name}-adobe-v2.wav", audio["adobe"], RATE, subtype="PCM_24")
        renders.append({"sample": name, "duration_seconds": n / RATE,
                        "input_sha256": sha256(source_path), "adobe_sha256": sha256(adobe_path),
                        "cap60_sha256": sha256(render_path), "metrics": variant_metrics})
    (output_dir / "runtime.json").write_text(json.dumps(runtimes, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    xs = np.arange(len(renders)); width = .36
    for ax, key, title, ylabel in (
        (axes[0], "speech_spectrum_mae_db_vs_adobe", "Distance spectrale à Adobe", "MAE (dB; plus bas = plus proche)"),
        (axes[1], "speech_envelope_corr_vs_input", "Enveloppe vocale conservée", "Corrélation à l’entrée (plus haut = plus proche)"),
    ):
        before = [row["metrics"]["cap60"][key] for row in renders]
        after = [row["metrics"]["cap60+global-EQ"][key] for row in renders]
        ax.bar(xs - width / 2, before, width, label="DeepFilterNet cap 60", color="#0072b2")
        ax.bar(xs + width / 2, after, width, label="cap 60 + EQ commune", color="#e69f00")
        ax.set_xticks(xs, ["Pauses", "Parole continue", "Variations"])
        ax.set_title(title); ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=.22); ax.legend(fontsize=8)
    fig.suptitle("Test sur trois extraits longs Adobe v2 · même voix, contenus différents")
    fig.savefig(OUT / "longform-adobe-pairs.png", dpi=170)
    plt.close(fig)
    return renders, runtimes


def main() -> None:
    data = {sample: load_sample(sample) for sample in SAMPLES}
    # Small stationary shelves only: no dynamics or time-varying gain. The
    # candidate EQ and global gain are selected from the other two voices.
    candidates = []
    for low_hz in (100.0, 200.0):
        for low_db in (-2.0, -1.0, 0.0, 1.0, 2.0):
            for high_hz in (3_500.0, 5_000.0):
                for high_db in (-3.0, -2.0, -1.0, 0.0, 1.0):
                    candidates.append({"low_db": low_db, "low_hz": low_hz,
                                       "high_db": high_db, "high_hz": high_hz})
    all_rows = []
    fold_rows = []
    renders = {}
    baseline_by_sample = {sample: score(data[sample]["base"], data[sample]) for sample in SAMPLES}
    fixed_eq_by_sample = {sample: score(tone(data[sample]["base"], **GLOBAL_EQ), data[sample])
                          for sample in SAMPLES}
    for held_out in SAMPLES:
        train = [sample for sample in SAMPLES if sample != held_out]
        trial_scores = []
        for params in candidates:
            per_train = []
            adjusted_level_errors = []
            for sample in train:
                result = score(tone(data[sample]["base"], **params), data[sample])
                per_train.append(result["mae_db_vs_adobe"])
                adjusted_level_errors.append(result["active_level_db_vs_clean"]
                                             - result["adobe_level_db_vs_clean"])
            trial_scores.append({**params,
                                 "train_mae_db": float(np.mean(per_train)),
                                 "train_level_error_db": float(np.median(adjusted_level_errors))})
        # Only choose tonal shape using training voices. The gain then centers
        # their median active speech level on Adobe's, without using held-out data.
        chosen = min(trial_scores, key=lambda row: (row["train_mae_db"],
                                                     abs(row["train_level_error_db"]),
                                                     abs(row["low_db"]) + abs(row["high_db"])))
        params = {key: chosen[key] for key in ("low_db", "low_hz", "high_db", "high_hz")}
        gain_db = -chosen["train_level_error_db"]
        base = data[held_out]["base"]
        filtered = tone(base, **params) * 10 ** (gain_db / 20)
        baseline = base * 10 ** (gain_db / 20)
        base_result = score(baseline, data[held_out])
        result = score(filtered, data[held_out])
        result.update({"gain_applied_db": gain_db, **params})
        fold_rows.append({"held_out": held_out, "train_voices": train,
                          "train_mae_db": chosen["train_mae_db"], "gain_db": gain_db,
                          **params, "baseline": base_result, "challenger": result})
        renders[held_out] = (baseline, filtered, data[held_out]["adobe"])
        for variant, metrics in (("cap60+train-level-match", base_result),
                                 ("cap60+LOVO-tone+train-level-match", result)):
            all_rows.append({"held_out": held_out, "variant": variant, **metrics})
        all_rows.append({"held_out": held_out, "variant": "cap60+fixed-global-EQ",
                         **fixed_eq_by_sample[held_out], **GLOBAL_EQ})

    OUT.mkdir(parents=True, exist_ok=True)
    rir_rows = rir_cross_domain()
    longform_rows, longform_runtimes = longform_adobe_pairs()
    with (OUT / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(dict.fromkeys(key for row in all_rows for key in row)),
            lineterminator="\n",
        )
        writer.writeheader(); writer.writerows(all_rows)
    (OUT / "leave-one-voice-out.json").write_text(json.dumps({
        "purpose": "Exploratory global tone and level correction on DeepFilterNet cap60; select parameters on two voices, report the held-out third.",
        "sample_rate_hz": RATE,
        "sweep": {"bass_db": [-2, -1, 0, 1, 2], "bass_corner_hz": [100, 200],
                  "treble_db": [-3, -2, -1, 0, 1], "treble_corner_hz": [3500, 5000]},
        "fixed_global_eq_challenger": GLOBAL_EQ,
        "fixed_global_eq_scores": {sample: fixed_eq_by_sample[sample] for sample in SAMPLES},
        "fixed_global_eq_median": {key: summary_median(key, fixed_eq_by_sample)
                                    for key in fixed_eq_by_sample[SAMPLES[0]]},
        "fixed_global_eq_baseline_median": {key: summary_median(key, baseline_by_sample)
                                             for key in baseline_by_sample[SAMPLES[0]]},
        "rir_cross_domain": rir_rows,
        "longform_adobe_pairs_same_speaker": longform_rows,
        "longform_cap60_runtime": longform_runtimes,
        "selection": "Lowest mean training Adobe spectral MAE; tie break by training active-level distance, then smaller absolute EQ. Global gain is the training median level difference to Adobe.",
        "baseline_cap60_median": {key: summary_median(key, baseline_by_sample)
                                  for key in baseline_by_sample[SAMPLES[0]]},
        "folds": fold_rows,
        "limitations": ["Only three controlled speech/noise pairs; this is not a universal-preset validation.",
                        "The target Adobe output is a style reference and may itself suppress speech.",
                        "Static EQ cannot perform dereverberation or improve time-varying separation.",
                        "Choose by objective curves only; no listening test was performed in this pass."],
    }, indent=2, ensure_ascii=False) + "\n")
    listen_dir = OUT / "listening-levelmatched"
    listen_dir.mkdir(exist_ok=True)
    for sample, (baseline, filtered, adobe) in renders.items():
        target_env = rms_frames(adobe)
        target_level = float(np.median(target_env[data[sample]["speech"]]))
        base_native = data[sample]["base"]
        eq_native = tone(base_native, **GLOBAL_EQ)
        base_env = rms_frames(base_native)
        eq_env = rms_frames(eq_native)
        base_native = base_native * (target_level / max(float(np.median(base_env[data[sample]["speech"]])), 1e-12))
        eq_native = eq_native * (target_level / max(float(np.median(eq_env[data[sample]["speech"]])), 1e-12))
        for name, audio in (("cap60-at-adobe-level", base_native), ("cap60-global-eq-at-adobe-level", eq_native),
                            ("adobe-v2", adobe)):
            sf.write(listen_dir / f"{sample}-{name}.wav", audio, RATE, subtype="PCM_24")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    x = np.arange(len(SAMPLES)); width = .36
    for ax, key, title, ylabel in (
        (axes[0], "mae_db_vs_adobe", "Distance de spectre", "MAE vs Adobe (dB; plus bas = plus proche)"),
        (axes[1], "weak_p10_db_vs_clean", "Voix faible", "P10 vs voix propre (dB; plus haut = moins atténué)"),
    ):
        base_values = [baseline_by_sample[sample][key] for sample in SAMPLES]
        eq_values = [fixed_eq_by_sample[sample][key] for sample in SAMPLES]
        ax.bar(x - width / 2, base_values, width, label="DeepFilterNet cap 60", color="#0072b2")
        ax.bar(x + width / 2, eq_values, width, label="cap 60 + EQ commune", color="#e69f00")
        ax.set_xticks(x, ["Emy", "Rémi", "Stéphanie"]); ax.set_title(title)
        ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=.22); ax.legend(fontsize=8)
    fig.suptitle("EQ globale fixe testée sur trois voix appariées à Adobe v2")
    fig.savefig(OUT / "fixed-global-eq-heldout.png", dpi=170)
    plt.close(fig)
    print(json.dumps({"baseline_median": {key: summary_median(key, baseline_by_sample)
                                           for key in baseline_by_sample[SAMPLES[0]]},
                      "folds": fold_rows}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
