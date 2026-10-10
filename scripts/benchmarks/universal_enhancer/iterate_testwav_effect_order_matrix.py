#!/usr/bin/env python3
"""Small effect-order/dynamics screen on the already-consumed test.wav only.

Outputs remain in ignored results/. Adobe is descriptive only and is never used
to optimize parameters. The two pre-EQ model passes use one fixed StuPASE seed.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter, sosfiltfilt, stft

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "results/user-recording-test-2026-10-09"
OUT = BASE / "effect-order-matrix-01"
SR = 48000
FRAME, HOP = 960, 480
TAIL = (5.79, 5.94)
SEED = 20261010


def read(path: Path) -> tuple[np.ndarray, int]:
    x, sr = sf.read(path, dtype="float64")
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x, sr


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2) + 1e-30))


def db(x: float) -> float:
    return float(20 * np.log10(max(float(x), 1e-15)))


def env(x: np.ndarray) -> np.ndarray:
    n = 1 + (len(x) - FRAME) // HOP
    return np.asarray([rms(x[i * HOP:i * HOP + FRAME]) for i in range(max(n, 0))])


def peaking(x: np.ndarray, fc: float, q: float, gain_db: float) -> np.ndarray:
    w = 2 * np.pi * fc / SR
    A = 10 ** (gain_db / 40)
    alpha, c = np.sin(w) / (2 * q), np.cos(w)
    b = np.array([1 + alpha * A, -2 * c, 1 - alpha * A])
    a = np.array([1 + alpha / A, -2 * c, 1 - alpha / A])
    return lfilter(b / a[0], a / a[0], x)


def low_shelf(x: np.ndarray, fc: float, gain_db: float, slope: float = .8) -> np.ndarray:
    A = 10 ** (gain_db / 40)
    w = 2 * np.pi * fc / SR
    c, s = np.cos(w), np.sin(w)
    alpha = s / 2 * np.sqrt((A + 1 / A) * (1 / slope - 1) + 2)
    beta = 2 * np.sqrt(A) * alpha
    b = np.array([A * ((A + 1) - (A - 1) * c + beta),
                  2 * A * ((A - 1) - (A + 1) * c),
                  A * ((A + 1) - (A - 1) * c - beta)])
    a = np.array([(A + 1) + (A - 1) * c + beta,
                  -2 * ((A - 1) + (A + 1) * c),
                  (A + 1) + (A - 1) * c - beta])
    return lfilter(b / a[0], a / a[0], x)


def high_shelf(x: np.ndarray, fc: float, gain_db: float, slope: float = .8) -> np.ndarray:
    A = 10 ** (gain_db / 40)
    w = 2 * np.pi * fc / SR
    c, s = np.cos(w), np.sin(w)
    alpha = s / 2 * np.sqrt((A + 1 / A) * (1 / slope - 1) + 2)
    beta = 2 * np.sqrt(A) * alpha
    b = np.array([A * ((A + 1) + (A - 1) * c + beta),
                  -2 * A * ((A - 1) + (A + 1) * c),
                  A * ((A + 1) + (A - 1) * c - beta)])
    a = np.array([(A + 1) - (A - 1) * c + beta,
                  2 * ((A - 1) - (A + 1) * c),
                  (A + 1) - (A - 1) * c - beta])
    return lfilter(b / a[0], a / a[0], x)


def matched(x: np.ndarray, active: np.ndarray, target: float) -> tuple[np.ndarray, float]:
    e = env(x)
    m = min(len(e), len(active))
    g = target / max(rms(e[:m][active[:m]]), 1e-15)
    return x * g, g


def make_gate(cap: np.ndarray, active: np.ndarray) -> np.ndarray:
    ce = env(cap)
    mask = ce > .035 * np.max(np.abs(cap))
    lookahead = 2  # 20 ms at 10 ms hop
    opened = np.zeros_like(mask)
    for ahead in range(lookahead + 1):
        opened[:len(mask) - ahead if ahead else len(mask)] |= mask[ahead:]
    state = 0.0
    smooth = np.zeros(len(opened))
    dt = HOP / SR
    for i, on in enumerate(opened):
        tau = .005 if on else .040
        coeff = np.exp(-dt / tau)
        state = (1 - coeff) * float(on) + coeff * state
        smooth[i] = state
    return np.interp(np.arange(len(cap)) / SR, np.arange(len(smooth)) * dt,
                     smooth, left=0, right=0)


def soft_compressor(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gentle RMS compressor, threshold set from active speech, 1.4:1."""
    frames = env(x)
    threshold = float(np.percentile(frames[frames > np.percentile(frames, 30)], 65))
    desired = np.minimum(1.0, (threshold / np.maximum(frames, 1e-12)) ** (1 - 1 / 1.4))
    smoothed = np.empty_like(desired)
    value = 1.0
    for i, goal in enumerate(desired):
        tau = .025 if goal < value else .120
        a = np.exp(-(HOP / SR) / tau)
        value = a * value + (1 - a) * goal
        smoothed[i] = value
    sample_gain = np.interp(np.arange(len(x)) / SR, np.arange(len(smoothed)) * HOP / SR,
                            smoothed, left=smoothed[0], right=smoothed[-1])
    return x * sample_gain, 20 * np.log10(np.maximum(smoothed, 1e-12))


def deesser(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Conservative 5.5–9 kHz detector; max reduction 1.5 dB, slow release."""
    sos = butter(4, [5500, 9000], btype="bandpass", fs=SR, output="sos")
    band = sosfiltfilt(sos, x)
    b_env, b_hop = 480, 240
    n = 1 + (len(x) - b_env) // b_hop
    be = np.asarray([rms(band[i * b_hop:i * b_hop + b_env]) for i in range(max(n, 0))])
    threshold = float(np.percentile(be[be > np.percentile(be, 30)], 75))
    reduction = np.minimum(1.5, np.maximum(0, 20 * np.log10(np.maximum(be, 1e-12) / max(threshold, 1e-12)) * .45))
    gain = 10 ** (-reduction / 20)
    # Blend only the detected sibilant band; no broadband dip.
    sample_gain = np.interp(np.arange(len(x)) / SR, np.arange(len(gain)) * b_hop / SR,
                            gain, left=1, right=1)
    return x + band * (sample_gain - 1), reduction


def infer_preeq(cap: np.ndarray, kind: str, out_path: Path) -> tuple[np.ndarray, float]:
    """Run the official local StuPASE model on a modest pre-EQ Cap60 input."""
    source16 = librosa.resample(cap.astype(np.float32), orig_sr=SR, target_sr=16000).astype(np.float32)
    if kind == "presence":
        # Build the RBJ peaking EQ at the model sample rate.
        w = 2 * np.pi * 3500 / 16000
        A = 10 ** (1.0 / 40)
        alpha, c = np.sin(w) / (2 * .65), np.cos(w)
        b = np.array([1 + alpha * A, -2 * c, 1 - alpha * A]); a = np.array([1 + alpha / A, -2 * c, 1 - alpha / A])
        source16 = lfilter(b / a[0], a / a[0], librosa.resample(cap.astype(np.float32), orig_sr=SR, target_sr=16000)).astype(np.float32)
    elif kind == "bass":
        # 1 dB low shelf @180 Hz, RBJ, at model sample rate.
        f = 180; A = 10 ** (1.0 / 40); w = 2 * np.pi * f / 16000; c, s = np.cos(w), np.sin(w); S = .8
        alpha = s / 2 * np.sqrt((A + 1 / A) * (1 / S - 1) + 2); beta = 2 * np.sqrt(A) * alpha
        b = np.array([A * ((A + 1) - (A - 1) * c + beta), 2 * A * ((A - 1) - (A + 1) * c), A * ((A + 1) - (A - 1) * c - beta)])
        a = np.array([(A + 1) + (A - 1) * c + beta, -2 * ((A - 1) + (A + 1) * c), (A + 1) + (A - 1) * c - beta])
        source16 = lfilter(b / a[0], a / a[0], source16).astype(np.float32)
    else:
        raise ValueError(kind)

    repo = ROOT / ".tools/stupase-src/stupase"
    weights = ROOT / ".tools/stupase-weights"
    sys.path.insert(0, str(repo))
    import torch
    from models.stupase import StuPASE
    checkpoints = [weights / n for n in ("DeWavLM-R.pt", "CFM.pt", "Vocoder_Mel-16k.pt")]
    model = StuPASE(*(str(p) for p in checkpoints)).to("cpu").eval()
    peak = float(np.max(np.abs(source16)))
    torch.manual_seed(SEED)
    inp = torch.from_numpy((source16 / max(peak, 1e-8)).copy()).unsqueeze(0)
    t0 = time.perf_counter()
    with torch.inference_mode():
        y = model(inp, sr=16000, steps=8, cfg_strength=.25, sway_sampling_coef=-1.0)
    elapsed = time.perf_counter() - t0
    del model
    y = y.squeeze().detach().cpu().numpy().astype(np.float32)
    y *= peak / max(float(np.max(np.abs(y))), 1e-8)
    y48 = librosa.resample(y, orig_sr=16000, target_sr=SR).astype(np.float64)
    if len(y48) < len(cap):
        y48 = np.pad(y48, (0, len(cap) - len(y48)))
    y48 = y48[:len(cap)]
    return y48, elapsed


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cap, sr = read(BASE / "cap60-dereverb-screen-02/deepfilternet-cap60-aligned.wav")
    stu, sr_stu = read(BASE / "stupase-testwav-cfg025-01/stupase-cap60-raw-48k.wav")
    adobe, sr_adobe = read(BASE / "06_adobe_v2.wav")
    original, sr_original = read(BASE / "01_input_mono_48k.wav")
    if sr != SR or sr_stu != SR:
        raise ValueError("Expected existing Cap60 and StuPASE assets at 48 kHz.")
    if sr_original != SR:
        raise ValueError("Expected original working input at 48 kHz.")
    n = min(len(cap), len(stu), len(adobe), len(original))
    cap, stu, adobe, original = cap[:n], stu[:n], adobe[:n], original[:n]
    ce = env(cap)
    active = ce > .035 * np.max(np.abs(cap))
    speech_level = rms(ce[active])
    gate = make_gate(cap, active)
    hp = sosfiltfilt(butter(6, 7800, btype="highpass", fs=SR, output="sos"), cap)
    hf = .5 * hp * gate

    def gentle_eq(z: np.ndarray) -> np.ndarray:
        return high_shelf(peaking(z, 3500, .72, 1.5), 5500, 2.0)

    current, current_sr = read(BASE / "stupase-presence-matrix-02/12.wav")
    if current_sr != SR or len(current) < n:
        raise ValueError("Missing or misaligned frozen current-chain reference.")
    current = current[:n]
    base = low_shelf(current, 180, 1.5)
    # Same EQ is moved before HF residual addition, so the restored branch is untouched.
    order = gentle_eq(stu) + hf
    order = low_shelf(order, 180, 1.5)
    comp, comp_gr = soft_compressor(base)
    deess, deess_gr = deesser(base)

    # Genuine pre-model tests: only small, fixed, documented input changes.
    pre, timing = {}, {}
    for kind in ("presence", "bass"):
        pre[kind], timing[kind] = infer_preeq(cap, kind, OUT / f"_internal_stupase_preeq_{kind}.wav")
    pre_presence = low_shelf(gentle_eq(pre["presence"] + hf), 180, 1.5)
    pre_bass = gentle_eq(pre["bass"] + hf)

    variants = {
        "A": ("Current chain + 1.5 dB low shelf", base),
        "B": ("EQ on StuPASE before HF restore", order),
        "C": ("Soft 1.4:1 compression", comp),
        "D": ("Sibilance de-esser, max 1.5 dB", deess),
        "E": ("+1 dB presence before StuPASE", pre_presence),
        "F": ("+1 dB bass shelf before StuPASE", pre_bass),
    }
    refs = {"ORIGINAL": original, "CAP60": cap, "ADOBE": adobe}
    rows, matched_audio = [], {}
    for key, (label, x) in variants.items():
        y, gain = matched(x, active, speech_level)
        matched_audio[key] = y
        sf.write(OUT / f"{key}.wav", y.astype(np.float32), SR, subtype="PCM_24")
        ey = env(y)
        m = min(len(ey), len(active))
        tail = y[round(TAIL[0] * SR):round(TAIL[1] * SR)]
        f, t, z = stft(y, fs=SR, nperseg=2048, noverlap=1536, boundary=None)
        am = np.interp(t, np.arange(len(active)) * HOP / SR, active.astype(float), left=0, right=0) > .5
        band_metrics = {}
        for lo, hi, name in [(80, 140, "80_140"), (140, 250, "140_250"), (4000, 8000, "4_8k"), (8000, 12000, "8_12k")]:
            use = (f >= lo) & (f < hi)
            band_metrics[name + "_active_db_rel_speech"] = db(np.sqrt(np.mean(np.abs(z[use][:, am]) ** 2)) / speech_level)
        p10 = float(np.percentile(20 * np.log10(np.maximum(ey[:m], 1e-12) / np.maximum(ce[:m], 1e-12))[active[:m]], 10))
        row = {"id": key, "candidate": label, "level_match_gain_db": db(gain),
               "tail_db_rel_speech": db(rms(tail) / speech_level), "active_p10_vs_cap60_db": p10,
               "peak_dbfs": db(np.max(np.abs(y))), "clipped_samples": int(np.count_nonzero(np.abs(y) >= 1)),
               **band_metrics}
        if key == "C":
            cm = min(len(comp_gr), len(active))
            active_gr = -comp_gr[:cm][active[:cm]]
            row["compression_gr_median_db"] = float(np.median(active_gr))
            row["compression_gr_p95_db"] = float(np.percentile(active_gr, 95))
        if key == "D":
            row["deesser_active_fraction"] = float(np.mean(deess_gr > .1))
            row["deesser_reduction_p95_db"] = float(np.percentile(deess_gr, 95))
        rows.append(row)

    # Reference rows are descriptive only; their playback is level matched to the same mask.
    for key, x in refs.items():
        y, gain = matched(x, active, speech_level)
        sf.write(OUT / f"{key}.wav", y.astype(np.float32), SR, subtype="PCM_24")
        ey = env(y)
        m = min(len(ey), len(active))
        tail = y[round(TAIL[0] * SR):round(TAIL[1] * SR)]
        rows.append({"id": key, "candidate": key + " descriptive reference", "level_match_gain_db": db(gain),
                     "tail_db_rel_speech": db(rms(tail) / speech_level),
                     "active_p10_vs_cap60_db": float(np.percentile(20 * np.log10(np.maximum(ey[:m], 1e-12) / np.maximum(ce[:m], 1e-12))[active[:m]], 10)),
                     "peak_dbfs": db(np.max(np.abs(y))), "clipped_samples": int(np.count_nonzero(np.abs(y) >= 1))})

    report = {"scope": "one already-consumed user recording; exploratory only; Adobe descriptive, not optimized",
              "sample_rate_hz": SR, "seed_for_pre_eq_stupase": SEED, "steps": 8, "cfg": .25,
              "pre_eq_model_inference_seconds": timing,
              "shared_hf_restore": "0.5 * HP(7.8kHz, Cap60) * frozen smoothed speech gate; 20ms lookahead, 5ms attack, 40ms release",
              "audition_level_match": "constant gain to shared Cap60 active-speech RMS using the same 20ms/10ms mask",
              "variants": rows,
              "limits": ["single recording", "single terminal tail window", "metrics do not predict perceived timbre", "pre-EQ changes StuPASE's generated output", "not a universal preset"]}
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)

    fig, axs = plt.subplots(2, 1, figsize=(12, 9), constrained_layout=True)
    colors = ["#555555", "#007f5f", "#bc6c25", "#457b9d", "#9b5de5", "#d1495b"]
    for (key, (label, _)), color in zip(variants.items(), colors):
        x = matched_audio[key]
        f, t, z = stft(x, fs=SR, nperseg=2048, noverlap=1536, boundary=None)
        am = np.interp(t, np.arange(len(active)) * HOP / SR, active.astype(float), left=0, right=0) > .5
        p = np.mean(np.abs(z[:, am]) ** 2, axis=1)
        axs[0].plot(f, 10 * np.log10(np.maximum(p, 1e-16)), label=f"{key}: {label}", color=color)
        e = env(x)
        axs[1].plot(np.arange(len(e)) * HOP / SR, 20 * np.log10(np.maximum(e / speech_level, 1e-9)), label=key, color=color)
    for key, color in [("ADOBE", "#d08c00"), ("CAP60", "#8b5e3c"), ("ORIGINAL", "#222222")]:
        y, _ = matched(refs[key], active, speech_level)
        f, t, z = stft(y, fs=SR, nperseg=2048, noverlap=1536, boundary=None)
        am = np.interp(t, np.arange(len(active)) * HOP / SR, active.astype(float), left=0, right=0) > .5
        p = np.mean(np.abs(z[:, am]) ** 2, axis=1)
        axs[0].plot(f, 10 * np.log10(np.maximum(p, 1e-16)), label=key + " (descriptive)", color=color, lw=1.8, ls="--")
    axs[0].set(xlim=(60, 14000), ylabel="Active PSD (dB)", title="Speech-active spectrum — descriptive comparison")
    axs[0].grid(alpha=.25); axs[0].legend(fontsize=8, ncol=2)
    axs[1].axvspan(*TAIL, color="gold", alpha=.2)
    axs[1].set(xlim=(5.2, 6.2), ylim=(-80, 5), xlabel="Time (s)", ylabel="RMS / active RMS (dB)", title="Speech envelope and short terminal window")
    axs[1].grid(alpha=.25); axs[1].legend(fontsize=8, ncol=2)
    fig.savefig(OUT / "effect-order-comparison.png", dpi=170)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
