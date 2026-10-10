#!/usr/bin/env python3
"""Synthetic-only, calibration/evaluation split for delayed LP and mono WPE."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
SR, N, PAUSE = 16_000, 32_000, 23_200
ROOMS, K, DELAY, WPE_ITERS = 12, 30, 4, 3
SEED = 610_2026
N_FFT, HOP = 512, 128


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_room(room_id: int):
    rng = np.random.default_rng(SEED + room_id * 1009)
    h = np.zeros(round(.48 * SR), np.float64)
    h[0] = 1.0
    # Deterministic per-room early reflections and diffuse exponentially
    # decaying late taps. Both independent sources in a room share this RIR.
    for _ in range(8):
        delay = int(rng.uniform(.025, .145) * SR)
        h[delay] += rng.choice([-1., 1.]) * rng.uniform(.035, .16)
    decay = rng.uniform(.15, .27)
    for delay in range(round(.11 * SR), h.size, 80):
        h[delay] += rng.normal() * .11 * np.exp(-(delay / SR - .11) / decay)
    return h


def make_dry(room_id: int, utterance_id: int):
    rng = np.random.default_rng(SEED + room_id * 65_537 + utterance_id * 9_973 + 41)
    t = np.arange(N, dtype=np.float64) / SR
    f0 = rng.uniform(92, 225)
    phase = 2 * np.pi * np.cumsum(f0 + rng.uniform(1., 4.) *
        np.sin(2 * np.pi * rng.uniform(3., 6.) * t)) / SR
    env = np.zeros(N, np.float64)
    starts = np.array([.14, .45, .79, 1.10]) + rng.uniform(-.03, .03, 4)
    for start, width in zip(starts, rng.uniform(.18, .28, 4)):
        u = (t - start) / width
        pulse = np.where((u >= 0) & (u < 1),
            np.sin(np.pi * np.clip(u, 0, 1)) ** .65, 0.)
        env = np.maximum(env, pulse)
    env *= .74 + .26 * np.sin(2 * np.pi * rng.uniform(3.4, 6.8) * t + rng.uniform(0, 6))
    voice = np.zeros(N, np.float64)
    formants = rng.uniform([550, 1050, 2200], [950, 1900, 3400])
    widths = rng.uniform([350, 500, 750], [650, 900, 1200])
    for n in range(1, 20):
        hz = f0 * n
        amp = (np.exp(-((hz-formants[0])/widths[0])**2) +
               .62*np.exp(-((hz-formants[1])/widths[1])**2) +
               .28*np.exp(-((hz-formants[2])/widths[2])**2)) / n
        voice += amp * np.sin(n * phase + rng.uniform(-.2, .2))
    dry = (env * voice * (.13 / max(np.max(np.abs(voice)), 1e-8))).astype(np.float32)
    return dry


def convolve_taps(dry: np.ndarray, h: np.ndarray) -> np.ndarray:
    out = dry.astype(np.float64).copy()
    for lag in np.flatnonzero(h[1:]) + 1:
        out[lag:] += h[lag] * dry[:-lag]
    return out.astype(np.float32)


def stft(x: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(x.astype(np.float32))[None]
    win = torch.hann_window(N_FFT)
    return torch.stft(tensor, n_fft=N_FFT, hop_length=HOP,
        win_length=N_FFT, window=win, center=True, return_complex=True)[0].numpy()


def istft(spec: np.ndarray, length: int) -> np.ndarray:
    win = torch.hann_window(N_FFT)
    value = torch.from_numpy(np.asarray(spec, dtype=np.complex64))[None]
    return torch.istft(value, n_fft=N_FFT, hop_length=HOP,
        win_length=N_FFT, window=win, center=True, length=length)[0].numpy()


def delayed_matrix(spec: np.ndarray):
    first = DELAY + K - 1
    times = np.arange(first, spec.shape[-1], dtype=np.int64)
    indices = times[:, None] - (DELAY + np.arange(K)[None, :])
    # [frequency,time,tap], oldest to newest.
    return spec[:, indices], times


def ridge_coefficients(xspec: np.ndarray, target: np.ndarray,
                       weights: np.ndarray | None = None):
    z, times = delayed_matrix(xspec)
    z = z.astype(np.complex128)
    y = target[:, times].astype(np.complex128)
    if weights is None:
        w = np.ones((xspec.shape[0], len(times)), dtype=np.float64)
    else:
        w = np.maximum(weights[:, times], 1e-12)
    gram = np.einsum("fti,ftj,ft->fij", z.conj(), z, w, optimize=True)
    cross = np.einsum("fti,ft,ft->fi", z.conj(), y, w, optimize=True)
    diagonal = np.trace(gram, axis1=1, axis2=2).real / K
    loading = 1e-4 * np.maximum(diagonal, 1e-12)
    gram += loading[:, None, None] * np.eye(K, dtype=np.complex128)[None]
    coeff = np.linalg.solve(gram, cross[..., None])[..., 0]
    return coeff, times, loading


def apply_coefficients(x: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    spec = stft(x)
    z, times = delayed_matrix(spec)
    pred = np.einsum("fti,fi->ft", z, coeff, optimize=True)
    out = spec.copy()
    out[:, times] -= pred.astype(out.dtype)
    return istft(out, len(x))


def wpe_coefficients(xspec: np.ndarray):
    sigma = np.abs(xspec).astype(np.float64) ** 2
    floor = np.maximum(np.mean(sigma, axis=1, keepdims=True) * 1e-3, 1e-12)
    # Three pre-fixed blind WPE iterations on calibration A only.
    for _ in range(WPE_ITERS):
        weights = 1.0 / np.maximum(sigma, floor)
        coeff, times, loading = ridge_coefficients(xspec, xspec, weights)
        z, _ = delayed_matrix(xspec)
        pred = np.einsum("fti,fi->ft", z, coeff, optimize=True)
        y = xspec.copy()
        y[:, times] -= pred.astype(y.dtype)
        power = np.abs(y).astype(np.float64) ** 2
        updated = np.empty_like(power)
        updated[:, 0] = np.maximum(power[:, 0], floor[:, 0])
        for t in range(1, power.shape[1]):
            updated[:, t] = .9 * updated[:, t-1] + .1 * power[:, t]
        sigma = np.maximum(updated, floor)
        if not np.isfinite(coeff).all() or not np.isfinite(sigma).all():
            raise FloatingPointError("non-finite WPE calibration")
    return coeff, loading


def activity_masks(dry: np.ndarray):
    x = torch.from_numpy(dry.astype(np.float32))
    frames = x.unfold(0, 320, 160)
    rms = frames.square().mean(-1).sqrt()
    peak = rms.max().clamp_min(1e-8)
    active = rms > torch.maximum(peak * .02, rms.new_tensor(1e-5))
    weak = active & (rms < peak * .35)
    onset = active & (rms > F.pad(rms[:-1], (1, 0)) * 1.5)
    return active.numpy(), weak.numpy(), onset.numpy()


def frame_level_db(y: np.ndarray, x: np.ndarray):
    yt = torch.from_numpy(y.astype(np.float32))
    xt = torch.from_numpy(x.astype(np.float32))
    yf, xf = yt.unfold(0, 320, 160), xt.unfold(0, 320, 160)
    return (20 * torch.log10((yf.square().mean(-1).sqrt() + 1e-8) /
                              (xf.square().mean(-1).sqrt() + 1e-8))).numpy()


def quantile(values, q):
    if values.size == 0:
        return None
    return float(np.quantile(values, q))


def band_window_db(y: np.ndarray, x: np.ndarray, pause: int, lo_ms: int, hi_ms: int):
    ys, xs = stft(y), stft(x)
    freqs = np.fft.rfftfreq(N_FFT, d=1/SR)
    bins = (freqs >= 150) & (freqs <= 300)
    centers = np.arange(xs.shape[1]) * HOP
    frames = (centers >= pause + lo_ms * 16) & (centers < pause + hi_ms * 16)
    ei = np.abs(xs[np.ix_(bins, frames)]) ** 2
    eo = np.abs(ys[np.ix_(bins, frames)]) ** 2
    return float(10 * np.log10((ei.mean() + 1e-20) / (eo.mean() + 1e-20)))


def dbfs_rms(x):
    return float(20 * np.log10(np.sqrt(np.mean(np.square(x.astype(np.float64))) + 1e-20)))


def band_level_db(x: np.ndarray, pause: int, lo_ms: int, hi_ms: int):
    spec = stft(x)
    freqs = np.fft.rfftfreq(N_FFT, d=1/SR)
    bins = (freqs >= 150) & (freqs <= 300)
    centers = np.arange(spec.shape[1]) * HOP
    frames = (centers >= pause + lo_ms * 16) & (centers < pause + hi_ms * 16)
    return float(10 * np.log10(np.mean(np.abs(spec[np.ix_(bins, frames)]) ** 2) + 1e-20))


def room_metrics(dry_b, wet_b, out_wet, out_dry):
    active, weak, onset = activity_masks(dry_b)
    gain = frame_level_db(out_wet, wet_b)
    n = min(gain.size, active.size)
    active, weak, onset, gain = active[:n], weak[:n], onset[:n], gain[:n]
    def q(mask, p): return quantile(gain[mask], p)
    return {"active_p50_db": q(active, .5), "active_p10_db": q(active, .1),
        "weak_p10_db": q(weak, .1), "onset_p10_db": q(onset, .1),
        "weak_below_minus3_fraction": float(np.mean(gain[weak] < -3)),
        "weak_below_minus6_fraction": float(np.mean(gain[weak] < -6)),
        "dry_active_p50_db": quantile(frame_level_db(out_dry, dry_b)[:n][active], .5),
        "dry_active_p10_db": quantile(frame_level_db(out_dry, dry_b)[:n][active], .1),
        "dry_weak_p10_db": quantile(frame_level_db(out_dry, dry_b)[:n][weak], .1),
        "dry_onset_p10_db": quantile(frame_level_db(out_dry, dry_b)[:n][onset], .1)}


def score_arm(rooms, name):
    all_metrics = []
    for r in rooms:
        dry_b, wet_b = r["dry_b"], r["wet_b"]
        out_wet, out_dry = r[f"{name}_wet"], r[f"{name}_dry"]
        w1 = band_window_db(out_wet, wet_b, PAUSE, 150, 300)
        w2 = band_window_db(out_wet, wet_b, PAUSE, 300, 600)
        dry_w1_db, wet_w1_db = band_level_db(dry_b, PAUSE, 150, 300), band_level_db(wet_b, PAUSE, 150, 300)
        dry_w2_db, wet_w2_db = band_level_db(dry_b, PAUSE, 300, 600), band_level_db(wet_b, PAUSE, 300, 600)
        eligible = wet_w1_db > max(-60., dry_w1_db + 6.)
        coeff = r[f"{name}_coeff"]
        all_metrics.append({"room": r["room"], "w1_mlow_db": w1, "w2_mlow_db": w2,
            "dry_w1_dbfs": dry_w1_db, "wet_w1_dbfs": wet_w1_db,
            "eligible_w1": bool(eligible), "dry_w2_dbfs": dry_w2_db,
            "wet_w2_dbfs": wet_w2_db,
            "eligible_w2": bool(wet_w2_db > max(-60., dry_w2_db + 6.)),
            "median_filter_l2_norm_per_frequency": float(np.median(np.linalg.norm(coeff, axis=1))),
            "output_to_input_rms_db": float(20*np.log10((np.sqrt(np.mean(out_wet.astype(np.float64)**2))+1e-20)/
                                                         (np.sqrt(np.mean(wet_b.astype(np.float64)**2))+1e-20))),
            "output_abs_peak": float(np.max(np.abs(out_wet))),
            "speech": room_metrics(dry_b, wet_b, out_wet, out_dry)})
    eligible = [m for m in all_metrics if m["eligible_w1"]]
    w1 = np.array([m["w1_mlow_db"] for m in eligible])
    informative_w2 = [m for m in all_metrics if m["eligible_w2"]]
    w2_all = np.array([m["w2_mlow_db"] for m in informative_w2])
    speech = [m["speech"] for m in all_metrics]
    # Across-room summary; per-room W1 robustness prevents one room dominating.
    med = lambda key: float(np.median([m["speech"][key] for m in all_metrics]))
    drymed = lambda key: float(np.median([m["speech"][key] for m in all_metrics]))
    gates = {
        "all_12_w1_eligible": len(eligible) == ROOMS,
        "w1_mlow_median_ge_2db": len(eligible) == ROOMS and float(np.median(w1)) >= 2.,
        "w1_positive_rooms_ge_10_of_12": len(eligible) == ROOMS and int(np.sum(w1 > 0)) >= 10,
        "w2_mlow_median_ge_minus_0_5db": w2_all.size > 0 and float(np.median(w2_all)) >= -.5,
        "w2_regressions_below_minus1_le_10pct": w2_all.size > 0 and float(np.mean(w2_all < -1.)) <= .10,
        "w2_informative_coverage_ge_25pct": len(informative_w2) / ROOMS >= .25,
        "active_p50_range": -.5 <= med("active_p50_db") <= .5,
        "active_p10_gt_minus1_5": med("active_p10_db") > -1.5,
        "weak_p10_gt_minus1_5": med("weak_p10_db") > -1.5,
        "onset_p10_gt_minus1": med("onset_p10_db") > -1.,
        "weak_below_minus3_le_10pct": float(np.median([m["weak_below_minus3_fraction"] for m in speech])) <= .10,
        "weak_below_minus6_le_3pct": float(np.median([m["weak_below_minus6_fraction"] for m in speech])) <= .03,
        "dry_active_p50_range": -.25 <= drymed("dry_active_p50_db") <= .25,
        "dry_active_p10_gt_minus0_75": drymed("dry_active_p10_db") > -.75,
        "dry_weak_p10_gt_minus1": drymed("dry_weak_p10_db") > -1.,
        "dry_onset_p10_gt_minus0_75": drymed("dry_onset_p10_db") > -.75,
    }
    return {"arm": name, "overall": "PASS" if all(gates.values()) else "FAIL",
        "eligible_w1_count": len(eligible), "w1_mlow_median_db": float(np.median(w1)) if len(w1) else None,
        "w1_positive_room_count": int(np.sum(w1 > 0)) if len(w1) else 0,
        "w2_informative_count": len(informative_w2),
        "w2_mlow_median_db": float(np.median(w2_all)) if w2_all.size else None,
        "w2_regression_below_minus1_fraction": float(np.mean(w2_all < -1.)) if w2_all.size else 1.0,
        "speech_medians": {k: med(k) for k in ("active_p50_db", "active_p10_db", "weak_p10_db", "onset_p10_db",
            "weak_below_minus3_fraction", "weak_below_minus6_fraction")},
        "dry_safety_medians": {k: drymed(k) for k in ("dry_active_p50_db", "dry_active_p10_db", "dry_weak_p10_db", "dry_onset_p10_db")},
        "gates": gates, "rooms": all_metrics}


def regression_sanity():
    rng = np.random.default_rng(SEED + 17)
    x = (rng.normal(size=(3, 160)) + 1j*rng.normal(size=(3, 160))).astype(np.complex64)
    z, times = delayed_matrix(x)
    truth = (rng.normal(size=(3, K)) + 1j*rng.normal(size=(3, K))).astype(np.complex128)
    y = np.zeros_like(x)
    y[:, times] = np.einsum("fti,fi->ft", z, truth, optimize=True)
    fitted, fit_times, _ = ridge_coefficients(x, y)
    predicted = np.einsum("fti,fi->ft", z, fitted, optimize=True)
    error = float(np.linalg.norm(predicted - y[:, fit_times]) /
                  (np.linalg.norm(y[:, fit_times]) + 1e-12))
    return {"passed": error < .002, "relative_fit_error": error}


def run():
    torch.use_deterministic_algorithms(True)
    np.random.seed(SEED)
    sanity = regression_sanity()
    if not sanity["passed"]:
        raise RuntimeError(f"complex ridge orientation sanity failed: {sanity}")
    rooms = []
    loading_oracle, loading_wpe = [], []
    for room_id in range(ROOMS):
        h = make_room(room_id)
        dry_a = make_dry(room_id, 0)
        dry_b = make_dry(room_id, 1)
        wet_a, wet_b = convolve_taps(dry_a, h), convolve_taps(dry_b, h)
        xa, xb, sa, sb = stft(wet_a), stft(wet_b), stft(dry_a), stft(dry_b)
        oracle, _, load_o = ridge_coefficients(xa, xa-sa)
        blind, load_w = wpe_coefficients(xa)
        # Coefficients are frozen before either B waveform is processed.
        r = {"room": room_id, "dry_b": dry_b, "wet_b": wet_b,
             "oracle_coeff": oracle, "wpe_coeff": blind}
        r["oracle_wet"] = apply_coefficients(wet_b, oracle)
        r["oracle_dry"] = apply_coefficients(dry_b, oracle)
        r["wpe_wet"] = apply_coefficients(wet_b, blind)
        r["wpe_dry"] = apply_coefficients(dry_b, blind)
        rooms.append(r)
        loading_oracle.extend(load_o.tolist())
        loading_wpe.extend(load_w.tolist())
    oracle_result = score_arm(rooms, "oracle")
    wpe_result = score_arm(rooms, "wpe")
    zero = np.zeros(N, np.float32)
    zero_checks = 0
    for r in rooms:
        for name in ("oracle", "wpe"):
            # A fixed nonzero filter must map exact digital silence to exact silence.
            coeff = r[f"{name}_coeff"]
            zero_checks += int(np.array_equal(apply_coefficients(zero, coeff), zero))
    result = {"experiment": "K-delayed-LP-WPE-synthetic-calibration-holdout-v1",
        "overall": "PASS" if oracle_result["overall"] == "PASS" and wpe_result["overall"] == "PASS" else "FAIL",
        "seed": SEED, "rooms": ROOMS, "calibration_eval_sources_disjoint": True,
        "same_room_rir_shared": True, "eval_used_to_fit_or_update": False,
        "stft": {"sample_rate": SR, "n_fft": N_FFT, "hop": HOP,
                 "hop_ms": 1000*HOP/SR, "delay_frames": DELAY,
                 "delay_ms": 1000*HOP*DELAY/SR, "taps": K,
                 "tap_history_ms": 1000*HOP*K/SR},
        "oracle": oracle_result, "blind_wpe": wpe_result,
        "regularization": "1e-4 * trace(Z^H Z)/K per frequency",
        "wpe_iterations": WPE_ITERS, "wpe_variance_ema_alpha": .9,
        "loading_median": {"oracle": float(np.median(loading_oracle)), "wpe": float(np.median(loading_wpe))},
        "exact_zero_checks": zero_checks, "exact_zero_required": ROOMS*2,
        "finite_coefficients_outputs": bool(all(np.isfinite(r[f"{name}_coeff"]).all() and
            np.isfinite(r[f"{name}_wet"]).all() and np.isfinite(r[f"{name}_dry"]).all()
            for r in rooms for name in ("oracle", "wpe"))),
        "complex_ridge_sanity": sanity,
        "corpus_read": False, "dev_or_holdout_read": False, "test_wav_accessed": False,
        "source_sha256": {"audit": sha(Path(__file__))}}
    result["overall"] = "PASS" if result["overall"] == "PASS" and zero_checks == ROOMS*2 else "FAIL"
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    if result["overall"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
