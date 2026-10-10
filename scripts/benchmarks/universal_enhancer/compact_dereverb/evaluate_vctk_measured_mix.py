#!/usr/bin/env python3
"""Score a VCTK-trained dereverberator on frozen Rochester measured-RIR pairs."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
from model import CompactDereverb16k, parameter_count  # noqa: E402

DEFAULT_PROTOCOL = ROOT / "results/noise-rir-truth-01/dpdfnet8-vs2-independent-01/protocol-manifest.json"
DEFAULT_CHECKPOINT = ROOT / ".tools/compact-dereverb/vctk-measured-room-pilot-2026-10-11/checkpoints/best-dev.pt"
DEFAULT_OUTPUT = ROOT / ".tools/compact-dereverb/vctk-measured-room-pilot-2026-10-11/evaluation"
RATE = 16_000
HELDOUT_SPEAKERS = {"p239", "p255", "p287", "p300"}


def read16(path: Path, expected_sha: str | None = None) -> np.ndarray:
    import hashlib

    raw = path.read_bytes()
    if expected_sha and hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError(f"source hash mismatch: {path}")
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr == 48_000:
        x = resample_poly(x[:, 0], 1, 3).astype(np.float32)[:, None]
    elif sr == RATE:
        pass
    else:
        raise ValueError(f"unsupported sample rate {sr} for {path}")
    if x.shape[1] != 1 or not np.isfinite(x).all():
        raise ValueError(f"expected finite mono audio: {path}")
    return x[:, 0]


def projection_coefficients(basis: np.ndarray, signal: np.ndarray) -> tuple[float, float]:
    n = min(basis.shape[0], signal.size)
    design = basis[:n]
    # Normalize each component before least squares so coefficients are stable.
    norms = np.sqrt(np.mean(design * design, axis=0)).clip(1e-12)
    coef, *_ = np.linalg.lstsq(design / norms, signal[:n], rcond=1e-7)
    return tuple((coef / norms).tolist())


def db_ratio(after: float, before: float) -> float:
    return 20 * math.log10(max(abs(after), 1e-9) / max(abs(before), 1e-9))


def main(args: argparse.Namespace) -> None:
    args.output = args.output.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.protocol = args.protocol.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(args.protocol.read_text())
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = CompactDereverb16k(base_channels=int(ckpt["base_channels"]))
    model.load_state_dict(ckpt["model"])
    device = torch.device(args.device)
    model.to(device).eval()
    count = parameter_count(model)

    cases = [c for c in protocol["cases"]
             if c["speaker_id"] in HELDOUT_SPEAKERS and c["condition"] == "rir_only"
             and c["variant"] == "normal"]
    if len(cases) != 16:
        raise RuntimeError(f"expected 16 frozen measured-RIR cases; found {len(cases)}")
    rows = []
    output_renders = args.output / "renders"
    output_renders.mkdir(exist_ok=True)
    infer_seconds = []
    peak_vram = 0
    with torch.inference_mode():
        for case in cases:
            components = []
            for key in ("direct", "late", "input", "dry"):
                path = ROOT / case["paths"][key]
                components.append(read16(path, case["file_sha256"][key]))
            direct, late, mixture, dry = components
            tin = torch.from_numpy(mixture).unsqueeze(0).to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            enhanced = model(tin).squeeze(0).float().cpu().numpy()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                peak_vram = max(peak_vram, int(torch.cuda.max_memory_allocated(device)))
            infer_seconds.append(time.perf_counter() - start)
            n = min(direct.size, late.size, mixture.size, enhanced.size, dry.size)
            basis = np.column_stack((direct[:n], late[:n])).astype(np.float64)
            beta_in = projection_coefficients(basis, mixture[:n])
            beta_out = projection_coefficients(basis, enhanced[:n])
            direct_shift_db = db_ratio(beta_out[0], beta_in[0])
            late_shift_db = db_ratio(beta_out[1], beta_in[1])
            # Scale-invariant waveform error against the dry close-mic proxy.
            target, estimate = dry[:n].astype(np.float64), enhanced[:n].astype(np.float64)
            alpha = np.dot(estimate, target) / max(np.dot(target, target), 1e-12)
            si_sdr = 10 * math.log10(max(np.dot(alpha * target, alpha * target), 1e-12) /
                                     max(np.dot(estimate - alpha * target, estimate - alpha * target), 1e-12))
            input_estimate = mixture[:n].astype(np.float64)
            input_alpha = np.dot(input_estimate, target) / max(np.dot(target, target), 1e-12)
            input_si_sdr = 10 * math.log10(max(np.dot(input_alpha * target, input_alpha * target), 1e-12) /
                                           max(np.dot(input_estimate - input_alpha * target,
                                                      input_estimate - input_alpha * target), 1e-12))
            direct_target = direct[:n].astype(np.float64)
            output_direct_alpha = np.dot(estimate, direct_target) / max(np.dot(direct_target, direct_target), 1e-12)
            output_direct_si_sdr = 10 * math.log10(
                max(np.dot(output_direct_alpha * direct_target, output_direct_alpha * direct_target), 1e-12) /
                max(np.dot(estimate - output_direct_alpha * direct_target,
                           estimate - output_direct_alpha * direct_target), 1e-12))
            input_direct_alpha = np.dot(input_estimate, direct_target) / max(np.dot(direct_target, direct_target), 1e-12)
            input_direct_si_sdr = 10 * math.log10(
                max(np.dot(input_direct_alpha * direct_target, input_direct_alpha * direct_target), 1e-12) /
                max(np.dot(input_estimate - input_direct_alpha * direct_target,
                           input_estimate - input_direct_alpha * direct_target), 1e-12))
            output_to_input_rms = 20 * math.log10(
                max(float(np.sqrt(np.mean(estimate ** 2))), 1e-12) /
                max(float(np.sqrt(np.mean(input_estimate ** 2))), 1e-12))
            render = output_renders / f"{case['case_id']}-vctk-measured.wav"
            sf.write(render, enhanced, RATE, subtype="PCM_16")
            rows.append({
                "case_id": case["case_id"], "speaker": case["speaker_id"], "room": case["rir"],
                "direct_component_change_db": direct_shift_db,
                "late_component_change_db": late_shift_db,
                "input_dry_proxy_si_sdr_db": input_si_sdr,
                "dry_proxy_si_sdr_db": si_sdr,
                "dry_proxy_si_sdr_improvement_db": si_sdr - input_si_sdr,
                "input_early_target_si_sdr_db": input_direct_si_sdr,
                "output_early_target_si_sdr_db": output_direct_si_sdr,
                "early_target_si_sdr_improvement_db": output_direct_si_sdr - input_direct_si_sdr,
                "output_to_input_rms_change_db": output_to_input_rms,
                "input_peak": float(np.max(np.abs(mixture))),
                "output_peak": float(np.max(np.abs(enhanced))),
                "output_clipped": bool(np.max(np.abs(enhanced)) >= 0.999),
                "inference_seconds": infer_seconds[-1], "render": str(render.relative_to(ROOT)),
            })

            # Run matching dry control once per speaker+utterance to expose speech damage.
            dry_id = f"{case['speaker_id']}-{case['utterance_id']}"
            dry_control_path = ROOT / "results/noise-rir-truth-01/dpdfnet8-vs2-independent-01/inputs" / f"{dry_id}-dry-control.wav"
            if dry_control_path.is_file() and not any(r.get("dry_control_id") == dry_id for r in rows):
                dry_control = read16(dry_control_path)
                out_dry = model(torch.from_numpy(dry_control).unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
                m = min(dry_control.size, out_dry.size)
                ref, enh = dry_control[:m].astype(np.float64), out_dry[:m].astype(np.float64)
                projection = np.dot(enh, ref) / max(np.dot(ref, ref), 1e-12)
                control_si_sdr = 10 * math.log10(max(np.dot(projection * ref, projection * ref), 1e-12) /
                                                 max(np.dot(enh - projection * ref, enh - projection * ref), 1e-12))
                gain = 20 * math.log10(max(np.sqrt(np.mean(enh**2)), 1e-12) /
                                       max(np.sqrt(np.mean(ref**2)), 1e-12))
                rows.append({"dry_control_id": dry_id, "speaker": case["speaker_id"],
                             "dry_control_rms_change_db": gain,
                             "dry_control_si_sdr_db": control_si_sdr,
                             "dry_control_max_abs_change": float(np.max(np.abs(enh - ref)))})

    measured = [r for r in rows if "late_component_change_db" in r]
    controls = [r for r in rows if "dry_control_rms_change_db" in r]
    def quantile(items, key, q):
        return float(np.quantile([x[key] for x in items], q))
    report = {
        "benchmark": "frozen Rochester measured-RIR VCTK dereverb pilot",
        "checkpoint": str(args.checkpoint), "checkpoint_step": ckpt.get("step"),
        "parameter_count": count, "device": str(device), "sample_rate_hz": RATE,
        "cases": len(measured), "dry_controls": len(controls),
        "speakers": sorted(HELDOUT_SPEAKERS), "rooms": sorted({c["rir"] for c in cases}),
        "late_component_change_db": {f"p{int(q*100)}": quantile(measured, "late_component_change_db", q)
                                     for q in (.1, .5, .9)},
        "direct_component_change_db": {f"p{int(q*100)}": quantile(measured, "direct_component_change_db", q)
                                       for q in (.1, .5, .9)},
        "dry_proxy_si_sdr_db": {f"p{int(q*100)}": quantile(measured, "dry_proxy_si_sdr_db", q)
                                for q in (.1, .5, .9)},
        "input_dry_proxy_si_sdr_db": {f"p{int(q*100)}": quantile(measured, "input_dry_proxy_si_sdr_db", q)
                                      for q in (.1, .5, .9)},
        "dry_proxy_si_sdr_improvement_db": {f"p{int(q*100)}": quantile(measured, "dry_proxy_si_sdr_improvement_db", q)
                                            for q in (.1, .5, .9)},
        "early_target_si_sdr_improvement_db": {f"p{int(q*100)}": quantile(measured, "early_target_si_sdr_improvement_db", q)
                                               for q in (.1, .5, .9)},
        "output_to_input_rms_change_db": {f"p{int(q*100)}": quantile(measured, "output_to_input_rms_change_db", q)
                                          for q in (.1, .5, .9)},
        "dry_control_rms_change_db": {f"p{int(q*100)}": quantile(controls, "dry_control_rms_change_db", q)
                                      for q in (.1, .5, .9)} if controls else None,
        "dry_control_si_sdr_db": {f"p{int(q*100)}": quantile(controls, "dry_control_si_sdr_db", q)
                                  for q in (.1, .5, .9)} if controls else None,
        "per_room": {room: {
            "median_late_component_change_db": float(np.median([r["late_component_change_db"] for r in measured if r["room"] == room])),
            "median_dry_proxy_si_sdr_improvement_db": float(np.median([r["dry_proxy_si_sdr_improvement_db"] for r in measured if r["room"] == room])),
            "median_early_target_si_sdr_improvement_db": float(np.median([r["early_target_si_sdr_improvement_db"] for r in measured if r["room"] == room])),
        } for room in sorted({r["room"] for r in measured})},
        "clipped_renders": sum(r["output_clipped"] for r in measured),
        "median_inference_seconds_per_3s_clip": float(np.median([r["inference_seconds"] for r in measured])),
        "realtime_factor_16khz_mono": 3.0 / float(np.median([r["inference_seconds"] for r in measured])),
        "peak_allocated_vram_mib": peak_vram / 2**20 if device.type == "cuda" else None,
        "limitations": ["Rochester rooms and selected speakers are held out from this fit, but have been used in prior project experiments.",
                        "VCTK mic1 is a hemi-anechoic close-mic proxy, not an anechoic target.",
                        "Projection coefficients are diagnostic, not a perceptual quality score."]
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.output / "cases.csv").open("w", newline="") as f:
        fields = sorted({k for row in rows for k in row})
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    main(parser.parse_args())
