"""Inference-only worker for an externally installed Resemble environment."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from voxrefine.backends.resemble_compat import install_numpy_fsolve_compat
from voxrefine.toneshape import apply_deesser, apply_gentle_compression, apply_shelves


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--nfe", type=int, choices=(16, 32, 64), default=64)
    parser.add_argument("--chunk-seconds", type=float, default=3.0)
    parser.add_argument("--tone", choices=("flat", "C", "soft-edges"), default="C")
    parser.add_argument("--treble-trim-db", type=float, default=-2.5)
    parser.add_argument("--deesser", choices=("off", "gentle"), default="off")
    parser.add_argument("--dynamics", choices=("off", "gentle"), default="gentle")
    args = parser.parse_args()

    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio.functional as AF
    from scipy.signal import butter, sosfiltfilt
    from scipy.signal import resample_poly
    import pyloudnorm as pyln

    upstream = args.upstream.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    if not (upstream / "resemble_enhance").is_dir():
        raise SystemExit(f"Invalid Resemble source checkout: {upstream}")
    if not (model_dir / "hparams.yaml").is_file():
        raise SystemExit(f"Local model directory is incomplete: {model_dir}")
    revision = subprocess.run(["git", "-C", str(upstream), "rev-parse", "HEAD"],
                              check=True, capture_output=True, text=True).stdout.strip()
    expected_revision = "8e978149bfe8abab3eb77d965d579a111afdb0ff"
    if revision != expected_revision:
        raise SystemExit(f"Unsupported Resemble code revision {revision}; expected {expected_revision}.")
    source_diff = subprocess.run(["git", "-C", str(upstream), "diff", "--binary"],
                                 check=True, capture_output=True).stdout
    expected_source_diff = "9f2fde7dc1aa7ab1b54948fe758c43ca410c57bfcfedbae16b9a0c3b23648e66"
    if hashlib.sha256(source_diff).hexdigest() != expected_source_diff:
        raise SystemExit("Resemble source modifications do not match the reviewed inference-only patch.")
    expected_hparams = "80c3f15bc5a5b2cacf2c698699a0f6599d62911c0d53e1d6dee895c0d7cbaeac"
    if sha256(model_dir / "hparams.yaml") != expected_hparams:
        raise SystemExit("Resemble hparams do not match the benchmarked pinned model.")
    sys.path.insert(0, str(upstream))
    try:
        from resemble_enhance.enhancer.inference import load_enhancer
        from resemble_enhance import inference as inference_module
        from resemble_enhance.inference import inference
    except ImportError as error:
        raise SystemExit(f"Resemble runtime dependencies are not installed in this Python: {error}") from error
    install_numpy_fsolve_compat()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    initial_device = device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable in this model runtime.")
    if not 2.0 <= args.chunk_seconds <= 30.0:
        raise SystemExit("Chunk duration must be between 2 and 30 seconds.")
    raw = np.fromfile(args.input, dtype="<f4")
    if len(raw) != args.sample_count or len(raw) == 0 or not np.isfinite(raw).all():
        raise SystemExit("Input PCM is truncated, empty, or contains non-finite samples.")
    weights = model_dir / "ds/G/default/mp_rank_00_model_states.pt"
    if not weights.is_file():
        raise SystemExit(f"Local Resemble checkpoint not found: {weights}")
    expected_weights = "f9d035f318de3e6d919bc70cf7ad7d32b4fe92ec5cbe0b30029a27f5db07d9d6"
    if sha256(weights) != expected_weights:
        raise SystemExit("Resemble checkpoint hash differs from the benchmarked pinned weights.")
    started_load = time.perf_counter()
    model = load_enhancer(model_dir, device)
    load_seconds = time.perf_counter() - started_load
    fallback_model_load_seconds = 0.0
    fallback_reason = None
    chunk_attempts = [args.chunk_seconds]
    next_chunk = args.chunk_seconds
    while next_chunk > 3.0:
        next_chunk = max(3.0, round(next_chunk / 2.0, 2))
        if next_chunk not in chunk_attempts:
            chunk_attempts.append(next_chunk)
    inference_seconds = None
    enhanced = None
    rate = 48000
    used_chunk = None
    merge_offsets_samples: list[int] = []
    original_compute_offset = inference_module.compute_offset

    def capture_offset(*values, **kwargs):
        offset = original_compute_offset(*values, **kwargs)
        merge_offsets_samples.append(int(offset))
        return offset

    inference_module.compute_offset = capture_offset
    for chunk in chunk_attempts:
        try:
            merge_offsets_samples.clear()
            model.configurate_(nfe=args.nfe, solver="midpoint", lambd=1.0, tau=0.5)
            torch.manual_seed(1701)
            if device == "cuda":
                torch.cuda.manual_seed_all(1701)
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            started = time.perf_counter()
            enhanced, rate = inference(model=model, dwav=torch.from_numpy(raw.copy()), sr=48000,
                                       device=device, chunk_seconds=chunk, overlap_seconds=1.0)
            if device == "cuda":
                torch.cuda.synchronize()
            inference_seconds = time.perf_counter() - started
            used_chunk = chunk
            break
        except torch.cuda.OutOfMemoryError as error:
            if device != "cuda":
                raise
            torch.cuda.empty_cache()
            if chunk != chunk_attempts[-1]:
                fallback_reason = f"CUDA OOM at {chunk:g}s window; retrying smaller chunks."
                continue
            if args.device != "auto":
                raise
            fallback_reason = f"CUDA OOM down to {chunk:g}s window; retrying on CPU."
            del model
            gc.collect()
            torch.cuda.empty_cache()
            started_load = time.perf_counter()
            model = load_enhancer(model_dir, "cpu")
            fallback_model_load_seconds = time.perf_counter() - started_load
            device = "cpu"
            model.configurate_(nfe=args.nfe, solver="midpoint", lambd=1.0, tau=0.5)
            torch.manual_seed(1701)
            started = time.perf_counter()
            enhanced, rate = inference(model=model, dwav=torch.from_numpy(raw.copy()), sr=48000,
                                       device=device, chunk_seconds=chunk, overlap_seconds=1.0)
            inference_seconds = time.perf_counter() - started
            used_chunk = chunk
            break
    if enhanced is None or inference_seconds is None:
        raise SystemExit("Resemble inference produced no output.")
    inference_module.compute_offset = original_compute_offset
    enhanced = enhanced.detach().cpu().numpy().astype(np.float32)
    if rate != 48000:
        enhanced = AF.resample(torch.from_numpy(enhanced), rate, 48000).numpy().astype(np.float32)
    expected = len(raw)
    if len(enhanced) < expected:
        enhanced = np.pad(enhanced, (0, expected-len(enhanced)))
    else:
        enhanced = enhanced[:expected]
    if not np.isfinite(enhanced).all():
        raise SystemExit("Resemble output contains non-finite samples.")

    if args.tone in {"C", "soft-edges"}:
        cutoff = 4000.0
        low = sosfiltfilt(butter(2, cutoff, btype="lowpass", fs=48000, output="sos"), enhanced)
        enhanced = (low + 10 ** (args.treble_trim_db / 20) * (enhanced-low)).astype(np.float32)
    if args.tone == "soft-edges":
        enhanced = apply_shelves(enhanced, 48000, bass_db=-3.0, bass_corner_hz=100.0,
                                 treble_db=-2.5, treble_corner_hz=3500.0)
    deesser_reduction_db = 0.0
    if args.deesser == "gentle":
        enhanced, deesser_reduction_db = apply_deesser(enhanced, 48000)
    compressor_reduction_db = 0.0
    if args.dynamics == "gentle":
        enhanced, compressor_reduction_db = apply_gentle_compression(enhanced, 48000)
    meter = pyln.Meter(48000)
    input_lufs = float(meter.integrated_loudness(raw.astype(np.float64)))
    headroom_gain_db = 0.0
    peak = float(np.max(np.abs(enhanced)))
    if peak > 0.99:
        headroom_gain_db = float(20*np.log10(0.99/peak))
        enhanced *= 10**(headroom_gain_db/20)
    peak = float(np.max(np.abs(enhanced)))
    true_peak = float(np.max(np.abs(resample_poly(enhanced.astype(np.float64), 4, 1))))
    output_lufs = float(meter.integrated_loudness(enhanced.astype(np.float64)))
    enhanced.astype("<f4", copy=False).tofile(args.output)
    report = {
        "model_code_revision": "8e978149bfe8abab3eb77d965d579a111afdb0ff",
        "model_weights_revision": "e10d34b312433b41a8eeeec43ebb6fa3219dab3f",
        "model_weights_sha256": sha256(weights),
        "device": device,
        "initial_device": initial_device,
        "fallback_reason": fallback_reason,
        "nfe": args.nfe,
        "requested_chunk_seconds": args.chunk_seconds,
        "chunk_seconds": used_chunk,
        "chunk_attempts_seconds": chunk_attempts,
        "overlap_seconds": 1.0,
        "solver": "midpoint",
        "lambda": 1.0,
        "tau": 0.5,
        "tone": args.tone,
        "dynamics": args.dynamics,
        "dynamics_compressor": ({
            "threshold_dbfs": -16.0,
            "ratio": 1.5,
            "knee_db": 6.0,
            "attack_ms": 10.0,
            "release_ms": 120.0,
            "detector_lookahead_ms": 10.0,
            "max_gain_reduction_db": compressor_reduction_db,
        } if args.dynamics == "gentle" else None),
        "treble_trim_db": args.treble_trim_db if args.tone in {"C", "soft-edges"} else 0.0,
        "tone_cutoff_hz": 5500 if args.tone in {"C", "soft-edges"} else None,
        "deesser": args.deesser,
        "deesser_config": ({
            "band_hz": [4000, 10000],
            "threshold_dbfs": -46.0,
            "max_reduction_db": 3.0,
            "ratio": 3.0,
            "window_ms": 20.0,
            "attack_ms": 6.0,
            "release_ms": 90.0,
            "max_observed_reduction_db": deesser_reduction_db,
        } if args.deesser == "gentle" else None),
        "soft_edge_bass_trim_db": -3.0 if args.tone == "soft-edges" else 0.0,
        "soft_edge_bass_cutoff_hz": 100 if args.tone == "soft-edges" else None,
        "soft_edge_treble_trim_db": -2.5 if args.tone == "soft-edges" else 0.0,
        "soft_edge_treble_cutoff_hz": 3500 if args.tone == "soft-edges" else None,
        "headroom_constant_gain_db": headroom_gain_db,
        "model_load_seconds": load_seconds,
        "fallback_model_load_seconds": fallback_model_load_seconds,
        "model_inference_seconds": inference_seconds,
        "model_rtf": inference_seconds / (len(raw)/48000),
        "peak_allocated_vram_mib": torch.cuda.max_memory_allocated()/1024**2 if device == "cuda" else None,
        "peak_reserved_vram_mib": torch.cuda.max_memory_reserved()/1024**2 if device == "cuda" else None,
        "output_sample_peak_dbfs": float(20*np.log10(np.max(np.abs(enhanced))+1e-12)),
        "output_samples": len(enhanced),
        "model_native_sample_rate": int(model.hp.wav_rate),
        "output_sample_rate": 48000,
        "input_integrated_lufs": input_lufs,
        "output_integrated_lufs": output_lufs,
        "sample_peak_dbfs": float(20*np.log10(peak+1e-12)),
        "true_peak_4x_dbfs": float(20*np.log10(true_peak+1e-12)),
        "merge_offset_samples_at_model_rate": merge_offsets_samples,
        "merge_offset_ms": [1000*offset/model.hp.wav_rate for offset in merge_offsets_samples],
    }
    print(json.dumps(report, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
