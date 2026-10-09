#!/usr/bin/env python3
"""Compare DPDFNet2 against published STSubNet outputs on REVERB examples.

Exploratory only: the upstream repository supplies processed audio samples,
not its model weights or inference code. No speech target is available here,
so the report is descriptive, not a quality ranking.
"""
from __future__ import annotations

import csv
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / ".tools/stsubnet-screen/fullband/reverb_challenge"
MODEL_DIR = ROOT / ".tools/deepvqe-screen"
OUTPUT_BASE = ROOT / "results"
MODEL_PATHS = {
    "dpdfnet2": MODEL_DIR / "dpdfnet2_48khz_hr.onnx",
    "dpdfnet8": MODEL_DIR / "dpdfnet8_48khz_hr.onnx",
}
INPUT_SR = 16_000
SR = 48_000
ALIGN = round(.040 * SR)  # measured DPDFNet2 streaming alignment in project runner


def read(path: Path) -> tuple[np.ndarray, int]:
    x, sr = sf.read(path, dtype="float32")
    if x.ndim != 1:
        raise ValueError(f"Expected mono audio: {path}")
    return x, sr


def metrics(x: np.ndarray, sr: int) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    frames = int(.020 * sr)
    n = len(x) // frames
    x = x[:n * frames].reshape(n, frames)
    rms = np.sqrt(np.mean(x * x, axis=1) + 1e-18)
    db = 20 * np.log10(rms + 1e-12)
    # Descriptive only: low-energy frames may contain pauses, quiet speech, or tails.
    threshold = max(float(np.max(db)) - 35.0, -65.0)
    low = rms[db < threshold]
    spectrum = np.abs(np.fft.rfft(x, axis=1)) ** 2
    freq = np.fft.rfftfreq(frames, 1 / sr)
    bands = {}
    for name, lo, hi in (("80_300", 80, 300), ("300_2k", 300, 2000),
                         ("2k_4k", 2000, min(4000, sr / 2)),
                         ("4k_8k", 4000, sr / 2)):
        mask = (freq >= lo) & (freq < hi)
        if np.any(mask):
            bands[name] = float(10 * np.log10(np.mean(spectrum[:, mask]) + 1e-18))
    return {
        "rms_dbfs": float(20 * np.log10(np.sqrt(np.mean(x*x)) + 1e-12)),
        "peak_dbfs": float(20 * np.log10(np.max(np.abs(x)) + 1e-12)),
        "crest_db": float(20 * np.log10(np.max(np.abs(x)) / (np.sqrt(np.mean(x*x)) + 1e-12))),
        "frame_envelope_std_db": float(np.std(db)),
        "low_energy_frame_fraction": float(len(low) / max(len(rms), 1)),
        "low_energy_frame_median_dbfs": float(np.median(20*np.log10(low+1e-12))) if len(low) else float("nan"),
        **{f"band_power_db_{k}": v for k, v in bands.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(MODEL_PATHS), default="dpdfnet2")
    args = parser.parse_args()
    output = OUTPUT_BASE / f"stsubnet-real-reverb-{args.model}-compare-2026-10-09"
    if output.exists():
        raise SystemExit(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    render_dir = output / args.model
    render_dir.mkdir()
    sys.path.insert(0, str(MODEL_DIR))
    from dpdfnet.stream import StreamEnhancer
    from dpdfnet_benchmark_adapter import DpdfNetBenchmarkAdapter

    model = StreamEnhancer(f"{args.model}_48khz_hr", onnx_path=MODEL_PATHS[args.model])
    cases = sorted(SOURCE.glob("*/**/*_stsubnet.wav"))
    if len(cases) != 10:
        raise RuntimeError(f"Expected 10 STSubNet reverb examples, found {len(cases)}")
    rows = []
    for st_path in cases:
        key = st_path.name.removesuffix("_stsubnet.wav")
        input_path = st_path.with_name(key + ".wav")
        x, sr = read(input_path)
        y_st, sr_st = read(st_path)
        if sr != INPUT_SR or sr_st != SR:
            raise ValueError(f"Unexpected rates: {input_path} {sr}; {st_path} {sr_st}")
        x48 = resample_poly(x, 3, 1).astype(np.float32)
        # DPDFNet's own causal streaming path; alignment is cropped to the
        # measured 40 ms pipeline offset, and no per-file level matching occurs.
        adapter = DpdfNetBenchmarkAdapter(model, sample_rate=SR, alignment_samples=ALIGN)
        start = time.perf_counter()
        for offset in range(0, len(x48), SR // 100):
            adapter.process(x48[offset:offset + SR // 100].astype(np.float32, copy=False))
        y_dp = adapter.finalize()
        elapsed = time.perf_counter() - start
        if len(y_dp) != len(x48):
            raise RuntimeError(f"duration mismatch {key}: {len(y_dp)} != {len(x48)}")
        dp_path = render_dir / f"{key}-{args.model}.wav"
        sf.write(dp_path, y_dp, SR, subtype="PCM_16")
        # Compare all signals inside the original 16 kHz bandwidth; this avoids
        # treating either 48 kHz header as evidence of restored high frequencies.
        x16 = resample_poly(x48.astype(np.float64), 1, 3).astype(np.float32)
        y_st16 = resample_poly(y_st.astype(np.float64), 1, 3).astype(np.float32)
        y_dp16 = resample_poly(y_dp.astype(np.float64), 1, 3).astype(np.float32)
        n = min(len(x), len(x16), len(y_st16), len(y_dp16))
        row = {"clip": key, "model": args.model,
               "input_seconds": len(x)/INPUT_SR, "model_elapsed_seconds": elapsed,
               "model_rtf": elapsed/(len(x)/INPUT_SR),
               "duration_exact": len(y_dp) == len(x48),
               "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
               "stsubnet_output_sha256": hashlib.sha256(st_path.read_bytes()).hexdigest(),
               "model_output": str(dp_path.relative_to(ROOT))}
        for name, signal in (("input", x[:n]), ("stsubnet", y_st16[:n]), (args.model, y_dp16[:n])):
            row.update({f"{name}_{k}": v for k, v in metrics(signal, INPUT_SR).items()})
        rows.append(row)
        print(f"DONE {key}: {elapsed:.2f}s ({elapsed/(len(x)/INPUT_SR):.2f} RTF)", flush=True)

    with (output / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader(); writer.writerows(rows)
    report = {
        "purpose": "Descriptive model output comparison on 10 published REVERB example pairs.",
        "source_repository": "https://github.com/ffxiong/stsubnet",
        "source_note": "STSubNet repository provided WAV examples only; no model checkpoint or inference code was used.",
        "input_corpus": "REVERB challenge AMI real-room examples; 16 kHz mono inputs.",
        "stsubnet_outputs": "48 kHz mono outputs, downsampled to 16 kHz for common-band descriptive metrics.",
        "model": args.model,
        "model_path": str(MODEL_PATHS[args.model].relative_to(ROOT)),
        "dpdfnet_sample_rate_hz": SR,
        "alignment_ms": 40,
        "quality_claim": "None: no clean reference, no Adobe output, and the clips are a small example set. Envelope low-energy frames can include quiet speech and are not a reverberation-only measure.",
        "rows": len(rows),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"WROTE {output.relative_to(ROOT)} ({len(rows)} clips)")


if __name__ == "__main__":
    main()
