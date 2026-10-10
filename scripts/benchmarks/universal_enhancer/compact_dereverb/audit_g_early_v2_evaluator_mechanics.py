#!/usr/bin/env python3
"""Deterministically check G2 descriptive actual/gain/phase output shapes."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
EXPERIMENT = ROOT / ".tools/compact-dereverb/g-early-v2-2026-10-10"
sys.path.insert(0, str(HERE))
from model_g_early_v2 import CompactAttenuationOnlyDereverbG2  # noqa: E402
from evaluate_g_early_v2 import infer_with_mechanism, sha  # noqa: E402


def audit() -> dict:
    torch.manual_seed(20261010)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    model = CompactAttenuationOnlyDereverbG2(base_channels=16).eval()
    sample_rate, samples = 16_000, 32_000
    t = np.arange(samples, dtype=np.float32) / sample_rate
    audio = (0.2 * np.sin(2 * np.pi * 180 * t) +
             0.08 * np.sin(2 * np.pi * 420 * t)).astype(np.float32)
    output, mechanism, elapsed_s = infer_with_mechanism(model, audio,
        {"clip_relative_start_sample": 8_000}, torch.device("cpu"))
    diagnostic_keys = ("w1_actual_reduction_db", "w1_gain_only_reduction_db",
        "w1_phase_only_reduction_db")
    if (output.shape != audio.shape or not np.isfinite(output).all() or
            any(mechanism[key] is not None and not np.isfinite(mechanism[key])
                for key in diagnostic_keys)):
        raise RuntimeError("G2 evaluator synthetic mechanism shape/finite check failed")
    evaluator_path = HERE / "evaluate_g_early_v2.py"
    report = {"name": "G-early-v2-synthetic-evaluator-mechanism-check",
        "passed": True, "sample_rate": sample_rate, "input_samples": samples,
        "input_sha256": hashlib.sha256(audio.tobytes()).hexdigest(),
        "output_shape": list(output.shape),
        "output_sha256": hashlib.sha256(output.tobytes()).hexdigest(),
        "mechanism_diagnostics_descriptive_only": {
            key: mechanism[key] for key in diagnostic_keys},
        "elapsed_s": elapsed_s,
        "evaluator_source_sha256": sha(evaluator_path),
        "audit_source_sha256": sha(Path(__file__)),
        "device": "cpu", "torch_version": torch.__version__,
        "test_wav_accessed": False, "dev_or_holdout_accessed": False}
    output_path = EXPERIMENT / "synthetic-evaluator-mechanism-check.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    report["receipt_sha256"] = sha(output_path)
    print(json.dumps(report, indent=2, allow_nan=False))
    return report


if __name__ == "__main__":
    audit()
