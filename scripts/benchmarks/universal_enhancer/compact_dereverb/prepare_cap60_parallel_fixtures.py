#!/usr/bin/env python3
"""Materialize four deterministic Cap60 concurrency fixtures from the frozen schedule."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts/benchmarks/universal_enhancer"))
from data import make_procedural_rir  # noqa: E402
from prepare_cap60_conditioned_pairs import (  # noqa: E402
    LIBRISPEECH, make_fan_mix, sha256,
)
from prepare_f2_tailbank import load_schedule, resolve_source  # noqa: E402
from train_measured_mix import measured_pair  # noqa: E402

F2_PROGRESS = ROOT / ".tools/compact-dereverb/e2-f2-tailbank-2026-10-10/data/progress.json"
DEFAULT_OUT = ROOT / ".tools/compact-dereverb/cap60-parallel-determinism-2026-10-10/inputs"


def run(output: Path) -> dict:
    progress = json.loads(F2_PROGRESS.read_text())
    if progress.get("test_wav_accessed") is not False:
        raise RuntimeError("fixture source manifest violates sealed test.wav policy")
    rows = load_schedule()
    clean_row = next(row for row in rows if row["kind"] == "identity")
    selected_tail_row = next(row for row in progress["completed_rows"]
                             if row["kind"] == "rir_only")
    tail_schedule = rows[int(selected_tail_row["index"])]
    clean_short, _, _ = resolve_source(clean_row, LIBRISPEECH / "train-clean-100")
    clean_long, _, _ = resolve_source(tail_schedule, LIBRISPEECH / "train-clean-100")
    recipe = selected_tail_row["rir_recipe"]
    rir, _, _ = make_procedural_rir(16_000, int(recipe["seed"]),
        float(recipe["t60_s"]), float(recipe["direct_to_reverb_db"]))
    pair = measured_pair(clean_long, rir)
    wet = pair["reverberant"]
    noisy = make_fan_mix(clean_long, wet, 0xC0FFEE, 10.0)
    fixtures = {
        "clean-short-quiet.wav": clean_short[:80_000] * np.float32(.18),
        "clean-long-normal.wav": clean_long,
        "reverb-long.wav": wet,
        "reverb-noise-long.wav": noisy,
    }
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty fixture directory {output}")
    details = {}
    for name, audio in fixtures.items():
        path = output / name
        audio = np.asarray(audio, np.float32)
        sf.write(path, audio, 16_000, subtype="PCM_16")
        details[name] = {"sha256": sha256(path), "samples": int(len(audio)),
            "duration_s": len(audio) / 16_000,
            "peak": float(np.max(np.abs(audio))), "rms": float(np.sqrt(np.mean(audio ** 2)))}
    manifest = {"name": "Cap60-concurrency-fixtures-v1",
        "schedule_sha256": sha256(ROOT / ".tools/compact-dereverb/cap60-conditioned-2026-10-10/pairs.jsonl"),
        "cap60_sha256": sha256(ROOT / ".tools/deepfilternet/deep-filter"),
        "fixtures": details, "test_wav_accessed": False}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
