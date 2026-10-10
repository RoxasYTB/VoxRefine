#!/usr/bin/env python3
"""Build licensed synthetic input/DeepFilterNet teacher pairs (never Adobe)."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly

from train_fullband_student import DATA, RATE, load_clips, make_noise_bank

ROOT = Path(__file__).resolve().parents[3]
TEACHER = ROOT / ".tools/deepfilternet/deep-filter"
OUT = ROOT / "corpus/samples/open-teacher-student-01"
LICENSES = {
    "speech": "CSTR VCTK 0.92, CC BY 4.0; cite Yamagishi, Veaux, MacDonald (2019), DOI 10.7488/ds/2645",
    "teacher_code_and_weights": "DeepFilterNet upstream dual MIT OR Apache-2.0; https://github.com/Rikorose/DeepFilterNet",
    "noise": "Local corpus/raw/cc0-*.mp3 samples; CC0 source metadata must be checked in corpus/raw/heldout-noise-sources.json",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def make_rir(rng: random.Random) -> tuple[np.ndarray, float]:
    n = rng.randint(1200, 18_000)
    rir = np.zeros(n, np.float32)
    rir[0] = 1.0
    for _ in range(rng.randint(2, 8)):
        rir[rng.randint(180, min(n - 1, 14_000))] += rng.uniform(-0.35, 0.35)
    t = np.arange(n - 1, dtype=np.float32) / RATE
    rt60 = rng.uniform(0.2, 0.65)
    tail_rng = np.random.default_rng(rng.randrange(0, 2**32))
    rir[1:] += tail_rng.normal(0, 1, n - 1).astype(np.float32) * np.power(10, -3 * t / rt60) * rng.uniform(0.02, 0.09)
    return rir, rt60


def corrupt(clean: np.ndarray, rng: random.Random, noises: list[np.ndarray]) -> tuple[np.ndarray, dict]:
    x = clean.astype(np.float32, copy=True)
    condition = rng.choice(("noise", "room_noise", "narrowband_noise", "room"))
    meta: dict = {"condition": condition}
    if condition in ("room_noise", "room"):
        rir, rt60 = make_rir(rng)
        x = fftconvolve(x, rir, mode="full")[:x.size].astype(np.float32)
        meta["synthetic_rt60_seconds"] = round(rt60, 3)
    if condition in ("noise", "room_noise", "narrowband_noise"):
        noise = rng.choice(noises)
        if noise.size < x.size:
            noise = np.tile(noise, math.ceil(x.size / noise.size))
        start = rng.randint(0, noise.size - x.size)
        n = noise[start:start + x.size]
        snr = rng.uniform(2, 24)
        n *= (np.sqrt(np.mean(x * x) + 1e-10) /
              (np.sqrt(np.mean(n * n) + 1e-10) * 10 ** (snr / 20)))
        x += n
        meta["snr_db"] = round(snr, 3)
    if condition == "narrowband_noise":
        x = resample_poly(resample_poly(x, 1, 3), 3, 1).astype(np.float32)
        if x.size < clean.size:
            x = np.pad(x, (0, clean.size - x.size))
        x = x[:clean.size]
        meta["bandwidth_input"] = "simulated_16kHz_then_48kHz"
    return x, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-per-speaker", type=int, default=8)
    parser.add_argument("--validation-per-speaker", type=int, default=4)
    parser.add_argument("--variants", type=int, default=2)
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not TEACHER.is_file():
        raise SystemExit(f"Local DeepFilterNet teacher missing: {TEACHER}")
    if OUT.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite {OUT}; use --force")
    rng = random.Random(args.seed)
    py_noises = [n.numpy() if hasattr(n, "numpy") else np.asarray(n, np.float32) for n in make_noise_bank()]
    if not py_noises:
        raise SystemExit("No CC0 noise source found under corpus/raw/cc0-*.mp3")
    train, valid = load_clips("train"), load_clips("validation")
    n_samples = round(args.seconds * RATE)
    rows: list[dict] = []
    temp_inputs: list[Path] = []
    if OUT.exists():
        shutil.rmtree(OUT)
    for split, clips, per_speaker in (("train", train, args.train_per_speaker),
                                       ("validation", valid, args.validation_per_speaker)):
        by_speaker: dict[str, list[tuple[Path, np.ndarray]]] = {}
        for path, audio in clips:
            sid = path.name.split("_")[0]
            by_speaker.setdefault(sid, []).append((path, audio.numpy()))
        for sid, speaker_clips in sorted(by_speaker.items()):
            chosen = rng.sample(speaker_clips, min(per_speaker, len(speaker_clips)))
            for source_path, audio in chosen:
                if audio.size < n_samples:
                    continue
                start = rng.randint(0, audio.size - n_samples)
                clean = audio[start:start + n_samples]
                clean = clean / max(float(np.sqrt(np.mean(clean * clean))), 1e-4)
                for variant in range(args.variants):
                    corrupted, metadata = corrupt(clean, rng, py_noises)
                    rel = f"{split}/{sid}_{source_path.stem}_{variant:02d}"
                    clean_path = OUT / "clean" / f"{rel.split('/',1)[1]}.wav"
                    input_path = OUT / "input" / f"{rel.split('/',1)[1]}.wav"
                    teacher_path = OUT / "teacher" / f"{rel.split('/',1)[1]}.wav"
                    clean_path.parent.mkdir(parents=True, exist_ok=True)
                    input_path.parent.mkdir(parents=True, exist_ok=True)
                    teacher_path.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(clean_path, clean, RATE, subtype="PCM_16")
                    sf.write(input_path, corrupted, RATE, subtype="PCM_16")
                    temp_inputs.append(input_path)
                    rows.append({"split": split, "speaker_id": sid, "source_clip": str(source_path.relative_to(ROOT)),
                                 "source_clip_sha256": sha(source_path), "clean_path": str(clean_path.relative_to(ROOT)),
                                 "input_path": str(input_path.relative_to(ROOT)), "teacher_path": str(teacher_path.relative_to(ROOT)),
                                 "clean_sha256": sha(clean_path), "input_sha256": sha(input_path),
                                 "sample_rate_hz": RATE, "duration_seconds": args.seconds, "variant": variant,
                                 "seed": args.seed, **metadata})
    with tempfile.TemporaryDirectory(prefix="voxrefine-open-teacher-") as tmp:
        output = Path(tmp) / "teacher"
        output.mkdir()
        command = [str(TEACHER), "--compensate-delay", "--atten-lim-db", "30", "-o", str(output),
                   *map(str, temp_inputs)]
        proc = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if proc.returncode:
            raise RuntimeError(f"DeepFilterNet teacher failed:\n{proc.stdout[-4000:]}")
        for row, input_path in zip(rows, temp_inputs):
            source_out = output / input_path.name
            destination = ROOT / row["teacher_path"]
            if not source_out.is_file():
                raise RuntimeError(f"Teacher output missing: {source_out}")
            y, sr = sf.read(source_out, dtype="float32", always_2d=True)
            x, _ = sf.read(input_path, dtype="float32", always_2d=True)
            y = y.mean(axis=1)
            if sr != RATE:
                raise RuntimeError(f"Unexpected teacher sample rate {sr}: {source_out}")
            # Compensated CLI may differ by a handful of samples at EOF; preserve onset and exact input duration.
            if y.size > x.shape[0]:
                y = y[:x.shape[0]]
            elif y.size < x.shape[0]:
                y = np.pad(y, (0, x.shape[0] - y.size))
            sf.write(destination, y, RATE, subtype="PCM_16")
            row["teacher_sha256"] = sha(destination)
            row["teacher_frames"] = int(y.size)
    manifest = {"dataset": "Open licensed DeepFilterNet student pilot; no Adobe inputs or outputs",
                "source_licenses": LICENSES, "teacher_command": "deep-filter --compensate-delay --atten-lim-db 30",
                "teacher_version": "deep_filter 0.5.6; upstream DeepFilterNet dual MIT OR Apache-2.0",
                "seed": args.seed, "speaker_disjoint_split": True,
                "train_speakers": sorted({r["speaker_id"] for r in rows if r["split"] == "train"}),
                "validation_speakers": sorted({r["speaker_id"] for r in rows if r["split"] == "validation"}),
                "records": rows}
    manifest_path = OUT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"records": len(rows), "train": sum(r["split"] == "train" for r in rows),
                      "validation": sum(r["split"] == "validation" for r in rows),
                      "train_speakers": len(manifest["train_speakers"]),
                      "validation_speakers": len(manifest["validation_speakers"]),
                      "dataset": str(OUT.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
