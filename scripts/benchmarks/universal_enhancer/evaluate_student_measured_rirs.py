#!/usr/bin/env python3
"""Evaluate a 16 kHz student on VCTK dry speech convolved with measured OpenSLR28 RIRs."""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from fullband_student import stft_l1
from multilingual_student import RATE, RIR_ROOT, MultilingualStudent, read_rir
from train_multilingual_student import si_sdr

ROOT = Path(__file__).resolve().parents[3]
VCTK = ROOT / "corpus/samples/fullband-student-vctk-01"
REAL_RIR = ROOT / "corpus/rir/openslr28/RIRS_NOISES/real_rirs_isotropic_noises"
OUT = ROOT / "results/measured-rir-vctk-student-01"


def measured_mono_rirs() -> list[Path]:
    paths = []
    for path in sorted(REAL_RIR.glob("*.wav")):
        info = sf.info(path)
        if info.samplerate == RATE and info.channels == 1:
            paths.append(path)
    if not paths:
        raise FileNotFoundError(f"No mono 16 kHz measured RIRs under {REAL_RIR}")
    return paths


def read_speech(path: Path) -> torch.Tensor:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate < RATE:
        raise ValueError(f"VCTK source unexpectedly below 16 kHz: {path} ({rate})")
    mono = audio.mean(axis=1)
    if rate != RATE:
        import librosa
        mono = librosa.resample(mono, orig_sr=rate, target_sr=RATE)
    return torch.from_numpy(np.asarray(mono, dtype=np.float32).copy())


def make_pair(path: Path, rir_path: Path, seconds: float, device: torch.device,
              rng: random.Random) -> tuple[torch.Tensor, torch.Tensor]:
    clean = read_speech(path)
    size = round(seconds * RATE)
    if clean.numel() < size + RATE:
        clean = clean.repeat(max(1, (size + RATE + clean.numel() - 1) // clean.numel()))
    start = rng.randint(0, clean.numel() - size - RATE)
    context = clean[start:start + size + RATE]
    rir = read_rir([rir_path], rng, device, recenter=False)
    abs_rir = rir.abs()
    peak_index = int(abs_rir.argmax().item())
    direct_candidates = torch.nonzero(abs_rir[:peak_index + 1] >= abs_rir[peak_index] * 0.316)
    direct_index = int(direct_candidates[0].item()) if direct_candidates.numel() else peak_index
    wet = F.conv1d(F.pad(context.to(device)[None, None], (rir.numel() - 1, 0)),
                   rir.flip(0)[None, None])[0, 0]
    offset = RATE // 2
    reference = context[offset:offset + size].to(device)
    # Keep measured propagation delay and direct/reverberant ratio. Normalize
    # only the *overall* pair gain so input SNR/level does not dominate metrics.
    degraded = wet[offset + direct_index:offset + direct_index + size]
    rms = reference.square().mean().sqrt().clamp_min(1e-5)
    degraded = degraded * (rms / degraded.square().mean().sqrt().clamp_min(1e-5))
    return reference, degraded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=int, default=64)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=260101)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    voices = sorted((VCTK / "validation").glob("*_mic1.flac"))
    if not voices:
        raise FileNotFoundError(f"No held-out VCTK validation files under {VCTK}")
    rirs = measured_mono_rirs()
    model = MultilingualStudent().to(device).eval()
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state.get("model", state))
    rng = random.Random(args.seed)
    records = []
    window = torch.hann_window(512, device=device)
    for index in range(args.clips):
        voice = voices[index % len(voices)]
        rir_path = rirs[rng.randrange(len(rirs))]
        clean, wet = make_pair(voice, rir_path, args.seconds, device, rng)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16,
                                             enabled=device.type == "cuda"):
            enhanced = model(wet[None])[0]
        for name, prediction in (("degraded", wet), ("student", enhanced)):
            records.append({
                "index": index, "speaker": voice.stem.split("_")[0],
                "rir": rir_path.name, "system": name,
                "si_sdr_db": float(si_sdr(prediction[None], clean[None]).item()),
                "wave_l1": float((prediction - clean).abs().mean().item()),
                "mrstft_l1": float(stft_l1(prediction[None], clean[None], 512, 128, window).item()),
                "peak": float(prediction.abs().max().item()),
                "clean_identity_l1": float((enhanced - wet).abs().mean().item()) if name == "student" else 0.0,
            })
    with (args.output / "per_clip.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    summary = {}
    for system in ("degraded", "student"):
        rows = [r for r in records if r["system"] == system]
        summary[system] = {key: float(np.mean([r[key] for r in rows]))
                           for key in ("si_sdr_db", "wave_l1", "mrstft_l1", "peak", "clean_identity_l1")}
    summary.update({"clips": args.clips, "speakers": len({r["speaker"] for r in records}),
                    "rir_count": len(rirs), "rir_license": "Apache-2.0 (OpenSLR28)",
                    "speech": "CSTR VCTK 0.92 held-out validation split",
                    "speech_license": "Open Data Commons Attribution License 1.0",
                    "sample_rate_hz": RATE, "seed": args.seed,
                    "checkpoint": str(args.checkpoint) if args.checkpoint else None})
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
