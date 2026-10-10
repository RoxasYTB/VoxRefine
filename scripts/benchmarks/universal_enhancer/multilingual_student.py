"""Small 16 kHz voice enhancer and multilingual FLEURS data utilities."""
from __future__ import annotations

import json
import math
import random
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from fullband_student import FullbandStudent, parameter_count

ROOT = Path(__file__).resolve().parents[3]
RATE = 16_000
DATA = ROOT / "corpus/samples/fleurs-multilingual-16k-01"
RIR_ROOT = ROOT / "corpus/rir/openslr26/simulated_rirs_16k"


class MultilingualStudent(FullbandStudent):
    """Bounded-footprint mono 16 kHz student; output keeps input sample count."""

    def __init__(self, base_channels: int = 8, residual_scale: float = 0.15):
        super().__init__(n_fft=512, hop_length=128, base_channels=base_channels,
                         residual_scale=residual_scale)


def load_manifest() -> dict:
    path = DATA / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"FLEURS manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("dataset") != "FLEURS" or manifest.get("audio_source_sample_rate_hz") != RATE:
        raise ValueError(f"Unexpected FLEURS manifest metadata: {path}")
    return manifest


def records_for_split(manifest: dict, split: str) -> list[dict]:
    rows = [r for r in manifest.get("records", []) if r.get("split") == split]
    good = []
    for row in rows:
        path = ROOT / row["path"]
        if not path.is_file():
            continue
        if row.get("sample_rate_hz") != RATE or row.get("channels") != 1:
            raise ValueError(f"Invalid rate/channels for {path}")
        good.append(row)
    if not good:
        raise RuntimeError(f"No valid {split} records in {DATA / 'manifest.json'}")
    return good


def group_by_locale(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row["locale"]), []).append(row)
    return groups


def read_crop(row: dict, seconds: float, rng: random.Random,
              context_seconds: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    path = ROOT / row["path"]
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != RATE or audio.shape[1] != 1:
        raise ValueError(f"Expected mono 16 kHz FLAC: {path}")
    signal = torch.from_numpy(audio[:, 0].copy())
    size = round(RATE * seconds)
    context = round(RATE * context_seconds)
    total = size + 2 * context
    if signal.numel() < size:
        # Repeat only very short utterances to make a crop, with crossfade-free
        # join probability low; the audio is still paired identically.
        if signal.numel() < RATE // 2:
            raise ValueError(f"Utterance shorter than 0.5 s: {path}")
        signal = signal.repeat(math.ceil(size / signal.numel()))
    if signal.numel() < total:
        # Pad with edge silence only for short utterances, while retaining exact crop length.
        signal = F.pad(signal, (context, context))
        start = rng.randint(0, signal.numel() - total)
        crop = signal[start:start + total]
    else:
        start = rng.randint(0, signal.numel() - total)
        crop = signal[start:start + total]
    return crop[context:context + size], crop


def make_noise_bank() -> list[torch.Tensor]:
    result = []
    source_dir = ROOT / "corpus/raw"
    for path in sorted(source_dir.glob("cc0-*.mp3")):
        command = ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(RATE), "-"]
        decoded = subprocess.run(command, check=True, stdout=subprocess.PIPE).stdout
        noise = np.frombuffer(decoded, dtype="<f4").copy()
        if noise.size >= RATE:
            result.append(torch.from_numpy(noise))
    # Randomized colored noise broadens noise shapes without adding a dataset.
    return result


def load_rir_index() -> list[Path]:
    """Return validated 16 kHz OpenSLR26 RIR paths without loading them all."""
    if not RIR_ROOT.is_dir():
        raise FileNotFoundError(f"Extract OpenSLR26 RIRs first: {RIR_ROOT}")
    paths = sorted(RIR_ROOT.glob("*room/Room*/*.wav"))
    if not paths:
        raise RuntimeError(f"No OpenSLR26 impulse responses found under {RIR_ROOT}")
    return paths


def read_rir(paths: list[Path], rng: random.Random, device: torch.device,
             recenter: bool = True) -> torch.Tensor:
    path = rng.choice(paths)
    rir, rate = sf.read(path, dtype="float32")
    if rate != RATE or rir.ndim != 1 or rir.size < 2:
        raise ValueError(f"Expected mono 16 kHz RIR, got {rate} Hz/shape={rir.shape}: {path}")
    impulse = torch.from_numpy(np.asarray(rir, dtype=np.float32).copy()).to(device)
    # Simulated OpenSLR26 includes propagation delay; optionally remove it for
    # training. Real OpenSLR28 measurements can be kept intact for evaluation.
    if recenter:
        abs_impulse = impulse.abs()
        peak_index = int(abs_impulse.argmax().item())
        direct_candidates = torch.nonzero(abs_impulse[:peak_index + 1] >= abs_impulse[peak_index] * 0.316)
        direct_index = int(direct_candidates[0].item()) if direct_candidates.numel() else peak_index
        start = max(0, direct_index - round(0.002 * RATE))
        impulse = impulse[start:]
    impulse = impulse[:min(impulse.numel(), round(0.6 * RATE))]
    # Remove DC; normalize by the direct peak so RIR gain is not a nuisance.
    impulse = impulse - impulse.mean()
    peak = impulse.abs().max().clamp_min(1e-6)
    return impulse / peak


def corrupt(clean: torch.Tensor, noise_bank: list[torch.Tensor], rng: random.Random,
            device: torch.device, mode: str = "mixed",
            rir_paths: list[Path] | None = None,
            clean_context: torch.Tensor | None = None) -> torch.Tensor:
    x = clean.to(device)
    if mode in {"mixed", "room"} and rng.random() < (0.8 if mode == "mixed" else 1.0):
        rir = read_rir(rir_paths, rng, device) if rir_paths else None
        if rir is None:
            raise RuntimeError("Measured/simulated RIR bank required for room corruption")
        source = clean_context.to(device) if clean_context is not None else x
        offset = max(0, (source.numel() - x.numel()) // 2)
        reverberant = F.conv1d(F.pad(source[None, None], (rir.numel() - 1, 0)),
                               rir.flip(0)[None, None])[0, 0]
        x = reverberant[offset:offset + x.numel()]
        # Normalize only overall convolution gain; preserve the direct/reverb ratio.
        x_rms = x.square().mean().sqrt().clamp_min(1e-5)
        clean_rms = clean.to(device).square().mean().sqrt().clamp_min(1e-5)
        x = x * (clean_rms / x_rms)
    if mode in {"mixed", "noise"} and noise_bank and rng.random() < (0.85 if mode == "mixed" else 1.0):
        source = rng.choice(noise_bank)
        if source.numel() < x.numel():
            source = source.repeat(math.ceil(x.numel() / source.numel()))
        start = rng.randint(0, source.numel() - x.numel())
        n = source[start:start + x.numel()].to(device)
        snr = rng.uniform(0.0, 25.0)
        x_rms = x.square().mean().sqrt().clamp_min(1e-5)
        n_rms = n.square().mean().sqrt().clamp_min(1e-5)
        x = x + n * x_rms / (n_rms * (10 ** (snr / 20)))
    # Low-band capture emulation; train only to the real 16 kHz target.
    if mode == "mixed" and rng.random() < 0.20:
        half = F.interpolate(x[None, None], size=max(1, x.numel() // 2), mode="area")
        x = F.interpolate(half, size=clean.numel(), mode="linear", align_corners=False)[0, 0]
    if mode == "mixed" and rng.random() < 0.10:
        drive = rng.uniform(1.2, 2.5)
        x = torch.tanh(drive * x) / math.tanh(drive)
    return x


def normalize_pair(clean: torch.Tensor, noisy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = clean.square().mean().sqrt().clamp_min(1e-4)
    return noisy / scale, clean / scale


__all__ = ["DATA", "RATE", "RIR_ROOT", "MultilingualStudent", "corrupt", "group_by_locale",
           "load_manifest", "make_noise_bank", "normalize_pair", "parameter_count",
           "load_rir_index", "read_crop", "records_for_split"]
