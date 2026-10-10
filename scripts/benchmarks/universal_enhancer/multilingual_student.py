"""Small 16 kHz voice enhancer and multilingual FLEURS data utilities."""
from __future__ import annotations

import json
import math
import random
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from fullband_student import FullbandStudent, parameter_count

ROOT = Path(__file__).resolve().parents[3]
RATE = 16_000
DATA = ROOT / "corpus/samples/fleurs-multilingual-16k-01"


class MultilingualStudent(FullbandStudent):
    """Bounded-footprint mono 16 kHz student; output keeps input sample count."""

    def __init__(self, base_channels: int = 8):
        super().__init__(n_fft=512, hop_length=128, base_channels=base_channels)


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


def read_crop(row: dict, seconds: float, rng: random.Random) -> torch.Tensor:
    path = ROOT / row["path"]
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != RATE or audio.shape[1] != 1:
        raise ValueError(f"Expected mono 16 kHz FLAC: {path}")
    signal = torch.from_numpy(audio[:, 0].copy())
    size = round(RATE * seconds)
    if signal.numel() < size:
        # Repeat only very short utterances to make a crop, with crossfade-free
        # join probability low; the audio is still paired identically.
        if signal.numel() < RATE // 2:
            raise ValueError(f"Utterance shorter than 0.5 s: {path}")
        signal = signal.repeat(math.ceil(size / signal.numel()))
    start = rng.randint(0, signal.numel() - size)
    return signal[start:start + size]


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


def random_rir(device: torch.device, rng: random.Random) -> torch.Tensor:
    length = rng.randint(round(0.12 * RATE), round(0.55 * RATE))
    rir = torch.zeros(length, device=device)
    rir[0] = 1.0
    for _ in range(rng.randint(2, 8)):
        delay = rng.randint(1, length - 1)
        rir[delay] += rng.choice((-1.0, 1.0)) * rng.uniform(0.04, 0.34)
    rt60 = rng.uniform(0.12, 0.70)
    t = torch.arange(1, length, device=device, dtype=torch.float32) / RATE
    decay = torch.pow(10.0, -3.0 * t / rt60)
    rir[1:] += torch.randn(length - 1, device=device) * decay * rng.uniform(0.02, 0.09)
    return rir


def corrupt(clean: torch.Tensor, noise_bank: list[torch.Tensor], rng: random.Random,
            device: torch.device, mode: str = "mixed") -> torch.Tensor:
    x = clean.to(device)
    if mode in {"mixed", "room"} and rng.random() < (0.8 if mode == "mixed" else 1.0):
        rir = random_rir(device, rng)
        x = F.conv1d(F.pad(x[None, None], (rir.numel() - 1, 0)),
                     rir.flip(0)[None, None])[0, 0, :clean.numel()]
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


__all__ = ["DATA", "RATE", "MultilingualStudent", "corrupt", "group_by_locale",
           "load_manifest", "make_noise_bank", "normalize_pair", "parameter_count",
           "read_crop", "records_for_split"]
