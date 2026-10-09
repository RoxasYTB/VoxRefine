"""Licensed speech loading and procedural dry/RIR pairs for the 16 kHz proof."""

from __future__ import annotations

from pathlib import Path
import random
import re

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, lfilter
import torch
from torch.utils.data import IterableDataset, get_worker_info


SPLITS = {"train-clean-100", "dev-clean", "dev-other", "test-clean", "test-other"}
SAMPLE_RATE = 16_000
CROP_SAMPLES = 32_000


def read_librispeech(root: Path, split: str) -> list[tuple[Path, str]]:
    """Return sorted (FLAC path, speaker ID) entries for one allowed split."""
    if split not in SPLITS:
        raise ValueError(f"unsupported LibriSpeech split: {split!r}")
    split_root = Path(root) / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"missing split directory: {split_root}")
    rows: list[tuple[Path, str]] = []
    for path in sorted(split_root.glob("*/*/*.flac")):
        speaker_id = path.parent.parent.name
        if not re.fullmatch(r"\d+", speaker_id):
            raise ValueError(f"invalid speaker directory: {path}")
        rows.append((path, speaker_id))
    if not rows:
        raise FileNotFoundError(f"no FLAC files under {split_root}")
    return rows


def make_procedural_rir(
    sr: int,
    seed: int,
    t60_s: float,
    direct_to_reverb_db: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a simple randomized RIR and its <=50 ms / >50 ms components.

    The direct impulse is at sample zero. The late noise envelope reaches -60 dB
    at `t60_s`; a random one-pole low-pass adds mild frequency-dependent decay.
    This is a controlled synthetic operator, not a measured-room simulator.
    """
    if sr <= 0 or not 0.25 <= t60_s <= 1.20:
        raise ValueError("sr must be positive and t60_s must be in [0.25, 1.20]")
    if not -6.0 <= direct_to_reverb_db <= 18.0:
        raise ValueError("direct_to_reverb_db must be in [-6, 18]")

    rng = np.random.default_rng(seed)
    # `t60_s` is the time to a 60 dB amplitude decay, not the RIR duration.
    # Keep at least 1.2 s so the held-out 300–600 ms post-event windows contain
    # the late response instead of an artificially hard-truncated zero tail.
    length = max(int(round(2.0 * t60_s * sr)), int(round(1.20 * sr)))
    early = np.zeros(length, dtype=np.float64)
    early[0] = 1.0
    n_reflections = int(rng.integers(4, 13))
    lo, hi = int(round(0.005 * sr)), min(int(round(0.050 * sr)), length - 1)
    for delay in rng.integers(lo, hi + 1, size=n_reflections):
        decay = np.exp(-6.907755 * (delay / sr) / t60_s)
        early[delay] += float(rng.choice((-1.0, 1.0))) * float(rng.uniform(0.08, 0.30)) * decay

    late = np.zeros_like(early)
    start = min(int(round(0.050 * sr)), length)
    if start < length:
        t = np.arange(length - start, dtype=np.float64) / sr
        noise = rng.standard_normal(length - start)
        cutoff_hz = float(rng.uniform(3_000.0, 9_000.0))
        pole = float(np.exp(-2.0 * np.pi * min(cutoff_hz, 0.49 * sr) / sr))
        damped = lfilter([1.0 - pole], [1.0, -pole], noise)
        envelope = np.power(10.0, -3.0 * (t + start / sr) / t60_s)
        late[start:] = damped * envelope
        direct_energy = float(np.sum(early * early))
        late_energy = float(np.sum(late * late))
        desired_late_energy = direct_energy / (10.0 ** (direct_to_reverb_db / 10.0))
        if late_energy > 0:
            late *= np.sqrt(desired_late_energy / late_energy)

    early = early.astype(np.float32)
    late = late.astype(np.float32)
    return early + late, early, late


def convolve_same_length(signal: np.ndarray, impulse: np.ndarray) -> np.ndarray:
    """Causal FFT convolution cropped to the source length."""
    signal = np.asarray(signal, dtype=np.float32)
    impulse = np.asarray(impulse, dtype=np.float32)
    if signal.ndim != 1 or impulse.ndim != 1:
        raise ValueError("signal and impulse must be mono 1-D arrays")
    return fftconvolve(signal, impulse, mode="full")[: len(signal)].astype(np.float32)


def make_validation_pair(
    clean: np.ndarray,
    rir_params: dict,
    sr: int = SAMPLE_RATE,
) -> dict[str, np.ndarray]:
    """Return aligned clean, early, late, and reverberant signals.

    Every output uses the same single scalar, derived from the maximum of the
    dry and reverberant signals. There is no local event-level gain matching.
    """
    clean = np.asarray(clean, dtype=np.float32)
    if clean.ndim != 1 or clean.size == 0:
        raise ValueError("clean must be a nonempty mono signal")
    rir, early_rir, late_rir = make_procedural_rir(
        sr=sr,
        seed=int(rir_params["seed"]),
        t60_s=float(rir_params["t60_s"]),
        direct_to_reverb_db=float(rir_params["direct_to_reverb_db"]),
    )
    early = convolve_same_length(clean, early_rir)
    late = convolve_same_length(clean, late_rir)
    reverberant = early + late
    shared_peak = max(float(np.max(np.abs(clean))), float(np.max(np.abs(reverberant))), 1e-8)
    scale = 0.80 / shared_peak
    return {
        "clean": (clean * scale).astype(np.float32),
        "early": (early * scale).astype(np.float32),
        "late": (late * scale).astype(np.float32),
        "reverberant": (reverberant * scale).astype(np.float32),
        "rir": rir,
    }


def crop_audio(audio: np.ndarray, crop_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Select a seeded crop; repeat only when the recording is shorter."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or audio.size == 0 or crop_samples <= 0:
        raise ValueError("audio must be nonempty mono and crop_samples positive")
    if audio.size < crop_samples:
        repeats = int(np.ceil(crop_samples / audio.size))
        audio = np.tile(audio, repeats)
    start = int(rng.integers(0, audio.size - crop_samples + 1))
    return audio[start : start + crop_samples].copy()


class RandomPairedSpeech(IterableDataset):
    """Endless clean/reverb pairs with fresh speech crops and procedural rooms."""

    def __init__(self, rows: list[tuple[Path, str]], seed: int = 0):
        super().__init__()
        if not rows:
            raise ValueError("rows must not be empty")
        self.rows = list(rows)
        self.seed = int(seed)

    def __iter__(self):
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        seed = self.seed + worker_id * 1_000_003
        rng = np.random.default_rng(seed)
        py_rng = random.Random(seed)
        while True:
            path, speaker_id = self.rows[int(rng.integers(0, len(self.rows)))]
            audio, sr = sf.read(path, dtype="float32", always_2d=False)
            if sr != SAMPLE_RATE:
                raise ValueError(f"expected 16 kHz LibriSpeech audio, got {sr}: {path}")
            if audio.ndim == 2:
                audio = audio.mean(axis=1, dtype=np.float32)
            clean = crop_audio(audio, CROP_SAMPLES, rng)
            level = float(10.0 ** (rng.uniform(-18.0, 0.0) / 20.0))
            t60 = float(rng.uniform(0.25, 1.20))
            drr = float(rng.uniform(-6.0, 18.0))
            rir_seed = py_rng.randrange(0, 2**32)
            pair = make_validation_pair(
                clean,
                {"seed": rir_seed, "t60_s": t60, "direct_to_reverb_db": drr},
            )
            dry_control = bool(rng.random() < 0.20)
            mixture = pair["clean"] if dry_control else pair["reverberant"]
            late = np.zeros_like(pair["late"]) if dry_control else pair["late"]
            yield {
                "mixture": torch.from_numpy(mixture * level),
                "clean": torch.from_numpy(pair["clean"] * level),
                "late": torch.from_numpy(late * level),
                "dry_control": dry_control,
                "speaker_id": speaker_id,
                "t60_s": t60,
                "drr_db": drr,
                "rir_seed": rir_seed,
            }
