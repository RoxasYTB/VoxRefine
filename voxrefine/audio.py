"""Strict PCM WAV input and streaming technical measurements."""

from array import array
from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import TypedDict
import wave


class VoxRefineError(Exception):
    """An actionable input or engine error."""


@dataclass(frozen=True)
class AudioInfo:
    frames: int
    sample_rate: int

    @property
    def seconds(self) -> float:
        return self.frames / self.sample_rate


class AudioMetrics(TypedDict):
    frames: int
    duration_seconds: float
    peak_dbfs: float | None
    rms_dbfs: float | None
    clipped_samples: int


def inspect_wav(path: Path) -> AudioInfo:
    try:
        with wave.open(str(path), "rb") as source:
            if (
                source.getnchannels() != 1
                or source.getsampwidth() != 2
                or source.getframerate() != 48000
                or source.getcomptype() != "NONE"
            ):
                raise VoxRefineError(
                    f"{path}: WAV PCM 16 bits, mono, 48000 Hz required; "
                    "convert the file explicitly before processing."
                )
            frames = source.getnframes()
            if not frames:
                raise VoxRefineError(f"{path}: empty audio.")
            remaining = frames
            while remaining:
                block = source.readframes(min(remaining, 48000))
                if not block or len(block) % 2:
                    raise VoxRefineError(f"{path}: truncated PCM data.")
                remaining -= len(block) // 2
            return AudioInfo(frames, source.getframerate())
    except (wave.Error, EOFError) as error:
        raise VoxRefineError(f"{path}: invalid PCM WAV ({error}).") from error


def decode_pcm(data: bytes) -> array:
    samples = array("h")
    samples.frombytes(data)
    if sys.byteorder != "little":
        samples.byteswap()
    return samples


def encode_pcm(samples: array) -> bytes:
    if sys.byteorder == "little":
        return samples.tobytes()
    copy = array("h", samples)
    copy.byteswap()
    return copy.tobytes()


def configure_wav(target: wave.Wave_write) -> None:
    target.setnchannels(1)
    target.setsampwidth(2)
    target.setframerate(48000)


def measure_wav(path: Path) -> AudioMetrics:
    info = inspect_wav(path)
    peak = 0
    square_sum = 0
    clipped = 0
    with wave.open(str(path), "rb") as source:
        while block := source.readframes(48000):
            for sample in decode_pcm(block):
                peak = max(peak, abs(sample))
                square_sum += sample * sample
                clipped += sample in (-32768, 32767)
    rms = math.sqrt(square_sum / info.frames) / 32768
    return {
        "frames": info.frames,
        "duration_seconds": info.seconds,
        "peak_dbfs": 20 * math.log10(peak / 32768) if peak else None,
        "rms_dbfs": 20 * math.log10(rms) if rms else None,
        "clipped_samples": clipped,
    }
