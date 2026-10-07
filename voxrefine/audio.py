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
    channels: int = 1
    channel_layout: str | None = None
    duration_s: float | None = None
    codec: str | None = None
    container: str | None = None
    bit_rate: int | None = None
    sample_format: str | None = None

    @property
    def seconds(self) -> float:
        return self.duration_s if self.duration_s is not None else self.frames / self.sample_rate


@dataclass(frozen=True)
class AudioSource:
    """Original media and its probed properties; no decoded PCM is stored here."""

    path: Path
    info: AudioInfo


@dataclass(frozen=True)
class AudioDomain:
    sample_rate: int
    channels: int
    sample_format: str
    container: str = "wav"
    interleaved: bool = True
    frame_samples: int | None = None

    def __post_init__(self) -> None:
        if self.sample_rate <= 0 or self.channels <= 0:
            raise ValueError("AudioDomain sample rate and channel count must be positive.")
        if self.sample_format not in {"s16", "f32"}:
            raise ValueError("AudioDomain sample_format must be 's16' or 'f32'.")
        if self.container not in {"wav", "raw"}:
            raise ValueError("AudioDomain container must be 'wav' or 'raw'.")
        if self.frame_samples is not None and self.frame_samples <= 0:
            raise ValueError("AudioDomain frame size must be positive.")


@dataclass(frozen=True)
class AdaptationInfo:
    source_sample_rate: int
    target_sample_rate: int
    source_channels: int
    target_channels: int
    sample_format: str
    resampler: str
    channel_policy: str
    ffmpeg_version: str | None = None
    ffmpeg_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class PreparedAudio:
    domain: AudioDomain
    path: Path | None
    stream: object | None
    source_info: AudioInfo
    adaptation: AdaptationInfo

    def __post_init__(self) -> None:
        if (self.path is None) == (self.stream is None):
            raise ValueError("Exactly one of path or stream must be set.")


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
