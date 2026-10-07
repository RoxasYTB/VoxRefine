"""Strict PCM WAV input and streaming technical measurements."""

from array import array
from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Iterator, TypedDict
import wave

SAMPLE_RATE = 48000
CHANNELS = 1
SAMPLE_WIDTH = 2
PCM_SCALE = 32768
FRAME_SAMPLES = SAMPLE_RATE // 100


class VoxRefineError(Exception):
    """An actionable input or engine error."""


@dataclass(frozen=True)
class AudioInfo:
    frames: int
    sample_rate: int
    channels: int

    @property
    def seconds(self) -> float:
        return self.frames / self.sample_rate


class AudioMetrics(TypedDict):
    frames: int
    duration_seconds: float
    channels: int
    peak_dbfs: float | None
    rms_dbfs: float | None
    clipped_samples: int


def compatible_wav(source: wave.Wave_read) -> bool:
    return (
        source.getnchannels() in (1, 2)
        and source.getsampwidth() == SAMPLE_WIDTH
        and source.getframerate() == SAMPLE_RATE
        and source.getcomptype() == "NONE"
    )


def wav_info(source: wave.Wave_read, path: Path) -> AudioInfo:
    if not compatible_wav(source):
        raise VoxRefineError(
            f"{path}: WAV PCM 16 bits, mono or stereo, {SAMPLE_RATE} Hz required; "
            "convert the file explicitly before processing."
        )
    if not source.getnframes():
        raise VoxRefineError(f"{path}: empty audio.")
    return AudioInfo(
        source.getnframes(), source.getframerate(), source.getnchannels()
    )


def pcm_blocks(source: wave.Wave_read, path: Path) -> Iterator[bytes]:
    remaining = source.getnframes()
    while remaining:
        count = min(remaining, SAMPLE_RATE)
        block = source.readframes(count)
        if len(block) != count * SAMPLE_WIDTH * source.getnchannels():
            raise VoxRefineError(f"{path}: truncated PCM data.")
        yield block
        remaining -= count


def inspect_wav(path: Path) -> AudioInfo:
    try:
        with wave.open(str(path), "rb") as source:
            info = wav_info(source, path)
            for _ in pcm_blocks(source, path):
                pass
            return info
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


def configure_wav(target: wave.Wave_write, channels: int = CHANNELS) -> None:
    target.setnchannels(channels)
    target.setsampwidth(SAMPLE_WIDTH)
    target.setframerate(SAMPLE_RATE)


def measure_wav(path: Path) -> AudioMetrics:
    peak = 0
    square_sum = 0
    samples_count = 0
    clipped = 0
    try:
        with wave.open(str(path), "rb") as source:
            info = wav_info(source, path)
            for block in pcm_blocks(source, path):
                for sample in decode_pcm(block):
                    peak = max(peak, abs(sample))
                    square_sum += sample * sample
                    clipped += sample in (-PCM_SCALE, PCM_SCALE - 1)
                    samples_count += 1
    except (wave.Error, EOFError) as error:
        raise VoxRefineError(f"{path}: invalid PCM WAV ({error}).") from error
    rms = math.sqrt(square_sum / samples_count) / PCM_SCALE
    return {
        "frames": info.frames,
        "duration_seconds": info.seconds,
        "channels": info.channels,
        "peak_dbfs": 20 * math.log10(peak / PCM_SCALE) if peak else None,
        "rms_dbfs": 20 * math.log10(rms) if rms else None,
        "clipped_samples": clipped,
    }
