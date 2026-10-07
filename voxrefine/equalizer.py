"""Optional bass and treble shelving equalization."""

from array import array
import math
from pathlib import Path
import shutil
import tempfile
import wave
from typing import Iterator, Protocol

from .audio import (
    PCM_SCALE,
    SAMPLE_RATE,
    VoxRefineError,
    configure_wav,
    decode_pcm,
    encode_pcm,
    inspect_wav,
    pcm_blocks,
)

MAX_EQ_DB = 12.0


class ProcessableEngine(Protocol):
    name: str

    def identity(self) -> dict[str, str | float]: ...

    def process(self, source: Path, target: Path) -> None: ...


class ShelfFilter:
    def __init__(self, frequency: float, gain_db: float, low: bool):
        amplitude = 10 ** (gain_db / 40)
        omega = 2 * math.pi * frequency / SAMPLE_RATE
        cosine = math.cos(omega)
        alpha = math.sin(omega) / math.sqrt(2)
        beta = 2 * math.sqrt(amplitude) * alpha
        if low:
            b0 = amplitude * ((amplitude + 1) - (amplitude - 1) * cosine + beta)
            b1 = 2 * amplitude * ((amplitude - 1) - (amplitude + 1) * cosine)
            b2 = amplitude * ((amplitude + 1) - (amplitude - 1) * cosine - beta)
            a0 = (amplitude + 1) + (amplitude - 1) * cosine + beta
            a1 = -2 * ((amplitude - 1) + (amplitude + 1) * cosine)
            a2 = (amplitude + 1) + (amplitude - 1) * cosine - beta
        else:
            b0 = amplitude * ((amplitude + 1) + (amplitude - 1) * cosine + beta)
            b1 = -2 * amplitude * ((amplitude - 1) + (amplitude + 1) * cosine)
            b2 = amplitude * ((amplitude + 1) + (amplitude - 1) * cosine - beta)
            a0 = (amplitude + 1) - (amplitude - 1) * cosine + beta
            a1 = 2 * ((amplitude - 1) - (amplitude + 1) * cosine)
            a2 = (amplitude + 1) - (amplitude - 1) * cosine - beta
        self.b0, self.b1, self.b2 = b0 / a0, b1 / a0, b2 / a0
        self.a1, self.a2 = a1 / a0, a2 / a0
        self.x1 = self.x2 = self.y1 = self.y2 = 0.0

    def process(self, sample: float) -> float:
        output = (
            self.b0 * sample + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1 - self.a2 * self.y2
        )
        self.x2, self.x1 = self.x1, sample
        self.y2, self.y1 = self.y1, output
        return output


def validate_eq(bass_db: float, treble_db: float) -> None:
    if any(
        not math.isfinite(gain) or not -MAX_EQ_DB <= gain <= MAX_EQ_DB
        for gain in (bass_db, treble_db)
    ):
        raise VoxRefineError(
            f"Bass and treble gains must be finite values between "
            f"-{MAX_EQ_DB:g} and {MAX_EQ_DB:g} dB."
        )


def _filtered_blocks(
    source: Path, bass_db: float, treble_db: float
) -> Iterator[array]:
    filters = []
    if bass_db:
        filters.append(ShelfFilter(150, bass_db, low=True))
    if treble_db:
        filters.append(ShelfFilter(4000, treble_db, low=False))

    with wave.open(str(source), "rb") as reader:
        for block in pcm_blocks(reader, source):
            processed = array("d")
            for sample in decode_pcm(block):
                value = sample
                for shelf in filters:
                    value = shelf.process(value)
                processed.append(value)
            yield processed


def equalize_wav(source: Path, target: Path, bass_db: float, treble_db: float) -> None:
    validate_eq(bass_db, treble_db)
    inspect_wav(source)
    if not bass_db and not treble_db:
        shutil.copyfile(source, target)
        return

    peak = max(
        (abs(sample) for block in _filtered_blocks(source, bass_db, treble_db)
         for sample in block),
        default=0.0,
    )
    scale = min(1.0, (PCM_SCALE - 2) / peak) if peak else 1.0
    with wave.open(str(target), "wb") as writer:
        configure_wav(writer)
        for block in _filtered_blocks(source, bass_db, treble_db):
            processed = array("h", (
                max(-PCM_SCALE, min(PCM_SCALE - 1, round(sample * scale)))
                for sample in block
            ))
            writer.writeframesraw(encode_pcm(processed))


class EqualizedEngine:
    def __init__(self, engine: ProcessableEngine, bass_db: float, treble_db: float):
        validate_eq(bass_db, treble_db)
        self.engine = engine
        self.name = engine.name
        self.bass_db = bass_db
        self.treble_db = treble_db

    def identity(self) -> dict[str, str | float]:
        return {
            **self.engine.identity(),
            "bass_db": self.bass_db,
            "treble_db": self.treble_db,
        }

    def process(self, source: Path, target: Path) -> None:
        with tempfile.TemporaryDirectory(prefix="voxrefine-eq-") as directory:
            cleaned = Path(directory) / "cleaned.wav"
            self.engine.process(source, cleaned)
            equalize_wav(cleaned, target, self.bass_db, self.treble_db)
