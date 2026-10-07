"""Adapters for explicitly installed, versioned local engines."""

from array import array
import ctypes
import hashlib
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Protocol
import wave

from .audio import (
    FRAME_SAMPLES,
    PCM_SCALE,
    SAMPLE_RATE,
    SAMPLE_WIDTH,
    VoxRefineError,
    configure_wav,
    decode_pcm,
    encode_pcm,
    inspect_wav,
)
from .conversion import prepared_audio
from .process import run_checked


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


class DeepFilterNet:
    name = "deepfilter"

    def __init__(self, executable: str, attenuation_limit_db: float = 100.0):
        if not math.isfinite(attenuation_limit_db) or not 0 <= attenuation_limit_db <= 100:
            raise VoxRefineError("Attenuation limit must be a finite number between 0 and 100 dB.")
        self.attenuation_limit_db = attenuation_limit_db
        resolved = shutil.which(executable)
        if resolved is None:
            raise VoxRefineError(
                f"DeepFilterNet executable not found: {executable}. "
                "See README.md for installation."
            )
        self.executable = Path(resolved).resolve()
        self.version = run_checked([str(self.executable), "--version"])
        if self.version != "deep_filter 0.5.6":
            raise VoxRefineError(
                f"Unsupported DeepFilterNet version: {self.version}; "
                "this adapter targets 0.5.6."
            )

    def identity(self) -> dict[str, str | float]:
        return {
            "engine": self.name,
            "version": self.version,
            "binary_sha256": file_hash(self.executable),
            "model": "upstream binary's built-in model",
            "attenuation_limit_db": self.attenuation_limit_db,
        }

    def process(self, source: Path, target: Path) -> None:
        info = inspect_wav(source)
        with tempfile.TemporaryDirectory(prefix="voxrefine-df-") as directory:
            temporary = Path(directory)
            padded = temporary / "input.wav"
            # The 0.5.6 CLI removes latency without flushing its final frames.
            # Add 100 ms plus hop alignment, then keep exactly the input length.
            padding = SAMPLE_RATE // 10 + (-info.frames % FRAME_SAMPLES)
            with wave.open(str(source), "rb") as reader:
                with wave.open(str(padded), "wb") as writer:
                    configure_wav(writer)
                    while block := reader.readframes(SAMPLE_RATE):
                        writer.writeframesraw(block)
                    writer.writeframesraw(b"\0" * SAMPLE_WIDTH * padding)
            output_directory = temporary / "out"
            run_checked([
                str(self.executable), "-D",
                "--atten-lim-db", str(self.attenuation_limit_db),
                "-o", str(output_directory), str(padded)
            ])
            enhanced = output_directory / "input.wav"
            output_info = inspect_wav(enhanced)
            if output_info.frames < info.frames:
                raise VoxRefineError("DeepFilterNet returned truncated audio.")
            with wave.open(str(enhanced), "rb") as reader:
                with wave.open(str(target), "wb") as writer:
                    configure_wav(writer)
                    remaining = info.frames
                    while remaining:
                        block = reader.readframes(min(remaining, SAMPLE_RATE))
                        writer.writeframesraw(block)
                        remaining -= len(block) // SAMPLE_WIDTH


class RNNoise:
    name = "rnnoise"
    frame_size = FRAME_SAMPLES

    def __init__(self, library: str):
        self.library = Path(library).expanduser().resolve()
        if not self.library.is_file():
            raise VoxRefineError(
                f"RNNoise library not found: {self.library}. "
                "Build the v0.1 library documented in README.md."
            )
        try:
            self.native = ctypes.CDLL(str(self.library))
            if hasattr(self.native, "rnnoise_model_from_file"):
                raise VoxRefineError(
                    "This RNNoise adapter targets the v0.1 ABI. "
                    "Build the pinned source documented in README.md."
                )
            self.native.rnnoise_create.argtypes = []
            self.native.rnnoise_create.restype = ctypes.c_void_p
            self.native.rnnoise_destroy.argtypes = [ctypes.c_void_p]
            self.native.rnnoise_destroy.restype = None
            pointer = ctypes.POINTER(ctypes.c_float)
            self.native.rnnoise_process_frame.argtypes = [
                ctypes.c_void_p, pointer, pointer
            ]
            self.native.rnnoise_process_frame.restype = ctypes.c_float
        except (OSError, AttributeError) as error:
            raise VoxRefineError(f"Cannot load RNNoise: {error}") from error

    def identity(self) -> dict[str, str | float]:
        return {
            "engine": self.name,
            "version": "native library; tested against upstream v0.1",
            "binary_sha256": file_hash(self.library),
            "model": "library's built-in model",
        }

    def speech_probabilities(self, source: Path) -> list[float]:
        """Return native speech confidence for each 10 ms input frame."""
        inspect_wav(source)
        state = self.native.rnnoise_create()
        if not state:
            raise VoxRefineError("RNNoise could not allocate its state.")
        buffer_type = ctypes.c_float * self.frame_size
        output = buffer_type()
        probabilities = []
        try:
            with wave.open(str(source), "rb") as reader:
                while block := reader.readframes(self.frame_size):
                    samples = decode_pcm(block)
                    samples.extend([0] * (self.frame_size - len(samples)))
                    probability = self.native.rnnoise_process_frame(
                        state, output, buffer_type(*samples)
                    )
                    if not math.isfinite(probability) or not 0 <= probability <= 1:
                        raise VoxRefineError("RNNoise returned invalid speech confidence.")
                    probabilities.append(float(probability))
        finally:
            self.native.rnnoise_destroy(state)
        return probabilities

    def process(self, source: Path, target: Path) -> None:
        info = inspect_wav(source)
        state = self.native.rnnoise_create()
        if not state:
            raise VoxRefineError("RNNoise could not allocate its state.")
        buffer_type = ctypes.c_float * self.frame_size
        output = buffer_type()
        remaining = info.frames
        skip = self.frame_size
        try:
            with wave.open(str(source), "rb") as reader:
                with wave.open(str(target), "wb") as writer:
                    configure_wav(writer)
                    # One trailing frame flushes the 10 ms synthesis delay.
                    for _ in range(
                        (info.frames + self.frame_size - 1) // self.frame_size + 1
                    ):
                        samples = decode_pcm(reader.readframes(self.frame_size))
                        samples.extend([0] * (self.frame_size - len(samples)))
                        self.native.rnnoise_process_frame(
                            state, output, buffer_type(*samples)
                        )
                        if skip:
                            skip = 0
                            continue
                        count = min(remaining, self.frame_size)
                        cleaned = array("h", (
                            max(-PCM_SCALE, min(PCM_SCALE - 1, round(value)))
                            for value in output[:count]
                        ))
                        writer.writeframesraw(encode_pcm(cleaned))
                        remaining -= count
        finally:
            self.native.rnnoise_destroy(state)


class Engine(Protocol):
    name: str

    def identity(self) -> dict[str, str | float]: ...

    def process(self, source: Path, target: Path) -> None: ...


def clean(
    source: Path, target: Path, engine: Engine, ffmpeg: str = "ffmpeg"
) -> None:
    source = source.expanduser().resolve()
    target = target.expanduser().resolve()
    if target.suffix.lower() != ".wav":
        raise VoxRefineError(
            "Output must have a .wav extension (mono/stereo PCM16 at 48000 Hz)."
        )
    if target.exists():
        raise VoxRefineError(
            f"Output already exists: {target}. Choose a new path; files are never overwritten."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    with prepared_audio(source, ffmpeg) as prepared, tempfile.TemporaryDirectory(
        prefix=".voxrefine-", dir=target.parent
    ) as directory:
        temporary = Path(directory) / "cleaned.wav"
        info = inspect_wav(prepared)
        if info.channels == 1:
            engine.process(prepared, temporary)
        else:
            left_input = Path(directory) / "left-input.wav"
            right_input = Path(directory) / "right-input.wav"
            left_output = Path(directory) / "left-output.wav"
            right_output = Path(directory) / "right-output.wav"
            with wave.open(str(prepared), "rb") as reader:
                with wave.open(str(left_input), "wb") as left, wave.open(
                    str(right_input), "wb"
                ) as right:
                    configure_wav(left)
                    configure_wav(right)
                    while block := reader.readframes(SAMPLE_RATE):
                        samples = decode_pcm(block)
                        left_samples = array("h", samples[0::2])
                        right_samples = array("h", samples[1::2])
                        left.writeframesraw(encode_pcm(left_samples))
                        right.writeframesraw(encode_pcm(right_samples))
            engine.process(left_input, left_output)
            engine.process(right_input, right_output)
            if (
                inspect_wav(left_output).frames != info.frames
                or inspect_wav(right_output).frames != info.frames
            ):
                raise VoxRefineError("Engine changed the audio duration.")
            with wave.open(str(left_output), "rb") as left, wave.open(
                str(right_output), "rb"
            ) as right, wave.open(str(temporary), "wb") as writer:
                configure_wav(writer, channels=2)
                remaining = info.frames
                while remaining:
                    count = min(remaining, SAMPLE_RATE)
                    left_samples = decode_pcm(left.readframes(count))
                    right_samples = decode_pcm(right.readframes(count))
                    interleaved = array("h")
                    for left_sample, right_sample in zip(left_samples, right_samples):
                        interleaved.extend((left_sample, right_sample))
                    writer.writeframesraw(encode_pcm(interleaved))
                    remaining -= len(left_samples)
        result_info = inspect_wav(temporary)
        if (result_info.frames, result_info.channels) != (info.frames, info.channels):
            raise VoxRefineError("Engine changed the audio duration or channel count.")
        # Publish atomically without replacing a file that appeared mid-run.
        os.link(temporary, target)
