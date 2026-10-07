"""Experimental speech-gated blending of two delay-compensated renders."""

from array import array
import math
from pathlib import Path
import tempfile
import wave

from .audio import (
    FRAME_SAMPLES, VoxRefineError, configure_wav, decode_pcm, encode_pcm,
    inspect_wav,
)
from .engines import DeepFilterNet, RNNoise

SPEECH_THRESHOLD = 0.8
MAX_BLEND = 0.75
ATTACK_FRAMES = 2
RELEASE_FRAMES = 10
LOSS_ENERGY_RATIO = 0.25
MIN_GENTLE_RMS = 32


def protect_render(
    strong: Path, gentle: Path, target: Path, probabilities: list[float]
) -> None:
    info = inspect_wav(strong)
    if inspect_wav(gentle).frames != info.frames:
        raise VoxRefineError("Protection renders have different durations.")
    expected = (info.frames + FRAME_SAMPLES - 1) // FRAME_SAMPLES
    if len(probabilities) != expected or any(
        not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities
    ):
        raise VoxRefineError("Invalid speech confidence for protection render.")
    mix = 0.0
    with wave.open(str(strong), "rb") as primary, wave.open(str(gentle), "rb") as backup:
        with wave.open(str(target), "wb") as writer:
            configure_wav(writer)
            for probability in probabilities:
                cleaned = decode_pcm(primary.readframes(FRAME_SAMPLES))
                softer = decode_pcm(backup.readframes(FRAME_SAMPLES))
                strong_energy = sum(x * x for x in cleaned) / len(cleaned)
                gentle_energy = sum(x * x for x in softer) / len(softer)
                # Require likely speech AND a >6 dB loss against the gentle render.
                # Near-silence is not evidence that speech needs restoring.
                suspect = (
                    probability >= SPEECH_THRESHOLD
                    and gentle_energy > MIN_GENTLE_RMS ** 2
                    and strong_energy < gentle_energy * LOSS_ENERGY_RATIO
                )
                desired = MAX_BLEND if suspect else 0.0
                step = MAX_BLEND / (ATTACK_FRAMES if suspect else RELEASE_FRAMES)
                next_mix = (
                    min(desired, mix + step) if suspect
                    else max(desired, mix - step)
                )
                samples = array("h")
                for index, (a, b) in enumerate(zip(cleaned, softer), start=1):
                    weight = mix + (next_mix - mix) * index / len(cleaned)
                    samples.append(round(a * (1 - weight) + b * weight))
                writer.writeframesraw(encode_pcm(samples))
                mix = next_mix


class ProtectedDeepFilterNet:
    name = "deepfilter"

    def __init__(self, primary: DeepFilterNet, detector: RNNoise):
        if primary.attenuation_limit_db <= 12:
            raise VoxRefineError(
                "Voice protection requires a DeepFilterNet attenuation limit above 12 dB."
            )
        self.primary = primary
        self.detector = detector
        self.gentle = DeepFilterNet(str(primary.executable), attenuation_limit_db=12.0)

    def identity(self) -> dict[str, str | float]:
        return {
            **self.primary.identity(),
            "voice_protection": "experimental-v1",
            "protection_gentle_limit_db": 12.0,
            "protection_speech_threshold": SPEECH_THRESHOLD,
            "protection_max_blend": MAX_BLEND,
            "protection_loss_energy_ratio": LOSS_ENERGY_RATIO,
            "protection_min_gentle_rms_pcm": float(MIN_GENTLE_RMS),
            "protection_attack_frames": float(ATTACK_FRAMES),
            "protection_release_frames": float(RELEASE_FRAMES),
            "protection_detector_sha256": self.detector.identity()["binary_sha256"],
        }

    def process(self, source: Path, target: Path) -> None:
        probabilities = self.detector.speech_probabilities(source)
        with tempfile.TemporaryDirectory(prefix="voxrefine-protect-") as directory:
            strong = Path(directory) / "strong.wav"
            gentle = Path(directory) / "gentle.wav"
            self.primary.process(source, strong)
            self.gentle.process(source, gentle)
            protect_render(strong, gentle, target, probabilities)
