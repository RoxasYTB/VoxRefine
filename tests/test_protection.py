from contextlib import redirect_stderr
import io
import math
import os
from pathlib import Path
import tempfile
import unittest
import wave

from tests.test_voxrefine import write_wav
from voxrefine.audio import VoxRefineError, decode_pcm, inspect_wav
from voxrefine.cli import main
from voxrefine.engines import DeepFilterNet, RNNoise, clean
from voxrefine.protection import ProtectedDeepFilterNet, protect_render


class ProtectionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.strong = self.root / "strong.wav"
        self.gentle = self.root / "gentle.wav"
        self.target = self.root / "protected.wav"
        write_wav(self.strong, [100] * 4801)
        write_wav(self.gentle, [1000] * 4801)

    def read_output(self):
        with wave.open(str(self.target), "rb") as reader:
            return decode_pcm(reader.readframes(reader.getnframes()))

    def test_no_speech_leaves_primary_unchanged(self):
        protect_render(self.strong, self.gentle, self.target, [0.0] * 11)
        self.assertEqual(self.target.read_bytes(), self.strong.read_bytes())

    def test_silence_does_not_trigger_protection(self):
        write_wav(self.strong, [0] * 4801)
        write_wav(self.gentle, [0] * 4801)
        protect_render(self.strong, self.gentle, self.target, [1.0] * 11)
        self.assertEqual(self.target.read_bytes(), self.strong.read_bytes())

    def test_no_large_attenuation_leaves_primary_unchanged(self):
        write_wav(self.gentle, [110] * 4801)
        protect_render(self.strong, self.gentle, self.target, [1.0] * 11)
        self.assertEqual(self.target.read_bytes(), self.strong.read_bytes())

    def test_blend_has_smooth_attack_and_release(self):
        write_wav(self.strong, [100] * (480 * 14))
        write_wav(self.gentle, [1000] * (480 * 14))
        protect_render(
            self.strong, self.gentle, self.target, [1.0] * 3 + [0.0] * 11
        )
        output = self.read_output()
        self.assertEqual(len(output), 480 * 14)
        self.assertEqual(max(output), 775)
        self.assertEqual(output[-1], 100)
        self.assertLessEqual(max(abs(a - b) for a, b in zip(output, output[1:])), 1)

    def test_short_file_and_bounds(self):
        write_wav(self.strong, [-32768])
        write_wav(self.gentle, [32767])
        protect_render(self.strong, self.gentle, self.target, [1.0])
        self.assertEqual(inspect_wav(self.target).frames, 1)

    def test_bad_confidence_is_explicit(self):
        for values in ([], [math.nan] * 11, [1.1] * 11):
            with self.subTest(values=values):
                with self.assertRaisesRegex(VoxRefineError, "confidence"):
                    protect_render(self.strong, self.gentle, self.target, values)
        self.assertFalse(self.target.exists())

    def test_mismatched_renders_are_rejected(self):
        write_wav(self.gentle, [1000])
        with self.assertRaisesRegex(VoxRefineError, "different durations"):
            protect_render(self.strong, self.gentle, self.target, [1.0] * 11)

@unittest.skipUnless(
    os.environ.get("VOXREFINE_DEEP_FILTER") and os.environ.get("VOXREFINE_RNNOISE"),
    "Native engines required.",
)
class NativeProtectionTests(unittest.TestCase):
    def test_native_detector_and_protected_render(self):
        primary = DeepFilterNet(os.environ["VOXREFINE_DEEP_FILTER"])
        detector = RNNoise(os.environ["VOXREFINE_RNNOISE"])
        engine = ProtectedDeepFilterNet(primary, detector)
        self.assertEqual(engine.identity()["voice_protection"], "experimental-v1")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for count in (1, 479, 480, 481, 48013):
                with self.subTest(count=count):
                    source = root / "silence.wav"
                    target = root / f"protected-{count}.wav"
                    write_wav(source, [0] * count)
                    probabilities = detector.speech_probabilities(source)
                    self.assertEqual(len(probabilities), (count + 479) // 480)
                    self.assertTrue(all(0 <= p <= 1 for p in probabilities))
                    clean(source, target, engine)
                    self.assertEqual(inspect_wav(target).frames, count)
        with self.assertRaisesRegex(VoxRefineError, "above 12"):
            ProtectedDeepFilterNet(
                DeepFilterNet(os.environ["VOXREFINE_DEEP_FILTER"], 12), detector
            )
