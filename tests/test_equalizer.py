from array import array
from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
import wave
from unittest.mock import patch

from voxrefine.audio import (
    SAMPLE_RATE,
    VoxRefineError,
    configure_wav,
    encode_pcm,
    inspect_wav,
    measure_wav,
)
from voxrefine.cli import main
from voxrefine.equalizer import EqualizedEngine, equalize_wav


class CopyEngine:
    name = "copy"

    def identity(self):
        return {"engine": self.name, "version": "test-only"}

    def process(self, source: Path, target: Path) -> None:
        target.write_bytes(source.read_bytes())


class EqualizerTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.source = self.root / "source.wav"
        self.target = self.root / "equalized.wav"

    def write_tone(self, frequency: int, amplitude: int = 4000):
        samples = array("h", (
            round(amplitude * math.sin(2 * math.pi * frequency * index / SAMPLE_RATE))
            for index in range(SAMPLE_RATE // 4)
        ))
        with wave.open(str(self.source), "wb") as writer:
            configure_wav(writer)
            writer.writeframes(encode_pcm(samples))

    def test_bass_shelf_boosts_low_tones_without_changing_duration(self):
        self.write_tone(80)
        original = measure_wav(self.source)["rms_dbfs"]
        equalize_wav(self.source, self.target, 6, 0)
        result = measure_wav(self.target)
        self.assertGreater(result["rms_dbfs"], original + 4)
        self.assertEqual(result["frames"], SAMPLE_RATE // 4)

    def test_treble_shelf_boosts_high_tones(self):
        self.write_tone(8000)
        original = measure_wav(self.source)["rms_dbfs"]
        equalize_wav(self.source, self.target, 0, 6)
        self.assertGreater(measure_wav(self.target)["rms_dbfs"], original + 4)

    def test_equalizer_prevents_clipping(self):
        self.write_tone(100, amplitude=30000)
        equalize_wav(self.source, self.target, 12, 0)
        metrics = measure_wav(self.target)
        self.assertEqual(metrics["clipped_samples"], 0)
        self.assertLessEqual(measure_wav(self.target)["peak_dbfs"], 0)

    def test_zero_gain_is_bit_identical(self):
        self.write_tone(220)
        equalize_wav(self.source, self.target, 0, 0)
        self.assertEqual(self.target.read_bytes(), self.source.read_bytes())

    def test_invalid_gains_are_rejected(self):
        self.write_tone(220)
        for bass, treble in ((13, 0), (0, -13), (math.inf, 0), (0, math.nan)):
            with self.subTest(bass=bass, treble=treble):
                with self.assertRaisesRegex(VoxRefineError, "between -12 and 12 dB"):
                    equalize_wav(self.source, self.target, bass, treble)

    def test_wrapper_records_equalizer_and_processes_audio(self):
        self.write_tone(80)
        engine = EqualizedEngine(CopyEngine(), 3, -2)
        self.assertEqual(engine.identity()["bass_db"], 3)
        self.assertEqual(engine.identity()["treble_db"], -2)
        engine.process(self.source, self.target)
        self.assertEqual(inspect_wav(self.target).frames, SAMPLE_RATE // 4)
        self.assertNotEqual(self.source.read_bytes(), self.target.read_bytes())

    def test_cli_applies_optional_equalization(self):
        self.write_tone(80)
        with patch("voxrefine.cli.DeepFilterNet", return_value=CopyEngine()), \
             patch("voxrefine.cli.RNNoise", return_value=CopyEngine()), \
             patch("voxrefine.cli.ProtectedDeepFilterNet", return_value=CopyEngine()):
            with redirect_stdout(io.StringIO()):
                code = main([
                    "clean", str(self.source), str(self.target),
                    "--bass-db", "3", "--treble-db", "-2",
                ])
        self.assertEqual(code, 0)
        self.assertEqual(inspect_wav(self.target).frames, SAMPLE_RATE // 4)
        self.assertNotEqual(self.source.read_bytes(), self.target.read_bytes())

    def test_cli_benchmark_records_equalizer_settings(self):
        self.write_tone(80)
        manifest = self.root / "corpus.json"
        manifest.write_text(json.dumps({"samples": [{
            "id": "voice",
            "path": self.source.name,
            "category": "synthetic",
            "rights": "Generated test audio",
        }]}))
        output = self.root / "benchmark"
        with patch("voxrefine.cli.DeepFilterNet", return_value=CopyEngine()), \
             patch("voxrefine.cli.RNNoise", return_value=CopyEngine()), \
             patch("voxrefine.cli.ProtectedDeepFilterNet", return_value=CopyEngine()):
            with redirect_stdout(io.StringIO()):
                code = main([
                    "benchmark", str(manifest), "--output", str(output),
                    "--engines", "deepfilter",
                    "--bass-db", "2.5", "--treble-db", "-1.5",
                ])
        self.assertEqual(code, 0)
        engine = json.loads((output / "report.json").read_text())["engines"][0]
        self.assertEqual(engine["bass_db"], 2.5)
        self.assertEqual(engine["treble_db"], -1.5)
