from contextlib import redirect_stderr
from array import array
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import wave

from tests.test_voxrefine import CopyEngine, write_wav
from voxrefine.audio import (
    VoxRefineError, configure_wav, encode_pcm, inspect_wav, measure_wav,
)
from voxrefine.benchmark import benchmark
from voxrefine.conversion import prepared_audio
from voxrefine.engines import DeepFilterNet, RNNoise, clean, file_hash
from voxrefine.protection import ProtectedDeepFilterNet


class ConversionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.source = self.root / "voice.wav"
        write_wav(self.source)

    def test_compatible_wav_needs_no_ffmpeg(self):
        with prepared_audio(self.source, "missing-ffmpeg") as prepared:
            self.assertEqual(prepared, self.source)

    def test_missing_ffmpeg_is_actionable(self):
        write_wav(self.source, rate=44100)
        with self.assertRaisesRegex(VoxRefineError, "Install FFmpeg"):
            with prepared_audio(self.source, "missing-ffmpeg"):
                self.fail("Conversion unexpectedly succeeded")

    def test_missing_input_is_explicit(self):
        with self.assertRaisesRegex(VoxRefineError, "not found"):
            with prepared_audio(self.root / "missing.mp3"):
                self.fail("Missing input unexpectedly succeeded")

    def test_output_must_be_wav(self):
        with self.assertRaisesRegex(VoxRefineError, ".wav extension"):
            clean(self.source, self.root / "output.mp3", CopyEngine())

    def test_existing_output_rejected_before_conversion(self):
        write_wav(self.source, rate=44100)
        target = self.root / "output.wav"
        target.write_bytes(b"keep me")
        with self.assertRaisesRegex(VoxRefineError, "already exists"):
            clean(self.source, target, CopyEngine(), "missing-ffmpeg")
        self.assertEqual(target.read_bytes(), b"keep me")


@unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is required for conversion integration tests.")
class FFmpegTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.source = self.root / "source.wav"
        write_wav(self.source, [
            round(6000 * math.sin(2 * math.pi * 300 * index / 48000))
            for index in range(48000)
        ])

    def encode(self, name, options):
        path = self.root / name
        subprocess.run([
            "ffmpeg", "-nostdin", "-loglevel", "error", "-n",
            "-i", str(self.source), *options, str(path),
        ], check=True, capture_output=True)
        return path

    def test_real_formats_conversion_and_cleanup(self):
        cases = [
            ("stereo 44k.wav", ["-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le"]),
            ("24bit.wav", ["-c:a", "pcm_s24le"]),
            ("float.wav", ["-c:a", "pcm_f32le"]),
            ("voice.mp3", ["-c:a", "libmp3lame"]),
            ("voice.m4a", ["-c:a", "aac"]),
        ]
        for name, options in cases:
            with self.subTest(format=name):
                source = self.encode(name, options)
                before = file_hash(source)
                stderr = io.StringIO()
                with redirect_stderr(stderr):
                    with prepared_audio(source) as prepared:
                        info = inspect_wav(prepared)
                        self.assertEqual(info.sample_rate, 48000)
                        self.assertEqual(info.channels, 2)
                        self.assertAlmostEqual(info.seconds, 1.0, delta=0.1)
                        self.assertIsNotNone(measure_wav(prepared)["rms_dbfs"])
                    self.assertFalse(prepared.exists())
                    target = self.root / f"{source.name}-clean.wav"
                    clean(source, target, CopyEngine())
                self.assertEqual(inspect_wav(target).frames, info.frames)
                self.assertEqual(file_hash(source), before)
                self.assertIn("stereo PCM16", stderr.getvalue())

    def test_invalid_audio_has_no_output(self):
        source = self.root / "broken.mp3"
        source.write_bytes(b"not audio")
        target = self.root / "out.wav"
        with redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(VoxRefineError, "Process exited"):
                clean(source, target, CopyEngine())
        self.assertFalse(target.exists())

    def test_benchmark_prepares_once_and_keeps_reference(self):
        source = self.encode("voice.m4a", ["-ac", "2", "-ar", "44100", "-c:a", "aac"])
        manifest = self.root / "corpus.json"
        manifest.write_text(json.dumps({"samples": [{
            "id": "voice", "path": source.name,
            "category": "synthetic", "rights": "Generated test signal",
        }]}))
        with redirect_stderr(io.StringIO()) as stderr:
            report = benchmark(manifest, self.root / "results", [CopyEngine()])
        self.assertEqual(stderr.getvalue().count("Converting"), 1)
        data = json.loads(report.read_text())
        self.assertEqual(data["status"], "complete")
        sample = data["samples"][0]
        reference = report.parent / sample["prepared_input_path"]
        self.assertEqual(sample["prepared_input_sha256"], file_hash(reference))
        self.assertEqual(sample["input_sha256"], file_hash(source))
        self.assertEqual(
            sample["input"]["frames"], sample["outputs"][0]["audio"]["frames"]
        )
        self.assertEqual(sample["input"]["channels"], 2)
        self.assertEqual(sample["outputs"][0]["audio"]["channels"], 2)

    def test_bad_conversion_marks_benchmark_failed(self):
        source = self.root / "broken.m4a"
        source.write_bytes(b"not audio")
        manifest = self.root / "corpus.json"
        manifest.write_text(json.dumps({"samples": [{
            "id": "broken", "path": source.name,
            "category": "invalid", "rights": "Generated invalid data",
        }]}))
        output = self.root / "results"
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(VoxRefineError):
                benchmark(manifest, output, [CopyEngine()])
        report = json.loads((output / "report.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertIn("broken/input", report["error"])

    @unittest.skipUnless(
        os.environ.get("VOXREFINE_DEEP_FILTER") and os.environ.get("VOXREFINE_RNNOISE"),
        "Native engines are required.",
    )
    def test_mp3_through_both_native_engines(self):
        source = self.encode("voice.mp3", ["-c:a", "libmp3lame"])
        with redirect_stderr(io.StringIO()):
            with prepared_audio(source) as prepared:
                expected = inspect_wav(prepared).frames
            for engine in [
                DeepFilterNet(os.environ["VOXREFINE_DEEP_FILTER"]),
                RNNoise(os.environ["VOXREFINE_RNNOISE"]),
            ]:
                with self.subTest(engine=engine.name):
                    target = self.root / f"{engine.name}.wav"
                    clean(source, target, engine)
                    self.assertEqual(inspect_wav(target).frames, expected)

    @unittest.skipUnless(
        os.environ.get("VOXREFINE_DEEP_FILTER") and os.environ.get("VOXREFINE_RNNOISE"),
        "Native engines are required.",
    )
    def test_stereo_through_protected_deepfilter(self):
        mono_samples = [
            round(4000 * math.sin(2 * math.pi * 220 * index / 48000))
            for index in range(4800)
        ]
        stereo = self.root / "stereo.wav"
        with wave.open(str(stereo), "wb") as writer:
            configure_wav(writer, channels=2)
            samples = array("h", (
                sample for value in mono_samples for sample in (value, round(value * 0.7))
            ))
            writer.writeframes(encode_pcm(samples))
        output = self.root / "stereo-clean.wav"
        engine = DeepFilterNet(os.environ["VOXREFINE_DEEP_FILTER"])
        protected = ProtectedDeepFilterNet(
            engine, RNNoise(os.environ["VOXREFINE_RNNOISE"])
        )
        clean(stereo, output, protected)
        result = inspect_wav(output)
        self.assertEqual(result.channels, 2)
        self.assertEqual(result.frames, len(mono_samples))
        with wave.open(str(output), "rb") as reader:
            interleaved = array("h", reader.readframes(reader.getnframes()))
        self.assertNotEqual(interleaved[0::2], interleaved[1::2])
