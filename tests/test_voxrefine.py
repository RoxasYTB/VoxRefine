from array import array
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
import unittest
from unittest.mock import patch
import wave

from voxrefine import __version__
from voxrefine.audio import (
    VoxRefineError, configure_wav, encode_pcm, inspect_wav, measure_wav
)
from voxrefine.benchmark import benchmark, read_corpus
from voxrefine.cli import main
from voxrefine.engines import DeepFilterNet, RNNoise, clean


def write_wav(path, samples=None, rate=48000, channels=1):
    if samples is None:
        samples = [1000] * 481
    with wave.open(str(path), "wb") as writer:
        configure_wav(writer)
        writer.setframerate(rate)
        writer.setnchannels(channels)
        writer.writeframes(encode_pcm(array("h", samples)))


class CopyEngine:
    name = "copy"

    def identity(self):
        return {"engine": self.name, "version": "test-only"}

    def process(self, source, target):
        shutil.copyfile(source, target)


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / "source.wav"
        write_wav(self.source)

    def manifest(self, **overrides):
        record = {
            "id": "test", "path": self.source.name,
            "category": "synthetic", "rights": "Generated for this test",
        }
        record.update(overrides)
        path = self.root / "corpus.json"
        path.write_text(json.dumps({"samples": [record]}), encoding="utf-8")
        return path

    def test_wav_validation(self):
        self.assertEqual(inspect_wav(self.source).frames, 481)
        for rate, channels in [(44100, 1)]:
            with self.subTest(rate=rate, channels=channels):
                write_wav(self.source, rate=rate, channels=channels)
                with self.assertRaisesRegex(VoxRefineError, "48000 Hz"):
                    inspect_wav(self.source)
        write_wav(self.source, channels=2, samples=[100, -100] * 481)
        self.assertEqual(inspect_wav(self.source).channels, 2)
        self.assertEqual(measure_wav(self.source)["channels"], 2)

    def test_clean_preserves_stereo_channels(self):
        samples = [value for _ in range(481) for value in (1200, -700)]
        write_wav(self.source, samples=samples, channels=2)
        target = self.root / "stereo-output.wav"
        clean(self.source, target, CopyEngine())
        self.assertEqual(inspect_wav(target).channels, 2)
        with wave.open(str(target), "rb") as reader:
            self.assertEqual(reader.readframes(481), encode_pcm(array("h", samples)))

    def test_empty_and_invalid_audio(self):
        write_wav(self.source, [])
        with self.assertRaisesRegex(VoxRefineError, "empty"):
            inspect_wav(self.source)
        self.source.write_bytes(b"not a wav")
        with self.assertRaisesRegex(VoxRefineError, "invalid PCM"):
            inspect_wav(self.source)

    def test_truncated_audio(self):
        data = self.source.read_bytes()
        self.source.write_bytes(data[:-2])
        with self.assertRaisesRegex(VoxRefineError, "truncated"):
            inspect_wav(self.source)
        with self.assertRaisesRegex(VoxRefineError, "truncated"):
            measure_wav(self.source)

    def test_measurements_read_the_file_once(self):
        with patch("voxrefine.audio.wave.open", wraps=wave.open) as opener:
            metrics = measure_wav(self.source)
        self.assertEqual(opener.call_count, 1)
        self.assertEqual(metrics["frames"], 481)

    def test_silence_is_json_safe(self):
        write_wav(self.source, [0] * 480)
        metrics = measure_wav(self.source)
        self.assertIsNone(metrics["rms_dbfs"])
        self.assertIsNone(metrics["peak_dbfs"])
        json.dumps(metrics, allow_nan=False)

    def test_clipping_and_level(self):
        write_wav(self.source, [-32768, 32767, 0])
        metrics = measure_wav(self.source)
        self.assertEqual(metrics["clipped_samples"], 2)
        self.assertEqual(metrics["peak_dbfs"], 0)

    def test_clean_preserves_input_and_refuses_overwrite(self):
        target = self.root / "nested" / "out.wav"
        original = self.source.read_bytes()
        clean(self.source, target, CopyEngine())
        self.assertEqual(target.read_bytes(), original)
        self.assertEqual(self.source.read_bytes(), original)
        with self.assertRaisesRegex(VoxRefineError, "already exists"):
            clean(self.source, target, CopyEngine())
        with self.assertRaises(VoxRefineError):
            clean(self.source, self.source, CopyEngine())

    def test_engine_failure_does_not_publish_output(self):
        class FailingEngine(CopyEngine):
            def process(self, source, target):
                target.write_bytes(b"partial")
                raise VoxRefineError("engine failed")

        target = self.root / "out.wav"
        with self.assertRaisesRegex(VoxRefineError, "engine failed"):
            clean(self.source, target, FailingEngine())
        self.assertFalse(target.exists())
        self.assertFalse(list(self.root.glob(".voxrefine-*")))

    def test_duration_change_is_rejected(self):
        class ShortEngine(CopyEngine):
            def process(self, source, target):
                write_wav(target, [0])

        with self.assertRaisesRegex(VoxRefineError, "duration"):
            clean(self.source, self.root / "out.wav", ShortEngine())

    def test_manifest_relative_paths_and_rights(self):
        records = read_corpus(self.manifest())
        self.assertEqual(records[0][1], self.source)
        for overrides in ({"rights": ""}, {"id": "../escape"}, {"path": 1}):
            with self.subTest(overrides=overrides):
                with self.assertRaises(VoxRefineError):
                    read_corpus(self.manifest(**overrides))

    def test_manifest_shape_errors(self):
        manifest = self.root / "bad.json"
        for payload in ("[1]", '{"samples": []}', "not json"):
            manifest.write_text(payload)
            with self.subTest(payload=payload):
                with self.assertRaises(VoxRefineError):
                    read_corpus(manifest)

    def test_duplicate_ids(self):
        manifest = self.manifest()
        data = json.loads(manifest.read_text())
        data["samples"] *= 2
        manifest.write_text(json.dumps(data))
        with self.assertRaisesRegex(VoxRefineError, "Duplicate"):
            read_corpus(manifest)

    def test_benchmark_report(self):
        output = self.root / "results"
        report = benchmark(self.manifest(), output, [CopyEngine()])
        data = json.loads(report.read_text())
        self.assertEqual(data["status"], "complete")
        self.assertEqual(data["voxrefine_version"], __version__)
        result = data["samples"][0]["outputs"][0]
        self.assertGreater(result["real_time_factor"], 0)
        self.assertEqual(result["audio"]["frames"], 481)
        self.assertEqual(len(result["output_sha256"]), 64)
        with self.assertRaises(FileExistsError):
            benchmark(self.manifest(), output, [CopyEngine()])

    def test_benchmark_rejects_duplicate_engines(self):
        with self.assertRaisesRegex(VoxRefineError, "duplicates"):
            benchmark(self.manifest(), self.root / "results", [CopyEngine(), CopyEngine()])

    def test_benchmark_failure_is_explicit_in_report(self):
        class FailingEngine(CopyEngine):
            def process(self, source, target):
                raise VoxRefineError("intentional failure")

        output = self.root / "results"
        with self.assertRaises(VoxRefineError):
            benchmark(self.manifest(), output, [FailingEngine()])
        report = json.loads((output / "report.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertIn("intentional failure", report["error"])

    def test_cli_reports_missing_engine(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = main([
                "clean", str(self.source), str(self.root / "out.wav"),
                "--engine", "rnnoise", "--rnnoise-library", "missing.so",
            ])
        self.assertEqual(code, 1)
        self.assertIn("RNNoise library not found", stderr.getvalue())

    def test_cli_defaults_to_protected_deepfilter(self):
        with patch("voxrefine.cli.DeepFilterNet", return_value=CopyEngine()), \
             patch("voxrefine.cli.RNNoise", return_value=CopyEngine()), \
             patch("voxrefine.cli.ProtectedDeepFilterNet", return_value=CopyEngine()) as protected:
            with redirect_stdout(io.StringIO()):
                code = main([
                    "clean", str(self.source), str(self.root / "out.wav"),
                ])
        self.assertEqual(code, 0)
        protected.assert_called_once()

    def test_cli_can_request_standard_deepfilter_without_detector(self):
        with patch("voxrefine.cli.DeepFilterNet", return_value=CopyEngine()) as engine:
            with redirect_stdout(io.StringIO()):
                code = main([
                    "clean", str(self.source), str(self.root / "out.wav"),
                    "--no-protect-voice",
                ])
        self.assertEqual(code, 0)
        engine.assert_called_once_with(".tools/deep-filter", attenuation_limit_db=100.0)

    def test_cli_default_engine_paths(self):
        with patch("voxrefine.cli.DeepFilterNet", return_value=CopyEngine()) as primary, \
             patch("voxrefine.cli.RNNoise", return_value=CopyEngine()) as detector, \
             patch("voxrefine.cli.ProtectedDeepFilterNet", return_value=CopyEngine()):
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main([
                    "clean", str(self.source), str(self.root / "out.wav"),
                ]), 0)
        primary.assert_called_once_with(".tools/deep-filter", attenuation_limit_db=100.0)
        detector.assert_called_once_with(".tools/librnnoise.so")

    def test_missing_native_engines(self):
        with self.assertRaises(VoxRefineError):
            DeepFilterNet(str(self.root / "missing"))
        with self.assertRaises(VoxRefineError):
            RNNoise(str(self.root / "missing.so"))

    def test_attenuation_limit_validation(self):
        for limit in (-1, 101, float("nan"), float("inf"), -float("inf")):
            with self.subTest(limit=limit):
                with self.assertRaisesRegex(VoxRefineError, "between 0 and 100"):
                    DeepFilterNet("missing", attenuation_limit_db=limit)
        for limit in (0, 12, 20, 100):
            with self.subTest(limit=limit):
                with patch("voxrefine.engines.shutil.which", return_value=str(self.source)):
                    with patch("voxrefine.engines.run_checked", return_value="deep_filter 0.5.6"):
                        engine = DeepFilterNet("mock", attenuation_limit_db=limit)
                self.assertEqual(engine.identity()["attenuation_limit_db"], limit)

    def test_cli_passes_attenuation_limit(self):
        for flag, expected in (
            (["--no-protect-voice"], 100.0),
            (["--no-protect-voice", "--attenuation-limit-db", "12"], 12.0),
        ):
            with self.subTest(expected=expected):
                target = self.root / f"out-{expected}.wav"
                with patch("voxrefine.cli.DeepFilterNet", return_value=CopyEngine()) as factory:
                    with redirect_stdout(io.StringIO()):
                        self.assertEqual(main([
                            "clean", str(self.source), str(target),
                            *flag,
                        ]), 0)
                    factory.assert_called_once_with(".tools/deep-filter", attenuation_limit_db=expected)

    def test_cli_rejects_limit_for_rnnoise_only(self):
        for command in (
            ["clean", str(self.source), str(self.root / "out.wav"), "--engine", "rnnoise"],
            ["benchmark", str(self.manifest()), "--output", str(self.root / "results"),
             "--engines", "rnnoise"],
        ):
            with redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main([*command, "--attenuation-limit-db", "12"])
            self.assertEqual(error.exception.code, 2)

    def test_cli_benchmark_passes_limit(self):
        with patch("voxrefine.cli.DeepFilterNet", return_value=CopyEngine()) as factory, \
             patch("voxrefine.cli.RNNoise", return_value=CopyEngine()), \
             patch("voxrefine.cli.ProtectedDeepFilterNet", return_value=CopyEngine()):
            with redirect_stdout(io.StringIO()):
                code = main([
                    "benchmark", str(self.manifest()), "--output", str(self.root / "results"),
                    "--engines", "deepfilter", "--rnnoise-library", "mock-rnnoise",
                    "--attenuation-limit-db", "20",
                ])
        self.assertEqual(code, 0)
        factory.assert_called_once_with(".tools/deep-filter", attenuation_limit_db=20.0)

    def test_cli_profiles_for_clean_and_benchmark(self):
        for profile, expected in (("balanced", 20.0), ("strong", 100.0)):
            for command in ("clean", "benchmark"):
                with self.subTest(profile=profile, command=command):
                    target = self.root / f"{command}-{profile}"
                    arguments = (
                        ["clean", str(self.source), str(target.with_suffix(".wav")),
                             "--rnnoise-library", "mock-rnnoise"]
                        if command == "clean"
                        else ["benchmark", str(self.manifest()), "--output", str(target),
                                  "--engines", "deepfilter", "--rnnoise-library", "mock-rnnoise"]
                    )
                    with patch("voxrefine.cli.DeepFilterNet", return_value=CopyEngine()) as factory, \
                         patch("voxrefine.cli.RNNoise", return_value=CopyEngine()), \
                         patch("voxrefine.cli.ProtectedDeepFilterNet", return_value=CopyEngine()):
                        with redirect_stdout(io.StringIO()):
                            self.assertEqual(main([*arguments, "--profile", profile]), 0)
                        factory.assert_called_once_with(
                            ".tools/deep-filter", attenuation_limit_db=expected
                        )

    def test_cli_natural_profile_requires_standard_mode(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                main([
                        "clean", str(self.source), str(self.root / "out.wav"),
                        "--profile", "natural",
                ])
        self.assertEqual(error.exception.code, 2)

    def test_cli_profiles_cannot_be_ignored_or_combined_with_limit(self):
        cases = (
            ["--engine", "rnnoise", "--profile", "natural"],
            ["--engine", "deepfilter", "--profile", "natural",
             "--attenuation-limit-db", "20"],
            ["--engine", "deepfilter", "--profile", "invalid"],
        )
        for options in cases:
            with self.subTest(options=options), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main([
                        "clean", str(self.source), str(self.root / "out.wav"),
                        "--rnnoise-library", "mock-rnnoise", *options,
                    ])
                self.assertEqual(error.exception.code, 2)

    def test_rnnoise_invalid_library_is_explicit(self):
        with self.assertRaisesRegex(VoxRefineError, "Cannot load RNNoise"):
            RNNoise(str(self.source))

    def test_newer_rnnoise_abi_is_rejected(self):
        with patch("voxrefine.engines.ctypes.CDLL"):
            with self.assertRaisesRegex(VoxRefineError, "v0.1 ABI"):
                RNNoise(str(self.source))


@unittest.skipUnless(
    os.environ.get("VOXREFINE_DEEP_FILTER") and os.environ.get("VOXREFINE_RNNOISE"),
    "Set VOXREFINE_DEEP_FILTER and VOXREFINE_RNNOISE to run real native engines.",
)
class NativeTests(unittest.TestCase):
    def test_native_limit_is_forwarded_and_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "signal.wav"
            write_wav(source, [
                round(3000 * math.sin(2 * math.pi * 180 * i / 48000))
                for i in range(48013)
            ])
            manifest = root / "corpus.json"
            manifest.write_text(json.dumps({"samples": [{
                "id": "signal", "path": source.name,
                "category": "synthetic", "rights": "Generated test signal",
            }]}))
            outputs = []
            for limit in (0, 12, 100):
                engine = DeepFilterNet(
                    os.environ["VOXREFINE_DEEP_FILTER"], attenuation_limit_db=limit
                )
                report = benchmark(manifest, root / f"results-{limit}", [engine])
                data = json.loads(report.read_text())
                self.assertEqual(data["engines"][0]["attenuation_limit_db"], limit)
                target = report.parent / data["samples"][0]["outputs"][0]["path"]
                self.assertEqual(inspect_wav(target).frames, 48013)
                outputs.append(target.read_bytes())
            self.assertNotEqual(outputs[0], outputs[1])
            self.assertNotEqual(outputs[1], outputs[2])

    def test_real_engines_preserve_exact_length_and_silence(self):
        engines = [
            DeepFilterNet(os.environ["VOXREFINE_DEEP_FILTER"]),
            RNNoise(os.environ["VOXREFINE_RNNOISE"]),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for engine in engines:
                for length in (1, 479, 480, 481, 48000, 48013):
                    with self.subTest(engine=engine.name, length=length):
                        source = root / "source.wav"
                        target = root / f"{engine.name}-{length}.wav"
                        write_wav(source, [0] * length)
                        clean(source, target, engine)
                        self.assertEqual(inspect_wav(target).frames, length)
                        self.assertIsNone(measure_wav(target)["rms_dbfs"])

    def test_real_engines_produce_nonempty_audio_and_benchmark(self):
        random_generator = random.Random(42)
        samples = [
            int(
                3000 * math.sin(2 * math.pi * 180 * index / 48000)
                + random_generator.uniform(-1000, 1000)
            )
            for index in range(48013)
        ]
        engines = [
            DeepFilterNet(os.environ["VOXREFINE_DEEP_FILTER"]),
            RNNoise(os.environ["VOXREFINE_RNNOISE"]),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_wav(root / "signal.wav", samples)
            manifest = root / "corpus.json"
            manifest.write_text(json.dumps({"samples": [{
                "id": "signal", "path": "signal.wav",
                "category": "synthetic-not-speech",
                "rights": "Generated diagnostic signal, no recording used",
            }]}))
            report = benchmark(manifest, root / "results", engines)
            outputs = json.loads(report.read_text())["samples"][0]["outputs"]
            self.assertEqual(len(outputs), 2)
            for result in outputs:
                self.assertEqual(result["audio"]["frames"], len(samples))
                self.assertIsNotNone(result["audio"]["rms_dbfs"])
