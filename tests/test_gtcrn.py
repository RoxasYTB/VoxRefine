"""GTCRN optional backend contracts; no model download or user audio required."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from voxrefine.audio import AudioDomain, AudioInfo, AudioSource, VoxRefineError
from voxrefine.conversion import prepared_for_backend
from voxrefine.ffmpeg_io import probe_audio
from voxrefine.gtcrn import GTCRN, GTCRNSession


class GTCRNDependencyTests(unittest.TestCase):
    def test_optional_runtime_error_is_actionable(self):
        with patch("voxrefine.gtcrn._optional_runtime", side_effect=VoxRefineError("missing optional deps")):
            with self.assertRaisesRegex(VoxRefineError, "missing optional deps"):
                GTCRN(Path("model.onnx"))


@unittest.skipUnless(os.environ.get("VOXREFINE_GTCRN_MODEL"), "Set VOXREFINE_GTCRN_MODEL to run ONNX integration checks.")
class GTCRNModelTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.np = np
        self.model = Path(os.environ["VOXREFINE_GTCRN_MODEL"])

    def run_chunks(self, chunks):
        session = GTCRNSession(self.model)
        output = b"".join(session.process(chunk) for chunk in chunks)
        output += session.flush()
        self.addCleanup(session.close)
        return self.np.frombuffer(output, dtype="<f4").copy(), session

    def test_chunk_boundaries_do_not_change_stream(self):
        samples = self.np.random.default_rng(15).normal(0, .01, 16000).astype(self.np.float32)
        chunks_a = [samples]
        chunks_b = [samples[i:i+137] for i in range(0, len(samples), 137)]
        out_a, state_a = self.run_chunks(chunks_a)
        out_b, state_b = self.run_chunks(chunks_b)
        self.assertEqual(len(out_a), len(samples))
        self.assertEqual(len(out_b), len(samples))
        self.assertTrue(self.np.allclose(out_a, out_b, atol=1e-7, rtol=1e-6))
        self.assertEqual(state_a.frames, state_b.frames)
        self.assertEqual(state_a.output_emitted, len(samples))

    def test_centered_output_is_exact_for_hop_aligned_and_partial_lengths(self):
        for length in (256, 257, 512, 1000, 16000):
            with self.subTest(length=length):
                samples = self.np.random.default_rng(length).normal(0, .01, length).astype(self.np.float32)
                output, session = self.run_chunks([samples])
                self.assertEqual(len(output), length)
                self.assertEqual(session.output_emitted, length)

    def test_a_b_a_runs_are_independent(self):
        rng = self.np.random.default_rng(52)
        a = rng.normal(0, .01, 4096).astype(self.np.float32)
        b = rng.normal(0, .01, 6144).astype(self.np.float32)
        out_a1, _ = self.run_chunks([a])
        self.run_chunks([b])
        out_a2, _ = self.run_chunks([a])
        self.assertTrue(self.np.array_equal(out_a1, out_a2))

    def test_upstream_fixture_prefix_parity(self):
        mix_path = os.environ.get("VOXREFINE_GTCRN_REFERENCE_MIX")
        reference_path = os.environ.get("VOXREFINE_GTCRN_REFERENCE_OUTPUT")
        if not mix_path or not reference_path:
            self.skipTest("Set upstream mix/output fixture paths to run numerical parity.")

        def read_mono_16k(path):
            raw = subprocess.check_output([
                "ffmpeg", "-v", "error", "-i", path, "-f", "f32le",
                "-acodec", "pcm_f32le", "-ac", "1", "-ar", "16000", "-",
            ])
            return self.np.frombuffer(raw, dtype="<f4").copy()

        source = read_mono_16k(mix_path)
        session = GTCRNSession(self.model)
        result = b""
        for offset in range(0, len(source), 4096):
            result += session.process(source[offset:offset + 4096])
        result += session.flush()
        actual = self.np.frombuffer(result, dtype="<f4")
        expected = read_mono_16k(reference_path)
        self.assertEqual(len(actual), len(source))
        common = min(len(actual), len(expected))
        actual = actual[:common]
        expected = expected[:common]
        error = actual - expected
        correlation = self.np.corrcoef(actual, expected)[0, 1]
        rmse = self.np.sqrt(self.np.mean(error ** 2))
        self.assertGreaterEqual(correlation, 0.9999)
        self.assertLessEqual(rmse, 5e-4)
        self.assertLessEqual(float(self.np.max(self.np.abs(error))), 1e-2)
        for start in range(0, common, 16000):
            window = error[start:start + 16000]
            if window.size:
                self.assertLessEqual(float(self.np.sqrt(self.np.mean(window ** 2))), 1e-3)

    def test_one_session_state_does_not_leak_across_files(self):
        samples = self.np.zeros(7000, dtype=self.np.float32)
        first, _ = self.run_chunks([samples])
        second, _ = self.run_chunks([samples])
        self.assertTrue(self.np.array_equal(first, second))
