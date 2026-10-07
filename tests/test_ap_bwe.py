"""Optional AP-BWE parity checks; weights and upstream source stay local."""

import os
from pathlib import Path
import unittest

from voxrefine.ap_bwe import APBWE16to48


@unittest.skipUnless(
    os.environ.get("VOXREFINE_APBWE_SOURCE") and os.environ.get("VOXREFINE_APBWE_CHECKPOINT"),
    "Set local AP-BWE source and checkpoint paths to run model checks.",
)
class APBWEModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = APBWE16to48(
            Path(os.environ["VOXREFINE_APBWE_SOURCE"]),
            Path(os.environ["VOXREFINE_APBWE_CHECKPOINT"]),
            device="cpu", threads=2, chunk_frames=512,
        )

    def test_chunked_synthesis_matches_full_sequence(self):
        import numpy as np
        import torchaudio.functional as audio_functional
        import torch

        signal = np.random.default_rng(1701).normal(0, .01, 80000).astype(np.float32)
        chunked = self.engine.process_samples(signal)
        audio = torch.from_numpy(signal).reshape(1, -1)
        wide = audio_functional.resample(audio, orig_freq=16000, new_freq=48000)
        with torch.inference_mode():
            magnitude, phase, _ = self.engine._stft(
                wide, self.engine.config.n_fft, self.engine.config.hop_size,
                self.engine.config.win_size,
            )
            full_mag, full_phase, _ = self.engine.model(magnitude, phase)
            full = self.engine._istft(
                full_mag, full_phase, self.engine.config.n_fft,
                self.engine.config.hop_size, self.engine.config.win_size,
            )[0].numpy()
        full = full[:len(chunked)]
        self.assertEqual(len(chunked), len(signal) * 3)
        self.assertTrue(np.allclose(chunked, full, rtol=1e-5, atol=2e-6))

    def test_repeated_inference_is_deterministic(self):
        import numpy as np

        signal = np.random.default_rng(83).normal(0, .01, 12000).astype(np.float32)
        first = self.engine.process_samples(signal)
        second = self.engine.process_samples(signal)
        self.assertTrue(np.array_equal(first, second))


if __name__ == "__main__":
    unittest.main()
