from __future__ import annotations

import importlib.util
import unittest

try:
    import numpy as np
    from voxrefine.toneshape import apply_deesser, apply_gentle_compression, apply_peaking_eq, apply_shelves
except ImportError:
    np = None
    apply_shelves = None
    apply_gentle_compression = None
    apply_deesser = None
    apply_peaking_eq = None


@unittest.skipUnless(importlib.util.find_spec("numpy") and importlib.util.find_spec("scipy"),
                     "tone shelves require the optional NumPy/SciPy runtime")
class ToneShelfTests(unittest.TestCase):
    rate = 48_000

    def tone(self, hz: float, duration: float = 1.0) -> np.ndarray:
        t = np.arange(round(self.rate * duration)) / self.rate
        return np.sin(2 * np.pi * hz * t).astype(np.float32)

    def steady_gain_db(self, signal: np.ndarray, output: np.ndarray) -> float:
        start = self.rate // 10
        return float(20 * np.log10(np.sqrt(np.mean(output[start:] ** 2)) /
                                   np.sqrt(np.mean(signal[start:] ** 2))))

    def test_bass_shelf_cuts_subbass_and_preserves_upper_band(self) -> None:
        low = self.tone(5, 2.0)
        high = self.tone(2_000)
        low_out = apply_shelves(low, self.rate, bass_db=-3.0, bass_corner_hz=100)
        high_out = apply_shelves(high, self.rate, bass_db=-3.0, bass_corner_hz=100)
        self.assertAlmostEqual(self.steady_gain_db(low, low_out), -3.0, delta=0.08)
        self.assertAlmostEqual(self.steady_gain_db(high, high_out), 0.0, delta=0.08)

    def test_treble_shelf_cuts_high_band_and_preserves_low_band(self) -> None:
        low = self.tone(100)
        high = self.tone(18_000, 2.0)
        low_out = apply_shelves(low, self.rate, treble_db=-2.5, treble_corner_hz=3_500)
        high_out = apply_shelves(high, self.rate, treble_db=-2.5, treble_corner_hz=3_500)
        self.assertAlmostEqual(self.steady_gain_db(low, low_out), 0.0, delta=0.08)
        self.assertAlmostEqual(self.steady_gain_db(high, high_out), -2.5, delta=0.08)

    def test_preserves_length_finiteness_and_untrimmed_signal(self) -> None:
        source = self.tone(440, .25)
        result = apply_shelves(source, self.rate)
        self.assertEqual(len(result), len(source))
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_allclose(result, source, atol=1e-7)

    def test_rejects_stereo_and_invalid_gain(self) -> None:
        with self.assertRaises(ValueError):
            apply_shelves(np.zeros((100, 2)), self.rate)
        with self.assertRaises(ValueError):
            apply_shelves(np.zeros(100), self.rate, bass_db=-13)

    def test_gentle_compressor_reduces_loud_burst_not_quiet_speech(self) -> None:
        quiet = self.tone(220) * 0.03
        loud = self.tone(220) * 0.7
        source = np.concatenate([quiet, loud, quiet]).astype(np.float32)
        result, max_reduction = apply_gentle_compression(source, self.rate)
        self.assertEqual(len(result), len(source))
        self.assertTrue(np.isfinite(result).all())
        self.assertGreater(max_reduction, 1.0)
        quiet_gain = self.steady_gain_db(source[:self.rate // 2], result[:self.rate // 2])
        loud_gain = self.steady_gain_db(source[self.rate: self.rate + self.rate // 2],
                                        result[self.rate: self.rate + self.rate // 2])
        self.assertAlmostEqual(quiet_gain, 0.0, delta=0.2)
        self.assertLess(loud_gain, -1.0)

    def test_deesser_reduces_only_loud_sibilance_and_never_boosts(self) -> None:
        t = np.arange(self.rate) / self.rate
        quiet = 0.005 * np.sin(2 * np.pi * 6_000 * t)
        loud = 0.5 * np.sin(2 * np.pi * 6_000 * t)
        source = np.concatenate([quiet, loud]).astype(np.float32)
        output, maximum = apply_deesser(source, self.rate)
        quiet_gain = self.steady_gain_db(source[:self.rate // 2], output[:self.rate // 2])
        loud_gain = self.steady_gain_db(source[self.rate + self.rate // 4:],
                                        output[self.rate + self.rate // 4:])
        self.assertEqual(len(output), len(source))
        self.assertTrue(np.isfinite(output).all())
        self.assertAlmostEqual(quiet_gain, 0.0, delta=0.1)
        self.assertLess(loud_gain, -0.5)
        self.assertGreaterEqual(maximum, -1e-6)

    def test_peaking_eq_has_stationary_expected_center_gain(self) -> None:
        t = np.arange(self.rate * 2) / self.rate
        source = np.sin(2 * np.pi * 3_500 * t).astype(np.float32)
        result = apply_peaking_eq(source, self.rate, ((3_500, 0.65, -2.5),))
        measured = self.steady_gain_db(source, result)
        self.assertAlmostEqual(measured, -2.5, delta=0.1)
        self.assertEqual(len(result), len(source))
        self.assertTrue(np.isfinite(result).all())


if __name__ == "__main__":
    unittest.main()
