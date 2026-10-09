"""Low-latency first-order shelving filters used by the optional tone stage."""

from __future__ import annotations

from math import tan

import numpy as np


def _shelf(audio: np.ndarray, cutoff_hz: float, sample_rate: int,
           gain_db: float, *, low: bool) -> np.ndarray:
    if not 0.0 < cutoff_hz < sample_rate / 2:
        raise ValueError("Shelf corner must be between 0 Hz and Nyquist.")
    from scipy.signal import lfilter, lfilter_zi

    k = tan(np.pi * cutoff_hz / sample_rate)
    gain = 10.0 ** (gain_db / 20.0)
    den = np.asarray([1.0, (k - 1.0) / (k + 1.0)], dtype=np.float64)
    if low:
        num = np.asarray([(1.0 + gain * k) / (1.0 + k), (gain * k - 1.0) / (1.0 + k)])
    else:
        num = np.asarray([(gain + k) / (1.0 + k), (k - gain) / (1.0 + k)])
    initial_state = lfilter_zi(num, den) * audio[0]
    return lfilter(num, den, audio, zi=initial_state)[0]


def apply_shelves(
    audio: np.ndarray,
    sample_rate: int,
    *,
    bass_db: float = 0.0,
    bass_corner_hz: float = 100.0,
    treble_db: float = 0.0,
    treble_corner_hz: float = 3_500.0,
) -> np.ndarray:
    """Apply causal mono shelves with zero look-ahead and preserved sample count.

    The bass shelf changes frequencies below its corner while tending to unity
    above it. The treble shelf changes frequencies above its corner while
    tending to unity below it. A first-order transition is intentionally mild;
    this utility is a tone control, not a voice/noise separator.
    """
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("Tone shelves require a non-empty mono signal.")
    if sample_rate <= 0 or not np.isfinite(x).all():
        raise ValueError("Audio must be finite and sample rate must be positive.")
    if not -12.0 <= bass_db <= 12.0 or not -12.0 <= treble_db <= 12.0:
        raise ValueError("Shelf gains must be between -12 and +12 dB.")

    y = x.copy()
    if bass_db:
        y = _shelf(y, bass_corner_hz, sample_rate, bass_db, low=True)
    if treble_db:
        y = _shelf(y, treble_corner_hz, sample_rate, treble_db, low=False)
    return y.astype(np.float32)
