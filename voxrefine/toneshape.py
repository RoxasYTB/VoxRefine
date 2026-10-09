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


def apply_gentle_compression(
    audio: np.ndarray,
    sample_rate: int,
    *,
    threshold_db: float = -16.0,
    ratio: float = 1.5,
    knee_db: float = 6.0,
    attack_ms: float = 10.0,
    release_ms: float = 120.0,
) -> tuple[np.ndarray, float]:
    """Gently reduce only sustained loud peaks; return audio and max reduction.

    A centered RMS detector uses half-window lookahead (10 ms at the defaults).
    This is a finishing compressor, not a denoiser or a substitute for a limiter.
    """
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError("Compression requires a non-empty finite mono signal.")
    if sample_rate <= 0 or not -60.0 <= threshold_db <= 0.0:
        raise ValueError("Invalid sample rate or compressor threshold.")
    if not 1.0 <= ratio <= 10.0 or not 0.0 <= knee_db <= 24.0:
        raise ValueError("Compressor ratio must be 1–10 and knee 0–24 dB.")
    if attack_ms <= 0.0 or release_ms <= 0.0:
        raise ValueError("Compressor attack and release must be positive.")

    from scipy.ndimage import uniform_filter1d

    window = max(1, round(sample_rate * 0.020))
    envelope = np.sqrt(uniform_filter1d(x * x, size=window, mode="nearest") + 1e-12)
    level_db = 20.0 * np.log10(np.maximum(envelope, 1e-12))
    over = level_db - threshold_db
    half_knee = knee_db / 2.0
    if knee_db:
        reduction_db = np.where(
            over <= -half_knee,
            0.0,
            np.where(
                over >= half_knee,
                over * (1.0 - 1.0 / ratio),
                ((over + half_knee) ** 2 / (2.0 * knee_db)) * (1.0 - 1.0 / ratio),
            ),
        )
    else:
        reduction_db = np.maximum(over, 0.0) * (1.0 - 1.0 / ratio)
    target = 10.0 ** (-reduction_db / 20.0)
    gain = np.empty_like(target)
    state = 1.0
    attack = np.exp(-1.0 / (sample_rate * attack_ms / 1000.0))
    release = np.exp(-1.0 / (sample_rate * release_ms / 1000.0))
    for index, target_gain in enumerate(target):
        coefficient = attack if target_gain < state else release
        state = coefficient * state + (1.0 - coefficient) * target_gain
        gain[index] = state
    return (x * gain).astype(np.float32), float(np.max(-20.0 * np.log10(np.maximum(gain, 1e-12))))
