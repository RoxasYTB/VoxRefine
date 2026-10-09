# Treble balance against Adobe Podcast v2 — 2026-10-09

## Question

The listener reported audible treble harshness and little perceived change. This screen isolates tone shaping on the exact 6 s controlled pair: the known dry LibriVox stem, the current Resemble NFE16-C output, and Adobe Podcast v2. It does not rerun either enhancer.

## Result

![Static treble balance comparison](../../results/treble-balance-ab-2026-10-09/treble-balance-spectra.png)

| Candidate | 1/6-octave spectrum MAE vs Adobe (20 Hz–12 kHz) | 4–8 kHz delta | 8–12 kHz delta |
|---|---:|---:|---:|
| Current NFE16-C | 3.25 dB | +3.94 dB | +1.96 dB |
| Treble shelf −1.5 dB, 4 kHz corner | 2.62 dB | +3.20 dB | +0.98 dB |
| **Treble shelf −2.5 dB, 4 kHz corner** | **2.28 dB** | **+2.73 dB** | **+0.33 dB** |
| Treble shelf −3.5 dB, 4 kHz corner | 2.38 dB | +2.29 dB | −0.31 dB |

The −2.5 dB setting is the best of these four tested curves. It moves the upper octave close to Adobe without over-cutting it. The 4–8 kHz band remains somewhat higher than Adobe, suggesting further refinement should target the speech-presence/sibilance region with a better-designed dynamic processor rather than applying a much larger broadband shelf.

## Waveform and spectral plot

![Treble curves and residuals](../../results/treble-balance-ab-2026-10-09/treble-balance-spectra.png)

## Listening files

All candidates are 6 seconds and rendered at the same integrated loudness as the current NFE16-C output:

- A: current output
- B: −1.5 dB shelf
- C: −2.5 dB shelf
- D: −3.5 dB shelf
- E: Adobe v2 reference

Files are in `results/treble-balance-ab-2026-10-09/` as PCM WAV and 192 kbps MP3. The full-curve plot and raw metrics are alongside them.

## Measurement protocol

- Same source clip and same duration, truncated to common length.
- Speech mask from the known dry stem: 10 ms RMS, threshold max−35 dB, expanded ±60 ms.
- Each candidate gets a single constant gain to the dry stem integrated LUFS for spectral comparison.
- The displayed spectrum is 1/6-octave-smoothed active-speech power. MAE is a shape descriptor, not an objective perceptual quality score.
- Listening outputs separately use constant gain to match the current candidate's integrated LUFS.
- Script: `scripts/benchmarks/universal_enhancer/render_treble_balance_ab.py`.

## Limits

One voice, one synthetic classroom-noise mixture at nominal 10 dB SNR, one Adobe version/slider setting. Do not treat this as universal validation. A split-band de-esser prototype was evaluated and rejected: it shifted global spectral distance substantially, and a fixed absolute threshold was not robust enough. The experimental `--deesser gentle` option remains off by default.
