# Shared Adobe-curve EQ screen — 2026-10-09

## Purpose

Fit one stationary EQ to the average active-speech curve of local paired Resemble and Adobe Podcast v2 renders. The screen asks whether a shared, smooth correction can move the spectrum toward Adobe without learning an EQ for each individual recording.

![Fixed-gain same-source spectra before and after the shared EQ](../../results/adobe-curve-eq-2026-10-09/shared-eq-curves.png)

## Data and validation

Seven paired files were used: one synthetic controlled-noise clip, four short Christiane conditions (two dry controls and two RIR-only conditions), and two Naf conditions (one dry control and one RIR-only). There are only three distinct voice groups. Speaker-group leave-one-out cross-validation selected among 3-, 5-, and 7-bell RBJ equalizers with different smoothness and gain penalties. A simpler model was preferred when its validation score was within 0.02 dB of the best.

The selected 5-band model uses fixed, time-invariant peaking filters. In the worker it is applied after the modest legacy C trim (−1.5 dB above 5.5 kHz) that was already present in these paired candidate renders:

| Center | Q | Gain |
|---:|---:|---:|
| 180 Hz | 0.65 | +1.67 dB |
| 500 Hz | 0.65 | +0.16 dB |
| 1.5 kHz | 0.65 | −1.35 dB |
| 3.5 kHz | 0.65 | −2.50 dB |
| 8 kHz | 0.65 | −2.50 dB |

The curve MAE is computed from 1/6-octave-smoothed active-speech power. One constant gain matches each candidate and Adobe render to its dry reference's integrated LUFS; the speech mask comes from the known dry stem. Candidate audio files in the results folder are separately level-matched to the baseline candidate for listening.

## Results

| Speaker group | Baseline mean MAE | Shared EQ mean MAE | Pairs |
|---|---:|---:|---:|
| Controlled synthetic voice | 3.82 dB | 1.71 dB | 1 |
| Christiane | 4.12 dB | 2.49 dB | 4 |
| Naf | 5.27 dB | 4.05 dB | 2 |
| All pairs, median | 3.82 dB | 2.86 dB | 7 |

The score improves on six of seven individual pairs. The Naf dry-control pair gets worse (2.98 → 4.53 dB), while the Naf RIR-only pair improves (7.57 → 3.57 dB). This is evidence that matching average spectral shape is useful, but a universal fixed curve cannot fit every voice and condition equally well.

The fit cannot reproduce Adobe's time-varying noise removal, echo suppression, or speech-dependent decisions. Nor does lower spectral MAE alone imply better perceived audio. The result remains an experimental preset and is not the default profile.

## Actual pipeline A/B

The three profiles were then run end-to-end on the same 6 s controlled input with Resemble NFE16, 3 s chunks, dynamics off, de-esser off, and −2.5 dB output gain. Only the tone profile differed. All three took about 62–65 s on CPU (RTF about 10.4–10.9); the GTX 1050 Ti run failed while loading the model with CUDA out of memory, so GPU performance is not established.

| Actual CLI profile | Curve MAE vs Adobe | 4–8 kHz delta | 8–12 kHz delta |
|---|---:|---:|---:|
| Flat | 4.34 dB | +4.57 dB | +3.04 dB |
| C | 3.40 dB | +3.42 dB | +1.48 dB |
| **adobe-curve** | **1.77 dB** | **+0.67 dB** | **−0.50 dB** |

![Actual NFE16 CLI curves against Adobe v2](../../results/adobe-curve-eq-2026-10-09/actual-cli-spectra.png)

This shows the fixed EQ materially closes the average spectrum gap on this clip. It does not certify perceived quality or remove time-varying artifacts from the denoiser. A 4.2 kHz narrow notch sweep was also tested across all seven pairs; it worsened average MAE, so it was rejected.

Actual-pipeline listening files, all matched to the C-profile loudness:

- [Flat](../../results/adobe-curve-eq-2026-10-09/actual-flat-levelmatched.mp3)
- [C](../../results/adobe-curve-eq-2026-10-09/actual-C-levelmatched.mp3)
- [adobe-curve](../../results/adobe-curve-eq-2026-10-09/actual-adobe_curve-levelmatched.mp3)
- [Adobe v2](../../results/adobe-curve-eq-2026-10-09/actual-adobe-levelmatched.mp3)

## Listening files

All three files for each pair (baseline, shared EQ, Adobe) are WAV and 192 kbps MP3 in `results/adobe-curve-eq-2026-10-09/`. Example controlled pair:

- [Baseline](../../results/adobe-curve-eq-2026-10-09/controlled-pauses-snr10-baseline.mp3)
- [Shared EQ](../../results/adobe-curve-eq-2026-10-09/controlled-pauses-snr10-curve-eq.mp3)
- [Adobe v2](../../results/adobe-curve-eq-2026-10-09/controlled-pauses-snr10-adobe.mp3)

Held-out Naf dry pair, where the EQ worsens the mean curve distance:

- [Baseline](../../results/adobe-curve-eq-2026-10-09/naf-event-2-dry-control-baseline.mp3)
- [Shared EQ](../../results/adobe-curve-eq-2026-10-09/naf-event-2-dry-control-curve-eq.mp3)
- [Adobe v2](../../results/adobe-curve-eq-2026-10-09/naf-event-2-dry-control-adobe.mp3)

## Reproduction

```bash
.tools/resemble-venv/bin/python scripts/benchmarks/universal_enhancer/fit_adobe_curve_eq.py
```

Raw measurements and fold scores: `results/adobe-curve-eq-2026-10-09/report.json`, `metrics.csv`, and `curves.csv`. The profile is available for local experiments with `studio-resemble --tone adobe-curve`; `C` remains the default.
