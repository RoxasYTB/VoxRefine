# Adobe v2 student distillation: residual A/B result

## Decision

**NO-GO for the current local Adobe-output corpus.** The residual-structured student protected weak speech better than the naive student, but it did not improve the pre-registered teacher-distance gate. A final phase-aware audit found mixed results: the direct-control was closer to Adobe in paired waveform and complex-STFT distance on all three speakers, while removing little estimated nuisance; the structured residual failed to transfer consistently to Stéphanie. No checkpoint is integrated.

This is a small black-box imitation study. It does not recover Adobe's internal algorithm and does not demonstrate studio quality or perceptual equivalence.

## Protocol

- Three speaker-strict leave-one-speaker-out folds: Emy, Rémi, and Stéphanie. The held-out speaker is never in training.
- Fifteen locally available source→Adobe pairs, balanced by domain→pair→2 s crop. Pair alignment passed a five-window envelope-lag check; no time warping was applied.
- 24 kHz mono; same 69,638-parameter U-Net trunk; 5,000 updates per fold and arm, paired crop schedule and seed.
- Arms: pooled direct-output control; residual-naive (`r = x − Adobe`); residual-structured, adding clean-derived speech-dominance and weak/onset protections where a reliable clean stem exists.
- The user's `test.wav` and Adobe render were not opened. Adobe exports, WAVs, and checkpoints remain local. Their terms for training and derivative weights have not been verified.

## Teacher distance

Values are 1 kHz log-magnitude STFT MAE after fixed gain normalization to the clean active-speech level; lower is closer. `Gain = d(input, Adobe) − d(student, Adobe)`, so positive would be an improvement over input.

| Held-out speaker | Input→Adobe | Direct→Adobe | Naive residual→Adobe | Structured residual→Adobe | Structured gain vs input |
|---|---:|---:|---:|---:|---:|
| Emy | 6.850 dB | 7.086 dB | 8.019 dB | 7.373 dB | −0.523 dB |
| Rémi | 7.262 dB | 7.392 dB | 8.436 dB | 8.006 dB | −0.744 dB |
| Stéphanie | 7.444 dB | 7.633 dB | 8.253 dB | 7.489 dB | −0.046 dB |

The pre-registered gate required at least +0.20 dB on every fold and +0.30 dB median, while preserving weak-frame and onset p10 above −3 dB. Teacher-distance failed on all three folds. The structured residual did preserve weak-frame p10 at −1.24/−2.41/−1.91 dB and onset p10 at −1.24/−1.61/−1.30 dB. The naive residual's weak p10 was −12.34/−8.07/−9.49 dB, showing that teacher imitation can remove speech.

## Phase-aware and nuisance audit

The final analysis used existing held-out WAVs only. Waveform L1 is normalized by Adobe RMS without candidate-specific gain fitting. Complex-STFT L1 is normalized by Adobe complex magnitude and averaged over FFT sizes 256/512/1024/2048. The estimated nuisance is `n = input − clean`; bins are marked nuisance-dominant when `q = |C|²/(|C|²+|N|²) < 0.1`. Positive percentages mean the candidate's distance to Adobe is lower than the input's.

| Speaker | Arm | Waveform L1 improvement vs input | Complex-STFT improvement vs input | Removed/noise energy (dB) | Complex cosine, noise-dominant bins |
|---|---|---:|---:|---:|---:|
| Emy | Direct | +13.3% | +23.3% | −11.3 dB | +0.88 |
| Emy | Naive residual | +44.4% | +60.9% | −1.2 dB | +0.95 |
| Emy | Structured residual | +11.7% | +25.6% | −3.0 dB | +0.80 |
| Rémi | Direct | +19.8% | +22.3% | −12.1 dB | +0.90 |
| Rémi | Naive residual | +32.1% | +45.1% | −3.9 dB | +0.96 |
| Rémi | Structured residual | +11.1% | +23.9% | −3.9 dB | +0.80 |
| Stéphanie | Direct | +20.9% | +27.0% | −12.5 dB | +0.84 |
| Stéphanie | Naive residual | −13.6% | +20.9% | −1.7 dB | +0.97 |
| Stéphanie | Structured residual | −10.8% | −15.0% | −5.7 dB | −0.39 |

The direct-control improves waveform and complex-STFT distance on all three folds, but removes little nuisance energy. The naive residual often looks numerically closest to Adobe and captures more nuisance, while losing weak speech. The structured residual keeps the weak/onset guard on all speakers, but fails both distance metrics on Stéphanie and its removed component points away from the estimated nuisance in her noise-dominant bins. Thus spectral or waveform proximity alone would reward undesirable speech loss; no arm passes the combined quality gate.

These metrics answer different questions. The earlier log-magnitude distance normalizes each candidate to clean active level and ignores phase; the phase-aware audit compares paired native-level waveforms. Neither is a perceptual score. The subtraction `input − clean` is only a nuisance proxy and depends on exact pair alignment. Three held-out voices are not enough to claim universality.

## Runtime and artifacts

- Training: median about 274 s per fold for the structured arm on the local GTX 1050 Ti.
- Warm inference: median about 57.5 ms for 15 s audio (RTF about 0.0038); the first direct run included initialization and measured 337 ms.
- Peak allocated GPU memory: about 321 MiB.
- No audio, teacher output, or checkpoint is included in this repository report.

The scripts are [`adobe_student_residual_ab.py`](../../scripts/benchmarks/universal_enhancer/adobe_student_residual_ab.py) and [`analyze_adobe_student_residual_metrics.py`](../../scripts/benchmarks/universal_enhancer/analyze_adobe_student_residual_metrics.py). They operate on local, ignored files under `results/adobe-student-residual-ab-2026-10-11/`. Raw per-fold data and plots remain local. Reproduce the final metric audit with:

```bash
.tools/resemble-cuda-venv/bin/python \
  scripts/benchmarks/universal_enhancer/analyze_adobe_student_residual_metrics.py
```

## Next direction

Do not run another student training job on these same pairs. Continue the open-source pipeline with known clean references and controlled measured/procedural room responses. Keep denoising and dereverberation as independently measured stages; validate them on held-out speakers and room responses. Any future Adobe-output distillation requires a substantially broader, independently licensed corpus and separate terms review.
