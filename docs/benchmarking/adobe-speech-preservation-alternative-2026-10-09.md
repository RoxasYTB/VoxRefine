# Speech-preservation regression and realtime alternative — 2026-10-09

## Decision

The shared five-band EQ moved the average spectrum closer to Adobe, but the user reported lost words and crackling. We therefore withdrew `adobe-curve` from the Resemble CLI/backend/worker. A closer average curve does not make damaged speech acceptable.

The next candidate is the existing 48 kHz streaming DPDFNet2 path. On this exact controlled crop it preserved the clean-reference intelligibility proxy and strongly suppressed the known gaps, while running on CPU. Treat this as a candidate for broader validation, not proof of universal or Adobe-equivalent quality.

## Same-input comparison

All renders use the same six-second, 48 kHz LibriVox speech crop mixed with classroom noise at nominal +10 dB SNR. The clean stem and an Adobe Podcast v2 render for the exact crop are available. Resemble NFE16/32/64 were rerun on CPU with tone flat, dynamics/de-esser disabled, and the same chunk settings. DPDFNet2 used the local 48 kHz ONNX streaming backend on CPU, with the project's wall-clock paced virtual device. Metrics use a constant loudness match to the clean stem. STOI uses the best single global delay within ±100 ms; no time warping is applied.

| Render | STOI, best global delay | Best delay | Median active-frame level | Median gap-frame level | Total runtime / 6 s |
|---|---:|---:|---:|---:|---:|
| Noisy input | 0.987 | 0 ms | −28.87 dBFS | −35.70 dBFS | — |
| Resemble NFE16, flat | 0.675 | 5.8 ms | −31.27 dBFS | −45.54 dBFS | 64.97 s |
| Resemble NFE32, flat | 0.780 | 5.8 ms | −31.03 dBFS | −43.90 dBFS | 90.23 s |
| Resemble NFE64, flat | 0.716 | 19.4 ms | −30.63 dBFS | −43.79 dBFS | 102.55 s |
| DPDFNet2 stream | **0.990** | 0 ms | −30.97 dBFS | **−94.23 dBFS** | 6.00 s wall-clock paced |
| Adobe Podcast v2 | 0.942 | 0.02 ms | −30.01 dBFS | −69.94 dBFS | Reference render |

The gap metric is a median 10 ms RMS on frames marked non-speech from the known clean stem, after LUFS matching; it includes breaths and mask leakage. It is not a direct noise power or quality score. The DPDFNet2 run had zero underruns, zero dropped blocks, and exact 6.000 s output duration. Its worker service time was 3.71 ms median, 5.29 ms p95, 6.43 ms p99, and 10.16 ms maximum per 10 ms hop. This is one virtual-device run, not a hard-realtime or physical-device latency guarantee.

### Interpretation

- Raising Resemble from NFE16 to NFE32 improved STOI on this crop, but NFE64 did not. All three remained far below the noisy input and Adobe. More inference steps are not a repair for this failure mode.
- DPDFNet2 preserved the known speech best in this screen and left substantially less energy in known pauses than Adobe. Its active-speech median is about 2.1 dB below the noisy input after whole-file loudness matching; a level/control adjustment may be useful, but should not be confused with restoring removed phonemes.
- Adobe has lower STOI than the unprocessed input in this example, showing why STOI alone cannot rank perceived enhancement. It only flags possible speech alteration.
- The curve EQ benchmark's mean spectral MAE improved, but that objective win did not capture the user's reported word loss/crackle. The product profile is withdrawn until it passes intelligibility and perceptual checks.
- DPDFNet2's near-real-time CPU behavior and existing 48 kHz streaming implementation fit the project's constrained-hardware goal better than the tested Resemble offline path. Validate next on more voices, noise types, dry controls, and measured room responses before considering a product default.

## Listening and figures

Every listening file is normalized to −24 LUFS. The same-source comparison graph includes 10 ms envelopes, speech-active spectra and NFE16/DPDFNet2 spectrograms.

![Speech-preservation comparison: envelope, active speech spectrum, and time-frequency output](../../results/adobe-curve-eq-2026-10-09/speech-preservation-comparison.png)

The files below are ignored local benchmark outputs, not committed dataset assets:

- Metrics: `results/adobe-curve-eq-2026-10-09/speech-preservation-report.json` and `speech-preservation-metrics.csv`.
- DPDFNet2 timing/stream report: `results/adobe-curve-eq-2026-10-09/controlled-dpdfnet2-live-sim.json`.
- Audio: `results/adobe-curve-eq-2026-10-09/speech-preservation-listening/`.
- Reproduction: `.tools/resemble-venv/bin/python scripts/benchmarks/universal_enhancer/compare_conservative_realtime_candidate.py`.

The reproduction script requires `pystoi` in its Python environment. The input and outputs are paired for this one crop only; the report is not a substitute for a multi-speaker, multi-condition evaluation or listening.
