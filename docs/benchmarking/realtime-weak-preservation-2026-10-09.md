# Weak-first real-time candidate screen — 2026-10-09

## Frozen comparison

Reuses exactly the seven-speaker/three-event frozen inputs from weak-speech-backend-heldout-02. DPDFNet2 causal native 48 kHz was rendered locally; DFN18/100 and Resemble denoiser-only are reused from the same frozen input set. All projections use exact 250 ms attenuated speech/noise stems. Macro summaries median the three events within speaker first, then weight speakers equally.

The same 126 exact inputs were used by every listed method. This screen adds 126 DPDFNet2 causal 48 kHz CPU renders; it does not re-render DFN or Resemble. They total 378 seconds of audio and 131.63 seconds of measured inference wall time (aggregate RTF 0.348); the older manifest omitted model load time, so this is not end-to-end startup or microphone latency. Adobe v2 is not present on these inputs.

| Noise / SNR | Backend | Voice loss median / speaker p90 / worst (dB) | Events >6 dB | Noise reduction median (dB) | Projected SNR median (dB) | RTF median |
|---|---|---:|---:|---:|---:|---:|
| classroom-babble / 10 dB | Entrée | -0.00 / 0.00 / 0.00 | 0.0% | 0.00 | 18.47 | n.d. |
| classroom-babble / 10 dB | DPDFNet2 48 kHz | 0.13 / 0.21 / 0.23 | 0.0% | 25.13 | 48.66 | 0.342 |
| classroom-babble / 10 dB | DFN cap18 | 0.19 / 1.17 / 2.13 | 0.0% | 1.46 | 21.58 | 0.414 |
| classroom-babble / 10 dB | DFN cap100* | 0.22 / 1.36 / 2.49 | 0.0% | 1.69 | 21.76 | 0.408 |
| classroom-babble / 10 dB | Resemble denoiser-only | -0.00 / -0.00 / 0.00 | 0.0% | 0.30 | 21.36 | 0.065 |
| crowd-babble-cc0 / 10 dB | Entrée | -0.00 / 0.00 / 0.00 | 0.0% | 0.00 | 19.70 | n.d. |
| crowd-babble-cc0 / 10 dB | DPDFNet2 48 kHz | 0.18 / 0.28 / 0.35 | 0.0% | 9.10 | 29.79 | 0.338 |
| crowd-babble-cc0 / 10 dB | DFN cap18 | 0.43 / 1.97 / 2.50 | 0.0% | 8.93 | 27.35 | 0.413 |
| crowd-babble-cc0 / 10 dB | DFN cap100* | 0.50 / 2.30 / 2.92 | 0.0% | 11.52 | 30.06 | 0.414 |
| crowd-babble-cc0 / 10 dB | Resemble denoiser-only | 0.02 / 0.04 / 0.06 | 0.0% | 7.21 | 27.53 | 0.066 |
| fan-motor-cc0 / 10 dB | Entrée | -0.00 / 0.00 / 0.00 | 0.0% | 0.00 | 19.18 | n.d. |
| fan-motor-cc0 / 10 dB | DPDFNet2 48 kHz | 0.16 / 0.20 / 0.21 | 0.0% | 14.52 | 37.89 | 0.341 |
| fan-motor-cc0 / 10 dB | DFN cap18 | 0.26 / 1.11 / 1.58 | 0.0% | 6.75 | 26.75 | 0.417 |
| fan-motor-cc0 / 10 dB | DFN cap100* | 0.29 / 1.29 / 1.83 | 0.0% | 8.36 | 29.50 | 0.412 |
| fan-motor-cc0 / 10 dB | Resemble denoiser-only | -0.00 / 0.01 / 0.01 | 0.0% | 14.85 | 36.31 | 0.065 |
| synthetic-pink / 10 dB | Entrée | -0.00 / -0.00 / 0.00 | 0.0% | -0.00 | 19.89 | n.d. |
| synthetic-pink / 10 dB | DPDFNet2 48 kHz | 0.15 / 0.19 / 0.19 | 0.0% | 14.99 | 34.37 | 0.337 |
| synthetic-pink / 10 dB | DFN cap18 | 0.28 / 0.87 / 1.60 | 0.0% | 7.21 | 27.12 | 0.416 |
| synthetic-pink / 10 dB | DFN cap100* | 0.32 / 1.01 / 1.85 | 0.0% | 9.00 | 29.23 | 0.412 |
| synthetic-pink / 10 dB | Resemble denoiser-only | 0.00 / 0.02 / 0.03 | 0.0% | 6.60 | 28.49 | 0.066 |

*DFN cap100 is retained only as an aggressive historical comparator. Positive noise reduction means lower projected noise component than the input stem.*

## Survie des fenêtres faibles DPDFNet2

Les fenêtres sont définies depuis le stem vocal propre : trames actives puis quartile de plus faible RMS. Les niveaux sont comparés aux trames propres alignées; les agrégats ci-dessous sont des médianes des événements, sans pondérer davantage un locuteur. La valeur « trames exact-floor » est comptée dans tous les événements du bloc.

| Bruit / SNR | p10 niveau faible (dB) | p1 (dB) | Trames <−40 dB | Corrélation enveloppe faible | Trames exact-floor |
|---|---:|---:|---:|---:|---:|
| classroom-babble / 10 dB | -1.58 | -5.08 | 0.16% | 0.946 | 0 |
| crowd-babble-cc0 / 10 dB | -2.98 | -7.26 | 0.15% | 0.900 | 0 |
| fan-motor-cc0 / 10 dB | -2.02 | -5.53 | 0.00% | 0.922 | 0 |
| synthetic-pink / 10 dB | -3.62 | -6.17 | 0.15% | 0.908 | 0 |
| crowd-babble-cc0 / 5 dB | -6.26 | -17.51 | 1.08% | 0.787 | 0 |
| fan-motor-cc0 / 5 dB | -2.40 | -6.37 | 0.00% | 0.905 | 0 |

DPDFNet2 n'a aucune trame faible sous −80 dB ni au plancher dans ces six cellules. Le stress foule à 5 dB se dégrade toutefois : p10 −6,26 dB et corrélation 0,787; le risque de parole faible sur le babble dense reste visible, même si la projection voix totale n'indique pas d'événement >6 dB de perte. Les cellules expérimentales restent distinctes, et ces métriques de niveau ne mesurent pas l'identité des phonèmes.

## Figure

![Voice preservation against projected noise suppression](weak-preservation-pareto.png)

## Candidate gates from GPT Web

Use weak-frame survival first: clean-defined event mask; report p10/p1, fractions below −20/−40/−80 dB, exact-floor count, and weak-zone envelope correlation. The three Adobe pairs are a secondary behavior check only (candidate envelope correlation ≥ Adobe −0.03, weak p10 no more than 6 dB below Adobe, and no >40 dB collapse if Adobe has none).

The full proposed primary gates are recorded in the Adobe v2 plan: catastrophic-collapse zero tolerance; weak-event p10 ≥−24 dB at speaker level, no speaker below −30 dB, speaker p90 event loss ≤6 dB and no event above 10 dB; normal-voice median event loss <1 dB, speaker p90 <3 dB, envelope median ≥0.90; noise suppression medians ≥8 dB on chatter10 and ≥12 dB on stationary10 with ≥75% cases improving; early-target p90 loss <3 dB and measurable late-tail reduction ≥8 dB (medium RIR) / ≥10 dB (long RIR). No composite score.

## Decisions and limits

- This is a useful held-out weak-event screen, not proof of Adobe parity or universal quality.
- The exact local runtime has no DeepFilterNet cap6, so this screen cannot choose between cap6 and cap18. No product default changes are justified by this screen alone.
- DPDFNet2 runtime is CPU-only, causal, native 48 kHz. Its median RTF is recorded per case and is a throughput metric, not an end-to-end microphone latency guarantee.
- Adobe v2 comparisons remain limited to the separate 3-voice matched cohort; all Adobe comparisons use original mix outputs, not these heldout renders.

## Reproduction

```bash
.venv/bin/python scripts/benchmarks/universal_enhancer/analyze_realtime_weak_preservation.py
.venv/bin/python scripts/benchmarks/universal_enhancer/audit_adobe_v2_pairs.py
```

Audio renders are ignored local files under `results/noise-rir-truth-01/realtime-weak-preservation-01/`.
