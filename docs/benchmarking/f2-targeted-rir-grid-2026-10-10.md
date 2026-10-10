# Full-exact strong-reverb probe — 2026-10-10

## Question

The second frozen F2 tail slot had no eligible candidates among 256 procedural
RIRs using `RT60 U[0.45, 1.10] s` and `DRR U[-6, 18] dB`. Its best surrogate
scores were `−53.49 dB` (W1) and `−60.11 dB` (W2), well below the rejection
boundary. This probe asks whether a longer and more reverb-dominant RIR can
produce measurable post-Cap60 tails for that fixed voice.

## Frozen method

- Development source: frozen schedule row 5; one LibriSpeech speaker, selected
  before this probe. No holdout or `test.wav` data was used.
- Grid: `RT60 ∈ {1.10, 1.40, 1.70} s` × `DRR ∈ {−3, −6, −9, −12} dB`.
- One seed was reused across the full grid to hold reflection/noise geometry
  fixed while varying only RT60 and DRR.
- Each of 12 points received full-utterance Cap60 on both wet and paired clean
  branches. There was no prefix or scale surrogate.
- Eligibility stayed fixed: both full-exact windows must be strictly above
  `−50 dB`.

## Results

| RT60 (s) | DRR (dB) | W1 (dB) | W2 (dB) | Eligible |
|---:|---:|---:|---:|:---:|
| 1.1 | −3 | −57.57 | −63.62 | no |
| 1.1 | −6 | −54.27 | −60.06 | no |
| 1.1 | −9 | −51.25 | −56.74 | no |
| 1.1 | −12 | −48.31 | −53.82 | no |
| 1.4 | −3 | −55.83 | −61.44 | no |
| 1.4 | −6 | −52.34 | −57.86 | no |
| 1.4 | −9 | −49.59 | −54.46 | no |
| 1.4 | −12 | −46.64 | −51.37 | no |
| 1.7 | −3 | −53.01 | −59.16 | no |
| 1.7 | −6 | −49.81 | −55.84 | no |
| 1.7 | −9 | −47.62 | −52.44 | no |
| 1.7 | −12 | −44.64 | −49.27 | yes |

Only `1/12` points passed, at the most extreme tested combination. The
predeclared `3/12` gate therefore fails. W2 is the limiting window; changing
the threshold would conceal that rather than solve it. This single voice and
one random geometry cannot support a new global RIR distribution.

The probe completed 24 full-exact Cap60 runs in `144.06 s`. It did not modify
the F2 training cache. The optional RT60/DRR bounds added to
`make_procedural_rir` default to the former `[0.25, 1.20] s` and `[-6, 18] dB`
ranges; the extended bounds are used only by explicit probe scripts.

## Reproducibility

- Raw metrics and exact source/model hashes:
  [`report.json`](assets/f2-targeted-rir-grid-2026-10-10/report.json)
- Probe runner:
  `scripts/benchmarks/universal_enhancer/compact_dereverb/probe_f2_targeted_rir_grid.py`
- Every report records `test_wav_accessed: false`, `surrogate_used: false`, and
  `selection_or_training_data_modified: false`.
