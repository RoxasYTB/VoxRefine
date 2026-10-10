# F2 strong RIR distribution preflight (2026-10-10)

## Decision

The frozen full-exact preflight **fails**. Do not use this distribution to build the 48-slot F2 tailbank or start training. Queue-level coverage and speech-retention checks pass, but all three pre-registered shared-scale gates fail.

This is a distribution-level rejection, not a candidate-selection rule. All 64 cases were evaluated and retained in the report; none were selected for training.

## Frozen protocol

- Eight previously unprobed RIR-only schedule rows, selected before measurement: `7, 10, 15, 18, 20, 23, 27, 31` (speakers `6563, 696, 8123, 3112, 6181, 1743, 6925, 7367`).
- Eight deterministic independent draws per row: `RT60 ~ U[1.65, 1.85] s`, `DRR ~ U[-14.5, -12.5] dB`.
- Full Cap60 inference on both wet and clean target for every case. No surrogate, candidate ranking, or training-data modification.
- Tail eligibility remains unchanged: both measured windows must be strictly above `-50 dB`.
- Scale checks were frozen before execution: median `20 log10(shared_scale) > -6 dB`, p10 `> -12 dB`, and at most 6/64 scales below `0.25`.

## Results

| Measure | Result | Gate |
|---|---:|---|
| Full-exact cases / Cap60 runs | 64 / 128 | — |
| Tail eligible | 56/64 (87.5%) | Pass (>=48) |
| Eligible by schedule row | 7: 8; 10: 8; 15: 8; 18: 7; 20: 8; 23: 5; 27: 8; 31: 4 | Pass (>=6 on >=6 rows; none <4) |
| Active delta, median / p10 | +4.854 / -2.763 dB | Pass (>-6 / >-12 dB) |
| Target-speech retention, min / median / max | 0.9656 / 0.9740 / 0.9812 | Pass (all >=0.05) |
| Clipped cases | 0/64; peak 0.4749 | Pass |
| Shared scale, median / p10 (linear) | 0.17383 / 0.13917 | — |
| Shared scale, median / p10 (dB) | -15.197 / -17.133 dB | **Fail** (>-6 / >-12 dB) |
| Shared scale below 0.25 | 53/64 (82.8%) | **Fail** (<=6/64) |
| Total elapsed time | 770.42 s | — |

The scale failure is substantial: the shared scale is between 0.1196 and 0.3099 across these cases, so the strong-tail candidates generally require roughly 10–18 dB attenuation before Cap60. The frozen aggregate gates therefore reject this proposed global distribution even though the Cap60 tail criterion and target-retention checks pass.

## Artifacts and integrity

- Machine-readable results: `.tools/compact-dereverb/f2-strong-rir-preflight-2026-10-10/report.json` (local ignored benchmark artifact).
- Runner: [`preflight_f2_strong_rir_distribution.py`](../../scripts/benchmarks/universal_enhancer/compact_dereverb/preflight_f2_strong_rir_distribution.py).
- No training or selection manifest was modified. The protected `/home/yohan/Bureau/test.wav` was not accessed (`test_wav_accessed: false`).

This preflight does not claim perceptual parity with Adobe Podcast. It establishes that the proposed procedural RIR distribution is not acceptable under the agreed amplitude-integrity gates. The next experiment should be chosen after reviewing the complete measurements; do not relax the gates post hoc.
