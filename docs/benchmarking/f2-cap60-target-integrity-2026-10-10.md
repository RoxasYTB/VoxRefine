# F2 strong RIR target integrity (2026-10-10)

## Decision

**Do not train F2 on the current `target = Cap60(clean)` pairs.** The pre-registered integrity gates fail because Cap60 removes too much energy from weak speech frames. A cached attribution probe identifies Cap60 itself, rather than the pair's shared attenuation, as the main source of that loss. The next experiment should change the dereverb target; it should not select RIRs around this failure.

No training data or training run was created from these probes. The earlier scale-gate result remains recorded as a failure; it was not retrospectively reclassified.

## Experiments and results

### Distribution preflight v1

The full-exact 64-case distribution preflight used eight frozen speakers and `RT60 U[1.65, 1.85] s × DRR U[-14.5, -12.5] dB`. It passed tail coverage (56/64 cases, 8/8 speakers had at least 4 valid cases), clipping (0), active delta, and target retention. It failed the shared-scale gates: median `-15.197 dB`, p10 `-17.133 dB`, and 53/64 scales below 0.25. This is preserved as a historical `FAIL` under those frozen gates.

### Integrity v2

The follow-up reused all 64 scaled-clean Cap60 outputs and added only eight full-exact unscaled-clean Cap60 inferences (candidate 0 for each frozen speaker). The 64 cached input hashes matched exactly.

| Measure | Result | Frozen gate |
|---|---:|---|
| Active frame gain median across cases | -0.360 dB | [-0.5, +0.5], pass |
| Active frame gain p10, median across cases | -1.452 dB | > -1 dB, fail |
| Weak frame gain p10, median across cases | -2.789 dB | > -1.5 dB, fail |
| Onset frame gain p10, median across cases | -1.023 dB | > -1 dB, fail |
| Active RMS retention, min / median / max | 0.9656 / 0.9740 / 0.9812 | min >= 0.90, pass |
| Cap60 clipping / early path peak / realized DRR | 0 / 1.0 on 64/64 / within 0.000001 dB | pass |
| Amplitude equivariance, median active envelope correlation | 0.999962 | >= 0.98, pass |
| Amplitude equivariance, median of per-voice p90 absolute gain delta | 0.464 dB | <= 1 dB, pass |

The overall integrity v2 gate therefore fails. `shared_scale` is descriptive in this second protocol and was not used to reject the pairs.

### Cached attribution diagnostic

This diagnostic used the existing unscaled and scaled Cap60 caches only. It made no new Cap60 calls. Frames are 20 ms RMS windows at 10 ms hops; masks use the unscaled clean crop. Statistics are calculated per voice or per candidate, then summarized across records rather than pooling all frames.

| Clean frame band | Intrinsic Cap60 p10, median across voices | Scaling effect p10, median across 64 cases |
|---|---:|---:|
| 2–5% of peak RMS | -4.330 dB | -1.408 dB |
| 5–10% | -1.034 dB | -0.119 dB |
| 10–35% | -0.642 dB | -0.078 dB |
| Above 35% | -0.320 dB | -0.027 dB |
| Combined weak mask (active, below 35%) | **-2.152 dB** | **-0.140 dB** |
| Onset mask | -0.855 dB | -0.068 dB |

For the 2–5% band, the median per-voice fraction below -3 dB was 30.7%, below -6 dB was 4.0%, and below -20 dB was 2.0%; the longest contiguous run below -6 dB was 300 ms. Under the pre-registered attribution rule, this confirms intrinsic Cap60 suppression on weak frames while the typical scaling effect remains small.

## Interpretation

The strong-RIR candidate set is not approved for F2 training with the current target. The result does not show that the reverb model itself is defective: the current target already suppresses some weak frames before dereverberation training. Training to reproduce that target could teach the dereverberator to remove weak speech. The next target experiment should compare Cap60 wet input against unprocessed clean speech or a separately frozen preservation-first target, using held-out speakers and new numerical gates agreed before training.

## Reproduction artifacts

- [Distribution preflight runner](../../scripts/benchmarks/universal_enhancer/compact_dereverb/preflight_f2_strong_rir_distribution.py)
- [Integrity v2 evaluator](../../scripts/benchmarks/universal_enhancer/compact_dereverb/evaluate_f2_strong_rir_integrity_v2.py)
- [Cached attribution diagnostic](../../scripts/benchmarks/universal_enhancer/compact_dereverb/diagnose_f2_cap60_weak_frame_suppression.py)
- Local full reports (ignored by Git):
  - `.tools/compact-dereverb/f2-strong-rir-preflight-2026-10-10/report.json`
  - `.tools/compact-dereverb/f2-strong-rir-integrity-v2-2026-10-10/report.json`
  - `.tools/compact-dereverb/f2-cap60-weak-frame-diagnostic-2026-10-10/report.json`

The protected `/home/yohan/Bureau/test.wav` was not accessed (`test_wav_accessed: false`).
