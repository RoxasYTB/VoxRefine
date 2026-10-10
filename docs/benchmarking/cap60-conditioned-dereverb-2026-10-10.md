# Cap60-conditioned dereverberator C — 2026-10-10

## Verdict

**NO-GO for C step 3000.** It preserves most dry speech energy, but introduces a substantial floor in pauses and does not measurably shorten room tails. The same noise-floor failure appears on BUT development rooms and the previously examined AIR diagnostic. This checkpoint is retained as a negative result; it is not suitable as a product enhancement stage.

This experiment does not compare against Adobe output and provides no evidence of Adobe Podcast parity. The result only compares a frozen DeepFilterNet Cap60 route with that route followed by the 555,922-parameter dereverberator.

## Frozen experiment

- Training pairs: 128 distinct LibriSpeech train speakers: 32 identity, 48 RIR-only, 24 fan20, 24 fan10. Each wet target was the Cap60 output of its matching clean crop; the model was trained to remove only residual artifacts after Cap60.
- RIR training used six fixed BUT rooms, with measured and procedural responses balanced 50/50. Dev used three other BUT rooms. No AIR or dEchorate data entered training.
- Dev candidates were selected before the Cap60 retention screen: 18 RIR (6 per room) and 6 identity (2 per room). No candidate was replaced after exclusion. The final dev had 13 RIRs (6/5/2 by room) and 4 identity cases. Target retention rejected 26 of 154 attempts (16.88%).
- Training used seed `20261010`, AdamW at `2e-4`, batch size 1, and exactly 3,000 updates. Only step 3000 was evaluated. Training took 144.8 seconds on an NVIDIA GTX 1050 Ti (4 GiB); peak model-training VRAM was about 124.6 MiB.
- The cache preparation ran 360 Cap60 outputs and accumulated 1,788.4 seconds of Cap60 process time. It remains an offline CPU preprocessing stage, so these figures do not establish end-to-end real-time operation.

## BUT development results

The development room counts are imbalanced and L227 has only two retained RIRs. Positive room-level gates require at least four measurable cases per room; a favorable result with insufficient coverage is `NE`. A clear regression remains `FAIL` despite low coverage. Four identity cases report exact values but cannot establish a robust passing gate.

| Measurement | Result | Gate |
|---|---:|---|
| RIR tail gain, 150–300 ms | 1 measurable; 12 censored; measured pair regressed by 23.02 dB | FAIL |
| RIR tail gain, 300–600 ms | 0 measurable; 13 censored | NE |
| Fan20 common output-floor change | +46.68 dB median, +50.56 dB p90 (n=13) | FAIL |
| Fan10 common output-floor change | +39.45 dB median, +41.73 dB p90 (n=13) | FAIL |
| Dry active speech median | −0.082 dB (n=4) | NE: small identity sample |
| Dry onset p10 | −0.212 dB (n=4) | NE: small identity sample |
| Dry weak-speech p10 | −1.613 dB (n=4) | FAIL: threshold −1.5 dB |
| Dry weak-speech p50 | −0.638 dB (n=4) | NE: small identity sample |

The noise-floor regression is visible in absolute signal levels, not just as a ratio:

| Pause output | Cap60 raw RMS | Cap60→C raw RMS | Cap60→C after mean removal |
|---|---:|---:|---:|
| Dry target, digital pause (n=13) | Exactly zero | −60.22 dBFS | −77.68 dBFS |
| Noise-only, fan20 (n=13) | −100.45 dBFS | −53.48 dBFS | −69.00 dBFS |
| Noise-only, fan10 (n=13) | −92.67 dBFS | −53.45 dBFS | −65.97 dBFS |

On fan20/fan10, C's median absolute DC mean is approximately −53.7 dBFS. Removing that mean improves the reported RMS, but leaves roughly 31 dB (fan20) and 27 dB (fan10) of excess AC energy versus Cap60. Therefore DC bias explains part of the artifact, not the full defect. For the dry identity controls, Cap60 is exactly zero in most measured windows; a finite dB increase against zero is undefined, so the report records exact-zero counts and C's absolute dBFS instead of publishing the misleading finite `+194 dB` ratio produced by an epsilon denominator.

![Absolute raw, AC-only, and DC pause-floor measurements](/mnt/windata/C/Users/Yohan/Desktop/Projets/projects/VoxRefine/docs/benchmarking/assets/cap60-conditioned-dereverb-2026-10-10/absolute-pause-floors.png)

![Frozen BUT development gates](/mnt/windata/C/Users/Yohan/Desktop/Projets/projects/VoxRefine/docs/benchmarking/assets/cap60-conditioned-dereverb-2026-10-10/but-dev-gates.png)

## AIR one-shot diagnostic

The checkpoint was not changed after BUT evaluation. The exact 12-speaker × 8-RIR AIR factorial produced 768 output rows. It is a previously examined diagnostic set, not a fresh product holdout.

- RIR-only tail gain: median −0.022 dB at 150–300 ms (70 measurable; 26 censored), and +0.095 dB at 300–600 ms (40 measurable; 56 censored). Both are far below the predeclared +2 / +1.5 dB targets.
- Noise-only common floor: +38.69 / +39.10 dB at fan20, and +30.86 / +31.32 dB at fan10 (150–300 / 300–600 ms).
- Dry speech change: active p50 −0.006 dB, onset p10 +0.029 dB, weak p10 −0.044 dB, weak p50 +0.011 dB. Speech preservation is good on this diagnostic, but it does not offset the floor and dereverberation failures.
- The C forward passes took 16.50 seconds over the AIR run. The cached Cap60 stage took no additional wall time in this run. This is an offline batch result, not a streaming latency benchmark.

## Interpretation and next experiment boundary

The model learned to retain dry speech energy but produces a persistent output floor in low-energy regions. The measured raw RMS and AC-only RMS both fail, so simply removing DC is not an adequate correction. It also fails to create measurable tail reduction once the matched output floor is considered. The likely next step is a new, separately versioned experiment that controls silence/noise-floor generation and checks its objective before using the dereverberation metric. C remains frozen as the NO-GO comparator.

BUT and AIR have now been inspected and must be treated as development data for further iterations. Do not use either as an untouched confirmation set. Do not unseal the 12 reserved LibriSpeech speakers or access `/home/yohan/Bureau/test.wav`. A new corpus or fresh physical room recording is required for a later independent holdout.

## Reproducibility artifacts

- Training cache, checkpoint, manifests, and raw per-case outputs: ignored `.tools/compact-dereverb/cap60-conditioned-2026-10-10/`.
- Frozen checkpoint: `training/checkpoints/step-003000.pt`, SHA-256 `6d881e1ade866aa013ea432bc0a57bb6fd6e98aa5e79ef755af5749c519e6dbb`.
- Final BUT summary: `dev-evaluation-final3/summary.json`; 234 per-route/window records, including absolute raw/AC/DC pause levels.
- AIR summary and per-case records: ignored `.tools/compact-dereverb/air-cap60-c-factorial-2026-10-10/`.
- Evaluation uses candidate index, speaker, room, and RIR configuration to preserve every preselected RIR case. It reports raw RMS as the primary noise-floor measure, with mean-removed RMS and DC level as diagnostics.
