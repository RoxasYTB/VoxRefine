# Cap60 prefix screening calibration — 2026-10-10

## Purpose

The E2/F2 dereverberation study selects synthetic room impulse responses by
measuring two post-pause residual windows after DeepFilterNet Cap60. Full
utterance inference for every proposed response is expensive. This calibration
checks whether Cap60 can screen clearly unsuitable candidates from a short
prefix while preserving the frozen full-run selection rule.

This is a data-generation speed optimization. It does not establish perceptual
parity with Adobe Podcast or validate a trained dereverberation model.

## Frozen measurement

- Input: public LibriSpeech train-clean-100 utterances with deterministic
  procedural RIRs; 8 tail slots × candidate indices `[0, 1, 3, 7, 15, 31, 63,
  127]` (64 comparisons).
- The wet and target branches use the same candidate-specific scale from
  `measured_pair`; each Cap60 invocation is isolated to one file.
- Activity mask: raw clean signal over the fixed 2 s training crop. The
  reference energy uses active Cap60-clean samples from 500 to 50 ms before the
  inserted pause. At least 200 ms of active reference is required.
- Residual windows: 150–300 ms and 300–600 ms after pause start. Each full
  candidate is eligible only if both measurements are strictly above −50 dB.
- Prefix length is measured from the beginning of the utterance through
  pause+1600 ms. The wrapper adds the same 100 ms EOF guard used for full runs.
  Scoring remains limited to the reference and the two frozen windows.

## Findings

| Check | Result |
|---|---:|
| Multi-file CLI vs isolated runs, max absolute sample difference | 0.18973 |
| Same input first vs last in one CLI batch | 0.11148 |
| Same 26.5 s input in two isolated processes, max difference | 0 |
| Prefix ending at pause+600 ms, max difference in scored region | 0.00012207 |
| Prefix ending at pause+600 ms, max `ΔLX` | 0.50392 dB |
| Prefix ending at pause+600 ms, eligibility disagreements | 0 / 64 |
| Prefix ending at pause+1600 ms, max difference in scored region | 0 |
| Prefix ending at pause+1600 ms, max `ΔLX` | 0 dB |
| Prefix ending at pause+1600 ms, eligibility disagreements | 0 / 64 |
| Candidates with at least one window in [−55, −45] dB | 33 / 64 |

The short prefix reaches its EOF too close to the late measurement window, so
it fails the strict equivalence gate despite no observed eligibility change.
Adding one second of context after the second window removes the measured
boundary effect for all 64 calibrated cases. This is empirical evidence over
the frozen sample set, not a mathematical guarantee for every utterance.

## Selection policy

The prefix is rejection-only. With a frozen 1 dB safety margin, a candidate is
screened out only when either prefix window is at or below −51 dB. All other
candidates receive full-utterance Cap60 on both wet and candidate-scaled clean
signals. The first candidate whose **full** measurements pass both >−50 dB
conditions is selected. Therefore the prefix can save work on clearly weak
candidates, but cannot promote a candidate to the training set.

Batching multiple files in one Cap60 CLI process is prohibited by the measured
order dependence. Reuse is allowed only for an exact input-byte hash under the
same Cap60 binary/configuration/sample rate/guard policy.

## Reproducibility artifacts

Raw reports, per-candidate rows, batch-order validation, and the two-process
determinism check are in
[`assets/compact-dereverb/f2-prefix-calibration-2026-10-10/`](assets/compact-dereverb/f2-prefix-calibration-2026-10-10/).
The frozen schedule and Cap60 executable hashes are recorded in each report.
Every artifact records `test_wav_accessed: false`.

## Limits

Only 8 voice/room slots and 8 spaced candidate indices per slot were used for
calibration. The prefix gate must stay tied to these hashes and settings. The
subsequent 48-slot data-generation run and its independent coverage audit are
still required before either E2/F2 fit can start. A fresh sealed holdout remains
untouched until both fits clear the development gates.
