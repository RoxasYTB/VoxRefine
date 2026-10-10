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

## Guarded clean-reference scale surrogate

The exact prefix calibration above still requires two Cap60 calls per candidate.
A second, narrower calibration checks whether the active clean-reference energy
(`Eref`) can be estimated from an exact Cap60(clean) reference at another
candidate scale. Across 64 comparisons from 8 slots, the maximum observed
absolute `Eref` normalization error was `0.0395213443 dB`; the mean absolute
error was `0.0040095242 dB`. The positive control used a full-eligible candidate
(slot 0, candidate 85): its error was `−0.00652756 dB`, and the surrogate did not
reject it.

For a candidate clean scale `s_i` and exact reference scale `s_0`, the estimate
is

```text
Eref_hat_i = Eref_0 × (s_i / s_0)²
L_hat = 10 log10(Ewet_prefix / Eref_hat_i)
```

The observed maximum is not a guaranteed bound. The operational guard adds
1 dB: `B = 1.0395213443 dB`. A surrogate can reject only if either estimated
window is at or below `−50 − B = −51.0395213443 dB`. Otherwise both full
Cap60(wet) and Cap60(clean) are run and the original strict `> −50 dB`
criterion decides. Candidate scales outside `[0.3397839838, 1.0000001490]`
run full Cap60(clean) even for screening. The first in-range candidate in each
slot establishes an exact full-length clean reference. Thus this shortcut can
only reject; it cannot accept or promote a candidate.

Every full-processed surrogate candidate checks the actual `Eref` error and
compares exact-target prefix scores to full-utterance scores. Error above the
observed `0.0395213443 dB` is logged. Error above `1.0395213443 dB`, or any
prefix/full score drift above `0.01 dB`, writes an invalidation marker and stops
data generation. The independent tailbank audit refuses an invalidated cache.

This is a measured runtime optimization for the frozen Cap60 binary and the
specified scale range, not a general property of DeepFilterNet. The 64-case
study uses only 8 speakers/slots and 8 spaced candidate indices; the positive
control confirms that one eligible candidate is retained, but does not prove a
universal error bound. Reports are `scale-equivariance-report.json` and
`positive-scale-surrogate-check.json` in this directory. Both explicitly mark
`test_wav_accessed: false`.
