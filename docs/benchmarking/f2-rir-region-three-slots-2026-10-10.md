# Full-exact reverb-region map across three voices — 2026-10-10

## Purpose

A 12-point full-exact probe on one unresolved voice found only one eligible
point, at the extreme `RT60=1.7 s / DRR=−12 dB`. Before selecting a new RIR
distribution, this experiment maps a wider region on three voices fixed in
advance by the frozen schedule: rows 3, 5, and 6 (speakers 3857, 2159, and
8098). These are development voices, not a held-out validation set.

## Frozen method

- Grid per voice: `RT60 ∈ {1.6, 1.9, 2.2, 2.5} s` ×
  `DRR ∈ {−9, −12, −15, −18} dB`.
- One procedural seed per voice was reused across its 4×4 grid, holding the
  random reflection and late-noise realization fixed within each voice.
- Every point used full-utterance Cap60 on both wet and paired-clean signals;
  no prefix or scale surrogate was used.
- Both full-exact tail windows had to be strictly above `−50 dB`.
- These 48 points are a mapping probe only; they did not enter the F2 training
  cache.

## Results

| Frozen schedule row | Eligible points | Grid |
|---:|---:|:---|
| 3 | 16/16 | all RT60 and DRR combinations passed |
| 5 | 13/16 | failures at (1.6, −9), (1.9, −9), (2.5, −9) |
| 6 | 14/16 | failures at (1.6, −9), (2.5, −9) |

The same contiguous rectangle `RT60={1.6,1.9} s × DRR={−12,−15} dB` passed
all four points on all three voices. The broader cartography gates also passed:

- At least 6/16 eligible points on two voices and at least 3/16 on the third.
- A common contiguous 2×2 rectangle with at least 3/4 eligible points on at
  least two voices.
- At least one success away from both extreme grid edges.

The full-exact probe took `599.73 s` for 96 Cap60 runs. This shows that a
measurable post-Cap60 strong-reverb region exists across these three voices.
It does not establish universal room realism, speech quality, or parity with
Adobe Podcast; the grid intentionally maps a strong-reverb training domain.
A separate speech-preservation and unseen-speaker validation is required
before expanding the tailbank or starting training.

## Reproducibility

- Exact per-point scores, hashes, seeds, and timings:
  [`report.json`](assets/f2-rir-region-three-slots-2026-10-10/report.json)
- Runner:
  `scripts/benchmarks/universal_enhancer/compact_dereverb/probe_f2_rir_region_three_slots.py`
- Every artifact records `surrogate_used: false`,
  `selection_or_training_data_modified: false`, and `test_wav_accessed: false`.
