# Cap60 process concurrency check — 2026-10-10

## Question

Can independent DeepFilterNet Cap60 CLI processes screen the frozen tailbank
faster on the GTX 1050 Ti while returning exactly the same samples? The prior
multi-file CLI test showed order-dependent outputs, so every inference here
uses one WAV, one process, and private input/output directories.

## Frozen fixtures

Four deterministic 16 kHz mono WAVs were generated from the frozen LibriSpeech
training schedule and the already selected procedural RIR. The set contains a
5 s quiet clean excerpt and three 26.5 s excerpts: clean, reverberant, and
reverberant plus fan noise. Their SHA-256 hashes and levels are recorded in
[`fixtures-manifest.json`](assets/cap60-parallel-determinism-2026-10-10/fixtures-manifest.json).
The sealed `/home/yohan/Bureau/test.wav` was not accessed.

Each condition ran the same four inputs sequentially in isolated Cap60
processes, followed by two-worker batches in two waves. The second parallel
run reversed launch order. The executable hash and input hashes are included
in the raw reports.

## Results

| Threads per process | Sequential | Parallel A | Parallel B | Conservative speedup | PCM identical |
|---:|---:|---:|---:|---:|---:|
| 2 | 18.788 s | 12.522 s | 12.965 s | 1.449× | 8/8 |
| 1 | 18.450 s | 12.467 s | 12.589 s | 1.466× | 8/8 |

Across both thread settings, all 16 PCM comparisons had maximum absolute
sample error `0`; all four sequential output PCM hashes also matched between
thread settings. No process crashed or ran out of memory. The runs used at most
two concurrent workers and private directories.

## Decision

Keep Cap60 generation sequential. GPT Web's preregistered gate required both
at least `1.5×` speedup using the slower of the two parallel runs and, for the
one-thread configuration, a worst parallel wall time no greater than `12.32 s`.
Neither configuration passed: the one-thread case reached `1.466×` and
`12.589 s`. The PCM determinism gate passed, but the wall-time benefit did not
justify a more complex generator. No three- or four-worker run is planned.

The tailbank screening policy is unchanged: one process at a time, exact-hash
cache reuse, ordered candidate evaluation, surrogate-only rejection, and
full-exact acceptance.

## Raw reports and reproduction

- [`threads-2-report.json`](assets/cap60-parallel-determinism-2026-10-10/threads-2-report.json)
- [`threads-1-report.json`](assets/cap60-parallel-determinism-2026-10-10/threads-1-report.json)
- `benchmark_cap60_parallel_determinism.py` runs the frozen comparison.
- `prepare_cap60_parallel_fixtures.py` regenerates the fixture WAVs from the
  frozen schedule. The WAVs remain local under `.tools/` and are not committed.

Every report marks `test_wav_accessed: false`.
