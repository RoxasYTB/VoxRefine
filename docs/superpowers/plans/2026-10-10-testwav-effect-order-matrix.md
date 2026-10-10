# Test WAV Effect Order and Dynamics Matrix Plan

> **For agentic workers:** Execute inline with review checkpoints. Keep the consumed recording and all generated audio private. Steps use checkbox syntax for tracking.

**Goal:** Find out whether a small number of plausible effect-order changes or gentle dynamics treatments improve the locally consumed `test.wav` rendering toward the user's Adobe V2 preference.

**Architecture:** Freeze the current `Cap60 → StuPASE CFG .25 → gated Cap60 high-frequency restoration → gentle EQ` result as the baseline. Compare level-matched post-effect variants and a small number of fixed-seed pre-EQ StuPASE reruns. Adobe is a descriptive listening/spectral reference only; do not optimize directly against its waveform or claim equivalence from one recording.

**Tech Stack:** Python, NumPy, SciPy, SoundFile, existing local StuPASE weights and Cap60/Adobe comparison assets.

## Global Constraints

- Never upload `test.wav`, Adobe output, or processed speech. Keep audio and sample-specific measurements in ignored `results/`.
- Do not train or fit parameters to Adobe or this consumed recording.
- Keep the existing baseline, HF gate, resampling, output PCM settings, and active-speech mask frozen.
- Level-match all audition files to active-speech RMS within 0.05 dB; use one fixed StuPASE seed for pre-EQ reruns.
- Treat outcomes as exploratory for one file. No universal preset or Adobe-equivalence claim.
- Stage and push only the new generic script, protocol, and plan; leave all pre-existing workspace changes untouched.

## Tasks

- [x] Freeze and document a maximum six-render matrix: current baseline; EQ-before-HF-restoration order; mild post-compression; mild de-essing; small pre-presence EQ into StuPASE; small pre-bass shelf into StuPASE.
- [x] Add safeguards and measurements: tail level, active/weak/onset preservation, active band energy, peaks/clipping, compressor gain-reduction distribution, de-esser activity.
- [x] Render mono 48 kHz PCM WAVs, a compact metrics CSV/JSON, a comparison plot, and an audition README with anonymized A–F players.
- [x] Check output duration, levels, finite samples, and clipping; do not add new general tests.
- [ ] Review `git diff --cached`, commit only the three new generic artifacts, then push the current checkpoint to the existing `fork` branch.

## Review Checkpoints

1. Confirm that effect variants differ only in the documented processing stage/algorithm and all share the same mask and level normalization.
2. Confirm no user audio or sample-derived metrics enter Git.
3. Report the best measured candidate without claiming listening superiority; invite the user to compare the embedded, level-matched players.
