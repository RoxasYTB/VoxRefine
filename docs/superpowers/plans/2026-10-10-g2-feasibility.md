# G2 Feasibility Pilot Implementation Plan

> **Execution:** run inline in the current Codex task. GPT Web reviewed and confirmed this protocol; no model training is authorized by this plan.

**Goal:** Determine whether 64 fresh RIR pairs on the already-consumed G-v1 DEV speakers contain measurable two-window Cap60 reverb tails under a dry-residual-relative threshold.

**Architecture:** Close G-v1 as non-executable without training. Freeze a separate input-only G2 feasibility design over the same 16 diagnostic speakers, new procedural RIR seeds, and the first eligible exact Cap60 candidate per slot. Measure wet-tail and dry Cap60 residual energy against the same active-speech reference and persist every candidate decision for audit. A passing feasibility pilot only authorizes a separately frozen G2-v1 experiment on new speakers.

**Tech Stack:** Existing Python, NumPy, SciPy, soundfile, DeepFilterNet Cap60 executable, LibriSpeech `train-clean-360`.

## Global Constraints

- Never access `/home/yohan/Bureau/test.wav`.
- Do not train, create model checkpoints, or inspect model outputs during feasibility.
- G-v1's 46 resolved pairs are diagnostic history only and are never reused as G2 examples.
- Use only input Cap60 and the corresponding dry Cap60 reference to determine eligibility.
- Use a fresh namespace `G2-feasibility-dev-v1|speaker|pair|candidate` and new RIR seeds.
- A pair passes only if both W1=150–300 ms and W2=300–600 ms satisfy `Lx(W) > max(-60 dB, Ldry(W)+6 dB)`; use the first exact candidate `j=0..31` that passes.
- Require all 64 DEV pairs to pass; otherwise close G2 two-window feasibility as non-executable.
- Keep IDs, wave caches, and detailed traces under ignored `.tools/`; public reports omit speaker IDs.

---

### Task 1: Freeze the G2 feasibility design

**Files:**
- Create: `docs/superpowers/plans/2026-10-10-g2-feasibility.md`
- Create ignored output: `.tools/compact-dereverb/g2-feasibility-2026-10-10/design.json`

- [x] Record the input-only eligibility formula, seed namespace, speaker reuse constraints, stop rule, and prohibition on training.
- [x] Freeze the 16 diagnostic speakers, 64 source pairs and hashes, RIR parameter ranges, candidate seeds, and Cap60 identity before running the pilot.

### Task 2: Implement the resumable input-only pilot

**Files:**
- Create: `scripts/benchmarks/universal_enhancer/compact_dereverb/pilot_g2_feasibility.py`
- Create: `scripts/benchmarks/universal_enhancer/compact_dereverb/summarize_g2_feasibility.py`
- Create ignored output: `.tools/compact-dereverb/g2-feasibility-2026-10-10/`

- [x] Mirror the deterministic G-v1 DEV speaker selection and four source-pair slots per speaker, while assigning a fresh G2 feasibility RIR namespace.
- [x] For each candidate, build the common-scaled dry/wet pair, run exact Cap60 on both, calculate W1/W2 levels relative to active dry-speech energy, and accept only the first candidate satisfying both strict thresholds.
- [x] Record candidate index, seeds, RT60/DRR, shared scale, dry/wet Cap60 tail levels, margins, source hashes, Cap60 metadata, and rejection reasons.
- [x] Persist design before inference and atomically checkpoint each resolved pair so an interrupted run resumes from the next candidate without changing prior results.
- [x] Close a slot as unresolved only after all 32 candidates fail; never alter threshold, window, RIR range, or selected speakers in response to outcomes.
- [x] Write a summary with complete 64/64 coverage, runtime, hashes, and `test_wav_accessed=false`; the gate is NO-GO at 55/64.

### Task 3: Report feasibility and decide whether G2-v1 can start

**Files:**
- Create: `docs/benchmarking/g2-feasibility-2026-10-10.md`
- Create: `docs/benchmarking/assets/g2-feasibility-2026-10-10/` (if compact figures materially clarify distributions)
- Modify: `docs/superpowers/plans/2026-10-10-g-target-ablation.md` (append closure status only)

- [x] Report the pilot outcome and distribution of Lx, Ldry, margins, selected candidate indices, and per-slot failures without speaker IDs.
- [x] Close G-v1 and G2 two-window as non-executable; no training or model outputs were used.
- [x] Consume the 16 pilot speakers and exclude them from later fits/evaluations; do not reuse any of the 55 passing rows as G2 training data.
- [x] Compile the new scripts, inspect their audit and hashes, and run scoped whitespace checks.

**Outcome:** The G2 feasibility pilot examined 64 pairs; 55 passed both windows and 9 exhausted all 32 candidates. The frozen gate required 64/64, so no fit or model checkpoint was created. See [the public report](../../benchmarking/g2-feasibility-2026-10-10.md).

## Self-Review

- The old absolute `−50 dB` rule and its partial 46-pair results remain unchanged and are not reused.
- `Lx` and `Ldry` share the same dry active-speech energy reference and epsilon, preserving a paired comparison.
- No model output participates in screening; no model is trained in this pilot.
- Any follow-on fit requires a new GPT-Web-reviewed protocol and new speaker splits. GPT Web has proposed a separate G-early protocol with W1-only supervision and W2 non-regression.
