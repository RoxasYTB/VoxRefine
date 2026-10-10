# G Target Ablation Implementation Plan

> **Execution:** run inline in the current Codex task. GPT Web supplied and confirmed the scientific protocol; no subagents are used.

**Goal:** Compare dry-clean and preservation-first hybrid training targets for the zero-preserving E dereverberation model on a frozen, speaker-held-out strong-RIR experiment.

**Architecture:** Reuse the fixed 128-row training schedule and exact Cap60 preprocessing, while building a new input-only eligible procedural tailbank for its 48 RIR rows. Store the dry target, Cap60-clean target, and a separate identical tail reference; derive the G-hybrid target from frozen raw-clean masks. Freeze DEV-SPEAKER and HOLDOUT-G manifests before fitting, train both variants from identical initialization/order, and open evaluation sets only in the prescribed order.

**Tech Stack:** Existing Python, PyTorch, NumPy, Cap60 executable, LibriSpeech `train-clean-100`/`train-clean-360` assets.

## Global Constraints

- Do not read or process `/home/yohan/Bureau/test.wav`.
- Do not alter or open the already reserved final 12-speaker holdout.
- Do not select examples from model outputs; only input Cap60 tails and dry references determine eligibility.
- Keep previous strong-RIR preflight failures and artifacts immutable.
- Require 48/48 eligible training tail slots and frozen DEV/HOLDOUT manifests before any optimizer update.
- Train exactly two 3,000-step fits with shared init, data, batch order, architecture, losses, and tail reference.
- Do not publish speaker IDs; keep generated arrays, IDs, and caches under ignored `.tools/`.

---

### Task 1: Freeze the experiment specification

**Files:**
- Create: `docs/superpowers/plans/2026-10-10-g-target-ablation.md`

- [x] Record target definitions, masks, independent tail reference, split policy, evaluation gates, and stop conditions from GPT Web.
- [x] Confirm no changed parameters other than the training target. Model E, optimizer, batch sequence, step count, and tail loss weight stay fixed.

### Task 2: Prepare shared G training data and exact tailbank

**Files:**
- Create: `scripts/benchmarks/universal_enhancer/compact_dereverb/prepare_g_target_ablation.py`
- Create ignored output: `.tools/compact-dereverb/g-target-ablation-2026-10-10/data/`

**Interfaces:** consume the frozen C schedule, exact source hashes, Cap60 executable, fixed `make_procedural_rir`, and `measured_pair`; produce shared arrays `x`, dry target `c_q`, Cap60 target `a_q`, hybrid target, raw-clean activity masks, and `tail_ref=a_q` with per-array hashes.

- [ ] Reuse all fixed non-tail rows and construct 48 new strong-RIR rows with RT60 U[1.65,1.85], DRR U[-14.5,-12.5], a new seed namespace, candidates 0..31, full exact Cap60 only, first input-only eligible candidate.
- [ ] Require both Cap60 input tail windows strictly above -50 dB and exactly 48/48 RIR slots; otherwise stop before training.
- [ ] Create masks only from raw unscaled clean using 20 ms RMS, 10 ms hop, active >2% peak, weak <35% peak, onset >1.5× previous active-frame RMS.
- [ ] Dilate weak-or-onset frames by ±20 ms; make `m=1` in the full dilated region with 10 ms half-cosine fades outside it. Store `t_hybrid=m*c_q+(1-m)*a_q`.
- [ ] Write deterministic manifest with source, binary, seed, and array hashes; set `test_wav_accessed=false`.

### Task 3: Freeze disjoint speaker development and holdout sets

**Files:**
- Create: `scripts/benchmarks/universal_enhancer/compact_dereverb/freeze_g_splits.py`
- Create ignored outputs: `.tools/compact-dereverb/g-target-ablation-2026-10-10/dev/` and `sealed/`

- [ ] Derive prior-used speakers from existing experiment manifests and exclude the 128 train speakers and already reserved final 12.
- [ ] Deterministically hash-select 16 fresh DEV and 12 different fresh HOLDOUT speakers from `train-clean-360`; freeze utterance pairs, four RIRs per speaker, noise/fan seeds, crops, and input-only eligibility before fitting.
- [ ] Require complete speaker counts and pair coverage, hash the manifests, omit speaker IDs from public report fields, and mark HOLDOUT-G unopened.

### Task 4: Train matched G-clean and G-hybrid models

**Files:**
- Create: `scripts/benchmarks/universal_enhancer/compact_dereverb/train_g_target_ablation.py`
- Reuse: `model_e.py`, `cap60_conditioned_loss` and the frozen noise/quiet auxiliary behavior.

**Interfaces:** load one immutable shared data manifest; train `G-clean` against `c_q` and `G-hybrid` against `t_hybrid`; pass `tail_ref=a_q` explicitly to the common tail loss.

- [ ] Keep the zero-preserving E architecture (555,922 parameters), optimizer, seed, sampler seed/order, 3,000 updates, and all existing E loss terms fixed.
- [ ] Refactor the tail objective locally so `Eref` always comes from `tail_ref`, the mask from raw clean, and eligibility only from input `x`; assert per-row tail reference hashes match between variants.
- [ ] Save step-3000 checkpoint, batch-order hash, config, and training manifest for both variants. Do not evaluate DEV before both checkpoints exist.

### Task 5: Evaluate DEV, select once, then open HOLDOUT-G once

**Files:**
- Create: `scripts/benchmarks/universal_enhancer/compact_dereverb/evaluate_g_target_ablation.py`
- Create: `docs/benchmarking/g-target-ablation-2026-10-10.md`

- [ ] Implement the preregistered dry-clean truth metrics and gates: censor-aware tail lower bounds >=+2.0 dB (150–300 ms), >=+1.5 dB (300–600 ms), Theil–Sen tail slope delta <=+2 dB/s; active median ±0.5 dB, active/weak p10 >−1.5 dB, weak p50 ±1 dB, onset p10 >−1 dB; weak loss fractions <−3 dB <=10% and <−6 dB <=3%; fan20/fan10 floor median <=+3 dB and p90 <+6 dB; zero-invariant 100/100; DC median <−80 dBFS and p90 <−70 dBFS.
- [ ] Evaluate both checkpoints on DEV only after both are final. Any failed primary gate is NO-GO; if both pass, choose lexicographically by 300–600 ms lower confidence bound, 150–300 ms lower confidence bound, weak p10, then weak <−3 dB fraction.
- [ ] Freeze the winner before a single HOLDOUT-G evaluation. Apply identical gates without retuning. Leave final reserved holdout and `test.wav` unopened.
- [ ] Document conclusions as limited to unseen speakers under this procedural RIR domain; do not claim universal or measured-room equivalence.

### Task 6: Verify and checkpoint

**Files:** only the plan, new scripts, and new public report above.

- [ ] Use the fixed weak mask `active & frame_RMS < 0.35 * peak_RMS` (20 ms frame, 10 ms hop) for all G weak-speech gates and weak-loss fractions. Keep the historical bottom-20%-of-active metric descriptive only.
- [ ] Run Python syntax compilation for the new scripts and deterministic manifest/coverage integrity checks; do not run the general test suite unless requested.
- [ ] Review staged diff and `git diff --cached --check`; stage only the explicitly listed G files.
- [ ] Commit and push the checkpoint to the existing fork branch after the experiment artifacts are complete and reviewed.
