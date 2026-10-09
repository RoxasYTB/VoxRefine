# Adobe v2 Teacher–Student Distillation Pilot

> **For agentic workers:** Implement this plan in the VoxRefine workspace. Preserve all existing user changes and keep all audio, checkpoints, and derived Adobe outputs local. Do not train on the user's `test.wav` or its Adobe render.

## Goal

Run a small, falsifiable, local teacher–student experiment that tests whether a compact model can approximate Adobe Podcast v2's observed transformation on a held-out speaker while preserving speech content. This is black-box output imitation, not recovery of Adobe's internal algorithm or a claim of studio-equivalent quality.

## Architecture

Use three leave-one-speaker-out folds on the existing 15 s Emy, Rémi, and Stéphanie noisy/Adobe/clean triples. Verify global alignment first using multiple separated speech windows; reject a pair if lag drifts rather than applying local time warping. Resample paired signals to 24 kHz mono. Train a small bounded complex-STFT residual student (approximately 0.5–2 M parameters), with paired Adobe spectral/waveform supervision and clean-reference active/weak/onset speech-preservation penalties. Score complete held-out speakers against noisy input, Adobe, and clean. Keep `test.wav` and all other pairs sealed.

## Tech Stack

Python, PyTorch CUDA when available, NumPy, SciPy, SoundFile, Matplotlib. Run in the existing `.tools/resemble-cuda-venv` environment on the GTX 1050 Ti; no network access or audio uploads.

## Global Constraints

- Preserve the user's dirty worktree; only create the named plan, isolated benchmark script/module, and ignored local result artifacts.
- Keep the experiment local. Adobe-derived output/checkpoint terms have not been verified for redistribution; do not publish them.
- No time warping or per-candidate alignment; only one pair-level global offset, if required and stable across windows.
- No hyperparameter tuning against the held-out speaker or user voice.
- Report speaker/fold as the independent unit. Three speakers are a pilot, not statistical proof.
- Include identity/no-op, current VoxRefine pipeline if its comparable render exists, student, Adobe, and clean references in outputs.

## Tasks

1. **Pair audit:** measure envelope lag in five separated speech windows for each noisy→Adobe and clean→Adobe pair, record duration/rate, drift, correlation, peak, and reject unstable pairs.
2. **Student:** implement a compact 24 kHz STFT residual network with bounded residual output and deterministic file inference. Avoid changing product code in this experiment.
3. **Training:** execute 3-fold speaker-held-out pilot using only aligned triples. Use 2 s paired crops, modest fixed step budget, deterministic seeds, and the same crop indices for noisy/Adobe/clean. Combine Adobe waveform and multi-resolution log-STFT losses with clean-speech active/weak/onset under-gain constraints.
4. **Evaluation:** for each held-out speaker, render noisy input, student, and Adobe; report teacher distance, speech level/envelope, weak-frame and onset retention, clean-silence residual, spectrograms, peak, model size, train/inference time, and peak VRAM. Preserve unnormalized files.
5. **Decision note:** state what transferred, what failed, whether a larger licensed corpus or a non-distillation path is the next step, and explicitly bound claims. Keep all user-voice assets out of training and tuning.

## Follow-up After the Initial Pilot

The first leave-one-speaker-out pass did not improve Adobe distance, while a 10 s / contiguous 5 s Emy capacity check reduced both waveform and MR-STFT error on the adjacent holdout. Per the GPT Web review, the next experiment keeps the 69,638-parameter model and the MR-STFT + 2× waveform objective, then expands to aligned local non-user pairs. Audit each pair and keep speaker/domain groups explicit. Apply clean-based weak/onset protection only to groups with a verified dry speech stem. Train three models with Emy, Rémi, or Stéphanie fully excluded, sample balanced by source group, and keep the user's `test.wav` and Adobe export sealed. Do not distribute Adobe-derived audio, features, or checkpoints while derivative-use terms remain unverified.

## Residual-Decomposition A/B After Pooled LOSO

The balanced pooled LOSO (13 training pairs/fold, 2,000 steps) did not improve teacher distance on any of the three held-out voices. The next falsifiable test keeps the same held-out identities and paired corpus and compares three arms with the same steps and seeds:

1. `direct-control`: current direct-output model and pooled objective.
2. `residual-naive`: predict `r_hat`, output `x-r_hat`, train against `r_teacher=x-Adobe` without clean/noise decomposition terms.
3. `residual-structured`: same residual prediction plus clean/noise speech-leakage and weak/onset protections, used only when a clean stem is reliable.

GPT Web confirmed `r=x-Adobe` alone is algebraically only a reparameterization; the clean/noise terms must distinguish the structured arm. Use complex STFT residuals with waveform synthesis, not magnitude-only masks. Keep speaker-strict Emy/Rémi/Stéphanie LOSO and sealed `test.wav`. Report teacher-distance delta and clean-preservation separately, plus residual speech leakage where clean speech dominates.

Pre-registered gate: GO only if structured residual beats direct input on all three held-out voices, improves teacher-distance by at least +0.20 dB each and +0.30 dB median, has weak p10 and onset p10 no worse than −3 dB vs clean, causes no material clean-silence energy increase, and beats residual-naive on at least two of three folds. Otherwise do not integrate the checkpoint or claim residual decomposition helped. All teacher-derived weights remain local pending terms review.

### Result: residual A/B (2026-10-11)

The gate failed. Direct-control teacher-distance gains versus input were −0.236/−0.129/−0.189 dB for Emy/Rémi/Stéphanie. Residual-naive gains were −1.169/−1.173/−0.809 dB and its weak-frame p10 was −12.34/−8.07/−9.49 dB. Residual-structured gains were −0.523/−0.744/−0.046 dB; weak p10 was −1.24/−2.41/−1.91 dB and onset p10 −1.24/−1.61/−1.30 dB. It beat the naive arm on spectral distance and preservation on all folds, but did not beat the direct input distance on any fold. On Rémi, its median clean-silence RMS was −45.69 dBFS versus −70.85 dBFS for Adobe. Thus the structured constraints prevent destructive speech loss but leave substantial residual noise and fail the pre-registered proximity gate. Do not integrate these checkpoints. Full table, limitations, plot, and timing are in the ignored local research README at `results/adobe-student-pilot-2026-10-09/README.md`; raw JSON/CSV/WAV/checkpoints are in `results/adobe-student-residual-ab-2026-10-11/` and remain local.

### Final metric audit: phase and nuisance alignment

Per GPT Web review, rerun no training. On the existing held-out WAVs, compute normalized waveform L1 and complex-STFT distances to Adobe, plus estimated removed-component alignment to `n=x-clean`. The direct-control beats the no-op on waveform and complex-STFT distance on all three folds (+13–21% and +22–27%, respectively), despite its gain-normalized log-magnitude score being slightly worse than input. This difference is expected from the different metric definitions and means neither score alone establishes perceptual quality. Direct-control captures little nuisance energy (−11 to −13 dB relative to estimated noise). Residual-naive captures more, but destroys weak speech; residual-structured preserves weak/onset p10 above −3 dB but loses complex-distance improvement on Stéphanie and has negative complex nuisance alignment there. The final gate fails. Close this local Adobe-distillation path without another training run; continue product work with open models and clean/RIR ground truth. The complete local metrics, gate fields, and figure are under `results/adobe-student-residual-ab-2026-10-11/phase-aware-metrics/`; reproducible analysis is `scripts/benchmarks/universal_enhancer/analyze_adobe_student_residual_metrics.py`. Keep all teacher-derived files/checkpoints local.
