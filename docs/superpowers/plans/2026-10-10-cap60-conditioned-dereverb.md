# Cap60-Conditioned Dereverberator C Implementation Plan

> **For agentic workers:** keep `test.wav` sealed. Use only local, licensed LibriSpeech/BUT training material and the already frozen Cap60 executable; preserve unrelated workspace changes.

**Goal:** Train and evaluate a small dereverberator C that removes residual room tail from Cap60 output without raising its quiet-output floor or attenuating speech.

**Architecture:** Keep the existing 16 kHz `CompactDereverb16k` architecture (555,922 parameters). Train it on Cap60-preprocessed reverberant/noisy mixtures with Cap60-preprocessed clean speech as target, plus identity examples whose input and target are both Cap60(clean). Use a fixed offline precompute stage to keep training deterministic and avoid launching the CPU denoiser inside the GPU training loop.

**Tech Stack:** Python 3.13, PyTorch 2.6 CUDA (GTX 1050 Ti), local DeepFilterNet CLI at attenuation cap 60 dB, NumPy, SciPy, SoundFile, BUT ReverbDB measured RIRs.

## Global Constraints

- No Adobe output, training, export, or `test.wav` use.
- Use the same 555,922-parameter architecture and fixed 3,000 optimizer updates as the current compact model.
- Train speech speakers are LibriSpeech `train-clean-100`; development speech speakers are `dev-clean`; assert no overlap. Seal 12 `train-clean-360` speakers excluded from all previous manifests, selected by sorted SHA256 of speaker ID, for a later physical/new-corpus holdout.
- Train RIRs use 50% procedural and 50% measured BUT responses from the six predeclared training rooms only. The three previously held-out BUT rooms are now dev-only; do not train on AIR or dEchorate.
- C targets Cap60(clean_scaled), not dry clean, so it learns only the residual dereverberation/cleanup after Cap60. For the target-retention screen, compare `RMS(Cap60(clean_scaled)|M_active)` with `RMS(clean_scaled|M_active)` before the common post-Cap60 gain. Never derive the denominator from Cap60(wet). The shared anti-clipping scale is retained and recorded relative to original clean; active masks still come from original clean.
- Append a 100 ms zero guard before every Cap60 file inference; the CLI's 480-sample EOF trim then affects only the guard. Verify start alignment and trim to the source crop length.
- Keep waveforms, model checkpoints, staged inputs, and rendered outputs under ignored `.tools/`; publish code, exact aggregate statistics, and methods only.
- Treat the existing AIR factorial as a one-shot diagnostic for C, not a project-level external validation or Adobe-parity claim; never tune C or its gates from that evaluation.

## Files

- Create `scripts/benchmarks/universal_enhancer/compact_dereverb/prepare_cap60_conditioned_pairs.py` for deterministic pair generation, Cap60 preprocessing, guard/alignment verification, manifests, and mmap-ready arrays.
- Create `scripts/benchmarks/universal_enhancer/compact_dereverb/train_cap60_conditioned.py` for C loss, speaker-safe training/development, fixed-step checkpoint capture, and training logs.
- Create `scripts/benchmarks/universal_enhancer/compact_dereverb/evaluate_cap60_conditioned_dev.py` to measure the frozen checkpoint against all declared BUT development gates before the AIR diagnostic.
- Reuse `scripts/benchmarks/universal_enhancer/compact_dereverb/model.py`, `data.py`, `screen_measured_rirs.py`, and the existing Cap60 CLI; do not change B or its checkpoint.
- Evaluate with `scripts/benchmarks/universal_enhancer/compact_dereverb/bench_air_cap60_factorial.py` using the frozen Cap60 outputs and the one selected C checkpoint; add cache reuse so the already computed Cap60 stage is not run again.
- Create `docs/benchmarking/cap60-conditioned-dereverb-2026-10-10.md` with training recipe, exact held-out evaluation, measurements, limits, and GO/NO-GO.
- Keep all generated data and checkpoints in `.tools/compact-dereverb/cap60-conditioned-2026-10-10/`.

## Predeclared training/evaluation design

- 3,000 total optimizer updates over a fixed 128-example cache with one same-speaker pair from each of 128 train speakers (32 identity, 48 RIR-only, 24 fan20, 24 fan10). Exact sampling frequencies: 25% identity (`Cap60(clean) → Cap60(clean)`), 37.5% RIR-only, 18.75% fan-like 20 dB, and 18.75% fan-like 10 dB. For wet examples, use 50% procedural and 50% measured BUT; sample the BUT room uniformly, then an RIR uniformly inside it. Repeat the frozen cache with epoch shuffling; this is a constrained proof run, and later expansion requires a new protocol before seeing the AIR diagnostic.
- Each wet input is `Cap60(clean_scaled * RIR + noise)` and its target is `Cap60(clean_scaled)` from the exact same speech crop and gain convention. The target-retention denominator is that paired `clean_scaled` input, never the wet input; record its ratio to original clean and the ratio in dB.
- Loss: Hann STFTs with `n_fft=(256,512,1024,2048)`, hop `n_fft/4`, `center=False`; active masks from original clean. For each resolution `k`, let `a_k=mean_active(|T_k|)`, `L_cSTFT=mean_k(mean(|Y_k-T_k|)/(a_k+eps))`, with complex Euclidean magnitude. Wave L1 is divided by RMS of Cap60(clean) on original-clean active frames. For log magnitude use only target bins `|T_k| >= 10^-2.5*a_k`; average `|20log10((|Y_k|+tau_k)/(|T_k|+tau_k))|` over visible bins and resolutions. Use 20 ms RMS, 10 ms hop and `eps_r=1e-5*s`; `L_drop=mean_{weak∪onset}(max(0,(-1-g_t)/3)^2)` where `g_t=20log10((RMS(y_t)+eps_r)/(RMS(t_t)+eps_r))`. For silent clean frames, `r_t=max(RMS(t_t),1e-4*s)`, `g_floor=20log10((RMS(y_t)+eps_r)/(r_t+eps_r))`, `L_floor=mean_silence(max(0,(g_floor-3)/6)^2)`. Total: `L=L_wave+0.5*L_cSTFT+0.25*L_log+0.5*L_drop+0.5*L_floor`.
- Use full-utterance Cap60 preprocessing, append 100 ms zero guard, infer, remove only the added guard, then fixed resample to 16 kHz and crop. Never independently process training 2 s crops through Cap60. Validate 32 deterministic utterances for alignment, lengths, determinism, no clipping, and matching resampling.
- Development monitoring uses `dev-clean` plus only the three BUT dev rooms (`Hotel_SkalskyDvur_Room112`, `VUT_FIT_L212`, `VUT_FIT_L227`): preselect exactly 18 RIR candidates (six per room) and six identity candidates (two per room) before applying the Cap60 retention filter; do not replace rejected candidates. Every valid RIR candidate also gets fan20/fan10 plus matched noise-only controls. Freeze exactly step 3000 (no checkpoint cherry-picking). AIR and dEchorate may be post-freeze diagnostics only. The final holdout must be a new corpus or physical playback/recapture; seal 12 excluded LibriSpeech speakers for that future test.
- Candidate development GO gates relative to Cap60: RIR-only tail gain ≥+2 dB at 150–300 ms and ≥+1.5 dB at 300–600 ms, positive on at least 75% of measurable pairs and in all three dev rooms; median slope change ≤+2 dB/s (ideally ≤0); noise-only common-floor median absolute change ≤3 dB separately at fan20/fan10 and p90 increase <+6 dB; dry active median within [−0.5,+0.5] dB; onset p10 ≥−1 dB; weak p10 ≥−1.5 dB; weak p50 within ±1 dB; dry floor median increase ≤+3 dB. Report PASS/FAIL/NE; a positive room-dependent gate requires at least four measurable RIR cases per room, and insufficient coverage cannot turn an observed regression into NE.
- An AIR run can only be described as C-specific, held-out-from-training/dev diagnostic evidence because AIR has already been examined during this project. A product-level room-generalization claim requires a later, fresh-room dataset or new physical playback/recapture held out from all development.

## Tasks

### Task 1: Freeze the corpus split and preprocessing

- [x] Verify LibriSpeech train/dev speaker disjointness; exclude speakers present in previous manifests; hash all selected train/dev files and seal the 12-speaker future holdout list.
- [x] Select BUT RIRs from the six fixed training rooms and separate BUT RIRs from the three development rooms; verify no room/configuration overlap.
- [x] Generate deterministic 2 s paired crops, target Cap60(clean_scaled), identity samples, procedural/measured RIR mixtures, and seeded fan20/fan10 mixtures.
- [x] Preprocess in bounded Cap60 batches with the 100 ms EOF guard; verify 16 kHz, finite values, zero-sample start alignment, and exact source-length trim before saving the fixed arrays and manifest.
- [x] Record every target-retention decision with speaker, condition, room/RIR, paired-clean scale, ratio and dB value; publish exclusion counts and percentages grouped by speaker and condition (training) and speaker and room (development).
- [x] Before filtering, preselect exactly 18 BUT dev RIR candidates (six per room) and six identity candidates (two per room). For every RIR candidate that passes the target filter, generate seeded RIR+fan20/fan10 and matching noise-only controls; never replace target or crop exclusions.

### Task 2: Train C and select once on development data

- [ ] Implement the frozen complex-STFT, waveform, log-magnitude, weak/onset, and floor loss exactly as pre-registered by GPT Web.
- [x] Run a single-batch numerical smoke check (finite values, gradients, and every loss component logged).
- [x] Train exactly 3,000 updates with the frozen seed, optimizer, batch size, and schedule.
- [x] Validate on speaker-disjoint dev-clean speech and only development RIRs; retain the exact step-3000 checkpoint. Development is monitoring only; do not select a different step.
- [x] Run and report all predeclared BUT dev gates before exposing the frozen checkpoint to the AIR one-shot diagnostic.

### Task 3: One-shot C diagnostic, report, and checkpoint

- [x] Reuse the frozen Cap60 outputs and evaluate Cap60 vs Cap60→C on the exact 12-speaker × eight-AIR factorial, including dry/fan controls, floors, tail censoring, speech preservation, decay slope, and runtime.
- [x] Do not tune on this run. Apply the predeclared gates and label this a C-specific diagnostic, not a new external product holdout.
- [ ] Write the report and aggregate plots; verify all row/file counts, checkpoint/source hashes, and the sealed-input constraint; commit only this experiment's files.

## Observed C result (2026-10-10)

- Fixed step 3000 is a **NO-GO**: BUT noise-only common output floor rises by about +46.7 dB at fan20 and +39.5 dB at fan10. Raw and mean-removed dBFS measurements confirm this is not just a denominator artifact: C noise-only RMS is around −53 dBFS, versus Cap60 around −100 dBFS (fan20) and −93 dBFS (fan10); after removing DC, C remains around −69 / −66 dBFS.
- On dry digital pauses, Cap60 is exactly zero while C produces a measurable floor (about −60 dBFS raw, −78 dBFS after mean removal on the RIR target controls). Never publish a finite dB delta against digital zero; report both absolute levels and the exact-zero count.
- BUT tail metrics are mostly censored at the matched output floor (12/13 at 150–300 ms; 13/13 at 300–600 ms), so there is no useful evidence of dereverberation. One measurable 150–300 ms pair regresses by 23 dB. Coverage remains 6/5/2 RIRs by room, below the four-per-room positive-gate requirement for L227.
- The AIR one-shot is concordant on the main failure: dry speech is preserved, but the tail median gains are only −0.022 dB and +0.095 dB, while common noise floors rise by about +39 dB (fan20) and +31 dB (fan10). The dry floor increase against exact digital silence is reported in dBFS, with no finite delta.
- Gate handling is PASS/FAIL/NE. Four identity cases can expose a clear failure, but a passing result at n=4 remains NE. AIR is not a substitute for the under-covered BUT room.
- Aggregate report: `docs/benchmarking/cap60-conditioned-dereverb-2026-10-10.md`. The frozen checkpoint, waveforms, arrays, and raw case rows remain under ignored `.tools/`.
