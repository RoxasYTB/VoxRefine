# Compact Preservation-First Dereverberation Proof — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task with review checkpoints.

**Goal:** Determine whether a small deterministic 16 kHz model can remove measurable late room energy while preserving dry, weak, and onset speech on voices and synthetic rooms held out from training.

**Architecture:** Keep this as an isolated research prototype under `scripts/benchmarks/universal_enhancer/compact_dereverb/`; do not change VoxRefine's live or offline product pipeline. Train a small complex-STFT residual network on clean LibriSpeech speech convolved on the fly with procedural RIRs. Compare it to identity/input on frozen dry controls and matched reverberant renders from speaker-disjoint LibriSpeech test-clean voices and RIR families excluded from training.

**Tech Stack:** Python 3.13, existing local PyTorch 2.6 CUDA 12.4 research environment, SoundFile, NumPy, SciPy, matplotlib. Train and run at 16 kHz mono with 2 s crops; keep model weights, audio, and environment files out of Git.

## Global Constraints

- Use LibriSpeech train-clean-100, dev-clean, and test-clean under the OpenSLR CC BY 4.0 terms; preserve attribution and dataset URLs in a local manifest and the final report.
- Do not use Adobe outputs, Adobe spectral curves, or Adobe-derived pseudo-targets for training, validation, or loss design.
- Do not use EARS or BUDDy artifacts in training because the relevant source data/checkpoint licensing is non-commercial or unspecified.
- Keep all model, audio, and training outputs in ignored `.tools/` or `results/` paths; never add them to Git.
- Prototype scope is offline and 16 kHz only. No product integration, live claim, universal claim, or Adobe parity claim.
- Preserve the same deterministic inference output for a fixed input; no sampling seed at inference.
- A reduction in tail energy does not pass unless dry and weak/onset preservation gates also pass.

## File Map

- Create `scripts/benchmarks/universal_enhancer/compact_dereverb/data.py`: LibriSpeech indexing, random speech crops, procedural RIR creation, dry/early/late stems, deterministic validation pair generation.
- Create `scripts/benchmarks/universal_enhancer/compact_dereverb/model.py`: small complex-STFT residual U-Net, zero-initialized final layer, stable shape handling for 2 s 16 kHz windows.
- Create `scripts/benchmarks/universal_enhancer/compact_dereverb/train.py`: train/validation loops, preservation-first losses, checkpointing, reproducible seeds, VRAM and timing logs.
- Create `scripts/benchmarks/universal_enhancer/compact_dereverb/evaluate.py`: frozen dry/RIR evaluation, onset/weak/tail metrics, RTF/VRAM, CSV and plots.
- Create `scripts/benchmarks/universal_enhancer/compact_dereverb/test_data.py`: deterministic unit checks for RIR decay, dry identity pairs, crop bounds, and held-out split isolation.
- Create `scripts/benchmarks/universal_enhancer/compact_dereverb/test_model.py`: shape, finite-gradient, identity-init, deterministic inference, and 2 s CPU/GPU forward checks.
- Create `docs/benchmarking/compact-dereverb-proof-2026-10-09.md`: dataset manifest, training config/hash, held-out protocol, raw metrics, limitations, and plots after the run.

## Frozen Evaluation Design

- Training voices: all speaker IDs from LibriSpeech train-clean-100 only.
- Validation voices: LibriSpeech dev-clean only; validation RIR generator seed family `70000–79999`.
- Final test voices: LibriSpeech test-clean only; test RIR generator seed family `90000–99999`.
- Training crop length: 32,000 samples (2.0 s at 16 kHz), sampled on the fly.
- Training RIR: direct impulse plus 4–12 random early reflections (5–50 ms), exponential late decay with T60 uniformly sampled in `[0.25, 1.20]` s, random direct-to-reverb ratio, gentle frequency-dependent damping, and bounded gain. Normalize only by a single full-file/crop scalar shared by input and target.
- Validation/test outputs: fixed 2 s regions with a weak event and known direct/late target stems; no test-set tuning.
- Primary metrics: tail/event improvement at 150–300 ms and 300–600 ms; weak-event level change; onset 0–40 ms change; dry-silence output floor; peak; runtime factor; peak VRAM.
- Gates to justify scaling beyond the proof: both tail/event medians at least +3 dB, weak-event median loss under 1 dB and p10 above −3 dB, onset p10 above −3 dB, no event loss beyond 10 dB, dry event median within ±0.5 dB with no persistent new tail, and deterministic output.
- Report per speaker and per RIR family; do not collapse results into a single score.

---

### Task 1: Pin licensed speech data and create deterministic dry/RIR pairs

**Files:**
- Create: `scripts/benchmarks/universal_enhancer/compact_dereverb/data.py`
- Create: `scripts/benchmarks/universal_enhancer/compact_dereverb/test_data.py`
- Local only: `.tools/compact-dereverb/data/`
- Local only: `.tools/compact-dereverb/dataset-manifest.json`

**Interfaces:**
- `read_librispeech(root: Path, split: str) -> list[tuple[Path, str]]` returns ordered `(audio_path, speaker_id)` entries and rejects split roots other than `train-clean-100`, `dev-clean`, and `test-clean`.
- `make_procedural_rir(sr: int, seed: int, t60_s: float, direct_to_reverb_db: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]` returns `(rir, early_rir, late_rir)` with `rir == early_rir + late_rir` within float tolerance.
- `make_validation_pair(clean: np.ndarray, rir_params: dict, sr: int=16000) -> dict[str, np.ndarray]` returns equal-length `clean`, `early`, `late`, and `reverberant` arrays.

- [ ] Download train-clean-100, dev-clean, and test-clean from the EU mirrors linked by [OpenSLR SLR12](https://www.openslr.org/12/); verify each archive with its official MD5 before extraction.
- [ ] Save source URL, split, archive name, MD5, license URL, and extraction timestamp in `.tools/compact-dereverb/dataset-manifest.json`.
- [ ] Implement a deterministic LibriSpeech walker that parses speaker ID from the parent chapter path and sorts paths before any seeded shuffle.
- [ ] Implement direct + early-reflection + damped exponential late RIR generation with independent seeded parameter draws and explicit early/late stems.
- [ ] Implement FFT convolution with fixed output length and one shared scalar normalization; keep `clean`, `early`, `late`, and `reverberant` time-aligned.
- [ ] Test RIR decomposition, T60 decay estimate within ±15%, direct-path location, equal-length convolution outputs, reproducible seed behavior, and disjoint speaker IDs across train/dev/test.
- [ ] Run: `.tools/resemble-venv/bin/python -m pytest scripts/benchmarks/universal_enhancer/compact_dereverb/test_data.py -q`; expected: all data tests pass.

### Task 2: Implement a compact deterministic complex-STFT residual model

**Files:**
- Create: `scripts/benchmarks/universal_enhancer/compact_dereverb/model.py`
- Create: `scripts/benchmarks/universal_enhancer/compact_dereverb/test_model.py`

**Interfaces:**
- `CompactDereverb16k(n_fft: int=512, hop_length: int=128, base_channels: int=24) -> torch.nn.Module` accepts `[batch, samples]` float audio and returns same-shape float audio.
- `parameter_count(model: torch.nn.Module) -> int` reports trainable parameters; target range is 0.5–2.0 M.

- [ ] Write tests for output shape at 32,000 and non-multiple lengths, finite output/gradients, deterministic repeated inference, and identity output from zero-initialized residual head.
- [ ] Implement centered 512-point Hann STFT with 128-sample hop, two real-valued input channels (real/imaginary), four modest 2D encoder levels, a temporal residual bottleneck, and skip connections.
- [ ] Predict an additive complex residual over the observed mixture; initialize the final projection to zero so the untrained model is exactly the identity.
- [ ] Crop/pad decoder feature maps against skip tensors by explicit dimension, then invert STFT and crop/pad waveform to the exact input length.
- [ ] Assert parameter count in `[500_000, 2_000_000]`; do not add a vocoder or a sample-rate converter to this proof.
- [ ] Run: `.tools/resemble-venv/bin/python -m pytest scripts/benchmarks/universal_enhancer/compact_dereverb/test_model.py -q`; expected: all model tests pass.

### Task 3: Train with direct/weak-speech protection and measure the hardware cost

**Files:**
- Create: `scripts/benchmarks/universal_enhancer/compact_dereverb/train.py`
- Modify: `scripts/benchmarks/universal_enhancer/compact_dereverb/data.py` only if training batching requires it.
- Local only: `.tools/compact-dereverb/checkpoints/`, `.tools/compact-dereverb/logs/`

**Interfaces:**
- `loss_components(pred: torch.Tensor, clean: torch.Tensor, weak_mask: torch.Tensor, early_mask: torch.Tensor, late_mask: torch.Tensor) -> dict[str, torch.Tensor]` returns named `wave`, `mrstft`, `weak_under`, and `early_under` losses.
- `train(config: dict) -> Path` writes a local checkpoint and machine-readable training manifest.

- [ ] Add a tiny CPU batch smoke test that confirms loss is finite, backward gradients reach the model, and identity initialization produces zero or near-zero dereverb benefit without altering input.
- [ ] Implement losses as waveform L1 + multi-resolution log-magnitude STFT loss, plus asymmetric under-reconstruction penalties on clean weak-speech and early/attack windows derived from the clean training target only.
- [ ] Use AdamW, batch 1–2, random 2 s crops, on-the-fly fresh RIRs, fixed seed, gradient clipping, optional mixed precision only after verifying finite CPU and CUDA gradients.
- [ ] Run a 200-step calibration and record mean step time, peak allocated/reserved VRAM, loss curves, and dataset throughput.
- [ ] If calibration fits below 3.2 GiB and is stable, run at most 50,000 updates; checkpoint every 1,000 updates and stop early if validation dry or weak-speech gates regress.
- [ ] Keep the best checkpoint selected by a predeclared validation Pareto gate, not by training loss or test performance.
- [ ] Run the CPU smoke test and a one-batch CUDA smoke test; expected: finite losses and gradients, no OOM, logged step time/VRAM.

### Task 4: Evaluate on untouched speakers and write the evidence report

**Files:**
- Create: `scripts/benchmarks/universal_enhancer/compact_dereverb/evaluate.py`
- Create: `docs/benchmarking/compact-dereverb-proof-2026-10-09.md`
- Local only: `.tools/compact-dereverb/evaluation/`

**Interfaces:**
- `evaluate(checkpoint: Path, split: str="test-clean", rir_seed_start: int=90000) -> dict` writes per-case CSV, summary JSON, and diagnostic plots.

- [ ] Write a test that refuses any speaker ID present in train/dev or any RIR seed below `90000` when evaluating the final test split.
- [ ] Freeze 30 test-clean speakers, 2 utterances per speaker, 3 new RIR families per utterance, and dry controls before the first final evaluation.
- [ ] Evaluate identity input and the trained model using identical clean, early, late, and reverberant waveforms; report no event-local gain matching.
- [ ] Report all primary metrics by speaker/RIR, plus median, p10, p90, worst case, runtime factor, peak VRAM, output peak, clipping, and dry-silence floor.
- [ ] Plot tail/event change against weak/onset preservation and include one matched spectrogram/waveform panel with the dry target, reverberant input, and model output.
- [ ] Run: `.tools/resemble-venv/bin/python -m pytest scripts/benchmarks/universal_enhancer/compact_dereverb -q`; expected: all tests pass.
- [ ] If the predeclared gates fail, record the failed gate and do not integrate or scale the model. If they pass, plan a separate 24/48 kHz and real-room holdout study before any product code change.

## Plan Self-Review

- Data licensing and split isolation are covered in Task 1 and final evaluation; Adobe is explicitly excluded from training and tuning.
- GPU limits are addressed by a 16 kHz, 2 s, 0.5–2 M parameter model, measured calibration, checkpointed training, and a 3.2 GiB VRAM budget.
- The proof is not presented as production-ready: live, universal, French, and Adobe parity claims are explicitly out of scope.
- Evaluation measures both tail removal and the dry/weak/onset failure modes that invalidated earlier candidates.
- Training examples receive dynamically generated RIRs; held-out final speakers and RIR seed families remain separate from training and validation.
