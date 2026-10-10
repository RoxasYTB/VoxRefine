# VCTK Measured-Room Dereverberation Pilot

> **For agentic workers:** Follow this plan task by task. This is an isolated research pilot; do not change product defaults or reuse its checkpoint in production without the gates below passing.

**Goal:** Determine whether the existing compact dereverberator learns useful room-tail removal from high-quality VCTK speech convolved with measured room impulse responses, while preserving dry speech and staying within the GTX 1050 Ti memory budget.

**Architecture:** Keep the existing 16 kHz, 555,922-parameter compact dereverberation network and its offline forward path. Use VCTK mic1 as the close, hemi-anechoic speech source; convolve it with measured training-room RIRs plus procedural RIRs. Train reverberant examples toward direct plus the first 50 ms of early reflections, and retain dry controls. Keep speech speakers disjoint; reserve complete rooms for validation and final evaluation. This tests paired-data and target hypotheses without changing product behavior.

**Tech Stack:** Python, PyTorch, NumPy, SoundFile, existing `compact_dereverb` training/evaluation utilities, CSTR VCTK 0.92 (CC BY 4.0), measured BUT ReverbDB RIRs (CC BY 4.0).

## Global Constraints

- Use only VCTK mic1 and RIRs whose local source manifests record their source, license, and SHA-256.
- Treat mic1 as close/hemi-anechoic speech, not mathematically anechoic audio.
- Keep speech speakers disjoint between train and validation, RIR configurations disjoint for validation, and complete room IDs disjoint for final evaluation.
- Keep the existing 16 kHz model, loss family, output limits, and product defaults unchanged in this pilot.
- Do not open `/home/yohan/Bureau/test.wav` or use user recordings.
- Do not send Adobe-derived audio, measurements, NDA content, or contract details to GPT Web.
- Keep corpus, checkpoints, and renders under ignored local directories; commit only scripts, manifests without private audio, and aggregate documentation.
- Hard stop if the training peak exceeds 2 GiB VRAM, clean controls lose more than 1 dB median or 3 dB p10, or held-out room-tail suppression fails its predeclared gate.

---

### Task 1: Verify the VCTK and measured-RIR sources

**Files:**
- Read: `corpus/samples/fullband-student-vctk-01/source-manifest.json`
- Read: `scripts/benchmarks/universal_enhancer/compact_dereverb/screen_measured_rirs.py`
- Read: `scripts/benchmarks/universal_enhancer/compact_dereverb/train_measured_mix.py`
- Read: local AIR and dEchorate source manifests under `.tools/compact-dereverb/data/`

- [x] Record the exact train/validation speaker IDs from the existing VCTK sample manifest and confirm all referenced files exist and decode at 48 kHz mono.
- [x] Record 26 training RIRs from 13 rooms across BUT, AIR, and dEchorate; validation uses 3 different BUT rooms and final evaluation uses two Rochester rooms.
- [x] Record source licenses and note that mic1 was captured in a hemi-anechoic chamber, so it is a dry proxy but not an anechoic ground truth.
- [x] Verify manifest paths, licenses, speaker splits, and RIR selection before training.

### Task 2: Add a VCTK input adapter without changing the existing trainer

**Files:**
- Create: `scripts/benchmarks/universal_enhancer/compact_dereverb/train_vctk_measured_mix.py`
- Create: `scripts/benchmarks/universal_enhancer/compact_dereverb/test_vctk_measured_mix.py`
- Reuse: `scripts/benchmarks/universal_enhancer/compact_dereverb/train_measured_mix.py`

**Interface:**
- `read_vctk_rows(manifest_path, split) -> list[tuple[Path, str]]` returns each manifest WAV and its speaker ID, accepting only the requested split.
- `VctkMeasuredMixSpeech(rows, measured_rirs, seed)` yields the same `mixture`, `clean`, `late`, `dry_control`, `rir_kind`, and `speaker_id` fields consumed by `train_measured_mix` losses.
- Training uses 20% dry controls, 40% held-out procedural RIR draws, and 40% measured training-room RIR draws, sampled uniformly by room and then RIR configuration.

- [ ] Add focused unit tests for adapter validation and paired tensors before maintaining this as a reusable training path.
- [x] Implement the isolated runner with a `--steps` argument, batch size 1 default, deterministic seed, and a hard VRAM gate of 2048 MiB. AMP was unnecessary for this model; peak allocated VRAM was 117.9 MiB.
- [x] Keep the existing waveform/complex-STFT losses and weak/onset protection terms unchanged so the data-source change remains interpretable.

### Task 3: Run one bounded pilot and evaluate frozen room holdouts

**Files:**
- Create: `docs/benchmarking/vctk-measured-room-dereverb-2026-10-11.md`
- Write local-only: `.tools/compact-dereverb/vctk-measured-room-pilot-2026-10-11/`

- [x] Train dry-target, increased-identity, early-target, and room-diverse early-target variants; select the room-diverse checkpoint from validation on three complete BUT room IDs.
- [x] Evaluate on frozen Rochester measured-RIR pairs with speaker and room IDs absent from this candidate's fit. Keep the 16 cases fixed across all candidates.
- [x] Report direct/early-target SI-SDR, dry-proxy SI-SDR, diagnostic component projections, room-wise results, clipping, runtime, and VRAM. The protocol does not permit a defensible 50–300 ms speech-bearing tail metric, so no such gate is claimed.
- [x] Room-diverse candidate passes aggregate open-data screening gates (early-target SI-SDR +2.49 dB median / +1.56 p10; late projection -6.46 dB median; both rooms positive; clean controls within ±0.5 dB; no collapse), but one of 16 clips regresses and the final test covers only two rooms. GO to a broader held-out evaluation; no production or universal claim. Peak training VRAM is 117.9 MiB.
- [ ] Before release, expand final evaluation to more unseen rooms and speakers, validate the remaining outlier, and add a framewise late-energy metric while speech is active.

## Self-review and known limits

- The pilot tests whether better dry speech targets and measured RIRs improve this compact model. It does not prove parity with Adobe Podcast, universal room generalization, or real-time end-to-end performance.
- VCTK mic1 is a close recording in a hemi-anechoic chamber and carries its own microphone response. The model can learn that target's acoustic style, not recover arbitrary missing source information exactly.
- The dataset is a small pilot; a pass permits a larger speaker/room-disjoint study, not a release claim.
