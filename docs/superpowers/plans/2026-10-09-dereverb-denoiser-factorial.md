# Frozen AIR Dereverb–Denoiser Factorial Benchmark Plan

> **For agentic workers:** execute this plan inline, preserving all unrelated workspace changes. Treat `test.wav` as a sealed external listening case: never use it for tuning, thresholds, or model selection.

**Goal:** Determine whether the frozen compact 16 kHz dereverberator reduces measured-room decay independently of, and after, the shipped DPDFNet2 denoiser.

**Architecture:** Use the existing AIR measured-RIR speech pairing and metric helpers, fresh LibriSpeech train-clean-360 speakers, and four paired routes: input, dereverb, DPDFNet2, DPDFNet2→dereverb. Use the first four seconds of two deterministic successive utterances, joined by a controlled pause, to keep the CPU-only DPDFNet2 stream runtime tractable. Add stationary fan-like noise at 20 and 10 dB SNR as a separate factor. Keep clean speech, reverberant speech, additive noise, and model output measurements distinguishable.

**Tech Stack:** Python 3.13, PyTorch 2.6 CUDA (GTX 1050 Ti), ONNX Runtime, NumPy, SciPy, SoundFile, Matplotlib.

## Global Constraints

- Freeze both model checkpoints and the DPDFNet2 ONNX weights; do not select checkpoints or tune thresholds from these results.
- Do not load, render, or process the user's `test.wav`.
- Use a deterministic 12-speaker selection from train-clean-360 that excludes previously used speakers.
- Evaluate eight measured AIR room responses independently; do not treat 96 speaker-room pairs as 96 independent rooms.
- Separate tail decay, additive-noise output floor, and active-speech preservation.
- A tail result is censored whenever its level is within 3 dB of the higher measured dry/noise-only output floor.
- No Adobe-equivalence or universal-performance claim follows from this screen.

## File Structure

- Create `scripts/benchmarks/universal_enhancer/compact_dereverb/bench_air_dpdfnet_factorial.py` for selection, processing routes, metric rows, timing, manifests, and figures.
- Create `docs/benchmarking/air-dpdfnet-factorial-2026-10-09.md` for frozen protocol, aggregate measurements, limits, and the resulting go/no-go decision.
- Keep source audio and full machine-readable results under ignored `.tools/compact-dereverb/air-dpdfnet-factorial-2026-10-09/`.

## Tasks

### Task 1: Freeze the protocol and dependencies

- [x] Verify the local DPDFNet2 model, compact dereverb checkpoint, AIR extract, and CUDA Python environment exist. Install the missing CPU ONNX Runtime package into the existing local CUDA environment; the SDK pins CPUExecutionProvider.
- [x] Record hashes and selected speakers before inference; preserve the frozen run manifest and verify all 12 speakers and referenced files.
- [x] Confirm DPDFNet stream alignment preserves exact duration through the local live backend with a 2 s synthetic pilot.

### Task 2: Implement the factorial screen

- [x] Reuse deterministic AIR RIR and pause pair helpers; crop the first 4 s from each utterance after deterministic speaker/file selection.
- [x] Render input, dereverb, DPDFNet2, and DPDFNet2→dereverb for clean-RIR, fan-like 20 dB, and fan-like 10 dB conditions. Treat the 4 s cuts as controlled offsets, not natural phrase endings.
- [x] Run dry controls and noise-only branches through each relevant model route; reconstruct the omitted dry-control rows from the frozen manifest and verify hashes.
- [x] Record per-pair active-speech/onset preservation after excluding the first 200 ms, fixed-reference tail windows, censoring, noise floor, and runtime.

### Task 3: Analyze without retuning

- [x] Aggregate paired branch differences by speaker and room, including medians and bootstrap intervals over the 12 speakers; show each room separately.
- [x] Apply the predeclared tail and speech-preservation gates. B passes the wet-input tail gate, fails the post-DPDFNet2 tail/location gates, and fails the active-speech and onset gates.
- [x] Keep B out of the DPDFNet2 product cascade. No retraining is justified until a separate frozen screen tests B behind the acoustically preferred DeepFilterNet Cap60 route; noisy tail scores in this screen are largely floor-censored.
- [x] Publish findings and limitations in `docs/benchmarking/air-dpdfnet-factorial-2026-10-10.md`; annotate censored comparisons in the aggregate plot.
