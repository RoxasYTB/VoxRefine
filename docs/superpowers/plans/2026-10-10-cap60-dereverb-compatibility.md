# Cap60–Dereverb Compatibility Implementation Plan

> **For agentic workers:** execute this plan inline, preserving all unrelated workspace changes. `test.wav` remains sealed and is never loaded, rendered, or used for decisions.

**Goal:** Determine whether compact dereverberator B adds measurable tail reduction after the acoustically preferred DeepFilterNet Cap60 denoiser without raising its output floor or damaging dry speech.

**Architecture:** Replay the same frozen 12-speaker × 8-AIR-RIR factorial cases used in the completed DPDFNet2 cascade screen. Compare only Cap60 and Cap60→B, passing dry, RIR-only, noisy mixtures, and their noise-only controls through matching branches. Keep the source corpus, model weights, clips, SNRs, pause windows, censoring rule, and room strata frozen; add a pause-decay slope diagnostic.

**Tech Stack:** Python 3.13, local DeepFilterNet CLI, PyTorch CUDA compact dereverberator, NumPy, SciPy, SoundFile, Matplotlib.

## Global Constraints

- Use only the frozen LibriSpeech/AIR cases from `.tools/compact-dereverb/air-dpdfnet-factorial-2026-10-09/manifest.json`.
- No training or tuning; test only Cap60 and Cap60→B at attenuation limit 60 dB.
- Do not process `test.wav`.
- Preserve full 8.8 s controlled-offset clips and exact matched inputs for dry, RIR-only, fan20, fan10, and noise-only conditions.
- Keep audio, checkpoints, and row-level outputs under ignored `.tools`; publish aggregates and plot only.
- Apply common-floor censoring at floor +3 dB and cluster-bootstrap speakers, not frames or speaker/RIR rows.

## File Structure

- Create `scripts/benchmarks/universal_enhancer/compact_dereverb/bench_air_cap60_factorial.py` for deterministic rendering, Cap60 invocation, B processing, row metrics, summary, plot, and hash/audit checks.
- Create `docs/benchmarking/air-cap60-b-factorial-2026-10-10.md` for protocol, results, limitations, and go/no-go.
- Create `docs/benchmarking/assets/air-cap60-b-factorial-2026-10-10.png` as the aggregate tail/floor figure.
- Keep data and generated audio in `.tools/compact-dereverb/air-cap60-b-factorial-2026-10-10/`.

## Frozen decision gates

- B adds at least +2 dB median tail reduction in 150–300 ms and at least +1 dB in 300–600 ms on RIR-only cases.
- The B effect is positive in at least 3 of 4 AIR locations.
- Cap60→B noise-only floor is no more than 3 dB higher than Cap60 in either scored tail window.
- Incremental dry active-speech median and onset p10 are each greater than −1 dB.
- Weak-frame p10 and p50 changes remain within ±3 dB; do not treat weak-frame gain as automatically positive.
- Do not claim a tail improvement for a censored comparison. Report the robust 80–500 ms pause-decay slope only where samples remain above the common floor +3 dB.

## Tasks

### Task 1: Freeze and audit inputs

**Files:** Create the benchmark script above; read the prior frozen manifest and matching source files.

- [x] Check source hashes, checkpoint hash, 12-speaker identity, eight RIR identities, and that `test.wav` is not in the manifest.
- [x] Generate deterministic PCM inputs for clean controls, RIR-only, fan20/fan10, and matched noise-only. Record the exact DeepFilterNet CLI and PCM conversion.
- [x] Pilot clean and noisy signals; confirm 16 kHz, finite aligned output, B preserves Cap60's output length, and record any deterministic EOF trimming. The measured CLI trims 480 samples (30 ms) at the end with zero-sample offset; compare every route over the shared available prefix instead of padding synthetic samples.

### Task 2: Run both routes

**Files:** Write ignored outputs to `.tools/compact-dereverb/air-cap60-b-factorial-2026-10-10/`.

- [x] Run frozen DeepFilterNet with `--atten-lim-db 60 --compensate-delay` for all matched inputs.
- [x] Run frozen B on each Cap60 render on CUDA; retain Cap60 and cascade outputs only as local ignored artifacts.
- [x] Record aggregate elapsed time, sample counts, model hashes, and branch identity.

### Task 3: Analyze and decide

**Files:** Update the benchmark script, public report, and aggregate plot.

- [x] Calculate the fixed-reference 50–150/150–300/300–600 ms tail, matched dry/noise-only common floor, censor status, and 80–500 ms decay slope.
- [x] Calculate dry active median, weak p10/p50, and onset p10; summarize paired differences with speaker-cluster bootstrap and per-location medians.
- [x] Apply only the predeclared gates above. Floor, active-speech, and onset gates fail; do not integrate Cap60→B unchanged.
- [x] Document protocol, exact counts, runtime, censoring, limitations, and aggregate figure in `docs/benchmarking/air-cap60-b-factorial-2026-10-10.md`; verify 768 unique rows and all route/condition counts before checkpointing.
