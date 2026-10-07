# VoxRefine Studio Roadmap

> **For agentic workers:** Use `executing-plans` to execute this roadmap inline. The roadmap is split into separately reviewable workstreams; do not begin a production rewrite until the benchmark gate has selected models and measured latency on target hardware.

**Goal:** Evolve VoxRefine from an offline CLI prototype into a local voice-enhancement product with controllable speech, music, and background layers, plus an optional low-latency live mode.

**Architecture:** Keep Python as the experiment and offline orchestration layer while establishing a repeatable listening benchmark. Evaluate separate models for speech enhancement and speech/music/noise separation; do not assume one model can provide both. Once quality and latency are measured, put the real-time audio callback and model runtime behind a native streaming boundary (Rust or C++), with a desktop UI communicating through a narrow local API.

**Tech Stack:** Existing Python 3.10+ package and FFmpeg; candidate model runtimes PyTorch/ONNX Runtime; candidate native audio host Rust or C++ with a cross-platform audio library; desktop shell to be chosen after the model/runtime spike.

## Global Constraints

- Preserve local-only processing as the default; never upload user audio for inference.
- Treat the current machine, GeForce GTX 1050 Ti (4 GB VRAM), as the low-end GPU reference.
- Keep CPU-only offline processing supported.
- Do not claim Adobe Podcast parity without a controlled listening comparison.
- Keep the existing CLI usable until a validated successor replaces it.
- Record the model, exact weights, runtime, hardware, sample rate, latency, and license for every benchmark run.
- Keep voice enhancement, stem separation, and real-time audio as distinct workstreams with independent quality/performance gates.

---

## Repository Assessment

The current codebase is a compact, dependency-free Python CLI. `voxrefine/conversion.py` decodes inputs to mono PCM16/48 kHz; `voxrefine/engines.py` adapts DeepFilterNet 0.5.6 and RNNoise v0.1; `voxrefine/benchmark.py` preserves inputs, outputs, hashes, and timing. The conversion and publication path is careful about malformed inputs, accidental overwrite, output duration, and reproducibility. Tests cover those contracts and optional native engine integrations.

The prototype does not yet address the requested product shape: it has no GUI or microphone capture, only DeepFilterNet attenuation as a control, no speech/music/noise stems, no perceptual listening results, and no latency or memory measurements beyond wall-time factor. DeepFilterNet is invoked as a whole-file CLI; RNNoise is processed in 10 ms frames. Neither adapter is a complete user-facing real-time pipeline.

The Adobe reference page currently exposes separate voice, music, and background controls, an original/enhanced comparison, and individual stem downloads. These are product requirements to benchmark against, not evidence that the same algorithm or quality can be reproduced locally.

## Workstream A: Quality and Runtime Benchmark

### Task A1: Establish a lawful, representative listening corpus

**Files:**
- Create: `docs/benchmarking/corpus-policy.md`
- Modify: `examples/corpus.json` only when the corresponding audio files and provenance are available.

- [ ] Define categories: clean speech, stationary fan/hum, traffic, keyboard, reverberant room, music under speech, music-only, and difficult speech/noise overlap.
- [ ] For each clip, record source, license/permission, language/accent, duration, sample format, and whether a clean reference exists.
- [ ] Use paired synthetic mixtures for objective comparisons: mix a licensed clean French speech recording with a licensed noise/music recording at fixed SNRs, preserving clean speech and interferer stems as references.
- [ ] Keep Adobe's supplied sample, if its source terms permit, as a qualitative external reference; do not treat an Adobe-processed output as ground truth.
- [ ] Ensure no private or unlicensed recording enters `corpus/` or a committed manifest.

**Gate:** At least 30 short clips spanning the listed categories, including paired mixtures and clear rights metadata, before any model is ranked.

### Task A2: Add listening and perceptual evaluation outputs

**Files:**
- Modify: `voxrefine/benchmark.py`
- Create: `voxrefine/listening.py`
- Create: `docs/benchmarking/listening-protocol.md`
- Test: `tests/test_listening.py`

- [ ] Add blind randomized A/B or MUSHRA-style listening sheets with loudness-matched playback and anonymized model labels.
- [ ] Record separate ratings for speech naturalness/intelligibility, residual noise, artifacts, and preservation of music/background.
- [ ] Add optional objective diagnostics (SI-SDR for paired stems and a documented non-intrusive speech metric) without treating one scalar as a quality verdict.
- [ ] Extend reports with peak/RMS already present plus integrated loudness, true peak, process real-time factor, peak memory, model/runtime identity, and device.
- [ ] Keep per-clip observations and aggregate summaries; do not auto-select a winner from a single metric.

**Gate:** A reviewer can reproduce the same model run and find its original, cleaned audio, settings, and listening ratings from the report.

### Task A3: Run a controlled engine bake-off

**Files:**
- Create: `docs/benchmarking/model-bakeoff.md`
- Keep candidate adapters isolated until results justify adding them to `voxrefine/engines.py`.

- [ ] Compare existing RNNoise and DeepFilterNet against a current DeepFilterNet release and GTCRN's official streaming implementation.
- [ ] Evaluate one higher-quality 48 kHz speech enhancer such as MossFormer2_SE_48K in offline mode, after checking weight license and resource use.
- [ ] Measure CPU and GTX 1050 Ti runs separately; record VRAM, RAM, cold start, steady-state RTF, chunk latency, and whether the GPU is actually used.
- [ ] Include already-clean speech as a regression category to expose over-processing and voice coloration.

**Gate:** Pick separate defaults for `enhance` (offline quality) and `live` (latency/resource budget), or document why the same model wins both.

## Workstream B: Speech, Music, and Noise Controls

### Task B1: Determine the separation problem before selecting a separator

The requested outputs are *speech*, *music*, and *other background*. This differs from four-stem music separation (vocals/drums/bass/other), and a conventional speech enhancer's residual is not automatically a clean music stem. A cascade can leak music into speech or noise into the music stem; controls therefore need to mix explicit stems and expose the original/residual for recovery.

**Files:**
- Create: `docs/benchmarking/separation-models.md`
- Add a separation adapter only after the bake-off identifies a candidate with usable weights and redistribution/inference terms.

- [ ] Build paired mixtures with clean speech, music, and environmental noise stems.
- [ ] Compare MERL's MRX model from the Cocktail Fork project with BandIt v2 checkpoints trained on DnR v3; both target dialogue/speech, music, and effects stems. Include a speech-extraction-plus-residual cascade as a baseline.
- [ ] Use the DnR v3 French and multilingual variants for in-domain held-out cases where dataset terms allow; create separate licensed French noisy-speech mixtures because DnR is not a substitute for a voice-cleanup corpus.
- [ ] Inspect each isolated stem and the recombined output; test the `pass`, `music_sfx`, and residual-consistency settings rather than relying on default recombination.
- [ ] Evaluate ordinary speech, singing, overlapping voices, and music with vocals separately; document unsupported cases.
- [ ] Confirm the exact license for source code, pretrained weights, and training data before bundling any model.

**Gate:** The three faders must change their intended component independently on listening tests, and the default fader settings must reconstruct a natural mix without clipping or destructive loudness shifts.

### Task B2: Define the processing graph and controls

**Files:**
- Create: `voxrefine/mixer.py`
- Create: `tests/test_mixer.py`
- Modify: `voxrefine/audio.py` only if multichannel float processing is required.

- [ ] Represent speech/music/background as aligned float audio buffers with one shared sample clock and an explicit residual path.
- [ ] Give the voice control a documented meaning: blend between the separated speech stem and its enhanced version (0% natural stem, 100% full enhancement); give music and background controls independent stem levels. Validate these semantics against the current Adobe reference UI before matching labels.
- [ ] Add headroom management and an optional final limiter; keep fader operations downstream of separation/enhancement so they do not require model inference on every movement.
- [ ] Add invariants for equal lengths, sample rate, finite samples, channel layout, and recombination peak.
- [ ] Preserve stereo when the input and selected model support it; avoid unconditional mono downmix for final product exports.

**Gate:** Fader and mute operations are deterministic, reversible, and preserve duration; recombined stems do not clip at default settings.

## Workstream C: Low-Latency Live Mode and Product Shell

### Task C1: Prototype actual streaming before selecting the app stack

**Files:**
- Create: `docs/architecture/live-audio-spike.md`
- Prototype in a disposable branch or isolated module before changing the CLI's stable API.

- [ ] Capture and render real device audio with a bounded ring buffer; keep the audio callback free of allocation, disk I/O, Python, and blocking model initialization.
- [ ] Stream GTCRN and DeepFilterNet through chunk/stateful interfaces; document lookahead, algorithmic delay, chunk size, underruns, and RTF on the GTX 1050 Ti and CPU.
- [ ] Test GPU and CPU fallback while a microphone stream is active; ensure model loading and device changes do not stall the callback.
- [ ] Measure end-to-end capture-to-output latency, not only neural model inference time.
- [ ] Set the live latency target in the spike; report achieved latency and quality instead of calling any fast offline file job “real time.”

**Gate:** At least 30 minutes of stable live processing without underruns at the agreed latency on the reference machine, with a working CPU fallback.

### Task C2: Choose and implement the product shell

**Files (tentative; confirm after Task C1):**
- Create: `apps/desktop/` for the UI shell and mixer.
- Create: `native/voxrefine-audio/` for device I/O, buffers, and the low-latency inference boundary.
- Keep: `voxrefine/` for offline CLI, corpus tooling, and reproducible experiments.

- [ ] Compare a Rust/Tauri shell with a C++/Qt shell against audio-device integration, packaging, accessibility, and contributor skills.
- [ ] Keep UI/model communication local and typed; make GPU selection, CPU fallback, model download/provenance, and offline processing explicit.
- [ ] Implement original/enhanced audition, waveform playback, three aligned faders, mute/solo, export stems, and a low-latency live meter only after the graph is validated.
- [ ] Package models separately from the UI where licenses and download size require it; show checksums and exact model version.

**Gate:** User can import a local file, compare original and processed audio at matched level, adjust/export stems, and run the live path with visible latency/device state.

## Initial Decisions

- **Do not rewrite the project language now.** The current Python code is suitable for benchmark orchestration and offline inference experiments. Rewriting before model selection would move code without answering the quality or latency question.
- **Do not make the GTX 1050 Ti a hard GPU dependency.** It has 4 GB VRAM and the current machine reports about 1.4 GiB already occupied by desktop applications. Use CPU-capable low-compute enhancement as the baseline; GPU is an optional acceleration path. Three-stem separation is initially an offline feature until measurements prove otherwise.
- **Keep enhancement and separation as separate capabilities.** A speech denoiser can be the first practical quality win. Three independently adjustable stems require a different model problem and validation set.
- **Treat the existing engine adapters as experiment baselines.** Preserve their precise I/O checks and benchmark determinism; replace or extend adapters only when measured evidence supports it.
- **Start with a desktop app, not a browser-only inference UI.** Local device capture and dependable low-latency audio callbacks need native integration. A web UI can still be hosted inside a desktop shell.

## Candidate References to Verify During the Bake-Off

- DeepFilterNet upstream code/weights and real-time design: https://github.com/Rikorose/DeepFilterNet and https://arxiv.org/abs/2305.08227
- GTCRN official implementation, streaming example, and low-compute results: https://github.com/Xiaobin-Rong/gtcrn
- ClearerVoice-Studio's 48 kHz MossFormer2 speech enhancement candidate: https://github.com/modelscope/ClearerVoice-Studio
- MERL's Cocktail Fork MRX implementation and pretrained three-stem checkpoints: https://github.com/merlresearch/cocktail-fork-separation
- Divide and Remaster (DnR), targeting speech/music/sound-effects separation: https://zenodo.org/records/6949108
- BandIt source and inference checkpoints for cinematic audio separation: https://github.com/kwatcharasupat/bandit
- DnR v3 dataset documentation, including French/multilingual variants and stated CC BY-SA 4.0 dataset terms: https://github.com/kwatcharasupat/divide-and-remaster-v3
- Demucs is a useful music-stem baseline, not a three-way speech/music/environmental-noise solution: https://github.com/facebookresearch/demucs
- Adobe reference page and its controls/sample: https://podcast.adobe.com/fr/enhance

Before choosing a model, verify the current license of the exact pretrained weights independently of the repository's code license. Confirm GTX 10xx compatibility with the selected inference runtime and installed driver rather than inferring it from CUDA toolkit version alone.

## Audit Gaps and Open Questions

- No audio file or `corpus/` directory is present in the checkout; the checked-in example manifest points to ignored local files that are absent here.
- The Adobe page opened with a 38-second `Sample.mp4` and per-stem controls. Its provenance/usage terms were not established from the interface, so it has not been downloaded or added to the corpus.
- The environment has `/usr/bin/python3` 3.13.5 and FFmpeg 7.1.5, but no `python` executable. The checked-out project declares Python 3.10+, but the project's CI currently validates only 3.10 and 3.12.
- Choose an explicit live latency budget (for example, conversational monitoring vs. virtual microphone) before locking chunk size, audio backend, or UI messaging.
- Choose initial target OSes and distribution format before committing to the desktop shell.
- Decide whether the “background” stem includes room tone and environmental effects while keeping music as its own stem, including how to handle singing and overlapping speech.
