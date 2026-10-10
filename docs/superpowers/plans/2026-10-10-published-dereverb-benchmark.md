# Published Dereverberation Model Benchmark (Plan)

**Status:** protocol locked; no outputs or model tuning yet.

**Goal:** Determine whether any openly released dereverberation model can materially reduce room tails while preserving speech on fresh speakers and fresh room responses. This is a research screen, not a claim of Adobe parity or universal quality.

## Candidate provenance gate

Only these two candidates enter the initial screen:

1. **ROSE-CD**, checkpoint `logs/CT_Reverb_EARS_pesq5e-4_L2/q0j0rpuu/last.ckpt`, repo revision `38e7dcb2f132b1400490426eef18bda9d03bb428`. Its model card identifies this as the one-step EARS-REVERB dereverberation checkpoint and declares MIT. Capture the final checkpoint SHA-256 before inference. Do not use the noise-only / speech-enhancement checkpoints.
2. **StuPASE**, model revision `539963f425ae201ac8c379b6480e3b182a4840ad`, official implementation revision `45614efae73dc49bf36018d1ca6a9319b3050038`. The model card declares Apache-2.0 and explicitly describes noise plus reverberation removal. It lists `CFM.pt` (758,024,988 bytes; SHA-256 `99531c8869761aca731a70b1f89875100cb643b560e52a4551b58300bb1ea01f`), `DeWavLM-R.pt` (1,261,990,714 bytes; SHA-256 `f3782058ec72f8a5107d17c4cbf66efc3fb398331490806e64d2bc577689a80e`), and `Vocoder_Mel-16k.pt` (242,750,990 bytes; SHA-256 `56a816ba2f662ebb73d30a75f95c97192b82c5dc66c82c99aa4ef70543fc0e51`). The model card explicitly labels the model Apache-2.0; retain this licensing evidence. The official path uses 16 kHz mono, 8 sampling steps, CFG 0.5 and sway coefficient −1.0. Its file inference wrapper rescales each output peak to the input peak; preserve and document that published behavior, and report candidate output level/gain so it cannot be mistaken for unmodified level. Preflight exact official code/path and dtype on this GTX 1050 Ti. If it OOMs, record `NOT_EXECUTABLE_4GB`; no offload, quantization, chunk/context reduction, or architecture changes after OOM.

**Excluded:** SGMSE+ HF revision `b6485214b3662a7f90309f397cacf1384046783c` exposes only a VoiceBank-DEMAND checkpoint, while its README points to dereverb checkpoints hosted separately. Do not conflate the local EARS-Reverb weight from the earlier experiment with the HF repository or include it in this fresh-model comparison. Reconsider only after standalone weight provenance and applicable license are independently established.

Dataset license and model licenses do not by themselves establish that every training source or weight is suitable for redistribution. This screen records repository/model card and checkpoint provenance; it does not grant new rights. No candidate is integrated or redistributed by this experiment.

## Data and frozen pairing

- Use new public dry speech and a new, deterministic seed namespace for room impulse responses. Do not reuse any previous user recordings, Adobe outputs, speaker IDs, RIRs, rendered inputs, or tuning examples.
- FLEURS was considered, but its current published feature schema omits the legacy `speaker_id` field even though the older TSV has an unlabeled extra column. Do not treat that column as a verified speaker identity. Use a source whose speaker identity is explicit in its official manifest, or stop until it can be established.
- Common Voice was audited but is not currently an admissible source: the current MDC-derived release has CC0 metadata and explicit `client_id` splits, but carries no-speaker-identification and no-rehosting terms; the large official corpus contains audio whose room acoustics are uncontrolled. A mirror has no license metadata. Reconsider only after terms allow internal stable-ID exclusion and a fixed release is locally authorized.
- Before selecting speakers, freeze an input-only dry-clip filter: 16 kHz analysis, no clipped samples, active speech SNR at least 25 dB, 150–600 ms post-activity energy at least 20 dB below active speech RMS, and stationary background 20 dB below active speech RMS. These are screening criteria, not claims that the clips are anechoic. Apply them mechanically to all eligible clips; do not relax thresholds to reach 24 speakers. Reject the corpus if fewer than 24 distinct, policy-eligible speakers remain.
- Exclude any source whose speakers are in prior VoxRefine tests or in model training sets declared on the candidate model cards. If source speaker IDs cannot be joined or reliably audited against a model's declared training data, label the evaluation as potentially contaminated and do not call it independent.
- Before processing any candidate, freeze the manifest of **24 speakers × 4 independently seeded synthetic rooms = 96 reverberant pairs**, plus matched dry controls. Record source URL/revision/license, speaker key, source clip key, crop offsets, RIR seed and measured RT60, and hashes. Select eligible clips using input-only conditions; no candidate outputs may influence inclusion.
- Use the same pairs and adapters for every model. Preserve native model input requirements; record all resampling and channel conversion. No output normalization, post-EQ, post-gain, or candidate-specific tuning.
- Run each model's published inference path and official recommended deterministic configuration. If stochastic, derive one seed from the pair key and never best-of-N. Freeze implementation and configuration before the first pair.

## Controls and metrics

For each speaker include a dry identity/control and a matched reverberant input; add noise-only/fan controls only if a separately licensed, fresh noise set is selected before outputs. Compare candidate speech output with the clean source using a raw-activity mask. Primary outcomes are:

- Tail/event energy change over 150–300 ms and 300–600 ms after clean speech events, with censor-aware windows and per-speaker/per-RIR summaries.
- Active, weak-speech, and onset preservation; dry-control drift; clipping and output floor.
- Runtime factor, model load time, warm steady-state latency, peak allocated/reserved VRAM, peak RAM, and checkpoint size on the GTX 1050 Ti. Report model-only and pipeline runtime separately. Batch RTF is not a live guarantee.

Frozen historical gates: W1 median at least +2 dB and at least 75% of speakers positive; W2 median at least −0.5 dB, regressions below −1 dB on no more than 10% of informative speakers; active p50 within ±0.5 dB, active/weak p10 above −1.5 dB, onset p10 above −1 dB; severe weak losses below −3 dB on no more than 10% and below −6 dB on no more than 3%. A candidate passes only if every primary gate passes. Report uncertainty and coverage; no single pooled score can hide failures.

## Execution order

1. Verify source and model metadata, capture immutable revisions and checkpoint SHA-256 values, and freeze environment/configuration receipts.
2. Freeze the fresh pair manifest and verify speaker identity, clean/RIR construction, and input-only eligibility.
3. Preflight each official model path on the 4 GB GPU with no adaptation. Record exact failures as results.
4. Run paired inference, controls, metrics, and deterministic repeat checks. Preserve published defaults exactly (StuPASE: 8 steps, CFG 0.5, sway −1.0); its wrapper's peak rescaling is an official processing step and must be declared in level statistics. Add no other output level adjustment.
5. Publish a report with model cards, exact commands, hashes, limitations, runtime/memory, per-speaker results, plots, and a strict GO/NO-GO for further work. No product integration unless a candidate passes and licensing is separately reviewed.
