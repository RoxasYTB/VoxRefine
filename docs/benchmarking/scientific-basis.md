# Scientific basis and evaluation protocol

This document separates published evidence, measurements made in VoxRefine, and hypotheses that still require listening tests. The objective is a reproducible, offline speech-enhancement alternative; Adobe Podcast is an external listening reference, not a source of training targets or a known algorithm specification.

## Evidence levels

Use these labels in reports:

- **Published result:** a result reported by a cited paper or upstream project, under that authors' data, hardware, and protocol.
- **VoxRefine measurement:** a value measured from a versioned local input, model, settings, and machine. Include hashes and command/configuration.
- **Listener observation:** a human preference or artifact report, tied to blind candidate IDs and the listening conditions.
- **Hypothesis:** a possible interpretation, never presented as an explanation of Adobe's internals without direct evidence.

Do not turn one clip's preference into a universal default, and do not call matching one reference sample “Adobe parity.”

## Current model roles

### AP-BWE

AP-BWE predicts extended speech amplitude and phase using a dual-stream convolutional GAN. Its stated task is bandwidth extension: reconstructing plausible high-frequency speech from a narrower-band input. It is not, by itself, a general-purpose denoiser. The paper reports strong bandwidth-extension scores and high throughput on the authors' hardware; those published runtime figures are not a VoxRefine benchmark and say nothing about its quality on our noisy mixtures. See [Lu et al., 2024](https://arxiv.org/abs/2401.06387) and the [upstream implementation](https://github.com/yxlu-0102/AP-BWE).

### DeepFilterNet3

DeepFilterNet is a single-channel speech-enhancement system using perceptually motivated features and deep filtering. Its papers describe the method and report benchmark results under their own data and hardware. Our sequence experiments answer a separate empirical question: whether pre-denoising helps AP-BWE on a given noisy input. See [Schröter et al., Interspeech 2023](https://arxiv.org/abs/2305.08227) and the [upstream repository](https://github.com/Rikorose/DeepFilterNet).

### What the current results support

On the supplied short clip, the listener preferred G (AP-BWE → DeepFilterNet3) over the earlier candidates and later preferred its -0.75 dB output trim in one easy synthetic condition. On a single synthetic speaker/noise pair at target active-speech SNRs +5 dB and -5 dB, both variants of G were judged poor. The trim cannot explain that quality failure because each blind pair shares the same enhanced signal and differs only by scalar gain. This is a failure region on one synthetic scenario, not a general SNR threshold.

The subsequent +5 dB attenuation-strength blind set compared AP-BWE-only with DeepFilterNet attenuation limits 6, 18, and 100 dB, with output gains left untouched. The listener reported that the voices sounded intact in all candidates; the difference was background noise, and C (100 dB, strongest suppression) was preferred as the cleanest. This is a listener observation and corrects any earlier inference that C perceptually damaged speech. Output-level plots are descriptive only. An exploratory clean-reference envelope diagnostic marks more low-energy frames as attenuated in C after a scalar fit, but this does not establish missing phonemes: noise contributes strongly to other candidates, alignment/spectral measures are approximate, and the listener heard intact speech.

A follow-up pause-only HF attenuation test was considered but not run. At a conservative speech activity threshold of -35 dB relative to the clean stem peak, with 120 ms safety padding, this short 3.4 s utterance contains no eligible speech-free frames. A meaningful pause-only ablation needs a longer clip with real pauses, or a clearly labeled exploratory VAD-derived mask; an oracle mask should remain diagnostic and never be presented as deployable behavior.

The next sequence comparison is limited to those difficult conditions:

- **G:** AP-BWE → DeepFilterNet3.
- **D:** DeepFilterNet3 at 48 kHz → resample to 16 kHz → AP-BWE to 48 kHz.
- **H:** DeepFilterNet3 → AP-BWE → DeepFilterNet3.

If a listening copy is speech-level matched, report the exact fixed gain used. Never apply a second loudness normalization that erases the intentional -0.75 dB trim.

### Latest low-SNR listening result (2026-10-07)

The owner reports that both blind +5 dB and -5 dB algorithm-order panels (G/D/H) over-suppressed the audio, with the -5 dB panel worse; all candidates damaged the original voice perceptually. This rejects those order permutations for this one synthetic case. It does not identify a universal SNR failure threshold.

Next, the blind ablation at +5 dB isolates post-AP-BWE DeepFilterNet attenuation strength and includes AP-BWE-only as a control. The attenuation limit is a linear mixture of the denoiser input and enhanced output; in this chain the denoiser input is AP-BWE output. It is not a speech-aware protection control. Listening copies preserve each candidate's output gain (no loudness matching), since the strength change itself alters output level. Reveal metrics therefore report levels descriptively; they do not rank quality. The clean speech stem is 16 kHz, so it can support diagnostics only below its 8 kHz Nyquist limit. Alignment-sensitive weak-frame fidelity scores are withheld until time alignment is independently verified.

## Dataset and generalization requirements

The workspace currently has one 3.4 s clean-speech stem at 16 kHz, one 5 s noise stem, and their associated noisy example. SNR mixtures made from these stems are useful for reproducible ablations, but they do not supply independent voices, microphones, rooms, noise types, or a true wideband clean reference. The 16 kHz clean stem cannot validate reconstructed content above its Nyquist limit (8 kHz).

Before ranking a general-purpose default, build a rights-cleared corpus with multiple speakers and recording conditions, including clean-speech regression clips, stationary and non-stationary noise, reverberation, music-under-speech, and low-SNR cases. Keep tuning and held-out recordings speaker-disjoint. Preserve clean 48 kHz reference stems for objective full-band comparison. Record source, exact version, license/permission, sample rate, channels, SNR calculation, processing chain, and SHA-256 for every item. See [`corpus-policy.md`](corpus-policy.md).

Use Adobe outputs only as paired, qualitative references for the exact same input when their terms permit. Do not treat them as ground truth or distill them into training targets without a separate rights review.

## Listening and metrics

Human listening remains the deciding evidence for naturalness and artifacts. Structure later panels around distinct dimensions—speech distortion/naturalness, background intrusiveness, and overall preference—following the spirit of [ITU-T P.835](https://www.itu.int/ITU-T/recommendations/rec.aspx?rec=7041); use blind randomized labels and clear headphone/playback instructions. For remote larger panels, follow [ITU-T P.808](https://www.itu.int/rec/T-REC-P.808-202106-I/en). The current project-owner comparisons are valuable directional feedback, not a statistically powered listening study.

Use complementary diagnostics rather than a single winner score:

- With aligned clean 48 kHz references: SI-SDR and intelligibility measures, plus band-limited spectral error. Document alignment and resampling. SI-SDR has known definition pitfalls; use the formulation discussed by [Le Roux et al. (2019)](https://arxiv.org/abs/1811.02508).
- For speech enhancement, report active-speech level, residual-noise level by bands and during pauses, true peak, loudness, and temporal pumping/artifacts.
- A non-intrusive metric such as DNSMOS P.835 may be reported as a proxy, never as the verdict. Its three scores separate speech quality, background quality, and overall quality; see [Reddy et al. (2022)](https://arxiv.org/abs/2110.01763). Verify an actually local implementation and its model-weight license before using it in this local-first project; the paper describes public availability as an Azure service.
- On real clips without a clean reference, do not report SI-SDR as though the reference existed. Use listening plus clearly labeled non-intrusive diagnostics.

No single SNR, PESQ-like metric, loudness value, or spectrogram proves “studio quality.” Report per-clip observations and distributions, not only a grand average.

## Open-source and licensing record

Check source, pretrained weights, training data, and evaluation audio separately.

- The AP-BWE upstream repository states that both code and weights are MIT-licensed and includes a separate `weights_LICENSE.txt`; retain those notices when redistributing the checkpoint. Confirm the exact downloaded artifact and hash.
- The DeepFilterNet upstream repository offers its code under MIT or Apache-2.0. For any redistributed binary/checkpoint, record the exact release, artifact hash, and included notices, and verify its terms at packaging time.
- Dataset terms do not follow from model-code licenses. Store provenance and permissions for every benchmark audio file.

This project should stay local-first by default. Do not upload user recordings to a remote metric service without a separate explicit user choice.

## Reproducible benchmark record

Every result should record: input/output hashes; source and license; exact model/checkpoint and hash; model order and settings; software versions; sample rate and channel conversion; SNR and how it was measured; any delay compensation, crop, or scalar gain; device, RAM/VRAM, cold start, steady-state duration, wall time and RTF. For live operation, also measure chunk size, lookahead, end-to-end latency, and boundary artifacts. Whole-file throughput faster than 1× does not establish low-latency streaming.

## References

1. Ye-Xin Lu, Yang Ai, Hui-Peng Du, Zhen-Hua Ling. “Towards High-Quality and Efficient Speech Bandwidth Extension with Parallel Amplitude and Phase Prediction.” 2024. [arXiv:2401.06387](https://arxiv.org/abs/2401.06387).
2. Hendrik Schröter, Tobias Rosenkranz, Alberto N. Escalante-B., Andreas Maier. “DeepFilterNet: Perceptually Motivated Real-Time Speech Enhancement.” Interspeech 2023. [arXiv:2305.08227](https://arxiv.org/abs/2305.08227).
3. Jonathan Le Roux, Scott Wisdom, Hakan Erdogan, John R. Hershey. “SDR – HALF-BAKED OR WELL DONE?” ICASSP 2019. [arXiv:1811.02508](https://arxiv.org/abs/1811.02508).
4. Chandan K. A. Reddy, Vishak Gopal, Ross Cutler. “DNSMOS P.835: A Non-Intrusive Perceptual Objective Speech Quality Metric to Evaluate Noise Suppressors.” ICASSP 2022. [arXiv:2110.01763](https://arxiv.org/abs/2110.01763).
5. ITU-T Recommendation P.808. “Subjective evaluation of speech quality with a crowdsourcing approach.” [Recommendation page](https://www.itu.int/rec/T-REC-P.808-202106-I/en).
6. ITU-T Recommendation P.835. “Subjective test methodology for evaluating speech communication systems that include noise suppression algorithm.” [Recommendation record](https://www.itu.int/ITU-T/recommendations/rec.aspx?rec=7041). Check current revision/status before formal publication.
