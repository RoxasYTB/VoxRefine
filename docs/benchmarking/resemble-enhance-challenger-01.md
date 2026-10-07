# Challenger 01 — Resemble Enhance

Status: inference/runtime spike complete; perceptual selection is pending the user's listening comparison. This benchmark asks whether the perceptual enhancer sounds closer to Adobe Podcast on the existing matched French samples. SI-SDR is not used to rank it against Adobe, because Adobe output is not a clean reference.

## Candidate and provenance

- Upstream code: [`resemble-ai/resemble-enhance`](https://github.com/resemble-ai/resemble-enhance), commit `8e978149bfe8abab3eb77d965d579a111afdb0ff`.
- Model repository: [`ResembleAI/resemble-enhance`](https://huggingface.co/ResembleAI/resemble-enhance/tree/e10d34b312433b41a8eeeec43ebb6fa3219dab3f), revision `e10d34b312433b41a8eeeec43ebb6fa3219dab3f`. Its model card declares MIT and tags English; this spike therefore tests French generalization rather than assuming it.
- Code and checkpoint licenses are both shown as MIT by their respective upstream repositories. The model checkpoint was downloaded without authentication. Preserve both license notices if distributing the model.
- Model files used: `hparams.yaml` SHA-256 `80c3f15bc5a5b2cacf2c698699a0f6599d62911c0d53e1d6dee895c0d7cbaeac`; `ds/G/latest` SHA-256 `37a8eec1ce19687d132fe29051dca629d164e2c4958ba141d5f4133a33f0688f`; checkpoint (713,176,232 bytes) SHA-256 `f9d035f318de3e6d919bc70cf7ad7d32b4fe92ec5cbe0b30029a27f5db07d9d6`.
- The upstream project describes two stages: denoising, then perceptual enhancement/restoration. It supports denoise-only and enhancement with configurable NFE/solver/strength. See its [README](https://github.com/resemble-ai/resemble-enhance#readme).

The exact import-only compatibility patch for inference is included at [`patches/resemble-enhance-inference-only.patch`](patches/resemble-enhance-inference-only.patch), SHA-256 `9f2fde7dc1aa7ab1b54948fe758c43ca410c57bfcfedbae16b9a0c3b23648e66`. It redirects inference imports to the model classes, avoids eager imports of DeepSpeed training code, and pins the Hugging Face revision. It does not change model layers, weights, resampling, or inference math. On the pinned upstream commit, it passes `git apply --check --unidiff-zero`; use `git apply --unidiff-zero` to apply this zero-context patch. The unmodified VoxRefine package does not yet integrate this backend.

## Machine and protocol

- Python 3.13.5; PyTorch/Torchaudio `2.6.0+cu124`; CUDA 12.4.
- NVIDIA GeForce GTX 1050 Ti, compute capability 6.1, 4 GB VRAM. CUDA smoke operation and real inference succeeded.
- Desktop applications already used about 1.45 GiB VRAM before the run.
- Model audio rate is 44.1 kHz. Inputs are downmixed to mono by averaging channels; outputs are resampled to 48 kHz only for comparison and export.
- Enhancement settings matching the upstream CLI defaults: midpoint solver, NFE 64, lambda 1.0, tau 0.5. Reduced-NFE comparisons change only NFE (16). Torch CPU/CUDA seeds are fixed to `1701`.
- The 3.4 s user demo uses upstream 30 s chunking. Three 38–39 s French audiobook excerpts were rendered at NFE 16 using 3 s chunks and 1 s overlap after the upstream 30 s setting ran out of VRAM on this occupied GPU. This produces a chunk every 2 s; it is explicitly a low-VRAM experiment, not upstream-standard output, and could introduce seams or alter room tails.
- Cold model load was about 5.3–5.6 s. Reported RTF is warm inference-call time divided by audio duration and excludes model loading. Peak allocated and reserved VRAM are measured per inference call.
- Listening copies use 12 s excerpts starting at 5 s for each LibriVox track, and a 3.37 s crop for the user demo. All candidates within a track have exactly the same output sample count and receive constant gain to −24 LUFS; no dynamic loudness processing or limiter is used. Four-times oversampled true-peak estimates after matching range from −3.1 to −7.6 dBFS, so the comparison copies do not clip. Resampling/downmix is the only format adaptation.

## Runtime observations

| Audio | Mode | NFE | Chunk / overlap | Warm RTF | Peak allocated VRAM |
|---|---|---:|---:|---:|---:|
| User noisy demo, 3.4 s | Denoise | — | 30 s / 1 s | 0.131 | 1,525 MiB |
| User noisy demo, 3.4 s | Enhance | 64 | 30 s / 1 s | 1.975 | 2,227 MiB |
| User noisy demo, 3.4 s | Enhance | 16 | 30 s / 1 s | 1.068 | 2,227 MiB |
| User noisy demo, 3.4 s | Enhance | 16 | 3 s / 1 s | 1.331 | 2,129 MiB |
| LibriVox, 38–39 s (median-like range across 3) | Denoise | — | 3 s / 1 s | 0.069–0.082 | 1,503–1,508 MiB |
| LibriVox, 38–39 s (3 samples) | Enhance | 16 | 3 s / 1 s | 1.149–1.171 | 2,129 MiB |

RTF below 1 is faster than audio duration. The official NFE 64 enhancer is not real-time on this GTX 1050 Ti in these measurements. NFE 16 approaches real-time on the very short sample but remains slower than real-time on the long low-VRAM chunked runs. Denoise-only is fast, but the current upstream wrapper still loads the full enhancement bundle; its memory result is not a standalone small-denoiser benchmark. This offline spike does not establish microphone latency or streaming stability.

The upstream 30 s enhancer call OOMed on a 39 s clip with the desktop using approximately 1.5 GiB VRAM. A 3.4 s clip succeeds at the upstream chunk default. This is a constrained-machine observation, not a general claim that the model cannot process long audio on a 4 GB card.

## Listening set and decision gate

The local, loudness-matched listening copies are under `results/resemble-enhance-01/listening/` (ignored by Git because they contain audio). `manifest-levelmatched.json` records input hashes, crop points, measured pre-match LUFS/peaks, gains, and output hashes. All variants are cropped from the same time point with identical sample counts. A rough envelope cross-correlation is within one 10 ms frame for most paths; Enhance sometimes peaks around ±10–15 ms versus the raw input. This is only a diagnostic because enhancement changes the waveform; it is not a calibrated latency measurement. It does mean a future dry/wet slider must verify alignment instead of assuming generated speech is sample-identical. Compare the labels directly; this is not a blind test. The primary candidates are raw, Adobe v2, VoxRefine's earlier AP-BWE→DeepFilterNet output where available, Resemble denoise, and Resemble Enhance. The short user demo also includes NFE 64 and two NFE 16 chunk conditions to distinguish step count from chunking.

A simple 10 ms RMS-envelope check at the expected 2 s chunk stride is not a reliable seam detector: speech onsets and pauses create large frame-to-frame changes both at and away from chunk boundaries. The 12 s chunked comparison excerpts begin at 5 s, so listen especially around approximately 1, 3, 5, 7, 9, and 11 s for boundary-related changes. The short 3.37 s demo has no strong RMS step at its 3 s chunk condition, but that does not validate long-form seams. A go decision requires the user to hear a clear, repeatable gain in studio/natural voice quality on more than one French excerpt without worse consonants, weak speech, noise, or chunk seams. If the model sounds synthetic or only changes brightness while leaving the cheap-mic impression, it is not selected. If it is promising but the 3 s chunk version sounds worse, compare the same NFE at 30 s versus 3 s on the short sample before blaming the model. No default pipeline, production CLI backend, or live-mode claim changes until this gate passes.

Next after listening: if Resemble is compelling, find the largest stable chunk and benchmark the full NFE 64 path on the 1050 Ti, then implement an optional backend. If it is not compelling, test a single 48 kHz ClearerVoice enhancement model as Challenger 02, after checking that exact model's weights license and runtime requirements.
