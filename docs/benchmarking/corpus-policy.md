# VoxRefine Listening Corpus Policy

## Purpose

Use this policy to assemble audio for evaluating speech enhancement and speech/music/background separation. Keep the corpus local and untracked. A model is not considered better because its output is quieter or because one automatic score increases.

## Rights and provenance

- Use recordings created for the project, recordings for which the project owner has documented permission, or datasets whose terms explicitly allow the intended local evaluation.
- Keep a source URL or project reference, the exact dataset/version, license or permission statement, and any attribution requirement for every file.
- Check source-code, checkpoint, and dataset terms separately. A model's code license does not establish rights to its weights or training data.
- Do not commit audio. Keep `corpus/` ignored by Git. Do not include private conversations, identifiable recordings, or third-party Adobe output unless its source and use terms are clear.
- The Adobe page's built-in `Sample.mp4` may be used only after confirming it is a provider-supplied demo and that local evaluation/download is permitted; retain its original source URL and do not redistribute it.

## Initial categories

Target 30–50 clips, mostly 5–20 seconds, with French speech across several voices and recording conditions. Include:

1. Clean studio speech (over-processing regression check).
2. Speech with fan, air conditioning, or steady hum.
3. Speech with traffic, wind, or outdoor ambience.
4. Speech with keyboard, handling noise, or intermittent household sounds.
5. Speech with room reverberation and distant/muffled capture.
6. Speech over music, including instrumental music and music with vocals.
7. Music-only and ambience-only controls (false speech creation / leakage checks).
8. Hard cases: low SNR, clipped speech, overlapping speakers, and speech mixed with singing.

At least half of the test set should be held back from any model tuning. Avoid near-duplicate speakers or source recordings across tuning and held-out partitions.

## Paired synthetic mixtures

For objective signal comparisons, mix independently sourced clean speech, music, and environmental noise so their unprocessed stems are known. Resample with a documented high-quality resampler, align starts, and apply gains before summing in float32. Save the exact inputs, gains, sample rates, and mixture hash in a sidecar record. Do not peak-normalize each stem independently after mixing. Leave headroom to avoid clipping.

Cover speech-to-interference ratios of approximately -5, 0, 5, and 10 dB where source material supports it. Include noise-only, music-only, and music-plus-noise mixtures. The mixture must be rendered from the recorded source stems and preserved byte-for-byte; compute SI-SDR or related paired scores against those references only when alignment and mixture construction are known.

Synthetic mixtures are useful for repeatable measurements but do not replace real recordings. Score naturalness, artifacts, and generalization on held-out real clips by listening.

## Listening protocol

- Randomize and blind engine labels for every listening session.
- Compare original and processed clips at matched perceived loudness so quieter output does not win by default.
- Use the same headphones/playback chain and listen to the full clip, including pauses and low-level tails.
- Rate separately: word intelligibility, voice naturalness, residual noise, musical preservation, artifacts/pumping, and overall preference.
- Keep per-listener scores and comments; publish medians and ranges rather than only one mean.
- Include an untouched-original reference and a known-clean reference when available.

## Manifest fields

Each `examples/corpus.json` sample should provide:

- `id`: stable ASCII identifier.
- `path`: path relative to the manifest.
- `category`: one of the categories above or a precise extension.
- `rights`: concise permission/license statement and attribution note.
- `source`: dataset name/version or creator reference.
- `language`: language and known accent, or `unknown`.
- `reference_stems`: optional relative paths to clean speech/music/noise references.
- `split`: `tuning`, `validation`, or `held_out`.

The current manifest reader accepts only `id`, `path`, `category`, and `rights`. Extend its schema and validation before relying on the additional fields; do not add fields that are silently ignored to a benchmark report.

## Reporting

For each run, record source checksum, prepared-audio checksum, engine/checkpoint hash, runtime and version, settings, device, sample rate, wall time, real-time factor, peak memory/VRAM if available, and output checksum. Keep listening ratings in a separate versioned report tied to the output checksum. Technical metrics are diagnostics, not a substitute for human listening.
