# Frozen playback–recapture protocol — 2026-10-09

## Purpose

Resolve whether the remaining audible room quality is still reverberation or has reached the physical room/microphone noise floor. This is a one-position pilot on the existing PC; it is not a multi-room product validation.

## Hardware precondition

- Input: USB Nor-Tec microphone, mono, 48 kHz.
- Output: a physical loudspeaker connected by analog or HDMI output. The currently observed default sink is a Bluetooth headset, which is not a valid acoustic playback path for this test.
- Place the loudspeaker and microphone in the chosen room, mark their positions and orientation, and do not move either during the paired capture. The microphone must not point directly at the speaker capsule if that would clip.
- Disable OS/app AGC, noise suppression, echo cancellation, automatic gain, spatial audio, and other effects. Record the exact input/output device names and mixer levels.

## Stimulus and captures

Prepare a mono, 48 kHz, 24-bit PCM WAV with: 1 s digital silence; a short sync chirp; 15–30 s of clean speech from licensed local samples, including several utterances separated by exactly 1 s digital silence; a second chirp; 1 s silence. Keep the source file and SHA-256. Do not normalize individual utterances.

1. Start raw microphone capture, then play the stimulus once through the physical speaker. Save the unprocessed recording as `wet_capture.wav`.
2. Without moving the hardware, keep the speaker silent and record 15 s of room-only microphone input as `room_only.wav`.
3. Run the exact same `wet_capture.wav` through DeepFilterNet cap60, cap60→DPDFNet2 full, and cap60→frozen measured-RIR checkpoint B. Keep native output gain and all raw WAVs. Do not use Bluetooth output or replay the processed candidates for the primary measurement.
4. Before any changed placement, repeat the paired wet and room-only captures once. This is a repeatability check, not an independent room sample.

## Analysis frozen before seeing model outputs

- Detect the two chirps on raw wet capture only. Fit one affine time map `t_capture = a * t_source + b`; apply it to all event masks. Never align each model output separately.
- Use the known inserted sentence/pause boundaries, with a 50 ms guard after each speech endpoint; do not estimate endpoint from a peak-relative threshold.
- Normalize each tail-window RMS to the wet capture's active-speech RMS from the immediately preceding utterance. Report 50–150, 150–300, and 300–600 ms windows for every pause.
- Estimate physical floor from room-only RMS and matched digital-silence windows processed through each system. Per candidate, floor is the larger of the room-only level and the corresponding dry-model floor; mark values within 3 dB of that floor as censored. Report floor counts and interval bounds instead of treating below-floor values as exact suppression.
- On known speech frames, report extra level change vs wet and vs cap60: active median, weak-speech p10/p50, rising/onset p10, and fractions with more than 3 dB and 6 dB extra attenuation. Do not interpret wet-vs-dry spectral error as a clean quality score because speaker, room and microphone color the capture.
- Report raw and processed clipping, device rates, dropped frames, alignment fit residual, and repeat-capture variation. Keep Adobe out of the primary causal comparison; its channel/downmix cannot establish room-only ground truth.

## Prospective decision gates

Treat the one-room result as a mechanism check only. A promising result requires B to reduce uncensored 150–300 ms tail by at least 2 dB versus cap60 in at least 3 measured pauses, with 300–600 ms consistent when above floor; no more than 1.5 dB extra weak-speech p10 loss and no more than 1 dB onset p10 loss versus cap60; and no dropped syllable, clipping, or severe artifacts. If all methods are within 3 dB of room-only/model floor, conclude the residual is not measurable as dereverb in this setup and do not tune to that floor.

## Current system observation

PipeWire currently reports the Nor-Tec microphone as running and a Bluetooth headset as the active default sink. Analog and HDMI outputs are present but suspended. Therefore no playback/recapture result has been collected: a physical speaker must be connected and positioned before automation can produce a meaningful acoustic capture. All processing and analysis after capture can be automated on this single PC.
