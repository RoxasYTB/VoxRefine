# User recording room-depth screen — 2026-10-09

## Question

Does a dereverberation stage after the user's preferred DeepFilterNet attenuation cap 60 reduce the audible room tail toward the paired Adobe Podcast v2 result, while retaining speech?

## Paired material and protocol

- Source recording: `/home/yohan/Bureau/test.wav`; prepared mono 48 kHz input is `results/user-recording-test-2026-10-09/01_input_mono_48k.wav`.
- Adobe v2 result: `results/user-recording-test-2026-10-09/06_adobe_v2.wav` (paired recording, 48 kHz stereo).
- Cap 60 uses the local DeepFilterNet3 CLI with `--atten-lim-db 60 --compensate-delay`. The CLI output is 1,440 samples (30 ms) shorter than the aligned reference due to delay compensation; output candidates were zero-padded at EOF for measurement only. This endpoint padding is outside speech and excluded from interpretation.
- DPDFNet2 uses the native 48 kHz ONNX model, 10 ms chunks and the repository's sample-alignment adapter (1,920 samples / 40 ms model delay); CPU processing took 2.685 s on a 6.197 s candidate (RTF 0.433). This is model-stage timing, not end-to-end device or live latency.
- A compact 16 kHz dereverb proof model was also run on the GTX 1050 Ti. Forward inference took 0.326 s (RTF 0.053; 85.2 MiB peak allocated). It was trained using synthetic procedural RIRs, not Adobe outputs or this recording.
- Speech offsets and pauses were frozen once from the MossFormer2 reference, then measured identically for all candidates. Each audition WAV is matched to the same active speech RMS by one constant gain. The report measures descriptive decay; no dry/anechoic stem is available, so it is not an RT60 estimate.

## Results

| Candidate | Long-pause RMS (dBFS) | Tail 150–300 ms (dB vs pre-offset) | Tail 300–600 ms (dB vs pre-offset) | Peak (dBFS) |
|---|---:|---:|---:|---:|
| Adobe v2 | −62.19 | −32.88 | −32.14 | −10.69 |
| DeepFilterNet cap 60 | −50.11 | −23.81 | −21.70 | −7.63 |
| Cap 60 + compact dereverb (100% wet) | −51.03 | −23.58 | −23.31 | −8.38 |
| Cap 60 + DPDFNet2 (25% blend) | −51.76 | −26.28 | −23.52 | −7.64 |
| Cap 60 + DPDFNet2 (50% blend) | −53.61 | −29.76 | −25.73 | −7.66 |
| Cap 60 + DPDFNet2 (75% blend) | −55.45 | −35.68 | −28.36 | −7.67 |
| Cap 60 + DPDFNet2 (100% stage) | −56.64 | −55.92 | −30.94 | −7.69 |

At the one scored terminal event, DPDFNet2 full-stage's 300–600 ms value is within 1.20 dB of Adobe, and its long-pause RMS is another 6.53 dB lower than cap 60. Its 150–300 ms value is far lower than Adobe (−55.92 vs −32.88 dB); this suggests gating/over-suppression during some pauses rather than a uniformly Adobe-like decay. The 75% blend has a less extreme 150–300 ms value (−35.68 dB) and a 300–600 ms value 3.78 dB above Adobe. Neither is a validated universal setting.

**Correction to the table above:** the earlier analysis called these values medians across offsets. In fact, the 150–300 ms and 300–600 ms windows were eligible at only one terminal speech offset in this 6.2 s recording (`n=1` for each window). These are single-event descriptive values, not medians or stable estimates of room decay. They also combine room tail and any residual noise; the recording has no dry reference to separate them. The per-offset analysis should therefore be read as a diagnostic for this sample only.

The compact dereverb model only improves the single scored 300–600 ms terminal value by 1.61 dB against cap 60, so it does not close this real-recording gap in its current form. WPE suppresses some short tails but is inconsistent at the terminal tail and is too slow in the screened offline configuration (about 1.3–2.8x RTF); it is not the lead candidate.

## Interpretation and limitations

The paired measurements show a strong post-speech residual reduction from DPDFNet2 after cap 60 in this one recording, with output active RMS nearly unchanged after level matching. Because the terminal windows are `n=1` and mix room tail with residual noise, this is not a stable room-tail estimate or evidence of full Adobe parity. The unusually deep 150–300 ms suppression and the model's prior weak-speech failures mean the full cascade may sound gated or truncate quiet consonants. This is one French voice recording and one microphone/room; it cannot establish universal performance. No model has been trained on Adobe outputs, and waveform/spectrogram matching alone cannot reveal Adobe's proprietary processing chain.

## Artifacts

- Metrics and level-matched listening files: `results/user-recording-test-2026-10-09/dereverb-analysis-05/`.
- Spectrograms: `results/user-recording-test-2026-10-09/dereverb-analysis-05/spectrograms.png`.
- Event-by-event decay chart: `results/user-recording-test-2026-10-09/dereverb-analysis-05/offset-decay.png`.
- Candidate generation records: `results/user-recording-test-2026-10-09/cap60-dereverb-screen-02/inference.json` and the processing commands in this report.
