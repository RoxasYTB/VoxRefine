# Measured-RIR dereverberation on the user recording — 2026-10-09

## Question

Does the frozen measured-RIR dereverberator reduce the room-like residual after the user's preferred DeepFilterNet cap60 stage, and does it move the recording toward the paired Adobe v2 render?

## Protocol

- Input is the existing 6.227 s, mono, 48 kHz `DeepFilterNet cap60` render of `/home/yohan/Bureau/test.wav`.
- The paired Adobe v2 reference is the existing 48 kHz render in `results/user-recording-test-2026-10-09/06_adobe_v2.wav`.
- Both frozen checkpoints run at 16 kHz on the GTX 1050 Ti, then are resampled back to 48 kHz: A is the synthetic-RIR weak-over checkpoint; B is the procedural + measured BUT-RIR checkpoint. Their SHA-256 values and unmodified renders are in `results/user-recording-test-2026-10-09/measured-dereverb-adobe-01/report.json`.
- A common 20 ms RMS activity mask is derived from cap60. Research outputs remain at their native gain; a separate constant-gain copy of each file matches Adobe's RMS on that mask for listening only.
- Timing was remeasured after a warm-up pass, with seven synchronized GPU passes per checkpoint. This excludes model loading and resampling.

## Results

| Render | Active RMS vs cap60 | Post-speech residual 150–300 ms vs active cap60 RMS | Inference median |
|---|---:|---:|---:|
| Cap60 | 0.00 dB | −25.72 dB | — |
| A, synthetic-RIR | −0.64 dB | −26.23 dB | 25.3 ms |
| B, measured-RIR | −1.19 dB | −26.88 dB | 23.4 ms |
| Adobe v2 | −0.16 dB | −41.86 dB | Cloud service; not measured locally |

The final cap60 speech-active frame ends at about 5.64 s under an exploratory 8%-of-peak endpoint rule; the common speech reference uses a separate 3.5%-of-peak activity mask. The scored 150–300 ms window is 5.79–5.94 s. These are one terminal window from one recording (`n=1`), not an estimate across speech offsets. With endpoint thresholds between 10% and 25%, the end moves 20 ms earlier and these tail values move by less than 0.3 dB; at 6%, the rule includes the flat background floor and no longer identifies the speech end. Adobe is about 15.0 dB lower than B in the selected window; B is about 1.16 dB lower than cap60. The model's active-speech RMS is also 1.19 dB below cap60. This is a post-hoc diagnostic and does **not** show that B solves the remaining sound of room.

The spectrogram figure uses one fixed color scale and time axis. The dereverb model only processes 16 kHz audio, so its meaningful upper frequency is 8 kHz; the apparent 48 kHz output is an upsampled signal and does not add bandwidth. The image is descriptive, not a perceptual score.

![Cap60, frozen dereverb A/B and Adobe v2 spectrogram comparison, limited to the model’s native 0–8 kHz band](../../results/user-recording-test-2026-10-09/measured-dereverb-adobe-01/comparison.png)

![Zoom of the terminal post-speech residual](../../results/user-recording-test-2026-10-09/measured-dereverb-adobe-01/post-speech-tail.png)

Level-matched listening copies are available here:

- Cap60: ![Cap60, level-matched](/mnt/windata/C/Users/Yohan/Desktop/Projets/projects/VoxRefine/results/user-recording-test-2026-10-09/measured-dereverb-adobe-01/cap60-levelmatched.wav)
- Synthetic-RIR checkpoint A after cap60: ![Checkpoint A, level-matched](/mnt/windata/C/Users/Yohan/Desktop/Projets/projects/VoxRefine/results/user-recording-test-2026-10-09/measured-dereverb-adobe-01/synth_only_A-levelmatched.wav)
- Measured-RIR checkpoint B after cap60: ![Checkpoint B, level-matched](/mnt/windata/C/Users/Yohan/Desktop/Projets/projects/VoxRefine/results/user-recording-test-2026-10-09/measured-dereverb-adobe-01/measured_mix_B-levelmatched.wav)
- Adobe v2: already at its rendered level

![Adobe Podcast v2](/mnt/windata/C/Users/Yohan/Desktop/Projets/projects/VoxRefine/results/user-recording-test-2026-10-09/measured-dereverb-adobe-01/adobe_v2-levelmatched.wav)

## Interpretation and next step

The tail window after speech contains both room decay and residual noise; there is no dry, same-room recording of the user's voice to separate the two. The near-flat post-speech level after 150 ms is compatible with a noise floor, so the 16 dB difference from Adobe cannot be attributed to reverberation alone. A and B both reduce the measured residual only slightly, and B's active speech level drops more than A's. These findings reject B as an already-sufficient fix for this sample, but one recording cannot reject its measured-RIR result on controlled convolutional tests.

GPT Web recommends a frozen playback/recapture with a physical speaker and the existing USB microphone, including a separate room-only capture to measure the acoustic floor. The complete one-position pilot, alignment rule, censor-aware metrics, speech-preservation checks, and prospective gates are documented in [the playback–recapture protocol](playback-recapture-protocol-2026-10-09.md). The currently active output is Bluetooth, so this measurement has not been performed; a physical speaker connected to analog or HDMI is required. No checkpoint was changed or tuned.

This remains a research screen. It does not establish Adobe parity, universal performance, or live latency.
