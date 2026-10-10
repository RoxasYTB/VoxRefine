# Bass-presence audition protocol

This exploratory offline protocol compares small low-shelf boosts after the current dereverberation/presence chain.

## Render set

The listening set contains a level-matched source reference, an Adobe reference, the current processed baseline, and four low-shelf variants at +0.5, +1.0, +1.5, and +2.0 dB centered at 180 Hz. A constant gain matches active-speech RMS across all renders; the shelf itself uses an RBJ biquad with slope 0.8. The filter is applied after the existing processed baseline.

## Measurements

The script records active-spectrum energy in 80–250 Hz, 250–1000 Hz, and 1–3 kHz; the common terminal-window level; peak amplitude; and clipping count. Its graph shows the active spectrum, a 30–500 Hz zoom, and the speech envelope. The output renders and sample-specific report are local ignored artifacts, not part of a source commit.

## Limits

These renders are an exploratory listening comparison on one recording. They do not establish a universal bass setting, a production default, or perceptual equivalence to Adobe. The unchanged source and Adobe render are references; the processed variants are level matched for audition. Keep the audio and sample-derived measurements private unless the speaker explicitly authorizes sharing them.

Run `scripts/benchmarks/universal_enhancer/iterate_presence_bass_testwav.py` from a workspace with the local benchmark inputs available. It writes WAVs, metrics, and a figure under the ignored `results/user-recording-test-2026-10-09/stupase-bass-matrix-01/` directory.
