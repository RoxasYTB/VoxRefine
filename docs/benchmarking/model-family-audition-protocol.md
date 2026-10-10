# Model-Family Audition Protocol

Use this protocol when small EQ or dynamics changes are not perceptually distinguishable and a backend comparison is needed.

## Candidate families

Compare a denoiser-only baseline, a dedicated blind dereverberation model on both raw and denoised input, a classical linear-prediction dereverberator, a stronger neural dereverberator as an over-suppression control, and the current generative candidate. Keep all model outputs otherwise unprocessed.

## Listening controls

- Align the same interval and use one shared active-speech mask.
- Apply one constant active-speech RMS gain per file; do not use loudness automation or candidate-specific EQ.
- Randomize labels and compare twice: natural voice/timbre first, then room tail and weak consonants.
- Keep the commercial render descriptive, not a fitted target.

## Measurements and limits

Record the short tail window, active and weak speech-envelope retention, peak/clipping, and active spectral bands. A single terminal window is only a local indicator: it mixes acoustic decay, residual noise, and any final speech. A candidate with a very low tail but lost weak speech is an over-suppression failure. Listening and independent multi-speaker/room evaluation are still required before product use.

Keep experiment audio and sample-derived metrics in ignored `results/`; commit the reproducible script and protocol only.
