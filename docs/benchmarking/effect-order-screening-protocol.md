# Effect Order and Dynamics Screening Protocol

This protocol screens whether a small set of deterministic post-processing and stage-order changes are worth further listening tests after speech enhancement.

## Controlled comparisons

- Keep the base enhancement, high-frequency residual, gate, resampling, and active-speech mask frozen.
- Compare moving a gentle EQ across a branch-addition boundary, a mild compressor, a conservative sibilance de-esser, and modest EQ before the nonlinear enhancement model.
- For model-input changes, hold model version, settings, and seed fixed. Record inference time separately from the rest of the chain.
- Level-match every render using one shared active-speech RMS mask, then export common PCM settings.

## Measurements

Record active and weak/onset speech preservation, short tail level, voiced low-band energy, active 4–8 kHz and 8–12 kHz energy, peak/clipping, and the effect-specific activity or gain-reduction distribution. A single tail window includes residual noise and speech decay; it is only a local proxy, not an RT60 or direct dereverberation measurement.

## Interpretation limits

Frequency curves and tail energy do not determine perceived identity. Report numerical proximity to any commercial reference descriptively; do not tune an exact transfer curve from one recording or treat one sample as a universal validation. A listener remains the authority on timbre and artifacts.

The experiment-specific renders, audio, and detailed measurements belong under ignored `results/`, not in Git.
