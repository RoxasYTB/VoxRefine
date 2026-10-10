# K delayed linear prediction / mono WPE preflight (2026-10-10)

## Decision

**NO-GO for the frozen single-channel delayed-LP/WPE configuration.** The oracle filter, trained with the known synthetic dry reference from calibration utterance A and then frozen on independent utterance B, failed to reduce the target low band in most rooms. Blind WPE performed worse. Both arms amplified the measured W1 low band on most or all evaluation rooms, regressed W2, and failed speech/dry safety gates. Stop this particular delayed-subtraction path before any real recordings.

## Frozen experiment

- 12 newly generated synthetic rooms; each room uses independent dry calibration A and evaluation B utterances, sharing the same synthetic RIR. B is never used to fit coefficients or update WPE variance.
- 16 kHz STFT, 512 FFT, 128-sample hop (8 ms), four-frame delay (32 ms), 30 delayed taps (240 ms history).
- **Oracle LP:** per-frequency ridge regression estimates the late residual `R_A = X_A − S_A` from A only, with fixed diagonal loading `1e−4 trace(ZᴴZ)/K`; apply unchanged coefficients to B.
- **Blind mono WPE:** three fixed iterations on A only; per-frequency variance from A power with causal EMA α=0.9; same delay, taps and loading; apply final coefficients to B with no evaluation adaptation.
- W1 is 150–300 ms after the synthetic pause, W2 300–600 ms. Mlow uses STFT energy from 150–300 Hz. All 12 rooms passed the frozen W1 input-eligibility floor; 11/12 had informative W2 energy.
- Frozen primary gates: W1 median Mlow >=+2 dB and positive in >=10/12 rooms; W2 median >=−0.5 dB with <=10% certain regressions below −1 dB; fixed speech and dry-safety limits; exact-zero and finite output.

## Results

Both deterministic runs produced byte-identical receipts on the GTX 1050 Ti. A complex ridge sanity check using a known generated coefficient matrix had relative prediction error **0.000116**, supporting the complex regression orientation used in the experiment. The learned fixed filters returned exact zeros for digital silence **24/24** checks, and all coefficients/outputs were finite.

| Arm | W1 Mlow median | W1 rooms >0 | W2 Mlow median | W2 rooms <−1 dB | Median output RMS change |
|---|---:|---:|---:|---:|---:|
| Oracle delayed LP | −5.272 dB | 2/12 | −18.024 dB | 11/11 | −0.05 dB |
| Blind mono WPE | −9.085 dB | 0/12 | −19.051 dB | 11/11 | +5.90 dB |

Oracle median speech preservation was: active p50 approximately 0 dB, active p10 −1.56 dB, weak p10 −3.07 dB, onset p10 −1.87 dB; 13.36% of weak frames fell below −3 dB. Applying the same coefficients to dry B changed active p50 by +1.33 dB, outside the ±0.25 dB dry-safety gate.

WPE median speech change was active p50 +5.53 dB, active p10 near 0 dB, weak p10 +1.35 dB; dry active p50 changed +6.68 dB. While weak frames were not attenuated, this large gain violates speech-level and dry-safety gates. All 12 W1 rooms were eligible; WPE had 0/12 positive W1 reductions. W2 regressed by more than 1 dB in every one of the 11 informative rooms for both methods.

## Interpretation

The oracle result is the important discriminator: even with the exact room shared and the synthetic dry reference available for calibration A, the fitted delayed filter does not transfer safely to a different utterance B. In this setup the past contains predictable voiced structure as well as reverberation, and the filter cannot subtract one without perturbing the other. Blind WPE compounds that ambiguity and produces large level changes. The results reject these frozen single-channel estimators and this calibration/evaluation setup; they do not disprove multichannel WPE, all dereverberation, or neural systems generally.

No regularization, tap count, variance update, room, or gate was adjusted after seeing these metrics. Do not rerun against prior DEV/HOLDOUT to rescue this path. A new approach should exploit an explicit speech/reverberation distinction or additional channels rather than another unconstrained delayed subtraction. Any subsequent candidate needs a distinct frozen protocol and fresh evaluation.

## Provenance

- Auditor: `scripts/benchmarks/universal_enhancer/compact_dereverb/audit_k_delayed_lp_wpe.py`, SHA-256 `c79b21408cbc1246302203e34506b3fb6855148daa2dc19fd125962b50e81d3b`.
- Repeated receipts in ignored `.tools/compact-dereverb/k-delayed-lp-wpe-2026-10-10/` are byte-identical, SHA-256 `dfdfe3120ec1dae3067c56a71fa3e6d7b587df68b3ab0597aacb0efe20056be7`.
- No corpus, prior DEV/HOLDOUT, or `/home/yohan/Bureau/test.wav` was read.
