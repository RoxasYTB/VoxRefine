# K delayed linear prediction / WPE preflight

## Question

Can a delayed single-channel linear predictor trained on an independent calibration utterance remove a synthetic room tail on a different utterance from the same room, without removing speech? Separate representational ceiling (oracle) from blind filter estimation (WPE).

## Frozen protocol

- 12 newly generated synthetic rooms. Each room has independent dry calibration utterance A and evaluation utterance B, sharing only the room impulse response. B is never used to estimate coefficients or update WPE variance.
- 16 kHz STFT, `n_fft=512`, `hop=128` (8 ms), delay `D=4` (32 ms), `K=30` taps (240 ms history after the delay).
- Oracle LP: fit per-frequency ridge regression on A only, target `R_A=X_A-S_A` (known synthetic dry reference), diagonal load `lambda=1e-4 trace(ZᴴZ)/K`; apply the frozen filter to B.
- Blind WPE: estimate per-frequency delayed predictor on A only, 3 iterations; initialize variance from A power, update with causal EMA alpha=0.9 on A outputs; same D/K and diagonal-load rule. Apply frozen final filter to B with no adaptation.
- Held-out scoring: Mlow reduction (input/output STFT energy, 150–300 Hz) on W1 (150–300 ms post-pause), W2 (300–600 ms), active/weak/onset waveform preservation versus reverberant input, dry-safety by applying the same filter to dry B, exact-zero and finiteness. Synthetic room eligibility for W1/W2: B input low-band energy above −60 dB and at least 6 dB above dry B in that window. Require all 12 W1 rooms eligible; otherwise mark NON-EXECUTABLE, not a pass.
- Frozen gates: W1 median Mlow >=+2.0 dB and >=10/12 rooms with positive W1; W2 median Mlow >=−0.5 dB, certain regressions (<−1 dB) <=10%, coverage >=25%; speech active p50 in [−0.5,+0.5] dB, active p10 >−1.5 dB, weak p10 >−1.5 dB, onset p10 >−1 dB, weak frames <−3 dB <=10%, <−6 dB <=3%; dry-safety active p50 in [−0.25,+0.25], active p10 >−0.75, weak p10 >−1.0, onset p10 >−0.75; exact-zero 100/100, no NaN/Inf.
- Decision: oracle failure rejects delayed LP/subtraction for this task. Oracle pass + WPE failure isolates blind estimation. Both pass permits a fresh measured-RIR validation only; no Adobe-parity or universal claim.

## Execution

Implement one deterministic synthetic-only auditor, run two identical receipts, compare hashes, and publish the outcome without changing thresholds after evaluation. No recordings, prior DEV/HOLDOUT, or user-private audio are read.
