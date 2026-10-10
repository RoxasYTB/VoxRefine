# J-delay-predictor synthetic preflight (2026-10-10)

## Decision

**NO-GO. Stop before any real recordings or corpus preparation.** The causal delayed residual predictor did not generalize to four unseen synthetic source/RIR fixtures: all four W1 bands grew in energy instead of shrinking. Voice-preservation metrics pass, and exact-zero input remains exact zero. This is a synthetic feasibility result only; it says nothing about Adobe parity or real-world quality.

## Frozen protocol

- GPT Web proposal: causal GRU predicts a complex residual only from delayed history; no same-frame gain mask, free phase head, or clean-speech target.
- 16 kHz; `n_fft=512`, `hop_length=128`, four frame delay, 240 ms recurrent history, one unidirectional GRU (128 hidden units), zero-initialized bias-free linear complex residual head.
- Timing correction: with 128/16,000 hop, four frames equal **32 ms**, not 40 ms. The STFT configuration remains the same as the prior compact model; no timing knob was adjusted after results.
- Eight procedural TRAIN fixtures and four EVAL fixtures generated from separate fixed seeds and source/RIR parameters. EVAL is not used in optimization.
- 800 fixed AdamW updates (`lr=2e-4`, `weight_decay=1e-4`, gradient clipping 3), deterministic fixture schedule, seed 5302026.
- Unit-weight objective: W1 hinge toward 3 dB plus separate normalized waveform identity losses on clean-derived active, weak, and onset frames. Speech regions are compared to the noisy/reverberant input, not clean target audio.
- EVAL gates fixed before the run: per-fixture W1 >=2.5 dB and median >=3.0 dB; active p50 in [-0.25,+0.25] dB and p10 >−0.75 dB; weak p10 >−1 dB and <=3% / <=1% below −3 / −6 dB; onset p10 >−0.75 dB; per-fixture W2 >=−0.5 dB and median >=0 dB; exact zero invariant; finite metrics; two repeated deterministic runs.

## Results

Two full runs on an NVIDIA GTX 1050 Ti (PyTorch 2.7.1+cu118) produced byte-identical JSON receipts. EVAL W1 energy changes (positive means reduction) were **−0.161, −1.221, −0.107, −0.187 dB**, median **−0.174 dB**. So every fixture failed the W1 minimum of +2.5 dB, and the candidate amplified the target tail. W2 changes were **−0.420, −0.534, −0.680, −0.308 dB**, median **−0.477 dB**; two fixtures also crossed the −0.5 dB safety bound and the median safety gate failed.

Speech preservation passed all frozen gates: active p50 **+0.110, +0.001, −0.001, −0.053 dB**; active p10 **−0.106, −0.113, −0.078, −0.143 dB**; weak p10 **−0.017, −0.123, −0.091, −0.123 dB**; onset p10 **−0.083, −0.022, −0.098, −0.053 dB**. No weak frames fell below −3 dB or −6 dB. The exact-zero invariant passed. Initial loss gradient norms (W1 / active / weak / onset) were **0.597 / 0 / 0 / 0**; the identity penalties naturally have zero first derivative at the identity initialization, so these zeros are diagnostic, not a gate failure.

| Eval fixture | W1 reduction (dB) | W2 reduction (dB) | Active p50 (dB) | Active p10 (dB) | Weak p10 (dB) | Onset p10 (dB) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | −0.161 | −0.420 | +0.110 | −0.106 | −0.017 | −0.083 |
| 2 | −1.221 | −0.534 | +0.001 | −0.113 | −0.123 | −0.022 |
| 3 | −0.107 | −0.680 | −0.001 | −0.078 | −0.091 | −0.098 |
| 4 | −0.187 | −0.308 | −0.053 | −0.143 | −0.123 | −0.053 |

## Interpretation and next step

The predictor can leave dry/synthetic speech nearly unchanged, but this frozen setup does not learn a residual direction that transfers across unseen room/source pairs. On EVAL, estimated residual correlation with input W1 ranged from −0.186 to +0.943, indicating unstable residual phase/direction rather than a transferable late-tail estimate. Do not tune the same preflight after seeing these metrics or open prior consumed DEV/HOLDOUT sets. A new candidate would need a distinct, preregistered mechanism and fresh synthetic evaluation. A measured-RIR linear-prediction baseline could separately answer whether a causal predictor has enough information in the room domain, but cannot turn this failed model into a pass.

## Provenance

- Model: `scripts/benchmarks/universal_enhancer/compact_dereverb/model_j_delay_predictor.py`, SHA-256 `c44278b3a457d9ee70b85793b7e54abdc1cb00ee05e840161b37da01cfd6458b`.
- Auditor: `scripts/benchmarks/universal_enhancer/compact_dereverb/audit_j_delay_predictor.py`, SHA-256 `e0f6fedd6d5967e3763ded14d5f1feca9dae23b74381abdc48ad36ae4aafe33f`.
- Two receipts under ignored `.tools/compact-dereverb/j-delay-predictor-2026-10-10/` are byte-identical, SHA-256 `27374de1fb81955806350843f704cb80fdd207bceddf4cc9d5085e22ff71ff34`.
- No training corpus, DEV, HOLDOUT, or `/home/yohan/Bureau/test.wav` was read.
