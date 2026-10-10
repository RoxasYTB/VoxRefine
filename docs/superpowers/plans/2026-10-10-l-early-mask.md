# L-early-mask synthetic preflight

## Hypothesis

A gain-only STFT model can learn a transferable late-energy mask when its training target is explicitly derived from known synthetic RIR decomposition. This tests direct/early versus late structure, without a W1 energy hinge, learned confidence, free phase, or waveform residual.

## Frozen protocol

- New fixed namespace/seeds. 12 synthetic TRAIN rooms + 8 wholly new EVAL rooms; each RIR is split at 50 ms into direct/early and late components. Four independent dry TRAIN utterances and four disjoint dry EVAL utterances provide identity examples. No utterance or RIR crosses TRAIN/EVAL.
- For each reverberant pair, create full and early-only waveforms from the same dry source and apply one shared scalar level normalization to both. STFT is 16 kHz, 512 FFT, 128 hop, centered Hann.
- Oracle mask `m*=clip(|T_early|/(|X_full|+eps),0,1)`. Dry identity examples have target mask 1. Before training, evaluate `m*` on the 8 EVAL rooms; require oracle W1 median >=3 dB and at least 7/8 W1 positive. If this fails, stop with NO-GO before fitting.
- Model: five small 2D conv layers with time dilations 1,2,4,8 and a scalar head; gain-only `m=1−0.75 sigmoid(a)`, output initialized at `a=-3` with deterministic small nonzero head weights. No phase/confidence/residual branch.
- 1,000 fixed AdamW steps, `lr=2e−4`, `weight_decay=1e−4`, gradient clip 3, deterministic shuffled/cyclic order, 12 reverberant +4 dry TRAIN items. Loss is energy-weighted SmoothL1 between predicted and oracle masks; dry identity target is 1. There is no W1 hinge.
- Candidate EVAL gates: W1 Mlow median >=2 dB and positive in >=7/8 rooms; W2 Mlow median >=−0.5 dB and no room <−1 dB; dry EVAL active p50 [−0.25,+0.25] dB, active p10 >−0.75, weak p10 >−1, onset p10 >−0.75, weak frames <−3 dB <=3% and <−6 dB <=1%; exact-zero 100/100; no NaN/Inf; two identical runs within 1e−5.
- Diagnostics only: energy-weighted oracle mask loss, predicted attenuation versus true late-energy fraction, predicted mask statistics on dry active/weak/onset and W1. Do not gate on internal mask statistics.
- If the oracle ceiling fails, stop immediately. If oracle passes but fitted model fails, no tuning on this EVAL. Even a PASS licenses only fresh, speaker-disjoint real-data planning, not an Adobe-parity claim.

## Execution

Implement one synthetic-only auditor, run the frozen oracle check, then (only if eligible) train/evaluate twice. Save a report and focused checkpoint. Do not read recordings, prior DEV/HOLDOUT, or user-private audio.
