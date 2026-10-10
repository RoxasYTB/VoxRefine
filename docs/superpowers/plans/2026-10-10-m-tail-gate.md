# M-tail-gate synthetic preflight

## Hypothesis

A small causal detector can identify only exposed late tails after direct speech has ended. A fixed, mild temporal gain is applied only when both “tail” confidence is very high and “speech” confidence is very low; otherwise output is the input. This trades missed tails for a deliberately measurable false-positive risk on speech.

## Frozen protocol

- New seed namespace. 24 synthetic wet TRAIN room/voice fixtures +12 dry TRAIN voices; 8 wholly new wet EVAL room/voice fixtures +8 dry EVAL voices. Every RIR is decomposed at 50 ms into early and late components; speakers/sources and rooms do not cross splits.
- 16 kHz, 512 FFT, 128 hop. Features: 32 band log energies, their temporal derivative, spectral flux; causal TCN with hidden 64 and a 51-frame receptive field (408 ms); two sigmoid outputs `p_tail`, `p_speech`.
- Clean-source activity from 20 ms frames/10 ms hop, dilated ±20 ms, marks protected speech. TAIL labels are only frames with no protected speech, late-component level >−50 dBFS, and late energy fraction >=0.8. Speech labels use the protected clean activity. Dry items provide negative tail and positive/negative speech supervision. Ambiguous frames are excluded from BCE.
- 2,000 fixed AdamW updates (`lr=2e−4`, `weight_decay=1e−4`, clip 3), BCE weights 1/1, deterministic shuffled/cyclic schedule; two repeated runs required identical within 1e−5.
- Fixed postfilter: activate only after two consecutive frames with `p_tail>=0.95` and `p_speech<=0.05`; apply −4 dB amplitude (gain `10^(−4/20)`) across the whole frame, with a fixed 10 ms attack/release ramp. Otherwise gain is exactly 1 and the output is the input.
- Waveform gates on 8 wet EVAL: W1 Mlow median >=2 dB and >=7/8 positive; W2 median >=−0.5 dB and none <−1 dB. Dry speech gates on all 8 dry EVAL: active p50 [−0.25,+0.25], active p10 >−0.75, weak p10 >−1, onset p10 >−0.75, weak frames <−3 dB <=3% and <−6 dB <=1%. Detector actuation on dry active and onsets <=0.5% each. Exact-zero 100/100, finite outputs, deterministic rerun.
- Diagnostics only: tail precision/recall, false-positive dry rate, event lengths, W1 late-energy coverage, `p_tail`/`p_speech` by active/weak/onset/tail. Decisions use waveform and dry actuation gates, not class accuracy.
- No audio corpus, prior DEV/HOLDOUT, or user-private recording. A pass allows only a fresh real-data evaluation plan, not an Adobe-parity claim.
