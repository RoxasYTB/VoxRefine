# GTCRN upstream parity check

This is a numerical implementation check, not a speech-quality score.
The upstream assets were downloaded from the official
[GTCRN streaming fixture directory](https://github.com/Xiaobin-Rong/gtcrn/tree/main/stream/test_wavs)
and `stream/onnx_models/gtcrn_simple.onnx` for a local, one-time comparison.
They are not bundled into the application or redistributed by VoxRefine.

## Assets and provenance

| Asset | SHA-256 | Samples / role |
|---|---|---|
| `gtcrn_simple.onnx` | `b4718df6228e7bdf1a8a435cf98f838636eb2fd331acabf86ba87c5192ebcb87` | Streaming model; hash matches local `results/gtcrn-demo/gtcrn_simple.onnx` |
| `mix.wav` | `8d47e1d03eeb457c2549be79c8ec33a349ccd79f21c3add05f946f8f760c5a99` | 156,302 samples, mono 16 kHz |
| `enh_stream.wav` | `44e53d66614c6077351f2f3e1cc51159de750a5815b14137b1019904afcdd4bb` | 156,160 samples, mono 16 kHz, upstream streaming output |
| `enh.wav` | `7bd0ab23386ac2c075b579bcbee2dc0e3a72238d65cdbd0e508fdefeb2df37f4` | 156,160 samples, mono 16 kHz, upstream offline output |

## Measured comparison

VoxRefine processed `mix.wav` through the streaming ONNX graph using CPU, with 4096-sample input chunks. The PCM comparison is against the common prefix of the VoxRefine output and `enh_stream.wav`; length differences are reported separately and are not hidden by alignment or resampling.

- VoxRefine output: 156,302 samples (source duration preserved).
- Upstream `enh_stream.wav`: 156,160 samples.
- Pearson correlation on the common prefix: `0.99998782`.
- Float32 RMSE on the common prefix: `0.00031417`.
- Maximum absolute difference: `0.00792367`.
- RMSE in first 256 samples: `0.00212122`; first 1024: `0.00241499`; first 10,000: `0.00080256`.

The first/last samples are more sensitive to STFT padding, OLA normalization, and WAV output length than the steady-state body. The current implementation explicitly preserves the input duration, while the upstream fixture is shorter by 142 samples. This check therefore verifies strong aligned waveform agreement but does not claim bit-identical boundaries or identical padding policy. It also does not measure perceived enhancement quality.

## Regression policy

The test suite checks that streaming output length equals the number of input samples for hop-aligned and partial-hop inputs, output is invariant to chunk boundaries, and separate sessions reset state. A future fixture-based regression may use the pinned upstream model/output pair and compare the common prefix, while asserting both lengths independently. Do not apply a tolerance to a shifted signal or use correlation alone: record RMSE, maximum error, lengths, model hash, and runtime version.
