# Reproducible blind listening sets

`voxrefine blind-listen` prepares listening copies from outputs that already
exist. It does not run or rank enhancement engines. Source outputs are never
modified. Every clip's candidates are cropped to that clip's shortest decoded
duration after mono 48 kHz conversion, and loudness matched using
FFmpeg's two-pass EBU R128 normalization. The source hashes, normalization
measurements, settings, and label mapping are recorded in the reveal key.

Example candidate manifest:

```json
{
  "samples": [
    {
      "id": "clip-001",
      "candidates": [
        {"engine": "raw", "path": "../corpus/clip-001.wav"},
        {"engine": "gtcrn", "path": "../renders/gtcrn/clip-001.wav"},
        {"engine": "deepfilternet", "path": "../renders/deepfilternet/clip-001.wav"}
      ]
    }
  ]
}
```

Paths are relative to the candidate manifest. Generate a new output directory:

```sh
voxrefine blind-listen results/experiment/candidates.json \
  --output results/experiment/listening-01 --seed 17
```

The output contains `blind/` for the listener and a sibling `key/reveal.json`
for later decoding. Keep the key out of the listening folder. The seed makes
the label assignment reproducible, and the order is changed across consecutive
clips when the random permutation would otherwise repeat. The public listening
manifest has clip IDs and playback settings only; the reveal key contains the
engine names and source/output checksums. All files are local and `results/` is
git-ignored.

This is a preparation utility, not an evaluation dataset or a quality measure.
The current Adobe/GTCRN/raw example is a single smoke-test clip. A useful
listening study needs several diverse French recordings and independent
feedback from the listener.
