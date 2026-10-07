import json
import shutil
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path

from voxrefine.audio import VoxRefineError
from voxrefine.listening import create_blind_listening_set


@unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is required.")
class BlindListeningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.a = self.root / "candidate-a.wav"
        self.b = self.root / "candidate-b.wav"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                        "-i", "sine=frequency=440:duration=1.2", "-c:a", "pcm_s16le", str(self.a)], check=True)
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                        "-i", "sine=frequency=660:duration=1.0", "-c:a", "pcm_s16le", str(self.b)], check=True)
        self.manifest = self.root / "candidates.json"
        self.manifest.write_text(json.dumps({"samples": [{"id": "sample-1", "candidates": [
            {"engine": "alpha", "path": self.a.name},
            {"engine": "beta", "path": self.b.name},
        ]}]}))

    def tearDown(self):
        self.temp.cleanup()

    def test_creates_repeatable_randomized_matched_copies_and_separate_key(self):
        blind, key_path = create_blind_listening_set(self.manifest, self.root / "listen-1", seed=123)
        revealed = json.loads(key_path.read_text())
        public = json.loads((blind / "manifest.json").read_text())
        mapping = {item["label"]: item["engine"] for item in revealed["samples"][0]["candidates"]}
        self.assertEqual(set(mapping.values()), {"alpha", "beta"})
        self.assertNotIn("engine", json.dumps(public))
        self.assertNotIn("alpha", json.dumps(public))
        self.assertNotIn("beta", json.dumps(public))
        self.assertEqual(public["samples"][0]["duration_seconds"], 1.0)
        paths = [blind / relative for relative in public["samples"][0]["files"]]
        for path in paths:
            with wave.open(str(path), "rb") as wav:
                self.assertEqual(wav.getframerate(), 48000)
                self.assertEqual(wav.getnchannels(), 1)
                self.assertAlmostEqual(wav.getnframes() / wav.getframerate(), 1.0, delta=1 / 48000)
        blind2, key_path2 = create_blind_listening_set(self.manifest, self.root / "listen-2", seed=123)
        revealed2 = json.loads(key_path2.read_text())
        self.assertEqual(mapping, {item["label"]: item["engine"] for item in revealed2["samples"][0]["candidates"]})
        self.assertEqual([Path(item).name for item in public["samples"][0]["files"]],
                         [Path(item).name for item in json.loads((blind2 / "manifest.json").read_text())
                          ["samples"][0]["files"]])

    def test_never_overwrites_existing_experiment(self):
        output = self.root / "existing"
        output.mkdir()
        with self.assertRaisesRegex(VoxRefineError, "already exists"):
            create_blind_listening_set(self.manifest, output)

