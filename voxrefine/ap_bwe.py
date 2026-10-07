"""Pinned AP-BWE 16 kHz -> 48 kHz inference adapter.

The upstream source and checkpoint are supplied by the user. VoxRefine never
downloads code or weights from this module.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

from .audio import AudioDomain, VoxRefineError
from .ffmpeg_io import decode_raw, probe_audio


UPSTREAM_REPOSITORY = "https://github.com/yxlu-0102/AP-BWE"
UPSTREAM_COMMIT = "751710f22404c27e5bcc983248f8b856a04b8422"
MODEL_CONFIG = "configs/config_16kto48k.json"
MODEL_INPUT_RATE = 16000
MODEL_OUTPUT_RATE = 48000
CHECKPOINT_FILENAME = "g_01000000.pt"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _git_commit(source: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise VoxRefineError(
            f"AP-BWE source must be a Git checkout at {UPSTREAM_COMMIT}: {source}"
        ) from error
    return result.stdout.strip()


def _upstream_imports(source: Path):
    """Load only the official model modules from an explicitly pinned checkout."""
    source = source.resolve()
    if not (source / "env.py").is_file() or not (source / "models/model.py").is_file():
        raise VoxRefineError(f"Not an AP-BWE source checkout: {source}")
    for name in ("env", "datasets", "datasets.dataset", "models", "models.model", "utils"):
        module = sys.modules.get(name)
        module_path = getattr(module, "__file__", None) if module else None
        if module_path and not Path(module_path).resolve().is_relative_to(source):
            raise VoxRefineError(
                f"Python already loaded a conflicting '{name}' module from {module_path}. "
                "Run AP-BWE in a clean environment to avoid mixing model source trees."
            )
    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    try:
        env = importlib.import_module("env")
        dataset = importlib.import_module("datasets.dataset")
        model = importlib.import_module("models.model")
    except (ImportError, AttributeError) as error:
        raise VoxRefineError(
            f"Could not import AP-BWE at {source}; install its documented dependencies. {error}"
        ) from error
    for module in (env, dataset, model):
        if not Path(module.__file__).resolve().is_relative_to(source):
            raise VoxRefineError("AP-BWE imports resolved outside the pinned source checkout.")
    return env.AttrDict, dataset.amp_pha_stft, dataset.amp_pha_istft, model.APNet_BWE_Model


class APBWE16to48:
    """Deterministic AP-BWE inference with bounded model activation memory."""

    def __init__(
        self,
        source: Path,
        checkpoint: Path,
        *,
        device: str = "cpu",
        threads: int = 4,
        chunk_frames: int = 2048,
    ):
        try:
            import numpy as np
            import torch
            import torchaudio.functional as audio_functional
        except ImportError as error:
            raise VoxRefineError(
                "AP-BWE needs PyTorch, torchaudio and NumPy in the active Python environment. "
                "Install the AP-BWE dependencies in a dedicated environment."
            ) from error
        if threads < 1:
            raise VoxRefineError("AP-BWE thread count must be at least 1.")
        if chunk_frames < 128:
            raise VoxRefineError("AP-BWE chunk size must be at least 128 STFT frames.")
        if device not in {"cpu", "cuda"}:
            raise VoxRefineError("AP-BWE device must be 'cpu' or 'cuda'.")
        if device == "cuda" and not torch.cuda.is_available():
            raise VoxRefineError("CUDA was requested but is unavailable in this PyTorch environment.")

        self.np = np
        self.torch = torch
        self.audio_functional = audio_functional
        self.source = source.expanduser().resolve()
        self.checkpoint = checkpoint.expanduser().resolve()
        if not self.checkpoint.is_file():
            raise VoxRefineError(f"AP-BWE checkpoint not found: {self.checkpoint}")
        self.source_commit = _git_commit(self.source)
        if self.source_commit != UPSTREAM_COMMIT:
            raise VoxRefineError(
                f"Unsupported AP-BWE source revision {self.source_commit}; expected {UPSTREAM_COMMIT}."
            )
        self.config_path = self.source / MODEL_CONFIG
        if not self.config_path.is_file():
            raise VoxRefineError(f"AP-BWE config not found: {self.config_path}")
        config_data = json.loads(self.config_path.read_text(encoding="utf-8"))
        if (config_data.get("lr_sampling_rate") != MODEL_INPUT_RATE
                or config_data.get("hr_sampling_rate") != MODEL_OUTPUT_RATE):
            raise VoxRefineError("AP-BWE config must describe the pinned 16 kHz -> 48 kHz model.")
        torch.set_num_threads(threads)
        self.device = torch.device(device)
        if device == "cuda":
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        attr_dict, self._stft, self._istft, model_class = _upstream_imports(self.source)
        self.config = attr_dict(config_data)
        started = time.perf_counter()
        self.model = model_class(self.config).to(self.device)
        self.model_init_seconds = time.perf_counter() - started
        load_started = time.perf_counter()
        try:
            parameters = inspect.signature(torch.load).parameters
            load_options: dict[str, Any] = {"map_location": self.device}
            if "weights_only" in parameters:
                load_options["weights_only"] = True
            checkpoint_data = torch.load(str(self.checkpoint), **load_options)
            if not isinstance(checkpoint_data, dict) or "generator" not in checkpoint_data:
                raise ValueError("checkpoint has no 'generator' state dictionary")
            self.model.load_state_dict(checkpoint_data["generator"], strict=True)
        except Exception as error:
            raise VoxRefineError(f"Could not load AP-BWE checkpoint {self.checkpoint}: {error}") from error
        self.model.eval()
        self.checkpoint_load_seconds = time.perf_counter() - load_started
        self.chunk_frames = chunk_frames
        self.context_frames = 3 * (int(config_data["ConvNeXt_layers"]) + 1)
        if self.context_frames >= chunk_frames:
            raise VoxRefineError("AP-BWE chunk size must exceed its temporal context.")
        self.threads = threads
        self.last_run: dict[str, float | int] = {}

    def process_samples(self, samples):
        """Enhance a mono 16 kHz float32 array and return mono 48 kHz float32."""
        torch, np = self.torch, self.np
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if samples.size < 1024:
            raise VoxRefineError("AP-BWE needs at least 1024 input samples at 16 kHz.")
        audio = torch.from_numpy(samples.copy()).reshape(1, -1).to(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            wide = self.audio_functional.resample(
                audio, orig_freq=MODEL_INPUT_RATE, new_freq=MODEL_OUTPUT_RATE
            )
            magnitude, phase, _ = self._stft(
                wide, self.config.n_fft, self.config.hop_size, self.config.win_size
            )
            total_frames = magnitude.shape[-1]
            magnitude_parts = []
            phase_parts = []
            for core_start in range(0, total_frames, self.chunk_frames):
                core_end = min(total_frames, core_start + self.chunk_frames)
                input_start = max(0, core_start - self.context_frames)
                input_end = min(total_frames, core_end + self.context_frames)
                enhanced_mag, enhanced_phase, _ = self.model(
                    magnitude[..., input_start:input_end], phase[..., input_start:input_end]
                )
                local_start = core_start - input_start
                local_end = local_start + (core_end - core_start)
                magnitude_parts.append(enhanced_mag[..., local_start:local_end])
                phase_parts.append(enhanced_phase[..., local_start:local_end])
            enhanced_mag = torch.cat(magnitude_parts, dim=-1)
            enhanced_phase = torch.cat(phase_parts, dim=-1)
            output = self._istft(
                enhanced_mag, enhanced_phase,
                self.config.n_fft, self.config.hop_size, self.config.win_size,
            )
        output = output[0].detach().cpu().numpy().astype(np.float32, copy=False)
        expected_samples = samples.size * 3
        if output.size < expected_samples:
            output = np.pad(output, (0, expected_samples - output.size))
        else:
            output = output[:expected_samples]
        if not np.all(np.isfinite(output)):
            raise VoxRefineError("AP-BWE generated non-finite output samples.")
        self.last_run = {
            "input_samples": int(samples.size),
            "output_samples": int(output.size),
            "stft_frames": int(total_frames),
            "chunk_count": len(magnitude_parts),
            "inference_seconds": time.perf_counter() - started,
        }
        return output

    def process_file(
        self,
        source_file: Path,
        target_file: Path,
        *,
        ffmpeg: str = "ffmpeg",
        channel_policy: str = "reject",
    ) -> dict[str, Any]:
        source_file = source_file.expanduser().resolve()
        target_file = target_file.expanduser().resolve()
        if target_file.suffix.lower() != ".wav":
            raise VoxRefineError("AP-BWE output must use a .wav extension.")
        if target_file.exists():
            raise VoxRefineError(f"Output already exists: {target_file}")
        if channel_policy not in {"reject", "downmix", "first"}:
            raise VoxRefineError("AP-BWE channel policy must be 'reject', 'downmix', or 'first'.")
        media = probe_audio(source_file, ffmpeg)
        if channel_policy == "reject" and media.info.channels != 1:
            raise VoxRefineError(
                f"AP-BWE requires mono input; source has {media.info.channels} channels. "
                "Choose --channel-policy downmix or first to define the stereo conversion explicitly."
            )
        executable = shutil.which(ffmpeg)
        if executable is None:
            raise VoxRefineError(f"FFmpeg not found: {ffmpeg}")
        target_file.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".voxrefine-apbwe-", dir=target_file.parent
        ) as directory:
            raw_input = Path(directory) / "input.f32"
            raw_output = Path(directory) / "enhanced.f32"
            domain = AudioDomain(MODEL_INPUT_RATE, 1, "f32", "raw")
            decode_started = time.perf_counter()
            decode_raw(
                media, raw_input, domain, ffmpeg,
                "mono" if channel_policy == "downmix" else "first" if channel_policy == "first" else "reject",
            )
            decode_seconds = time.perf_counter() - decode_started
            input_samples = self.np.fromfile(raw_input, dtype="<f4")
            started = time.perf_counter()
            enhanced = self.process_samples(input_samples)
            inference_seconds = time.perf_counter() - started
            enhanced.astype("<f4", copy=False).tofile(raw_output)
            encoded = Path(directory) / "enhanced.wav"
            command = [executable, "-nostdin", "-hide_banner", "-loglevel", "error",
                       "-f", "f32le", "-ar", str(MODEL_OUTPUT_RATE), "-ac", "1",
                       "-i", str(raw_output), "-c:a", "pcm_s24le", "-f", "wav", str(encoded)]
            encode_started = time.perf_counter()
            try:
                subprocess.run(command, check=True, capture_output=True)
            except subprocess.CalledProcessError as error:
                detail = error.stderr.decode("utf-8", errors="replace") if error.stderr else str(error)
                raise VoxRefineError(f"Could not write AP-BWE output: {detail}") from error
            encode_seconds = time.perf_counter() - encode_started
            try:
                os.link(encoded, target_file)
            except FileExistsError as error:
                raise VoxRefineError(f"Output already exists: {target_file}") from error
        self.last_run.update({"decode_seconds": float(decode_seconds),
                              "inference_seconds": float(inference_seconds),
                              "encode_seconds": float(encode_seconds)})
        return {
            "input": {"path": str(source_file), "sample_rate": media.info.sample_rate,
                      "channels": media.info.channels, "codec": media.info.codec,
                      "duration_seconds": media.info.seconds},
            "output": {"path": str(target_file), "sample_rate": MODEL_OUTPUT_RATE,
                       "channels": 1, "duration_seconds": len(enhanced) / MODEL_OUTPUT_RATE},
            "model": self.identity(),
            "processing": {**self.last_run, "device": str(self.device),
                           "threads": self.threads, "chunk_frames": self.chunk_frames,
                           "context_frames": self.context_frames,
                           "channel_policy": channel_policy,
                           "resampler": "FFmpeg source decode to 16 kHz, then torchaudio sinc resampling to 48 kHz"},
        }

    def identity(self) -> dict[str, Any]:
        torch = self.torch
        return {
            "repository": UPSTREAM_REPOSITORY,
            "source_commit": self.source_commit,
            "config_path": str(self.config_path),
            "config_sha256": _sha256(self.config_path),
            "checkpoint_path": str(self.checkpoint),
            "checkpoint_sha256": _sha256(self.checkpoint),
            "checkpoint_size_bytes": self.checkpoint.stat().st_size,
            "weights_license": "MIT; see upstream weights_LICENSE.txt",
            "torch_version": torch.__version__,
            "torchaudio_version": importlib.import_module("torchaudio").__version__,
            "device": str(self.device),
            "parameter_count": sum(parameter.numel() for parameter in self.model.parameters()),
            "model_init_seconds": self.model_init_seconds,
            "checkpoint_load_seconds": self.checkpoint_load_seconds,
        }
