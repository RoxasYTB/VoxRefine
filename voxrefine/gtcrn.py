"""GTCRN streaming ONNX adapter (CPU only, model file supplied by the user)."""

import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import statistics

from .audio import AudioDomain, VoxRefineError
from .backends.base import BackendSpec
from .ffmpeg_io import probe_audio
from .conversion import prepared_for_backend


SAMPLE_RATE = 16000
FFT_SIZE = 512
HOP_SIZE = 256


def _optional_runtime():
    try:
        import numpy as np
        import onnxruntime as ort
        return np, ort
    except ImportError as error:
        raise VoxRefineError(
            "GTCRN needs optional dependencies. Install them with: pip install 'voxrefine[gtcrn]'"
        ) from error


class GTCRNSession:
    """Stateful one-frame GTCRN session. Each instance owns its own model caches."""

    def __init__(self, model_path: Path, threads: int = 1):
        if threads < 1:
            raise VoxRefineError("--ort-threads must be at least 1.")
        self.np, ort = _optional_runtime()
        self.model_path = model_path.expanduser().resolve()
        if not self.model_path.is_file():
            raise VoxRefineError(f"GTCRN ONNX model not found: {self.model_path}. No model is downloaded automatically.")
        options = ort.SessionOptions()
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = threads
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        try:
            self.session = ort.InferenceSession(
                str(self.model_path), sess_options=options, providers=["CPUExecutionProvider"]
            )
        except Exception as error:
            raise VoxRefineError(f"Could not load GTCRN ONNX model: {error}") from error
        if self.session.get_providers()[0] != "CPUExecutionProvider":
            raise VoxRefineError("GTCRN must run with ONNX Runtime CPUExecutionProvider.")
        self.input_names = {item.name for item in self.session.get_inputs()}
        self.output_names = [item.name for item in self.session.get_outputs()]
        self.runtime_metadata = {
            "onnxruntime_version": ort.__version__,
            "provider": "CPUExecutionProvider",
            "available_providers": ort.get_available_providers(),
            "intra_op_threads": threads,
            "inter_op_threads": 1,
            "graph_optimization": "ORT_ENABLE_ALL",
        }
        if not {"mix", "conv_cache", "tra_cache", "inter_cache"} <= self.input_names:
            raise VoxRefineError("Unsupported GTCRN model: expected official streaming inputs mix/conv_cache/tra_cache/inter_cache.")
        if len(self.output_names) < 4:
            raise VoxRefineError("Unsupported GTCRN model: expected enhanced spectrum and three cache outputs.")
        self.caches = {
            "conv_cache": self.np.zeros((2, 1, 16, 16, 33), dtype=self.np.float32),
            "tra_cache": self.np.zeros((2, 3, 1, 1, 16), dtype=self.np.float32),
            "inter_cache": self.np.zeros((2, 1, 33, 16), dtype=self.np.float32),
        }
        self.window = self.np.hanning(FFT_SIZE).astype(self.np.float32) ** 0.5
        # torch.stft(center=True) pads half a window at both boundaries.
        self.input_buffer = self.np.zeros(FFT_SIZE // 2, dtype=self.np.float32)
        self.total_input = 0
        self.frames = 0
        self.output_emitted = 0
        self.initial_frame = True
        self.ola = self.np.zeros(FFT_SIZE, dtype=self.np.float32)
        self.weight = self.np.zeros(FFT_SIZE, dtype=self.np.float32)
        self.closed = False
        self.inference_ms: list[float] = []
        self.pipeline_ms: list[float] = []

    def warmup(self, frames: int = 20) -> float:
        """Exercise ORT on a disposable signal, then reset all audio state."""
        if frames <= 0:
            return 0.0
        started = time.perf_counter()
        for _ in range(frames):
            self.process(self.np.zeros(HOP_SIZE, dtype=self.np.float32))
        elapsed = time.perf_counter() - started
        self.caches = {
            "conv_cache": self.np.zeros((2, 1, 16, 16, 33), dtype=self.np.float32),
            "tra_cache": self.np.zeros((2, 3, 1, 1, 16), dtype=self.np.float32),
            "inter_cache": self.np.zeros((2, 1, 33, 16), dtype=self.np.float32),
        }
        self.input_buffer = self.np.zeros(FFT_SIZE // 2, dtype=self.np.float32)
        self.total_input = self.frames = self.output_emitted = 0
        self.initial_frame = True
        self.ola = self.np.zeros(FFT_SIZE, dtype=self.np.float32)
        self.weight = self.np.zeros(FFT_SIZE, dtype=self.np.float32)
        self.inference_ms.clear()
        self.pipeline_ms.clear()
        return elapsed

    def _frame(self) -> bytes:
        np = self.np
        pipeline_started = time.perf_counter()
        frame = self.input_buffer[:FFT_SIZE]
        spec = np.fft.rfft(frame * self.window).astype(np.complex64)
        mix = np.stack((spec.real, spec.imag), axis=-1)[None, :, None, :].astype(np.float32)
        inputs = {"mix": mix, **self.caches}
        started = time.perf_counter()
        values = self.session.run(self.output_names, inputs)
        self.inference_ms.append((time.perf_counter() - started) * 1000)
        outputs = dict(zip(self.output_names, values))
        enhanced = outputs.get("enh")
        if enhanced is None:
            enhanced = values[0]
        self.caches = {
            "conv_cache": outputs.get("conv_cache_out", values[1]),
            "tra_cache": outputs.get("tra_cache_out", values[2]),
            "inter_cache": outputs.get("inter_cache_out", values[3]),
        }
        complex_spec = enhanced[0, :, 0, 0] + 1j * enhanced[0, :, 0, 1]
        segment = np.fft.irfft(complex_spec, n=FFT_SIZE).astype(np.float32) * self.window
        self.ola += segment
        self.weight += self.window * self.window
        self.frames += 1
        self.input_buffer = self.input_buffer[HOP_SIZE:]
        self.pipeline_ms.append((time.perf_counter() - pipeline_started) * 1000)
        valid = np.divide(self.ola[:HOP_SIZE], self.weight[:HOP_SIZE], out=np.zeros(HOP_SIZE, dtype=np.float32), where=self.weight[:HOP_SIZE] > 1e-8)
        if self.initial_frame:
            self.initial_frame = False
            valid = self.np.zeros(0, dtype=np.float32)
        self.ola[:HOP_SIZE] = self.ola[HOP_SIZE:]
        self.ola[HOP_SIZE:] = 0
        self.weight[:HOP_SIZE] = self.weight[HOP_SIZE:]
        self.weight[HOP_SIZE:] = 0
        self.output_emitted += len(valid)
        return valid.astype("<f4", copy=False).tobytes()

    def process(self, samples):
        if self.closed:
            raise VoxRefineError("GTCRN session is closed.")
        np = self.np
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        self.total_input += samples.size
        self.input_buffer = np.concatenate((self.input_buffer, samples))
        chunks = []
        while self.input_buffer.size >= FFT_SIZE:
            output = self._frame()
            if output:
                chunks.append(output)
        return b"".join(chunks)

    def flush(self):
        if self.closed:
            return b""
        # Complete the centered STFT with right-side padding like torch.stft(center=True).
        wanted_frames = 1 + (self.total_input + HOP_SIZE - 1) // HOP_SIZE
        already_emitted = self.output_emitted
        chunks = []
        while self.frames < wanted_frames:
            if self.input_buffer.size < FFT_SIZE:
                needed = FFT_SIZE - self.input_buffer.size
                self.input_buffer = self.np.concatenate(
                    (self.input_buffer, self.np.zeros(needed, dtype=self.np.float32))
                )
            output = self._frame()
            if output:
                chunks.append(output)
        result = b"".join(chunks)
        # Trim right-side padding to preserve the exact source duration.
        remaining = max(0, self.total_input - already_emitted)
        result = result[:remaining * 4]
        self.output_emitted = already_emitted + len(result) // 4
        return result

    def close(self):
        self.closed = True

    def spec(self) -> BackendSpec:
        domain = AudioDomain(SAMPLE_RATE, 1, "f32", "raw", frame_samples=HOP_SIZE)
        return BackendSpec("gtcrn", "GTCRN", "official streaming ONNX", "stream", domain,
                           domain, "gtcrn", True, False, False, False)


class GTCRN:
    name = "gtcrn"

    def __init__(self, model: Path, threads: int = 1):
        self.model_path = model.expanduser().resolve()
        # Validate dependencies/model/provider eagerly so CLI errors stay actionable.
        self._np, self._ort = _optional_runtime()
        if not self.model_path.is_file():
            raise VoxRefineError(f"GTCRN ONNX model not found: {self.model_path}. No model is downloaded automatically.")
        if threads < 1:
            raise VoxRefineError("--ort-threads must be at least 1.")
        self.threads = threads

    def spec(self) -> BackendSpec:
        domain = AudioDomain(SAMPLE_RATE, 1, "f32", "raw", frame_samples=HOP_SIZE)
        return BackendSpec("gtcrn", "GTCRN", "official streaming ONNX", "stream", domain,
                           domain, "gtcrn", True, False, False, False)

    def identity(self) -> dict[str, str]:
        return {"engine": self.name, "version": str(self._ort.__version__),
                "model": self.model_path.name, "model_path": str(self.model_path),
                "model_sha256": self._hash_model(), "provider": "CPUExecutionProvider"}

    def _hash_model(self):
        digest = hashlib.sha256()
        with self.model_path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def process_file(self, source: Path, target: Path, ffmpeg: str = "ffmpeg",
                     channel_policy: str = "reject", output_rate: str = "source",
                     warmup_frames: int = 0) -> dict:
        source = source.expanduser().resolve()
        target = target.expanduser().resolve()
        if target.suffix.lower() != ".wav":
            raise VoxRefineError("GTCRN output must have a .wav extension.")
        if channel_policy not in ("reject", "downmix"):
            raise VoxRefineError("channel_policy must be 'reject' or 'downmix'.")
        if target.exists():
            raise VoxRefineError(f"Output already exists: {target}. Choose a new path; files are never overwritten.")
        media = probe_audio(source, ffmpeg)
        if media.info.seconds <= 0:
            raise VoxRefineError(f"{source}: empty or zero-duration audio.")
        if media.info.channels != 1 and channel_policy != "downmix":
            raise VoxRefineError("GTCRN accepts mono audio. For an explicit mono downmix, pass --channel-policy downmix.")
        executable = shutil.which(ffmpeg)
        if not executable:
            raise VoxRefineError(f"FFmpeg not found: {ffmpeg}.")
        if output_rate not in ("source", "16000"):
            raise VoxRefineError("--output-rate must be 'source' or '16000'.")
        rate = media.info.sample_rate if output_rate == "source" else SAMPLE_RATE
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="voxrefine-gtcrn-", dir=target.parent) as directory:
            raw_output, wav_output = Path(directory) / "output.f32", Path(directory) / "output.wav"
            decode_started = time.perf_counter()
            with prepared_for_backend(source, self.spec().input_domain, ffmpeg,
                                      "mono" if channel_policy == "downmix" else "reject") as raw_input:
                decode_seconds = time.perf_counter() - decode_started
                session = GTCRNSession(self.model_path, self.threads)
                warmup_seconds = session.warmup(warmup_frames)
                model_started = time.perf_counter()
                try:
                    with raw_input.open("rb") as reader, raw_output.open("wb") as writer:
                        while block := reader.read(4096 * 4):
                            usable = block[:len(block) - len(block) % 4]
                            samples = self._np.frombuffer(usable, dtype="<f4")
                            writer.write(session.process(samples))
                        writer.write(session.flush())
                finally:
                    session.close()
                model_seconds = time.perf_counter() - model_started
            encode_started = time.perf_counter()
            encode_args = [executable, "-nostdin", "-hide_banner", "-loglevel", "error", "-f", "f32le", "-ar", str(SAMPLE_RATE),
                           "-ac", "1", "-i", str(raw_output), "-af", f"aresample={rate},apad=whole_dur={media.info.seconds},atrim=duration={media.info.seconds}", "-c:a", "pcm_s16le", "-f", "wav", str(wav_output)]
            try:
                subprocess.run(encode_args, check=True, capture_output=True)
            except subprocess.CalledProcessError as error:
                message = error.stderr.decode("utf-8", errors="replace") if error.stderr else str(error)
                raise VoxRefineError(f"FFmpeg could not encode GTCRN output: {message}") from error
            encode_seconds = time.perf_counter() - encode_started
            try:
                import os
                os.link(wav_output, target)
            except FileExistsError:
                raise VoxRefineError(f"Output already exists: {target}.")
            import wave
            with wave.open(str(wav_output), "rb") as output_wav:
                output_frames = output_wav.getnframes()
            expected_frames = round(media.info.seconds * rate)
            model_output_frames = session.output_emitted
            return {"input_sample_rate": media.info.sample_rate, "input_channels": media.info.channels,
                    "model_sample_rate": SAMPLE_RATE, "output_sample_rate": rate,
                    "effective_bandwidth_hz": 8000, "model_sha256": self._hash_model(),
                    "output_frames": output_frames,
                    "model_output_frames": model_output_frames,
                    "trimmed_samples": max(0, session.total_input - model_output_frames),
                    "padded_samples": max(0, model_output_frames - session.total_input),
                    "provider": "CPUExecutionProvider", "frames": session.frames,
                    "decode_seconds": decode_seconds, "model_pipeline_seconds": model_seconds,
                    "encode_seconds": encode_seconds, "warmup_frames": warmup_frames,
                    "warmup_seconds": warmup_seconds,
                    "inference_ms": _timing_summary(session.inference_ms),
                    "pipeline_frame_ms": _timing_summary(session.pipeline_ms),
                    "inference_rtf": (sum(session.inference_ms) / 1000) / (session.frames * HOP_SIZE / SAMPLE_RATE),
                    "model_pipeline_rtf": model_seconds / max(session.total_input / SAMPLE_RATE, 1e-12),
                    "resampler": "FFmpeg swr",
                    "ffmpeg_decode_args": ["<ffmpeg>", "-i", "<input>", "-map", "0:a:0", "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_f32le", "-f", "f32le", "<adapted-input>"],
                    "ffmpeg_encode_args": ["<ffmpeg>", *["<enhanced-f32>" if item == str(raw_output) else "<output>" if item == str(wav_output) else item for item in encode_args[1:]]],
                    "ffmpeg_version": subprocess.run([executable, "-version"], check=True, capture_output=True, text=True).stdout.splitlines()[0],
                    "source_info": media.info}


def _timing_summary(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None, "p50": None, "p95": None, "p99": None}
    ordered = sorted(values)

    def percentile(p: float) -> float:
        position = (len(ordered) - 1) * p
        left = int(position)
        right = min(left + 1, len(ordered) - 1)
        return ordered[left] + (ordered[right] - ordered[left]) * (position - left)

    return {"count": len(values), "mean": statistics.fmean(values), "min": ordered[0],
            "max": ordered[-1], "p50": percentile(.50), "p95": percentile(.95),
            "p99": percentile(.99), "percentile_method": "linear"}
