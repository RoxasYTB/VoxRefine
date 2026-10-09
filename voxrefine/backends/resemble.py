"""Optional offline Resemble Enhance adapter using a separate Python runtime."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

from ..audio import AudioDomain, VoxRefineError
from ..engines import DeepFilterNet
from ..ffmpeg_io import decode_raw, probe_audio


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def enhance_resemble(
    source: Path,
    target: Path,
    *,
    python: str = "python3",
    upstream: Path,
    model_dir: Path,
    device: str = "auto",
    nfe: int = 64,
    chunk_seconds: float = 3.0,
    tone: str = "C",
    treble_trim_db: float = -2.5,
    deesser: str = "off",
    dynamics: str = "gentle",
    output_gain_db: float = -2.0,
    post_denoise_limit_db: float | None = None,
    deep_filter: str = "deep-filter",
    ffmpeg: str = "ffmpeg",
    channel_policy: str = "reject",
) -> dict:
    """Run a locally installed Resemble checkpoint; no model downloads occur."""
    source = source.expanduser().resolve()
    target = target.expanduser().resolve()
    upstream = upstream.expanduser().resolve()
    model_dir = model_dir.expanduser().resolve()
    if target.suffix.lower() != ".wav":
        raise VoxRefineError("Resemble output must be a .wav file (mono PCM16 at 48000 Hz).")
    if target.exists():
        raise VoxRefineError(f"Output already exists: {target}. Choose a new path; files are never overwritten.")
    if device not in {"auto", "cpu", "cuda"}:
        raise VoxRefineError("device must be auto, cpu, or cuda.")
    if nfe not in {16, 32, 64}:
        raise VoxRefineError("NFE must be one of 16, 32, or 64.")
    if not 2.0 <= chunk_seconds <= 30.0:
        raise VoxRefineError("chunk_seconds must be between 2 and 30 seconds.")
    if tone not in {"flat", "C", "soft-edges"}:
        raise VoxRefineError("tone must be flat, C, or soft-edges.")
    if tone in {"C", "soft-edges"} and not -6 <= treble_trim_db <= 0:
        raise VoxRefineError("C/soft-edges base treble trim must be between -6 and 0 dB.")
    if deesser not in {"off", "gentle"}:
        raise VoxRefineError("deesser must be off or gentle.")
    if dynamics not in {"off", "gentle"}:
        raise VoxRefineError("dynamics must be off or gentle.")
    if not -12 <= output_gain_db <= 0:
        raise VoxRefineError("output_gain_db must be between -12 and 0 dB.")
    if post_denoise_limit_db is not None and not 0 <= post_denoise_limit_db <= 100:
        raise VoxRefineError("Post-denoise attenuation limit must be between 0 and 100 dB.")
    if channel_policy not in {"reject", "downmix", "first"}:
        raise VoxRefineError("channel_policy must be reject, downmix, or first.")
    if not upstream.is_dir() or not (upstream / "resemble_enhance").is_dir():
        raise VoxRefineError(f"Resemble source checkout not found: {upstream}")
    if not model_dir.is_dir() or not (model_dir / "hparams.yaml").is_file():
        raise VoxRefineError(f"Local Resemble model directory is incomplete: {model_dir}")
    python_exec = shutil.which(python)
    ffmpeg_exec = shutil.which(ffmpeg)
    if not python_exec:
        raise VoxRefineError(f"Model Python interpreter not found: {python}")
    if not ffmpeg_exec:
        raise VoxRefineError(f"FFmpeg not found: {ffmpeg}")
    media = probe_audio(source, ffmpeg)
    if media.info.seconds <= 0:
        raise VoxRefineError(f"{source}: empty or zero-duration audio.")
    if media.info.channels != 1 and channel_policy == "reject":
        raise VoxRefineError("Resemble accepts mono audio. Select --channel-policy downmix or first explicitly.")
    target.parent.mkdir(parents=True, exist_ok=True)
    worker = Path(__file__).resolve().parents[1] / "workers" / "resemble_worker.py"
    with tempfile.TemporaryDirectory(prefix="voxrefine-resemble-", dir=target.parent) as directory:
        tmp = Path(directory)
        raw_in, raw_out, wav_out = tmp / "input.f32", tmp / "output.f32", tmp / "output.wav"
        job_started = time.perf_counter()
        domain = AudioDomain(48000, 1, "f32", "raw")
        decode_raw(media, raw_in, domain, ffmpeg,
                   "first" if channel_policy == "first" else "mono")
        samples = raw_in.stat().st_size // 4
        if samples == 0:
            raise VoxRefineError("Decoded input contains no audio samples.")
        command = [python_exec, str(worker), "--input", str(raw_in), "--output", str(raw_out),
                   "--sample-count", str(samples), "--upstream", str(upstream), "--model-dir", str(model_dir),
                   "--device", device, "--nfe", str(nfe), "--tone", tone,
                   "--treble-trim-db", str(treble_trim_db), "--deesser", deesser,
                   "--dynamics", dynamics,
                   "--chunk-seconds", str(chunk_seconds)]
        started = time.perf_counter()
        try:
            process = subprocess.run(command, capture_output=True, text=True, timeout=1800)
        except subprocess.TimeoutExpired as error:
            raise VoxRefineError("Resemble processing exceeded the 30-minute timeout.") from error
        if process.returncode:
            detail = (process.stderr or process.stdout).strip()
            raise VoxRefineError(f"Resemble worker failed (exit {process.returncode}): {detail[-4000:]}")
        try:
            worker_report = json.loads(process.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError) as error:
            raise VoxRefineError("Resemble worker did not return a valid provenance report.") from error
        elapsed = time.perf_counter() - started
        if not raw_out.is_file() or raw_out.stat().st_size != samples * 4:
            raise VoxRefineError("Resemble worker returned an invalid output length.")
        try:
            subprocess.run([ffmpeg_exec, "-nostdin", "-hide_banner", "-loglevel", "error", "-xerror",
                            "-f", "f32le", "-ar", "48000", "-ac", "1", "-i", str(raw_out),
                            "-c:a", "pcm_s16le", "-f", "wav", str(wav_out)], check=True, capture_output=True)
        except subprocess.CalledProcessError as error:
            detail = error.stderr.decode("utf-8", errors="replace") if error.stderr else str(error)
            raise VoxRefineError(f"FFmpeg could not encode Resemble output: {detail}") from error
        import wave
        with wave.open(str(wav_out), "rb") as audio:
            if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getnframes()) != (1, 2, 48000, samples):
                raise VoxRefineError("Encoded Resemble WAV failed format or duration validation.")
        selected_wav = wav_out
        post_elapsed = 0.0
        post_identity = None
        if post_denoise_limit_db is not None:
            selected_wav = tmp / "output-post-denoise.wav"
            post_started = time.perf_counter()
            post_backend = DeepFilterNet(deep_filter, attenuation_limit_db=post_denoise_limit_db)
            post_backend.process(wav_out, selected_wav)
            post_elapsed = time.perf_counter() - post_started
            post_identity = post_backend.identity()
            with wave.open(str(selected_wav), "rb") as audio:
                if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getnframes()) != (1, 2, 48000, samples):
                    raise VoxRefineError("DeepFilterNet post-denoise output failed format or duration validation.")
        final_wav = selected_wav
        if output_gain_db != 0.0:
            final_wav = tmp / "output-final-gain.wav"
            try:
                subprocess.run([ffmpeg_exec, "-nostdin", "-hide_banner", "-loglevel", "error", "-xerror",
                                "-i", str(selected_wav), "-af", f"volume={output_gain_db:.6f}dB",
                                "-c:a", "pcm_s16le", "-f", "wav", str(final_wav)],
                               check=True, capture_output=True)
            except subprocess.CalledProcessError as error:
                detail = error.stderr.decode("utf-8", errors="replace") if error.stderr else str(error)
                raise VoxRefineError(f"FFmpeg could not apply final output gain: {detail}") from error
            with wave.open(str(final_wav), "rb") as audio:
                if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getnframes()) != (1, 2, 48000, samples):
                    raise VoxRefineError("Final output-gain WAV failed format or duration validation.")
        total_elapsed = time.perf_counter() - job_started
        try:
            os.link(final_wav, target)
        except FileExistsError as error:
            raise VoxRefineError(f"Output already exists: {target}.") from error
    return {
        "backend": "resemble-enhance",
        "input": str(source),
        "input_sha256": _sha256(source),
        "output": str(target),
        "output_sha256": _sha256(target),
        "input_sample_rate": media.info.sample_rate,
        "input_channels": media.info.channels,
        "output_sample_rate": 48000,
        "output_channels": 1,
        "channel_policy": channel_policy,
        "duration_seconds": samples / 48000,
        "wall_seconds_including_worker_start": elapsed,
        "rtf_including_worker_start": elapsed / (samples / 48000),
        **worker_report,
        "post_denoise": post_identity,
        "post_denoise_wall_seconds": post_elapsed,
        "final_output_gain_db": output_gain_db,
        "final_dynamics_mode": dynamics,
        "final_deesser_mode": deesser,
        "total_wall_seconds_including_decode_encode_and_post_denoise": total_elapsed,
        "total_rtf_including_decode_encode_and_post_denoise": total_elapsed / (samples / 48000),
    }
