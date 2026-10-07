"""Media probing and explicit FFmpeg adaptation helpers."""

import json
from pathlib import Path
import shutil
import subprocess

from .audio import AudioInfo, AudioSource, VoxRefineError


def probe_audio(path: Path, ffmpeg: str = "ffmpeg") -> AudioSource:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise VoxRefineError(f"Audio file not found: {path}")
    executable = shutil.which(ffmpeg)
    if executable is None:
        raise VoxRefineError(f"FFmpeg not found: {ffmpeg}. Install FFmpeg to inspect audio metadata.")
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        # WAV probing is available without FFmpeg's companion binary.
        import wave
        try:
            with wave.open(str(path), "rb") as wav:
                frames, rate, channels = wav.getnframes(), wav.getframerate(), wav.getnchannels()
                info = AudioInfo(frames, rate, channels, duration_s=frames / rate,
                                 codec="pcm", container="wav", sample_format=f"pcm_s{wav.getsampwidth() * 8}le")
                return AudioSource(path, info)
        except (wave.Error, EOFError):
            raise VoxRefineError("ffprobe was not found; it is required to inspect non-WAV audio.")
    command = [ffprobe, "-v", "error", "-select_streams", "a:0", "-show_entries",
               "stream=sample_rate,channels,channel_layout,codec_name,sample_fmt,bit_rate,duration,nb_frames",
               "-show_entries", "format=format_name,duration,bit_rate", "-of", "json", str(path)]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        data = json.loads(result.stdout)
        stream = data["streams"][0]
        rate = int(stream["sample_rate"])
        channels = int(stream["channels"])
        duration_raw = stream.get("duration") or data.get("format", {}).get("duration")
        duration = float(duration_raw) if duration_raw not in (None, "N/A") else None
        frames_raw = stream.get("nb_frames")
        frames = int(frames_raw) if frames_raw and frames_raw != "N/A" else int(duration * rate) if duration else 0
        fmt = data.get("format", {})
        bitrate_raw = stream.get("bit_rate") or fmt.get("bit_rate")
        bitrate = int(bitrate_raw) if bitrate_raw and bitrate_raw != "N/A" else None
        info = AudioInfo(frames, rate, channels, stream.get("channel_layout"), duration,
                         stream.get("codec_name"), fmt.get("format_name"),
                         bitrate, stream.get("sample_fmt"))
        if not rate or not channels:
            raise ValueError("missing sample rate or channel count")
        return AudioSource(path, info)
    except (subprocess.CalledProcessError, KeyError, ValueError, IndexError, json.JSONDecodeError) as error:
        detail = getattr(error, "stderr", None) or str(error)
        raise VoxRefineError(f"Could not inspect audio {path}: {detail}") from error


def decode_raw(
    source: AudioSource, target: Path, domain, ffmpeg: str = "ffmpeg",
    channel_policy: str = "preserve",
) -> dict:
    """Decode an original media file into an explicitly requested raw PCM domain."""
    executable = shutil.which(ffmpeg)
    if executable is None:
        raise VoxRefineError(f"FFmpeg not found: {ffmpeg}.")
    if channel_policy not in ("preserve", "mono", "first", "reject"):
        raise VoxRefineError("channel_policy must be preserve, mono, first, or reject.")
    if channel_policy == "reject" and source.info.channels != domain.channels:
        raise VoxRefineError(
            f"Backend requires {domain.channels} channel(s), input has {source.info.channels}; "
            "select an explicit channel policy."
        )
    channels = source.info.channels if channel_policy == "preserve" else domain.channels
    if channel_policy in ("mono", "first"):
        channels = 1
    if channels != domain.channels:
        raise VoxRefineError("Preserved channel count does not match backend domain.")
    fmt = {"f32": ("f32le", "pcm_f32le"), "s16": ("s16le", "pcm_s16le")}.get(domain.sample_format)
    if fmt is None:
        raise VoxRefineError(f"Unsupported raw sample format: {domain.sample_format}.")
    output_format, codec = fmt
    args = [executable, "-nostdin", "-hide_banner", "-loglevel", "error", "-xerror", "-i",
            str(source.path), "-map", "0:a:0", "-vn", "-sn", "-dn"]
    if channel_policy == "first":
        args += ["-af", "pan=mono|c0=c0"]
    elif channels == 1:
        args += ["-ac", "1"]
    args += ["-ar", str(domain.sample_rate), "-c:a", codec, "-f", output_format, str(target)]
    try:
        subprocess.run(args, check=True, capture_output=True)
    except subprocess.CalledProcessError as error:
        detail = error.stderr.decode("utf-8", errors="replace") if error.stderr else str(error)
        raise VoxRefineError(f"FFmpeg could not adapt {source.path}: {detail}") from error
    return {"input_sample_rate": source.info.sample_rate, "input_channels": source.info.channels,
            "target_sample_rate": domain.sample_rate, "target_channels": channels,
            "sample_format": domain.sample_format, "channel_policy": channel_policy,
            "resampler": "FFmpeg swr", "ffmpeg_args": args}
