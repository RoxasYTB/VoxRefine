from contextlib import contextmanager
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Iterator
import wave

from .audio import AudioDomain, VoxRefineError, inspect_wav
from .ffmpeg_io import probe_audio
from .process import run_checked


@contextmanager
def prepared_audio(source: Path, ffmpeg: str = "ffmpeg") -> Iterator[Path]:
    source = source.expanduser().resolve()
    if not source.is_file():
        raise VoxRefineError(f"Audio file not found: {source}")
    try:
        with wave.open(str(source), "rb") as reader:
            compatible = (
                reader.getnchannels() == 1
                and reader.getsampwidth() == 2
                and reader.getframerate() == 48000
                and reader.getcomptype() == "NONE"
            )
    except (wave.Error, EOFError):
        compatible = False
    if compatible:
        inspect_wav(source)
        yield source
        return
    executable = shutil.which(ffmpeg)
    if executable is None:
        raise VoxRefineError(
            f"FFmpeg not found: {ffmpeg}. Install FFmpeg to read MP3, M4A "
            "or WAV files that are not mono PCM16 at 48000 Hz."
        )
    print(
        f"Converting {source.name} to mono PCM16 at 48000 Hz "
        "(multiple channels are mixed; the original is unchanged).",
        file=sys.stderr,
    )
    with tempfile.TemporaryDirectory(prefix="voxrefine-convert-") as directory:
        target = Path(directory) / "prepared.wav"
        run_checked([
            executable, "-nostdin", "-hide_banner", "-loglevel", "error",
            "-xerror", "-n", "-protocol_whitelist", "file",
            "-i", str(source), "-map", "0:a:0", "-vn", "-sn", "-dn",
            "-map_metadata", "-1", "-ac", "1", "-ar", "48000",
            "-c:a", "pcm_s16le", "-f", "wav", str(target),
        ])
        inspect_wav(target)
        yield target


@contextmanager
def prepared_for_backend(source: Path, domain: AudioDomain, ffmpeg: str = "ffmpeg",
                          channel_policy: str = "mono") -> Iterator[Path]:
    """Adapt original media to a backend domain at the last responsible point."""
    audio_source = probe_audio(source, ffmpeg)
    if channel_policy not in ("preserve", "mono", "reject"):
        raise VoxRefineError("channel_policy must be preserve, mono, or reject.")
    if channel_policy == "reject" and audio_source.info.channels != domain.channels:
        raise VoxRefineError(
            f"Backend requires {domain.channels} channel(s); input has {audio_source.info.channels}."
        )
    with tempfile.TemporaryDirectory(prefix="voxrefine-domain-") as directory:
        if domain.sample_format == "s16" and domain.container == "wav":
            target = Path(directory) / "prepared.wav"
            executable = shutil.which(ffmpeg)
            if executable is None:
                raise VoxRefineError(f"FFmpeg not found: {ffmpeg}.")
            args = [executable, "-nostdin", "-hide_banner", "-loglevel", "error", "-xerror",
                    "-i", str(audio_source.path), "-map", "0:a:0", "-vn", "-sn", "-dn"]
            channels = 1 if channel_policy == "mono" else domain.channels
            if channels == 1:
                args += ["-ac", "1"]
            args += ["-ar", str(domain.sample_rate), "-c:a", "pcm_s16le", "-f", "wav", str(target)]
            run_checked(args)
        else:
            target = Path(directory) / "prepared.raw"
            from .ffmpeg_io import decode_raw
            decode_raw(audio_source, target, domain, ffmpeg, channel_policy)
        yield target
