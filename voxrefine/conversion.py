from contextlib import contextmanager
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Iterator
import wave

from .audio import SAMPLE_RATE, VoxRefineError, compatible_wav, inspect_wav
from .process import run_checked


@contextmanager
def prepared_audio(source: Path, ffmpeg: str = "ffmpeg") -> Iterator[Path]:
    source = source.expanduser().resolve()
    if not source.is_file():
        raise VoxRefineError(f"Audio file not found: {source}")
    try:
        with wave.open(str(source), "rb") as reader:
            compatible = compatible_wav(reader)
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
            "or WAV files that are not mono/stereo PCM16 at 48000 Hz."
        )
    print(
        f"Converting {source.name} to stereo PCM16 at 48000 Hz "
        "(multichannel inputs are downmixed to stereo; the original is unchanged).",
        file=sys.stderr,
    )
    with tempfile.TemporaryDirectory(prefix="voxrefine-convert-") as directory:
        target = Path(directory) / "prepared.wav"
        run_checked([
            executable, "-nostdin", "-hide_banner", "-loglevel", "error",
            "-xerror", "-n", "-protocol_whitelist", "file",
            "-i", str(source), "-map", "0:a:0", "-vn", "-sn", "-dn",
            "-map_metadata", "-1", "-ac", "2", "-ar", str(SAMPLE_RATE),
            "-c:a", "pcm_s16le", "-f", "wav", str(target),
        ])
        inspect_wav(target)
        yield target
