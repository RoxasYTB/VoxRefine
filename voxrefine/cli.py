import argparse
from pathlib import Path
import sys

from . import __version__
from .audio import VoxRefineError
from .benchmark import benchmark
from .equalizer import EqualizedEngine, validate_eq
from .engines import DeepFilterNet, Engine, RNNoise, clean
from .protection import ProtectedDeepFilterNet

PROFILES = {"natural": 12.0, "balanced": 20.0, "strong": 100.0}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="VoxRefine: local voice cleanup from WAV, MP3 or M4A."
    )
    parser.add_argument("--version", action="version", version=f"VoxRefine {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    cleanup = commands.add_parser("clean", help="Clean one file without overwriting.")
    cleanup.add_argument("input", type=Path)
    cleanup.add_argument("output", type=Path)
    cleanup.add_argument("--engine", choices=["deepfilter", "rnnoise"], default="deepfilter")
    comparison = commands.add_parser("benchmark", help="Compare engines on a corpus.")
    comparison.add_argument("manifest", type=Path)
    comparison.add_argument("--output", type=Path, required=True)
    comparison.add_argument(
        "--engines", nargs="+", choices=["deepfilter", "rnnoise"],
        default=["deepfilter", "rnnoise"],
    )
    for command in (cleanup, comparison):
        command.add_argument(
            "--deep-filter", default=".tools/deep-filter", metavar="EXECUTABLE"
        )
        command.add_argument(
            "--rnnoise-library", default=".tools/librnnoise.so", metavar="LIBRARY"
        )
        command.add_argument("--ffmpeg", default="ffmpeg", metavar="EXECUTABLE")
        command.add_argument(
            "--bass-db", type=float, default=0.0, metavar="DB",
            help="Optional low-shelf EQ at 150 Hz: -12 to +12 dB (default: 0).",
        )
        command.add_argument(
            "--treble-db", type=float, default=0.0, metavar="DB",
            help="Optional high-shelf EQ at 4 kHz: -12 to +12 dB (default: 0).",
        )
        command.add_argument(
            "--no-protect-voice", action="store_true",
            help="Use standard DeepFilterNet without speech-aware protection.",
        )
        settings = command.add_mutually_exclusive_group()
        settings.add_argument(
            "--attenuation-limit-db", type=float, metavar="DB",
            help="DeepFilterNet only: 0 to 100 dB; lower is gentler (default: 100).",
        )
        settings.add_argument(
            "--profile", choices=list(PROFILES),
            help="DeepFilterNet: natural (12 dB), balanced (20 dB), strong (100 dB).",
        )
    args = parser.parse_args(argv)
    try:
        validate_eq(args.bass_db, args.treble_db)
    except VoxRefineError as error:
        parser.error(str(error))
    selected = [args.engine] if args.command == "clean" else args.engines
    if (args.attenuation_limit_db is not None or args.profile is not None) and "deepfilter" not in selected:
        parser.error("--attenuation-limit-db and --profile require the deepfilter engine.")
    if args.no_protect_voice and "deepfilter" not in selected:
        parser.error("--no-protect-voice requires the deepfilter engine.")
    if "deepfilter" in selected:
        limit = (
            args.attenuation_limit_db if args.attenuation_limit_db is not None
            else PROFILES[args.profile or "strong"]
        )
        if not args.no_protect_voice and limit <= 12:
            parser.error(
                "Voice protection requires an attenuation limit above 12 dB; "
                "use --no-protect-voice for this profile or limit."
            )
        if not args.no_protect_voice and not args.rnnoise_library:
            parser.error("DeepFilterNet voice protection requires --rnnoise-library.")

    def engine(name: str) -> Engine:
        if name == "deepfilter":
            limit = (
                args.attenuation_limit_db if args.attenuation_limit_db is not None
                else PROFILES[args.profile or "strong"]
            )
            primary = DeepFilterNet(args.deep_filter, attenuation_limit_db=limit)
            if args.no_protect_voice:
                return primary
            return ProtectedDeepFilterNet(primary, RNNoise(args.rnnoise_library))
        if not args.rnnoise_library:
            raise VoxRefineError("--rnnoise-library is required for RNNoise.")
        return RNNoise(args.rnnoise_library)

    def configured_engine(name: str) -> Engine:
        selected_engine = engine(name)
        if args.bass_db or args.treble_db:
            return EqualizedEngine(selected_engine, args.bass_db, args.treble_db)
        return selected_engine

    try:
        if args.command == "clean":
            clean(args.input, args.output, configured_engine(args.engine), args.ffmpeg)
            print(f"Cleaned audio: {args.output}")
        else:
            result = benchmark(
                args.manifest, args.output,
                [configured_engine(name) for name in args.engines], args.ffmpeg,
            )
            print(f"Technical comparison: {result}")
    except (VoxRefineError, OSError) as error:
        print(f"VoxRefine: {error}", file=sys.stderr)
        return 1
    return 0
