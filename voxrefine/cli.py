import argparse
from pathlib import Path
import sys

from . import __version__
from .audio import VoxRefineError
from .benchmark import benchmark
from .engines import DeepFilterNet, Engine, RNNoise, clean


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="VoxRefine: local voice cleanup from WAV, MP3 or M4A."
    )
    parser.add_argument("--version", action="version", version=f"VoxRefine {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    cleanup = commands.add_parser("clean", help="Clean one file without overwriting.")
    cleanup.add_argument("input", type=Path)
    cleanup.add_argument("output", type=Path)
    cleanup.add_argument("--engine", choices=["deepfilter", "rnnoise"], required=True)
    comparison = commands.add_parser("benchmark", help="Compare engines on a corpus.")
    comparison.add_argument("manifest", type=Path)
    comparison.add_argument("--output", type=Path, required=True)
    comparison.add_argument(
        "--engines", nargs="+", choices=["deepfilter", "rnnoise"],
        default=["deepfilter", "rnnoise"],
    )
    for command in (cleanup, comparison):
        command.add_argument("--deep-filter", default="deep-filter", metavar="EXECUTABLE")
        command.add_argument("--rnnoise-library", metavar="LIBRARY")
        command.add_argument("--ffmpeg", default="ffmpeg", metavar="EXECUTABLE")
        command.add_argument(
            "--attenuation-limit-db", type=float, metavar="DB",
            help="DeepFilterNet only: 0 to 100 dB; lower is gentler (default: 100).",
        )
    args = parser.parse_args(argv)
    selected = [args.engine] if args.command == "clean" else args.engines
    if args.attenuation_limit_db is not None and "deepfilter" not in selected:
        parser.error("--attenuation-limit-db requires the deepfilter engine.")

    def engine(name: str) -> Engine:
        if name == "deepfilter":
            limit = 100.0 if args.attenuation_limit_db is None else args.attenuation_limit_db
            return DeepFilterNet(args.deep_filter, attenuation_limit_db=limit)
        if not args.rnnoise_library:
            raise VoxRefineError("--rnnoise-library is required for RNNoise.")
        return RNNoise(args.rnnoise_library)

    try:
        if args.command == "clean":
            clean(args.input, args.output, engine(args.engine), args.ffmpeg)
            print(f"Cleaned audio: {args.output}")
        else:
            result = benchmark(
                args.manifest, args.output, [engine(name) for name in args.engines],
                args.ffmpeg,
            )
            print(f"Technical comparison: {result}")
    except (VoxRefineError, OSError) as error:
        print(f"VoxRefine: {error}", file=sys.stderr)
        return 1
    return 0
