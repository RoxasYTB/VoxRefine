import argparse
from pathlib import Path
import sys

from . import __version__
from .audio import VoxRefineError
from .benchmark import benchmark
from .engines import DeepFilterNet, Engine, RNNoise, clean


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="VoxRefine: local voice cleanup, WAV mono PCM16 at 48 kHz."
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
    args = parser.parse_args(argv)

    def engine(name: str) -> Engine:
        if name == "deepfilter":
            return DeepFilterNet(args.deep_filter)
        if not args.rnnoise_library:
            raise VoxRefineError("--rnnoise-library is required for RNNoise.")
        return RNNoise(args.rnnoise_library)

    try:
        if args.command == "clean":
            clean(args.input, args.output, engine(args.engine))
            print(f"Cleaned audio: {args.output}")
        else:
            result = benchmark(
                args.manifest, args.output, [engine(name) for name in args.engines]
            )
            print(f"Technical comparison: {result}")
    except (VoxRefineError, OSError) as error:
        print(f"VoxRefine: {error}", file=sys.stderr)
        return 1
    return 0
