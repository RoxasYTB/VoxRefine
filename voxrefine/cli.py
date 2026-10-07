import argparse
from pathlib import Path
import sys

from . import __version__
from .audio import VoxRefineError
from .benchmark import benchmark, benchmark_gtcrn
from .engines import DeepFilterNet, Engine, RNNoise, clean
from .gtcrn import GTCRN
from .listening import create_blind_listening_set


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="VoxRefine: local voice cleanup from WAV, MP3 or M4A."
    )
    parser.add_argument("--version", action="version", version=f"VoxRefine {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    cleanup = commands.add_parser("clean", help="Clean one file without overwriting.")
    cleanup.add_argument("input", type=Path)
    cleanup.add_argument("output", type=Path)
    cleanup.add_argument("--engine", choices=["deepfilter", "rnnoise", "gtcrn"], required=True)
    comparison = commands.add_parser("benchmark", help="Compare engines on a corpus.")
    comparison.add_argument("manifest", type=Path)
    comparison.add_argument("--output", type=Path, required=True)
    comparison.add_argument(
        "--engines", nargs="+", choices=["deepfilter", "rnnoise", "gtcrn"],
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
    cleanup.add_argument("--model", type=Path, help="Local GTCRN streaming ONNX model (never downloaded).")
    cleanup.add_argument("--channel-policy", choices=["reject", "downmix"], default="reject")
    cleanup.add_argument("--output-rate", choices=["source", "16000"], default="source")
    cleanup.add_argument("--ort-threads", type=int, default=1)
    comparison.add_argument("--model", type=Path, help="Local GTCRN streaming ONNX model.")
    comparison.add_argument("--channel-policy", choices=["reject", "downmix"], default="reject")
    comparison.add_argument("--output-rate", choices=["source", "16000"], default="source")
    comparison.add_argument("--ort-threads", type=int, default=1)
    listening = commands.add_parser(
        "blind-listen", help="Create randomized, loudness-matched copies from candidate audio."
    )
    listening.add_argument("manifest", type=Path, help="JSON manifest of already-generated candidates.")
    listening.add_argument("--output", type=Path, required=True,
                           help="New directory; writes blind/ and separate key/ folders.")
    listening.add_argument("--seed", type=int, default=0)
    listening.add_argument("--target-lufs", type=float, default=-20.0)
    listening.add_argument("--true-peak-db", type=float, default=-1.5)
    listening.add_argument("--ffmpeg", default="ffmpeg", metavar="EXECUTABLE")
    args = parser.parse_args(argv)
    selected = [args.engine] if args.command == "clean" else args.engines if args.command == "benchmark" else []
    if args.command != "blind-listen" and args.attenuation_limit_db is not None and "deepfilter" not in selected:
        parser.error("--attenuation-limit-db requires the deepfilter engine.")
    if args.command == "clean" and args.engine == "gtcrn" and args.model is None:
        parser.error("--model is required when --engine gtcrn is selected.")
    if args.command == "clean" and args.engine != "gtcrn" and args.model is not None:
        parser.error("--model is only used with --engine gtcrn.")
    if args.command == "benchmark" and "gtcrn" in args.engines and len(args.engines) != 1:
        parser.error("GTCRN currently requires a standalone benchmark run: --engines gtcrn.")
    if args.command == "benchmark" and "gtcrn" in args.engines and args.model is None:
        parser.error("--model is required when benchmarking GTCRN.")
    if args.command == "benchmark" and "gtcrn" not in args.engines and args.model is not None:
        parser.error("--model is only used with --engines gtcrn.")

    def engine(name: str) -> Engine:
        if name == "deepfilter":
            limit = 100.0 if args.attenuation_limit_db is None else args.attenuation_limit_db
            return DeepFilterNet(args.deep_filter, attenuation_limit_db=limit)
        if not args.rnnoise_library:
            raise VoxRefineError("--rnnoise-library is required for RNNoise.")
        return RNNoise(args.rnnoise_library)

    try:
        if args.command == "blind-listen":
            blind_dir, reveal = create_blind_listening_set(
                args.manifest, args.output, args.ffmpeg, args.seed,
                args.target_lufs, args.true_peak_db,
            )
            print(f"Blind listening files: {blind_dir}")
            print(f"Reveal key (keep separate): {reveal}")
            return 0
        if args.command == "clean":
            if args.engine == "gtcrn":
                gtcrn = GTCRN(args.model, args.ort_threads)
                metadata = gtcrn.process_file(
                    args.input, args.output, args.ffmpeg, args.channel_policy, args.output_rate
                )
                print(f"GTCRN output: {args.output} ({metadata['model_sample_rate']} Hz model, "
                      f"{metadata['output_sample_rate']} Hz file, CPUExecutionProvider)")
                return 0
            clean(args.input, args.output, engine(args.engine), args.ffmpeg)
            print(f"Cleaned audio: {args.output}")
        else:
            if "gtcrn" in args.engines:
                report = benchmark_gtcrn(
                    args.manifest, args.output, GTCRN(args.model, args.ort_threads),
                    args.ffmpeg, args.channel_policy, args.output_rate,
                )
                print(f"GTCRN technical comparison: {report}")
                return 0
            result = benchmark(
                args.manifest, args.output, [engine(name) for name in args.engines],
                args.ffmpeg,
            )
            print(f"Technical comparison: {result}")
    except (VoxRefineError, OSError) as error:
        print(f"VoxRefine: {error}", file=sys.stderr)
        return 1
    return 0
