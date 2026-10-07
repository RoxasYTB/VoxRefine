"""Manifest-driven comparisons; technical metrics are not quality scores."""

import json
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import shutil
import subprocess
import threading
import time
import wave

from . import __version__
from .audio import VoxRefineError, measure_wav
from .conversion import prepared_audio
from .engines import Engine, clean, file_hash
from .ffmpeg_io import probe_audio
from .gtcrn import GTCRN


def read_corpus(manifest: Path) -> list[tuple[str, Path, str, str]]:
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise VoxRefineError(f"Invalid corpus JSON: {error}") from error
    if not isinstance(data, dict) or not isinstance(data.get("samples"), list):
        raise VoxRefineError("Corpus must contain a 'samples' list.")
    if not data["samples"]:
        raise VoxRefineError("Corpus must contain at least one sample.")
    records = []
    identifiers = set()
    for sample in data["samples"]:
        if not isinstance(sample, dict) or not all(
            isinstance(sample.get(key), str) and sample[key].strip()
            for key in ("id", "path", "category", "rights")
        ):
            raise VoxRefineError(
                "Each sample needs non-empty id, path, category and rights strings."
            )
        identifier = sample["id"]
        if not identifier.isascii() or not all(
            char.isalnum() or char in "-_" for char in identifier
        ):
            raise VoxRefineError("Sample ids may only contain ASCII letters, digits, - and _.")
        if identifier in identifiers:
            raise VoxRefineError(f"Duplicate sample id: {identifier}")
        identifiers.add(identifier)
        path = (manifest.parent / sample["path"]).expanduser().resolve()
        if not path.is_file():
            raise VoxRefineError(f"Audio file not found: {path}")
        records.append((identifier, path, sample["category"], sample["rights"]))
    return records


def benchmark(
    manifest: Path, output: Path, engines: list[Engine], ffmpeg: str = "ffmpeg"
) -> Path:
    if not engines or len({engine.name for engine in engines}) != len(engines):
        raise VoxRefineError("Select at least one engine, without duplicates.")
    samples = read_corpus(manifest)
    identities = [engine.identity() for engine in engines]
    # A fresh directory prevents mixing old results with a new evaluation.
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "voxrefine_version": __version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "manifest_sha256": file_hash(manifest),
        "engines": identities,
        "measurement_scope": (
            "Wall time includes initialization and file I/O, excluding input conversion. "
            "Input metrics describe the prepared mono PCM16 48000 Hz audio. RMS is not LUFS. "
            "No perceptual score, memory measurement or quality ranking is inferred."
        ),
        "samples": [],
    }
    report_path = output / "report.json"

    def save() -> None:
        temporary = output / "report.json.tmp"
        temporary.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(report_path)

    save()
    for identifier, source, category, rights in samples:
        prepared_path = output / f"{identifier}-input.wav"
        try:
            with prepared_audio(source, ffmpeg) as prepared:
                shutil.copyfile(prepared, prepared_path)
            original = measure_wav(prepared_path)
        except (VoxRefineError, OSError) as error:
            report["status"] = "failed"
            report["error"] = f"{identifier}/input: {error}"
            save()
            raise
        entry = {
            "id": identifier,
            "category": category,
            "rights": rights,
            "input_sha256": file_hash(source),
            "prepared_input_path": prepared_path.name,
            "prepared_input_sha256": file_hash(prepared_path),
            "input": original,
            "outputs": [],
        }
        report["samples"].append(entry)
        for engine in engines:
            target = output / f"{identifier}-{engine.name}.wav"
            started = time.perf_counter()
            try:
                clean(prepared_path, target, engine)
                elapsed = time.perf_counter() - started
                measurements = measure_wav(target)
                checksum = file_hash(target)
            except (VoxRefineError, OSError) as error:
                report["status"] = "failed"
                report["error"] = f"{identifier}/{engine.name}: {error}"
                save()
                raise
            entry["outputs"].append({
                "engine": engine.name,
                "path": target.name,
                "output_sha256": checksum,
                "wall_seconds": elapsed,
                "real_time_factor": elapsed / original["duration_seconds"],
                "audio": measurements,
            })
            save()
    report["status"] = "complete"
    save()
    return report_path


def benchmark_gtcrn(
    manifest: Path, output: Path, engine: GTCRN, ffmpeg: str = "ffmpeg",
    channel_policy: str = "reject", output_rate: str = "source",
) -> Path:
    """Run GTCRN directly on original sources and write the versioned report."""
    samples = read_corpus(manifest)
    try:
        ffmpeg_version = subprocess.run([ffmpeg, "-version"], check=True, capture_output=True,
                                        text=True).stdout.splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        ffmpeg_version = "unavailable"
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "report.json"
    report = {
        "schema_version": 2,
        "run": {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "voxrefine_version": __version__, "command": "benchmark --engines gtcrn"},
        "host": {"os": platform.platform(), "architecture": platform.machine(),
                 "cpu": platform.processor(), "logical_cores": os.cpu_count(),
                 "python_version": platform.python_version(), "ffmpeg": ffmpeg_version},
        "backend": engine.identity(),
        "runtime": {"provider": "CPUExecutionProvider", "available_providers": engine._ort.get_available_providers(),
                    "onnxruntime_version": engine._ort.__version__, "intra_op_threads": engine.threads,
                    "inter_op_threads": 1, "graph_optimization": "ORT_ENABLE_ALL"},
        "configuration": {"channel_policy": channel_policy, "output_rate": output_rate,
                          "percentile_method": "linear", "warmup_frames": 20,
                          "resource_sample_interval_ms": 50},
        "samples": [], "warnings": [], "status": "running",
    }

    def save() -> None:
        temporary = output / "report.json.tmp"
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(report_path)

    save()
    try:
        import psutil
    except ImportError:
        psutil = None
        report["warnings"].append("psutil is not installed; CPU/RSS sampling omitted. Install voxrefine[bench].")
    for identifier, source, category, rights in samples:
        media = probe_audio(source, ffmpeg)
        target = output / f"{identifier}-gtcrn.wav"
        resource_stop = threading.Event()
        resource = {"rss_peak_process_bytes": None, "rss_peak_tree_bytes": None,
                    "cpu_user_s": None, "cpu_system_s": None, "cpu_total_s": None,
                    "cpu_utilization_pct": None}
        sampler = None
        if psutil is not None:
            root = psutil.Process(os.getpid())
            cpu_by_pid = {}
            root_times = root.cpu_times()
            cpu_initial = {root.pid: (root_times.user, root_times.system)}

            def sample_resources():
                while not resource_stop.is_set():
                    processes = [root]
                    try:
                        processes += root.children(recursive=True)
                    except psutil.Error:
                        pass
                    tree_rss = 0
                    for process in processes:
                        try:
                            memory = process.memory_info().rss
                            times = process.cpu_times()
                            tree_rss += memory
                            previous = cpu_by_pid.get(process.pid, (0.0, 0.0))
                            cpu_by_pid[process.pid] = (max(previous[0], times.user), max(previous[1], times.system))
                            cpu_initial.setdefault(process.pid, (times.user, times.system))
                        except psutil.Error:
                            continue
                    resource["rss_peak_tree_bytes"] = max(resource["rss_peak_tree_bytes"] or 0, tree_rss)
                    resource["rss_peak_process_bytes"] = max(resource["rss_peak_process_bytes"] or 0, root.memory_info().rss)
                    resource_stop.wait(.05)

            resource["rss_peak_process_bytes"] = 0
            resource["rss_peak_tree_bytes"] = 0
            sampler = threading.Thread(target=sample_resources, daemon=True)
        started = time.perf_counter()
        if sampler:
            sampler.start()
        try:
            metadata = engine.process_file(source, target, ffmpeg, channel_policy, output_rate,
                                           warmup_frames=20)
        except (VoxRefineError, OSError) as error:
            report["status"] = "failed"
            report["error"] = f"{identifier}/gtcrn: {error}"
            save()
            raise
        finally:
            resource_stop.set()
            if sampler:
                sampler.join()
        wall_seconds = time.perf_counter() - started
        if psutil is not None:
            resource["cpu_user_s"] = sum(max(0, item[0] - cpu_initial.get(pid, item)[0])
                                          for pid, item in cpu_by_pid.items())
            resource["cpu_system_s"] = sum(max(0, item[1] - cpu_initial.get(pid, item)[1])
                                            for pid, item in cpu_by_pid.items())
            resource["cpu_total_s"] = resource["cpu_user_s"] + resource["cpu_system_s"]
            resource["cpu_utilization_pct"] = 100 * resource["cpu_total_s"] / wall_seconds if wall_seconds else 0
        with wave.open(str(target), "rb") as wav:
            output_duration = wav.getnframes() / wav.getframerate()
        duration = media.info.seconds
        entry = {"id": identifier, "category": category, "rights": rights,
                 "input": {"path": source.name, "sha256": file_hash(source),
                           "duration_s": duration, "sample_rate": media.info.sample_rate,
                           "channels": media.info.channels, "codec": media.info.codec,
                           "container": media.info.container},
                 "audio_adaptation": {"input_sample_rate": media.info.sample_rate,
                                      "model_sample_rate": metadata["model_sample_rate"],
                                      "input_channels": media.info.channels,
                                      "target_channels": 1, "channel_policy": channel_policy,
                                      "resampler": metadata["resampler"],
                                      "ffmpeg_decode_args": metadata["ffmpeg_decode_args"],
                                      "ffmpeg_encode_args": metadata["ffmpeg_encode_args"]},
                 "model": {"path": str(engine.model_path), "filename": engine.model_path.name,
                           "size_bytes": engine.model_path.stat().st_size,
                           "sha256": metadata["model_sha256"]},
                 "timing": {"decode_s": metadata["decode_seconds"],
                            "warmup_s": metadata["warmup_seconds"],
                            "inference_rtf": metadata["inference_rtf"],
                            "inference_frame_ms": metadata["inference_ms"],
                            "model_pipeline_s": metadata["model_pipeline_seconds"],
                            "model_pipeline_rtf": metadata["model_pipeline_rtf"],
                            "pipeline_frame_ms": metadata["pipeline_frame_ms"],
                            "encode_s": metadata["encode_seconds"], "wall_s": wall_seconds,
                            "end_to_end_rtf": wall_seconds / duration if duration else None},
                 "latency": {"measured_frames": metadata["frames"], "hop_samples": 256,
                             "hop_ms": 16, "algorithmic_window_ms": 32},
                 "resources": resource,
                 "output": {"path": target.name, "sha256": file_hash(target),
                            "sample_rate": metadata["output_sample_rate"],
                            "duration_s": output_duration,
                            "effective_bandwidth_hz": metadata["effective_bandwidth_hz"],
                            "trimmed_samples": metadata.get("trimmed_samples", 0),
                            "padded_samples": metadata.get("padded_samples", 0)}}
        report["samples"].append(entry)
        save()
    report["status"] = "complete"
    save()
    return report_path
