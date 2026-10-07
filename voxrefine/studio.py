"""Reusable offline AP-BWE -> DeepFilterNet enhancement pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import time
from typing import Any

from .ap_bwe import APBWE16to48
from .audio import VoxRefineError, measure_wav
from .benchmark import read_corpus
from .engines import DeepFilterNet, clean, file_hash


class StudioEnhancer:
    """Load model weights once and reuse them across a corpus batch."""

    def __init__(
        self,
        *,
        ap_bwe_source: Path,
        checkpoint: Path,
        deep_filter: str = "deep-filter",
        ffmpeg: str = "ffmpeg",
        attenuation_limit_db: float = 100.0,
        device: str = "cpu",
        threads: int = 4,
        chunk_frames: int = 2048,
        channel_policy: str = "reject",
    ):
        started = time.perf_counter()
        self.bwe = APBWE16to48(
            ap_bwe_source, checkpoint, device=device, threads=threads,
            chunk_frames=chunk_frames,
        )
        self.model_setup_seconds = time.perf_counter() - started
        self.dfn = DeepFilterNet(deep_filter, attenuation_limit_db)
        self.ffmpeg = ffmpeg
        self.channel_policy = channel_policy
        self.attenuation_limit_db = attenuation_limit_db

    def process_file(self, source: Path, target: Path) -> dict[str, Any]:
        source = source.expanduser().resolve()
        target = target.expanduser().resolve()
        if not source.is_file():
            raise VoxRefineError(f"Audio file not found: {source}")
        if target.exists():
            raise VoxRefineError(f"Output already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="voxrefine-studio-") as directory:
            intermediate = Path(directory) / "ap-bwe-48k-pcm24.wav"
            bwe_report = self.bwe.process_file(
                source, intermediate, ffmpeg=self.ffmpeg,
                channel_policy=self.channel_policy,
            )
            bwe_report["output"]["path"] = "temporary AP-BWE intermediate; removed after stage 2"
            dfn_started = time.perf_counter()
            clean(intermediate, target, self.dfn, self.ffmpeg)
            dfn_seconds = time.perf_counter() - dfn_started
        wall_seconds = time.perf_counter() - started
        try:
            ffmpeg_version = subprocess.run(
                [shutil.which(self.ffmpeg) or self.ffmpeg, "-version"], check=True,
                capture_output=True, text=True,
            ).stdout.splitlines()[0]
        except (OSError, subprocess.CalledProcessError, IndexError):
            ffmpeg_version = "unavailable"
        audio = measure_wav(target)
        return {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "pipeline": "AP-BWE 16k->48k then DeepFilterNet 0.5.6",
            "host": {"platform": platform.platform(), "python": platform.python_version(),
                     "ffmpeg": ffmpeg_version},
            "input": {"path": str(source), "sha256": file_hash(source)},
            "stages": {
                "ap_bwe": bwe_report,
                "deepfilternet": {
                    **self.dfn.identity(),
                    "wall_seconds_including_cli_startup_and_io": dfn_seconds,
                    "attenuation_limit_db": self.attenuation_limit_db,
                },
            },
            "timing": {"model_initialization_seconds": self.model_setup_seconds,
                       "wall_seconds_excluding_model_initialization": wall_seconds,
                       "duration_seconds": audio["duration_seconds"],
                       "real_time_factor_excluding_model_initialization": wall_seconds / audio["duration_seconds"]},
            "output": {"path": str(target), "sha256": file_hash(target), "audio": audio},
            "limitations": [
                "Offline file pipeline; AP-BWE is not a real-time streaming stage.",
                "AP-BWE widens bandwidth; it is not a dereverberation model.",
                "Quality must be judged on varied voices and listening samples; this report is technical metadata.",
            ],
        }


def enhance_studio(
    source: Path,
    target: Path,
    *,
    ap_bwe_source: Path,
    checkpoint: Path,
    deep_filter: str = "deep-filter",
    ffmpeg: str = "ffmpeg",
    attenuation_limit_db: float = 100.0,
    device: str = "cpu",
    threads: int = 4,
    chunk_frames: int = 2048,
    channel_policy: str = "reject",
) -> dict[str, Any]:
    enhancer = StudioEnhancer(
        ap_bwe_source=ap_bwe_source, checkpoint=checkpoint,
        deep_filter=deep_filter, ffmpeg=ffmpeg,
        attenuation_limit_db=attenuation_limit_db, device=device,
        threads=threads, chunk_frames=chunk_frames,
        channel_policy=channel_policy,
    )
    return enhancer.process_file(source, target)


def write_report(report: dict[str, Any], path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.exists():
        raise VoxRefineError(f"Report already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")
    return path


def benchmark_studio(
    manifest: Path,
    output: Path,
    *,
    ap_bwe_source: Path,
    checkpoint: Path,
    deep_filter: str = "deep-filter",
    ffmpeg: str = "ffmpeg",
    attenuation_limit_db: float = 100.0,
    device: str = "cpu",
    threads: int = 4,
    chunk_frames: int = 2048,
    channel_policy: str = "reject",
) -> Path:
    """Run the pinned enhancement chain once per corpus sample and save provenance."""
    manifest = manifest.expanduser().resolve()
    output = output.expanduser().resolve()
    if output.exists():
        raise VoxRefineError(f"Output directory already exists: {output}")
    samples = read_corpus(manifest)
    output.mkdir(parents=True)
    audio_dir = output / "audio"
    audio_dir.mkdir()
    report_path = output / "report.json"
    try:
        enhancer = StudioEnhancer(
            ap_bwe_source=ap_bwe_source, checkpoint=checkpoint,
            deep_filter=deep_filter, ffmpeg=ffmpeg,
            attenuation_limit_db=attenuation_limit_db, device=device,
            threads=threads, chunk_frames=chunk_frames,
            channel_policy=channel_policy,
        )
    except Exception as error:
        report_path.write_text(json.dumps({"schema_version": 1, "status": "failed",
                                           "error": str(error)}, indent=2) + "\n",
                               encoding="utf-8")
        raise
    report: dict[str, Any] = {
        "schema_version": 1,
        "run": {"started_utc": datetime.now(timezone.utc).isoformat(),
                "pipeline": "AP-BWE 16k->48k then DeepFilterNet 0.5.6",
                "manifest_path": str(manifest), "manifest_sha256": file_hash(manifest),
                "model_initialization_seconds": enhancer.model_setup_seconds,
                "device": device, "threads": threads, "chunk_frames": chunk_frames,
                "channel_policy": channel_policy,
                "attenuation_limit_db": attenuation_limit_db},
        "model": {"ap_bwe": enhancer.bwe.identity(), "deepfilternet": enhancer.dfn.identity()},
        "samples": [],
        "status": "running",
    }

    def save() -> None:
        temporary = report_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                        allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(report_path)

    save()
    for identifier, source, category, rights in samples:
        target = audio_dir / f"{identifier}-studio.wav"
        try:
            result = enhancer.process_file(source, target)
        except Exception as error:
            report["status"] = "failed"
            report["error"] = f"{identifier}: {error}"
            save()
            raise
        report["samples"].append({"id": identifier, "category": category,
                                  "rights": rights, "result": result})
        save()
    report["status"] = "complete"
    report["completed_utc"] = datetime.now(timezone.utc).isoformat()
    save()
    return report_path
