"""Manifest-driven comparisons; technical metrics are not quality scores."""

import json
from pathlib import Path
import platform
import time

from . import __version__
from .audio import VoxRefineError, inspect_wav, measure_wav
from .engines import Engine, clean, file_hash


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
        inspect_wav(path)
        records.append((identifier, path, sample["category"], sample["rights"]))
    return records


def benchmark(manifest: Path, output: Path, engines: list[Engine]) -> Path:
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
            "Wall time includes initialization and file I/O. RMS is not LUFS. "
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
        original = measure_wav(source)
        entry = {
            "id": identifier,
            "category": category,
            "rights": rights,
            "input_sha256": file_hash(source),
            "input": original,
            "outputs": [],
        }
        report["samples"].append(entry)
        for engine in engines:
            target = output / f"{identifier}-{engine.name}.wav"
            started = time.perf_counter()
            try:
                clean(source, target, engine)
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
