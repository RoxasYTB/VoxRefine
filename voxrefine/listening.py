"""Reproducible blind listening copies from already-generated candidates."""

import json
from pathlib import Path
import random
import re
import shutil
import subprocess

from .audio import VoxRefineError
from .engines import file_hash
from .ffmpeg_io import probe_audio


def _read_candidates(manifest: Path) -> list[dict]:
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VoxRefineError(f"Cannot read candidate manifest: {error}") from error
    if not isinstance(data, dict) or not isinstance(data.get("samples"), list) or not data["samples"]:
        raise VoxRefineError("Candidate manifest must contain a non-empty 'samples' list.")
    ids: set[str] = set()
    samples = []
    for sample in data["samples"]:
        if not isinstance(sample, dict):
            raise VoxRefineError("Each listening sample must be an object.")
        identifier = sample.get("id")
        candidates = sample.get("candidates")
        if (not isinstance(identifier, str) or not identifier.isascii()
                or not identifier or not all(c.isalnum() or c in "-_" for c in identifier)):
            raise VoxRefineError("Listening sample ids may contain only ASCII letters, digits, - and _.")
        if identifier in ids:
            raise VoxRefineError(f"Duplicate listening sample id: {identifier}")
        ids.add(identifier)
        if not isinstance(candidates, list) or len(candidates) < 2:
            raise VoxRefineError(f"{identifier}: at least two candidates are required.")
        names: set[str] = set()
        resolved = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise VoxRefineError(f"{identifier}: each candidate needs engine and path.")
            name, path = candidate.get("engine"), candidate.get("path")
            if not isinstance(name, str) or not name.strip() or name in names:
                raise VoxRefineError(f"{identifier}: candidate engine names must be unique and non-empty.")
            if not isinstance(path, str) or not path.strip():
                raise VoxRefineError(f"{identifier}/{name}: candidate path is required.")
            source = (manifest.parent / path).expanduser().resolve()
            if not source.is_file():
                raise VoxRefineError(f"Candidate audio not found: {source}")
            names.add(name)
            resolved.append({"engine": name, "source": source})
        samples.append({"id": identifier, "candidates": resolved})
    return samples


def _decoded_duration(path: Path, ffmpeg: str, sample_rate: int = 48000) -> float:
    """Measure decoded PCM length, avoiding container-duration/padding differences."""
    process = subprocess.Popen(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(path),
         "-map", "0:a:0", "-f", "f32le", "-ac", "1", "-ar", str(sample_rate), "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    byte_count = 0
    assert process.stdout is not None
    while block := process.stdout.read(1024 * 1024):
        byte_count += len(block)
    process.stdout.close()
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    if process.stderr:
        process.stderr.close()
    return_code = process.wait()
    if return_code:
        raise VoxRefineError(f"Could not decode candidate {path}: {stderr.strip() or return_code}")
    if byte_count == 0 or byte_count % 4:
        raise VoxRefineError(f"Candidate decoded to empty or malformed PCM: {path}")
    return (byte_count // 4) / sample_rate


def create_blind_listening_set(
    manifest: Path,
    output: Path,
    ffmpeg: str = "ffmpeg",
    seed: int = 0,
    target_lufs: float = -20.0,
    true_peak_db: float = -1.5,
) -> tuple[Path, Path]:
    """Write loudness-matched, randomized copies and a separate reveal key."""
    if not -70 <= target_lufs <= -5:
        raise VoxRefineError("target_lufs must be between -70 and -5 LUFS.")
    if not -9 <= true_peak_db <= 0:
        raise VoxRefineError("true_peak_db must be between -9 and 0 dBTP.")
    if output.exists():
        raise VoxRefineError(f"Output already exists: {output}. Choose a new path; files are never overwritten.")
    samples = _read_candidates(manifest)
    ffmpeg_path = shutil.which(ffmpeg)
    if not ffmpeg_path:
        raise VoxRefineError(f"FFmpeg executable not found: {ffmpeg}")
    blind_dir = output / "blind"
    key_dir = output / "key"
    blind_dir.mkdir(parents=True)
    key_dir.mkdir()
    rng = random.Random(seed)
    blind_manifest = {
        "schema_version": 1,
        "experiment_id": output.name,
        "audio": {"sample_rate": 48000, "channels": 1, "codec": "PCM s24le",
                  "target_lufs": target_lufs, "true_peak_db": true_peak_db,
        "duration_policy": "crop all candidates to the shortest decoded duration after mono 48 kHz conversion"},
        "samples": [],
    }
    reveal = {
        "schema_version": 1,
        "seed": seed,
        "candidate_manifest_sha256": file_hash(manifest),
        "audio": blind_manifest["audio"],
        "samples": [],
    }
    try:
        previous_order: list[str] | None = None
        for sample in samples:
            inspected = []
            for candidate in sample["candidates"]:
                # Probe early for a useful format/error message, then count actual
                # decoded output samples because MP3 container duration can include
                # encoder delay/padding not emitted by FFmpeg's decoder.
                probe_audio(candidate["source"], ffmpeg_path)
                duration = _decoded_duration(candidate["source"], ffmpeg_path)
                inspected.append((candidate, duration))
            common_duration = min(duration for _, duration in inspected)
            candidates = [candidate for candidate, _ in inspected]
            rng.shuffle(candidates)
            order = [candidate["engine"] for candidate in candidates]
            if previous_order is not None and len(order) > 1 and order == previous_order:
                candidates = candidates[1:] + candidates[:1]
                order = [candidate["engine"] for candidate in candidates]
            previous_order = order
            blind_sample = {"id": sample["id"], "duration_seconds": common_duration,
                            "files": [f"{sample['id']}/{chr(65 + i)}.wav" for i in range(len(candidates))]}
            (blind_dir / sample["id"]).mkdir()
            reveal_candidates = []
            for index, candidate in enumerate(candidates):
                label = chr(65 + index)
                relative = Path(sample["id"]) / f"{label}.wav"
                target = blind_dir / relative
                trim_filter = f"atrim=duration={common_duration:.9f},asetpts=PTS-STARTPTS"
                analysis_filter = f"{trim_filter},loudnorm=I={target_lufs}:LRA=7:TP={true_peak_db}:print_format=json"
                analysis = subprocess.run(
                    [ffmpeg_path, "-hide_banner", "-nostats", "-nostdin", "-i",
                     str(candidate["source"]), "-af", analysis_filter, "-f", "null", "-"],
                    check=True, capture_output=True, text=True,
                )
                blocks = re.findall(r"\{[^{}]*\}", analysis.stderr, flags=re.DOTALL)
                try:
                    measured = json.loads(blocks[-1])
                    measured_i = float(measured["input_i"])
                    if measured_i <= -99:
                        measured_i = target_lufs
                    measured_tp = float(measured["input_tp"])
                    measured_lra = float(measured["input_lra"])
                    measured_thresh = float(measured["input_thresh"])
                    offset = float(measured["target_offset"])
                except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                    raise VoxRefineError(f"Could not measure loudness for {candidate['source']}.") from error
                filter_chain = (
                    f"{trim_filter},loudnorm=I={target_lufs}:LRA=7:TP={true_peak_db}:"
                    f"measured_I={measured_i}:measured_TP={measured_tp}:"
                    f"measured_LRA={measured_lra}:measured_thresh={measured_thresh}:"
                    f"offset={offset}:linear=true:print_format=summary"
                )
                command = [ffmpeg_path, "-hide_banner", "-loglevel", "error", "-nostats", "-nostdin",
                           "-i", str(candidate["source"]), "-af", filter_chain,
                           "-ar", "48000", "-ac", "1", "-c:a", "pcm_s24le", str(target)]
                subprocess.run(command, check=True, capture_output=True, text=True)
                reveal_candidates.append({"label": label, "engine": candidate["engine"],
                                          "source_path": str(candidate["source"]),
                                          "source_sha256": file_hash(candidate["source"]),
                                          "measured_source_loudness": {
                                              "integrated_lufs": measured_i,
                                              "true_peak_dbfs": measured_tp,
                                              "lra_lu": measured_lra,
                                          },
                                          "listening_path": str(relative),
                                          "listening_sha256": file_hash(target)})
            blind_manifest["samples"].append(blind_sample)
            reveal["samples"].append({"id": sample["id"], "duration_seconds": common_duration,
                                      "candidates": reveal_candidates})
    except (OSError, subprocess.CalledProcessError, VoxRefineError) as error:
        shutil.rmtree(output, ignore_errors=True)
        if isinstance(error, subprocess.CalledProcessError):
            detail = error.stderr.strip() or str(error)
            raise VoxRefineError(f"Could not prepare blind listening copy: {detail}") from error
        raise
    (blind_dir / "manifest.json").write_text(
        json.dumps(blind_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (key_dir / "reveal.json").write_text(
        json.dumps(reveal, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return blind_dir, key_dir / "reveal.json"
