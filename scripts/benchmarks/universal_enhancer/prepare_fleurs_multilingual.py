#!/usr/bin/env python3
"""Prepare official FLEURS train/dev WAVs as attributed, lossless 16 kHz FLAC.

Archives are downloaded one at a time, resumed through HTTP Range, checked
against their Hub SHA-256, extracted without resampling, then removed. The
official test split is deliberately excluded from preparation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
import tarfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath

import soundfile as sf

ROOT = Path(__file__).resolve().parents[3]
API = "https://huggingface.co/api/datasets/google/fleurs"
BASE = "https://huggingface.co/datasets/google/fleurs/resolve"
DEFAULT_OUT = ROOT / "corpus/samples/fleurs-multilingual-16k-01"
CITATION = (
    "Conneau, A., Ma, M., Khanuja, S., Zhang, Y., Axelrod, V., Dalmia, S., "
    "Riesa, J., Rivera, C., & Bapna, A. (2022). FLEURS: Few-shot Learning "
    "Evaluation of Universal Representations of Speech. SLT 2022. "
    "https://arxiv.org/abs/2205.12446"
)
LICENSE = "Creative Commons Attribution 4.0 International (CC BY 4.0)"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
CHUNK = 8 * 1024 * 1024


def get_json(url: str) -> dict | list:
    request = urllib.request.Request(url, headers={"User-Agent": "VoxRefine-research/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def fetch_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "VoxRefine-research/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def download_resumable(url: str, path: Path, expected_size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    offset = path.stat().st_size if path.exists() else 0
    if offset == expected_size:
        return
    if offset > expected_size:
        path.unlink()
        offset = 0
    headers = {"User-Agent": "VoxRefine-research/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=180)
    except urllib.error.HTTPError as exc:
        if offset and exc.code == 416:
            path.unlink(missing_ok=True)
            return download_resumable(url, path, expected_size)
        raise
    with response:
        status = getattr(response, "status", 200)
        if offset and status != 206:
            # The endpoint ignored Range; restart instead of appending a full body.
            offset = 0
        mode = "ab" if offset and status == 206 else "wb"
        if mode == "wb":
            offset = 0
        with path.open(mode) as handle:
            while True:
                block = response.read(CHUNK)
                if not block:
                    break
                handle.write(block)
                handle.flush()
                current = handle.tell()
                if current and current % (256 * 1024 * 1024) < CHUNK:
                    print(f"  download {current / 1024**3:.2f}/{expected_size / 1024**3:.2f} GiB", flush=True)
    actual = path.stat().st_size
    if actual != expected_size:
        raise RuntimeError(f"Incomplete archive {path}: {actual} bytes, expected {expected_size}; rerun with --resume")


def hub_revision() -> str:
    info = get_json(API)
    revision = info.get("sha") if isinstance(info, dict) else None
    if not revision:
        raise RuntimeError("Hugging Face dataset API did not return a pinned commit SHA")
    return str(revision)


def load_tree(revision: str) -> dict[str, dict]:
    query = urllib.parse.urlencode({"recursive": "true", "expand": "false"})
    rows = get_json(f"{API}/tree/{revision}?{query}")
    if not isinstance(rows, list):
        raise RuntimeError("Unexpected Hub tree response")
    return {str(row["path"]): row for row in rows if row.get("type") == "file"}


def locales_from_files(files: dict[str, dict]) -> list[str]:
    locales = set()
    for path in files:
        parts = path.split("/")
        if len(parts) == 4 and parts[0] == "data" and parts[2] == "audio" and parts[3].endswith(".tar.gz"):
            locales.add(parts[1])
    return sorted(locales)


def row_manifest(tsv: bytes) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    reader = csv.reader(io.StringIO(tsv.decode("utf-8")), delimiter="\t")
    for fields in reader:
        if len(fields) < 2:
            continue
        audio_name = PurePosixPath(fields[1]).name
        if audio_name != fields[1] or not audio_name.lower().endswith(".wav"):
            raise ValueError(f"Unexpected unsafe FLEURS audio path: {fields[1]!r}")
        rows[audio_name] = {
            "row_id": fields[0],
            "gender": fields[5] if len(fields) > 5 else None,
            "num_samples": int(fields[4]) if len(fields) > 4 and fields[4].isdigit() else None,
        }
    return rows


def existing_manifest(out: Path) -> dict:
    path = out / "manifest.json"
    if not path.exists():
        return {"dataset": "FLEURS", "records": [], "completed_splits": []}
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("dataset") != "FLEURS":
        raise RuntimeError(f"Refusing to resume a non-FLEURS manifest at {path}")
    return value


def write_manifest(out: Path, manifest: dict) -> None:
    target = out / "manifest.json"
    temp = target.with_suffix(".json.tmp")
    temp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(target)


def extract_split(archive: Path, tsv_rows: dict[str, dict], out_dir: Path,
                  locale: str, split: str, limit: int | None) -> list[dict]:
    records: list[dict] = []
    found: set[str] = set()
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, mode="r:gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.lower().endswith(".wav"):
                continue
            name = PurePosixPath(member.name).name
            meta = tsv_rows.get(name)
            if meta is None:
                continue
            found.add(name)
            if limit is not None and len(records) >= limit:
                continue
            target = out_dir / name
            payload = None
            if target.exists():
                info = sf.info(target)
                if info.samplerate != 16_000 or info.channels != 1:
                    raise RuntimeError(f"Existing output has unexpected format: {target} ({info})")
            else:
                source = tar.extractfile(member)
                if source is None:
                    raise RuntimeError(f"Cannot read archive member {member.name}")
                payload = source.read()
                target.write_bytes(payload)
                info = sf.info(target)
                if info.samplerate != 16_000 or info.channels != 1:
                    target.unlink(missing_ok=True)
                    raise RuntimeError(f"Expected mono 16 kHz source, got {info.samplerate} Hz/{info.channels}ch: {member.name}")
            records.append({
                "locale": locale, "split": split, "row_id": meta["row_id"],
                "speaker_id": None, "gender": meta["gender"],
                "path": str(target.relative_to(ROOT)), "sha256": hashlib.sha256(payload).hexdigest() if payload is not None else sha256_file(target),
                "sample_rate_hz": int(info.samplerate), "channels": int(info.channels),
                "duration_seconds": float(info.duration), "frames": int(info.frames),
                "source_member": member.name,
            })
    if not found:
        raise RuntimeError(f"No audio entries matched the {locale}/{split} TSV")
    if len(found) != len(tsv_rows):
        missing = len(tsv_rows) - len(found)
        raise RuntimeError(f"{locale}/{split}: archive is missing {missing} TSV audio files")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locales", default="all", help="all or comma-separated locale codes")
    parser.add_argument("--splits", default="train,dev", help="official train/dev only; test is rejected")
    parser.add_argument("--max-clips-per-locale", type=int, default=None,
                        help="optional reproducible cap: first N ordered rows from each locale/split")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.max_clips_per_locale is not None and args.max_clips_per_locale < 1:
        parser.error("--max-clips-per-locale must be positive")
    splits = [part.strip() for part in args.splits.split(",") if part.strip()]
    if not splits or any(split not in {"train", "dev"} for split in splits):
        parser.error("Only official train and dev splits are allowed; test remains reserved")

    revision = hub_revision()
    files = load_tree(revision)
    available = locales_from_files(files)
    requested = available if args.locales == "all" else [p.strip() for p in args.locales.split(",") if p.strip()]
    unknown = sorted(set(requested) - set(available))
    if unknown:
        parser.error(f"Unknown FLEURS locales: {', '.join(unknown)}")
    archive_rows = []
    for locale in requested:
        for split in splits:
            archive_path = f"data/{locale}/audio/{split}.tar.gz"
            tsv_path = f"data/{locale}/{split}.tsv"
            if archive_path not in files or tsv_path not in files:
                raise RuntimeError(f"Missing official FLEURS files for {locale}/{split}")
            row = files[archive_path]
            sha = row.get("lfs", {}).get("oid")
            if not sha or len(str(sha)) != 64:
                raise RuntimeError(f"No SHA-256 LFS hash supplied for {archive_path}")
            archive_rows.append({"locale": locale, "split": split, "archive_path": archive_path,
                                 "tsv_path": tsv_path, "size": int(row["size"]), "sha256": str(sha)})
    total_bytes = sum(row["size"] for row in archive_rows)
    largest = max((row["size"] for row in archive_rows), default=0)
    free_bytes = shutil.disk_usage(args.output.parent).free
    conservative_required = int(total_bytes * 1.5) + 2 * 1024**3
    summary = {"dataset": "google/fleurs", "revision": revision, "license": LICENSE,
               "locales": len(requested), "splits": splits,
               "archives": len(archive_rows), "compressed_download_bytes": total_bytes,
               "compressed_download_gib": round(total_bytes / 1024**3, 2),
               "largest_archive_gib": round(largest / 1024**3, 2),
               "free_disk_gib": round(free_bytes / 1024**3, 2),
               "conservative_free_disk_required_gib": round(conservative_required / 1024**3, 2),
               "max_clips_per_locale_split": args.max_clips_per_locale}
    print(json.dumps(summary, indent=2), flush=True)
    if args.dry_run:
        return
    if free_bytes < conservative_required:
        raise SystemExit("Insufficient free disk for the conservative archive+extraction estimate")
    if args.output.exists() and not (args.resume or args.force):
        raise SystemExit(f"Refusing to reuse {args.output}; pass --resume or --force")
    if args.force and args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    cache = ROOT / "results/.fleurs-download-cache"
    cache.mkdir(parents=True, exist_ok=True)
    manifest = existing_manifest(args.output)
    if manifest.get("revision") not in {None, revision}:
        raise RuntimeError("The existing corpus uses another Hub revision; use a new output directory")
    manifest.update({
        "dataset": "FLEURS", "repository": "https://huggingface.co/datasets/google/fleurs",
        "revision": revision, "license": LICENSE, "license_url": LICENSE_URL,
        "citation": CITATION, "audio_source_sample_rate_hz": 16_000,
        "audio_source_channels": 1,
        "preprocessing": "Original FLEURS train/dev WAV retained byte-for-byte after 16 kHz mono validation; no resampling, filtering, trimming, gain change, or re-encoding.",
        "test_split_policy": "Official test archives are intentionally excluded and remain sealed for final evaluation.",
        "records": manifest.get("records", []),
        "completed_splits": manifest.get("completed_splits", []),
    })
    completed = set(manifest["completed_splits"])
    # The TSV's numeric row id is not unique; the archive member is.
    existing_records = {(r["locale"], r["split"], r["source_member"]): r for r in manifest["records"]}
    base_url = f"{BASE}/{revision}/"
    for index, row in enumerate(archive_rows, 1):
        key = f"{row['locale']}/{row['split']}"
        if key in completed:
            print(f"[{index}/{len(archive_rows)}] already complete: {key}", flush=True)
            continue
        print(f"[{index}/{len(archive_rows)}] fetching {key} ({row['size'] / 1024**3:.2f} GiB compressed)", flush=True)
        archive = cache / f"{row['locale']}-{row['split']}.tar.gz.part"
        url = base_url + urllib.parse.quote(row["archive_path"], safe="/") + "?download=true"
        download_resumable(url, archive, row["size"])
        checksum = sha256_file(archive)
        if checksum != row["sha256"]:
            archive.unlink(missing_ok=True)
            raise RuntimeError(f"SHA-256 mismatch for {key}: got {checksum}, expected {row['sha256']}")
        tsv_url = base_url + urllib.parse.quote(row["tsv_path"], safe="/")
        tsv_data = fetch_bytes(tsv_url)
        tsv_rows = row_manifest(tsv_data)
        destination = args.output / row["locale"] / row["split"]
        records = extract_split(archive, tsv_rows, destination, row["locale"], row["split"],
                                args.max_clips_per_locale)
        for record in records:
            existing_records[(record["locale"], record["split"], record["source_member"])] = record
        manifest["records"] = list(existing_records.values())
        manifest["completed_splits"] = sorted(completed | {key})
        completed.add(key)
        write_manifest(args.output, manifest)
        archive.unlink()
        print(f"  extracted {len(records)} clips; completed {key}", flush=True)
    durations = {}
    for record in manifest["records"]:
        durations.setdefault(record["split"], {"clips": 0, "seconds": 0.0, "locales": set()})
        item = durations[record["split"]]
        item["clips"] += 1
        item["seconds"] += record["duration_seconds"]
        item["locales"].add(record["locale"])
    manifest["summary"] = {split: {"clips": item["clips"], "hours": round(item["seconds"] / 3600, 2),
                                     "locales": len(item["locales"])}
                           for split, item in durations.items()}
    write_manifest(args.output, manifest)
    print(json.dumps({"event": "complete", "output": str(args.output), "summary": manifest["summary"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
