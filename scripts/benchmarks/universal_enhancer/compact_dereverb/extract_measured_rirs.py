"""Extract only BUT ReverbDB RIR waveforms and matching metadata from its archive."""
from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path, PurePosixPath


def extract(archive_path: Path, destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    extracted, rooms = [], set()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            raw = PurePosixPath(member.name)
            parts = tuple(part for part in raw.parts if part not in ("", ".", "/"))
            if not parts:
                continue
            is_rir = "RIR" in parts and member.isfile() and parts[-1].lower().endswith(".wav")
            is_metadata = member.isfile() and parts[-1] in {
                "mic_meta.txt", "spk_meta.txt", "env_meta.txt", "env_full_meta.txt"
            }
            if not (is_rir or is_metadata):
                continue
            target = (root / Path(*parts)).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"archive member escapes extraction directory: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise OSError(f"cannot read archive member: {member.name}")
            with source, target.open("wb") as output:
                output.write(source.read())
            extracted.append(str(target))
            if is_rir:
                mic_index = next((i for i, p in enumerate(parts) if p.startswith("MicID")), None)
                if mic_index is not None and mic_index > 0:
                    rooms.add(parts[mic_index - 1])
    return {"archive": str(archive_path), "destination": str(destination),
            "extracted_file_count": len(extracted),
            "rir_file_count": sum(Path(p).suffix.lower() == ".wav" for p in extracted),
            "room_names": sorted(rooms)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--destination", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(extract(args.archive, args.destination), indent=2))


if __name__ == "__main__":
    main()
