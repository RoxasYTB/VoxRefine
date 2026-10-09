"""Extract a deterministic, mono 16 kHz AIR evaluation subset from AIR v1.4."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import soundfile as sf
from scipy.io import loadmat
from scipy.signal import resample_poly


ROOMS = {"booth", "office", "meeting", "lecture"}
TARGET_SR = 16_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract(archive: Path, destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    candidates: dict[str, list[dict]] = {room: [] for room in ROOMS}
    with ZipFile(archive) as package:
        for member in sorted(package.namelist()):
            match = re.fullmatch(r"AIR_1_4/air_binaural_([a-z0-9_]+)_0_0_(\d+)\.mat", member)
            if not match or match.group(1) not in ROOMS:
                continue
            mat = loadmat(package.open(member), squeeze_me=True, struct_as_record=False)
            info, response = mat["air_info"], np.asarray(mat["h_air"], dtype=np.float64).reshape(-1)
            source_rate = int(np.asarray(info.fs).item())
            distance_cm = float(np.asarray(info.distance).item())
            channel, head = int(np.asarray(info.channel).item()), int(np.asarray(info.head).item())
            room = str(info.room)
            if room not in ROOMS or channel != 0 or head != 0:
                continue
            if source_rate != TARGET_SR:
                from math import gcd
                divisor = gcd(source_rate, TARGET_SR)
                response = resample_poly(response, TARGET_SR // divisor, source_rate // divisor)
            candidates[room].append({"member": member, "response": response.astype(np.float32),
                "distance_cm": distance_cm, "source_rate": source_rate, "channel": channel,
                "head": head, "room": room})

    selected = []
    for room, items in sorted(candidates.items()):
        if len(items) < 2:
            raise RuntimeError(f"AIR has fewer than two eligible RIRs for {room}")
        items.sort(key=lambda item: (item["distance_cm"], item["member"]))
        for role, item in (("short_distance", items[0]), ("long_distance", items[-1])):
            rel_config = f"{room}/MicID01/{role}_{item['distance_cm']:.0f}cm"
            config_dir = root / rel_config
            rir_dir = config_dir / "RIR"
            rir_dir.mkdir(parents=True, exist_ok=True)
            target = rir_dir / (Path(item["member"]).stem + ".wav")
            sf.write(target, item["response"], TARGET_SR, subtype="PCM_24")
            (config_dir / "mic_meta.txt").write_text(
                f"$EnvMic1RelDistance {item['distance_cm'] / 100:.6f}\n", encoding="utf-8")
            selected.append({"path": str(target), "room": room,
                "configuration": rel_config, "distance_m": item["distance_cm"] / 100,
                "distance_role": role, "sha256": sha256(target), "source_member": item["member"],
                "source_sample_rate": item["source_rate"], "selected_channel": item["channel"],
                "head_setting": item["head"], "normalization": "none here; evaluation applies training-domain early-path peak normalization"})
    manifest = {"source_archive": str(archive), "source_archive_sha256": sha256(archive),
        "dataset": "Aachen Impulse Response Database (AIR), v1.4",
        "source_page": "https://www.iks.rwth-aachen.de/en/research/tools-downloads/databases/aachen-impulse-response-database",
        "license": "MIT (AIR database, as stated in included readme and license.txt)",
        "attribution": "Marco Jeub et al., RWTH Aachen University, Aachen Impulse Response Database; cite Jeub et al., DSP 2009 and Jeub et al., ICA 2010.",
        "selection": "Single right-ear binaural path (channel 0), no dummy head (head 0), shortest and longest available source distances for booth, office, meeting, and lecture; 8 RIRs total; mono 16 kHz evaluation extracts.",
        "rirs": selected}
    (root / "air_subset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    result = extract(args.archive, args.destination)
    print(json.dumps({key: value for key, value in result.items() if key != "rirs"}, indent=2))
    for row in result["rirs"]:
        print(f"{row['room']} {row['distance_role']}: {row['path']} ({row['distance_m']:.2f} m)")


if __name__ == "__main__":
    main()
