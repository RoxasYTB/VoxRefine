"""Select and extract a small, metadata-driven dEchorate SOFA RIR subset."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

import h5py
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


BASE_URL = "https://sofacoustics.org/data/database/dechorate/"
DATASET = "dEchorate SOFA subset"


class SofaIndex(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "a":
            href = dict(attrs).get("href")
            if href:
                self.hrefs.append(href)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        request = urllib.request.Request(url, headers={"User-Agent": "VoxRefine research benchmark"})
        with urllib.request.urlopen(request, timeout=90) as response, destination.open("wb") as output:
            output.write(response.read())
    return destination


def decode(value) -> str:
    raw = value.item() if hasattr(value, "item") else value
    return raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)


def rt20_seconds(ir: np.ndarray, sample_rate: int) -> float:
    """Estimate T20 from a Schroeder energy-decay curve (-5 to -25 dB)."""
    signal = np.asarray(ir, dtype=np.float64).reshape(-1)
    peak = int(np.argmax(np.abs(signal)))
    signal = signal[peak:]
    energy = signal * signal
    edc = np.cumsum(energy[::-1])[::-1]
    if not edc.size or edc[0] <= 0:
        return float("nan")
    curve_db = 10.0 * np.log10(np.maximum(edc / edc[0], 1e-15))
    selected = (curve_db <= -5.0) & (curve_db >= -25.0)
    if int(selected.sum()) < 30:
        return float("nan")
    time_s = np.arange(signal.size, dtype=np.float64) / float(sample_rate)
    slope = float(np.polyfit(time_s[selected], curve_db[selected], 1)[0])
    return -60.0 / slope if slope < -1e-6 else float("nan")


def sofa_summary(path: Path) -> dict:
    with h5py.File(path, "r") as sofa:
        ir = np.asarray(sofa["Data.IR"][0], dtype=np.float64)
        sample_rate = int(round(float(sofa["Data.SamplingRate"][0])))
        distances = np.asarray(sofa["ReceiverPosition"][...], dtype=np.float64)[:, :, 0]
        source_pos = np.asarray(sofa["SourcePosition"][...], dtype=np.float64)[0]
        license_text = decode(sofa.attrs.get("License", ""))
        room_description = decode(sofa.attrs.get("RoomDescription", ""))
        title = decode(sofa.attrs.get("Title", path.name))
        delay_s = np.asarray(sofa["Data.Delay"][0], dtype=np.float64)
    if sample_rate != 48_000:
        raise ValueError(f"expected 48 kHz SOFA, got {sample_rate}: {path}")
    t20 = [rt20_seconds(channel, sample_rate) for channel in ir]
    return {"title": title, "room_description": room_description,
            "sample_rate": sample_rate, "rir_count": int(ir.shape[0]),
            "receiver_positions_m": distances.tolist(), "source_position_m": source_pos.tolist(),
            "data_delay_s": delay_s.tolist(), "t20_seconds_by_receiver": t20,
            "median_t20_seconds": float(np.nanmedian(t20)), "license": license_text}


def source_positions_by_id(annotation_path: Path) -> tuple[dict[int, np.ndarray], np.ndarray]:
    with h5py.File(annotation_path, "r") as annotations:
        directional = np.asarray(annotations["sources_directional_position"][...], dtype=np.float64).T
        omnidirectional = np.asarray(annotations["sources_omnidirection_position"][...], dtype=np.float64).T
        microphones = np.asarray(annotations["microphones"][...], dtype=np.float64).T
    sources = {index + 1: position for index, position in enumerate(directional)}
    offset = len(sources)
    sources.update({offset + index + 1: position for index, position in enumerate(omnidirectional)})
    return sources, microphones


def spatial_extremes(room: str, files: list[tuple[str, int, int, int, int]],
                     source_positions: dict[int, np.ndarray], microphones: np.ndarray) -> dict[str, dict]:
    """Choose geometric extremes only from source/array pairs outside the probe."""
    candidates = []
    for filename, source_id, array_id, first_mic, last_mic in files:
        # source1/array1 was used to rank room configurations; it is never a test RIR.
        if source_id == 1 and array_id == 1:
            continue
        source = source_positions[source_id]
        for microphone_id in range(first_mic, last_mic + 1):
            microphone = microphones[microphone_id - 1]
            candidates.append({"room_code": room, "filename": filename,
                "source_id": source_id, "array_id": array_id, "microphone_id": microphone_id,
                "source_position_m": source.tolist(), "microphone_position_m": microphone.tolist(),
                "distance_m": float(np.linalg.norm(source - microphone))})
    if len(candidates) < 2:
        raise RuntimeError(f"{room}: fewer than two non-probe source/array/microphone paths")
    candidates.sort(key=lambda row: (row["distance_m"], row["source_id"],
                                     row["array_id"], row["microphone_id"]))
    return {"near": candidates[0], "far": candidates[-1]}


def extract(sofa_root: Path, annotation_path: Path, cache: Path, output_root: Path) -> dict:
    sofa_root.mkdir(parents=True, exist_ok=True)
    index_path = fetch(BASE_URL, cache / "index.html")
    parser = SofaIndex()
    parser.feed(index_path.read_text(errors="replace"))
    filename_pattern = re.compile(r"dEchorate_(room\d+)_src(\d+)_arr(\d+)_mics(\d+)-(\d+)\.sofa")
    parsed_files = [(href, *map(int, m.groups()[1:])) for href in parser.hrefs
                    if (m := filename_pattern.fullmatch(href))]
    rooms = sorted({m.group(1) for href in parser.hrefs if (m := filename_pattern.fullmatch(href))})
    if len(rooms) < 3:
        raise RuntimeError(f"expected at least three dEchorate room/config codes, found {rooms}")

    probes = []
    for room in rooms:
        filename = f"dEchorate_{room}_src1_arr1_mics1-5.sofa"
        path = fetch(BASE_URL + filename, sofa_root / "probes" / filename)
        info = sofa_summary(path)
        if "The MIT License (MIT)" not in info["license"]:
            raise ValueError(f"SOFA file does not carry the expected MIT license text: {filename}")
        probes.append({"room_code": room, "filename": filename, "path": str(path),
                       "sha256": sha256(path), **info})
    probes.sort(key=lambda row: (row["median_t20_seconds"], row["room_code"]))
    selected_indices = sorted({0, (len(probes) - 1) // 2, len(probes) - 1})
    selected_rooms = [probes[index] for index in selected_indices]

    source_positions, microphone_positions = source_positions_by_id(annotation_path)

    extracted = []
    output_root.mkdir(parents=True, exist_ok=True)
    for room_info in selected_rooms:
        room_code = room_info["room_code"]
        room_files = [(name, source_id, array_id, first_mic, last_mic)
                      for name, source_id, array_id, first_mic, last_mic in parsed_files
                      if name.startswith(f"dEchorate_{room_code}_")]
        pairs = spatial_extremes(room_code, room_files, source_positions, microphone_positions)
        for role, pair in pairs.items():
            mic_id, source_id = pair["microphone_id"], pair["source_id"]
            array_id, filename = pair["array_id"], pair["filename"]
            sofa_path = fetch(BASE_URL + filename, sofa_root / filename)
            with h5py.File(sofa_path, "r") as sofa:
                ir = np.asarray(sofa["Data.IR"][0, (mic_id - 1) % 5], dtype=np.float64)
                sample_rate = int(round(float(sofa["Data.SamplingRate"][0])))
                delay_s = float(sofa["Data.Delay"][0, (mic_id - 1) % 5])
                license_text = decode(sofa.attrs.get("License", ""))
            if "The MIT License (MIT)" not in license_text:
                raise ValueError(f"selected SOFA file has no embedded MIT license: {filename}")
            delay_samples = max(0, int(round(delay_s * sample_rate)))
            if delay_samples:
                ir = np.pad(ir, (delay_samples, 0))
            relative = Path(f"dEchorate_{room_code}") / "MicID01" / role
            config_dir = output_root / relative
            rir_dir = config_dir / "RIR"
            rir_dir.mkdir(parents=True, exist_ok=True)
            target = rir_dir / f"source{source_id}_mic{mic_id}.wav"
            sf.write(target, ir, sample_rate, subtype="PCM_24")
            (config_dir / "mic_meta.txt").write_text(
                f"$EnvMic1RelDistance {pair['distance_m']:.9f}\n", encoding="utf-8")
            extracted.append({"path": str(target), "sha256": sha256(target),
                "source_sofa": filename, "source_sofa_sha256": sha256(sofa_path),
                "room_code": room_code, "room_t20_median_seconds": room_info["median_t20_seconds"],
                "distance_role": role, "source_id": source_id, "microphone_id": mic_id,
                "source_position_m": pair["source_position_m"],
                "microphone_position_m": pair["microphone_position_m"],
                "distance_m": pair["distance_m"], "sample_rate": sample_rate,
                "data_delay_s": delay_s, "license": license_text})

    manifest = {"dataset": DATASET, "official_download_index": BASE_URL,
        "source_annotation_file": str(annotation_path), "source_annotation_sha256": sha256(annotation_path),
        "selection_rule": "Download one source 1 / array 1 probe per every listed room configuration; rank the median T20 across its five receivers and retain minimum, median and maximum configurations. Probe RIRs are never scored. In each selected configuration, enumerate source/microphone 3D distances over the official source and microphone annotations and available SOFA source/array files, excluding source 1 / array 1; select the nearest and farthest paths, tie-breaking by source ID, array ID, then microphone ID. Neither selection uses model outputs.",
        "t20_method": "Schroeder energy-decay curve; linear fit from -5 to -25 dB, extrapolated to 60 dB; median of five receivers in source1/array1 probe.",
        "license_rule": "Every probe and selected SOFA file must embed the MIT license text in the SOFA License attribute; no Zenodo HDF5 audio archive is used.",
        "selected_configurations": selected_rooms,
        "selected_source_microphone_pairs_by_configuration": {
            item["room_code"]: spatial_extremes(item["room_code"],
                [(name, source_id, array_id, first_mic, last_mic)
                 for name, source_id, array_id, first_mic, last_mic in parsed_files
                 if name.startswith(f"dEchorate_{item['room_code']}_")],
                source_positions, microphone_positions)
            for item in selected_rooms},
        "rirs": extracted}
    manifest_path = output_root / "dechorate_subset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {"manifest": str(manifest_path), "selected_room_codes": [r["room_code"] for r in selected_rooms],
            "selected_room_t20_s": [r["median_t20_seconds"] for r in selected_rooms],
            "rir_count": len(extracted)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sofa-root", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(extract(args.sofa_root, args.annotation, args.cache, args.output_root), indent=2))


if __name__ == "__main__":
    main()
