#!/usr/bin/env python3
"""Create deterministic Cap60-conditioned dereverberation pairs."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, fftconvolve, resample_poly, sosfiltfilt

from data import SAMPLE_RATE, read_librispeech
from screen_measured_rirs import prepare_rir, select_rirs
from train_measured_mix import HOLDOUT_ROOMS, TRAIN_ROOMS, measured_pair

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / ".tools/compact-dereverb/cap60-conditioned-2026-10-10"
DEEP_FILTER = ROOT / ".tools/deepfilternet/deep-filter"
LIBRISPEECH = ROOT / ".tools/compact-dereverb/data/LibriSpeech"
BUT_ROOT = ROOT / ".tools/compact-dereverb/data/but-reverbdb/extracted"
SR = 16_000
GUARD_SAMPLES = SR // 10
CROP_SAMPLES = 2 * SR
SEED = 20261010


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf8")).hexdigest()


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim == 2:
        x = x.mean(axis=1, dtype=np.float32)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError(f"invalid audio: {path}")
    return x, int(sr)


def to_16k(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == SR:
        return np.asarray(x, dtype=np.float32)
    from math import gcd
    g = gcd(int(sr), SR)
    return resample_poly(x, SR // g, int(sr) // g).astype(np.float32)


def read_train_dev(data_root: Path, excluded_path: Path) -> tuple[list, list, list[str]]:
    train = read_librispeech(data_root, "train-clean-100")
    dev = read_librispeech(data_root, "dev-clean")
    train_ids, dev_ids = {s for _, s in train}, {s for _, s in dev}
    if train_ids & dev_ids:
        raise RuntimeError(f"LibriSpeech train/dev speaker overlap: {sorted(train_ids & dev_ids)}")
    previous_ids: set[str] = set()
    for path in (ROOT / ".tools/compact-dereverb/air-external-2026-10-09/excluded_prior-speakers.txt",
                 ROOT / ".tools/compact-dereverb/data/dechorate/excluded-speakers.txt"):
        if path.exists():
            previous_ids.update(line.strip() for line in path.read_text().splitlines()
                                if line.strip() and not line.lstrip().startswith("#"))
    # Exclude any speaker named in any existing experiment manifest; manifest
    # paths are local metadata only and never load sealed test.wav.
    for manifest in ROOT.glob(".tools/compact-dereverb/**/manifest.json"):
        try:
            doc = json.loads(manifest.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for entry in doc.get("speakers", []):
            if isinstance(entry, dict) and entry.get("speaker") is not None:
                previous_ids.add(str(entry["speaker"]))
    future_root = data_root / "train-clean-360"
    future_pool = {path.name for path in future_root.iterdir() if path.is_dir() and path.name.isdigit()}
    # A rerun that regenerates manifests must preserve the holdout list already
    # frozen for this experiment instead of treating its own IDs as prior data.
    frozen_future = ([line.strip() for line in excluded_path.read_text().splitlines()
                      if line.strip() and not line.lstrip().startswith("#")]
                     if excluded_path.exists() else [])
    if frozen_future:
        if (len(frozen_future) != 12 or len(set(frozen_future)) != 12 or
                not set(frozen_future) <= future_pool or
                set(frozen_future) & (train_ids | dev_ids | previous_ids)):
            raise RuntimeError("existing future holdout list is invalid or overlaps known speakers")
        future_ids = frozen_future
    else:
        future_ids = sorted((future_pool
                             - train_ids - dev_ids - previous_ids), key=hash_text)[:12]
    if len(future_ids) != 12:
        raise RuntimeError("could not reserve 12 unseen future holdout speakers")
    return train, dev, future_ids


def selected_rooms(rir_root: Path) -> tuple[list[dict], list[dict]]:
    all_rirs = select_rirs(rir_root)
    train = [x for x in all_rirs if x["room"] in TRAIN_ROOMS]
    dev = [x for x in all_rirs if x["room"] in HOLDOUT_ROOMS]
    if len(train) != 12 or len({x["room"] for x in train}) != 6:
        raise RuntimeError(f"expected 12 selected BUT training RIRs, got {len(train)}")
    if len(dev) != 6 or {x["room"] for x in dev} != set(HOLDOUT_ROOMS):
        raise RuntimeError(f"expected 6 selected BUT dev RIRs, got {len(dev)}")
    train_cfg = {x["configuration"] for x in train}
    dev_cfg = {x["configuration"] for x in dev}
    if train_cfg & dev_cfg:
        raise RuntimeError("BUT train/dev configuration overlap")
    return train, dev


def cap60_one(executable: Path, x: np.ndarray, sr: int, work: Path,
              cache_key: str, cache_dir: Path) -> tuple[np.ndarray, dict]:
    cache_path = cache_dir / f"{cache_key}.npy"
    meta_path = cache_dir / f"{cache_key}.json"
    if cache_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("source_sha256") == hashlib.sha256(np.asarray(x).tobytes()).hexdigest():
            y = np.load(cache_path, allow_pickle=False)
            if len(y) == len(to_16k(x, sr)) and np.isfinite(y).all():
                return y.astype(np.float32), meta
    canon = to_16k(np.asarray(x, dtype=np.float32), sr)
    if not len(canon):
        raise ValueError("empty canonical audio")
    guarded = np.pad(canon, (0, GUARD_SAMPLES)).astype(np.float32)
    source_hash = hashlib.sha256(np.asarray(x).tobytes()).hexdigest()
    # DeepFilterNet CLI accepts a file and writes a PCM WAV. The guard moves
    # its known 480-sample EOF trim into disposable samples.
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="cap60-") as td:
        td_path = Path(td)
        in_path, out_dir = td_path / "input.wav", td_path / "out"
        sf.write(in_path, guarded, SR, subtype="PCM_16")
        proc = subprocess.run([str(executable), "--atten-lim-db", "60",
                               "--compensate-delay", "-o", str(out_dir), str(in_path)],
                             check=True, capture_output=True, text=True)
        out_path = out_dir / in_path.name
        if not out_path.exists():
            wavs = list(out_dir.glob("*.wav"))
            if len(wavs) != 1:
                raise RuntimeError(f"Cap60 output missing; stdout={proc.stdout[-1000:]}")
            out_path = wavs[0]
        y, out_sr = load_audio(out_path)
        if out_sr != SR:
            raise RuntimeError(f"Cap60 returned {out_sr} Hz; expected {SR}")
        if len(y) < len(canon):
            raise RuntimeError(f"Cap60 output shorter than source: {len(y)} < {len(canon)}")
        y = y[:len(canon)]
    if not np.isfinite(y).all() or np.max(np.abs(y)) >= 1.0:
        raise RuntimeError("Cap60 output is non-finite or clipped")
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, y.astype(np.float32), allow_pickle=False)
    meta = {"source_sha256": source_hash, "sample_rate": SR, "input_samples": len(canon),
            "guard_samples": GUARD_SAMPLES, "output_samples": len(y),
            "alignment_samples": 0, "elapsed_s": time.perf_counter() - started,
            "cap60_stdout_tail": proc.stdout[-600:]}
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return y.astype(np.float32), meta


def fan_noise(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n).astype(np.float64)
    sos = butter(4, [90, 5_800], btype="bandpass", fs=SR, output="sos")
    noise = sosfiltfilt(sos, white)
    t = np.arange(n, dtype=np.float64) / SR
    noise += .10 * np.sin(2 * np.pi * 120 * t + .31)
    noise += .035 * np.sin(2 * np.pi * 240 * t + 1.17)
    noise /= max(float(np.sqrt(np.mean(noise * noise))), 1e-12)
    return noise.astype(np.float32)


def active_indices(x: np.ndarray) -> np.ndarray:
    frame, hop = 320, 160
    if len(x) < frame:
        return np.arange(len(x))
    frames = np.lib.stride_tricks.sliding_window_view(x, frame)[::hop]
    levels = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-24)
    selected = np.flatnonzero((levels >= max(float(levels.max()) * .02, 1e-5)) &
                              ((np.arange(len(levels)) % 2) == 0))
    return np.concatenate([np.arange(i * hop, min(i * hop + frame, len(x))) for i in selected])


def make_fan_components(clean: np.ndarray, wet: np.ndarray, seed: int,
                        snr_db: float) -> tuple[np.ndarray, np.ndarray]:
    indices = active_indices(clean)
    n = min(len(clean), len(wet))
    noise = fan_noise(n, seed)
    speech_rms = float(np.sqrt(np.mean(wet[indices] ** 2) + 1e-24))
    noise_rms = float(np.sqrt(np.mean(noise[indices] ** 2) + 1e-24))
    noise = (noise * (speech_rms / (noise_rms * 10 ** (snr_db / 20)))).astype(np.float32)
    return (wet + noise).astype(np.float32), noise


def make_fan_mix(clean: np.ndarray, wet: np.ndarray, seed: int, snr_db: float) -> np.ndarray:
    return make_fan_components(clean, wet, seed, snr_db)[0]


def choose_rir(rng: np.random.Generator, train_rirs: list[dict], by_room: dict) -> tuple[np.ndarray, str, str | None, dict]:
    if rng.random() < .5:
        seed = int(rng.integers(0, 2**32))
        t60 = float(rng.uniform(.25, 1.2))
        drr = float(rng.uniform(-6, 18))
        rir, _, _ = __import__("data").make_procedural_rir(
            SR, seed, t60, drr)
        return rir, "procedural", None, {"seed": seed, "t60_s": t60, "direct_to_reverb_db": drr}
    rooms = sorted(by_room)
    room = rooms[int(rng.integers(0, len(rooms)))]
    item = by_room[room][int(rng.integers(0, len(by_room[room])))]
    rir, _ = prepare_rir(item)
    return rir, "but_measured", room, {"configuration": item["configuration"],
                                        "rir_sha256": item["sha256"]}


def utterance_pairs(rows: list, max_files: int) -> list[dict]:
    grouped: dict[str, list[Path]] = {}
    for path, speaker in rows:
        grouped.setdefault(speaker, []).append(path)
    pairs = []
    for speaker, paths in sorted(grouped.items()):
        paths = paths[:max_files] if max_files else paths
        if len(paths) < 2:
            continue
        # Every utterance gets a stable pair with its next same-speaker file.
        for i in range(0, len(paths) - 1, 2):
            pairs.append({"speaker": speaker, "first": paths[i], "second": paths[i + 1]})
    return pairs


def make_clean(pair: dict, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    from screen_measured_rirs import make_inserted_pause_pair
    joined, pause = make_inserted_pause_pair(pair["first"], pair["second"])
    # Keep complete joined utterances through Cap60. Crop only after full-
    # utterance preprocessing to avoid making model startup an input variable.
    joined = joined.astype(np.float32)
    joined *= .80 / max(float(np.max(np.abs(joined))), 1e-8)
    frame, hop = 320, 160
    levels = np.sqrt(np.mean(np.lib.stride_tricks.sliding_window_view(joined, frame)[::hop] ** 2,
                              axis=1) + 1e-20)
    candidates = np.flatnonzero(levels > max(float(levels.max()) * .02, 1e-5))
    center = int(candidates[int(rng.integers(0, len(candidates)))] * hop)
    start = max(0, min(len(joined) - CROP_SAMPLES, center - CROP_SAMPLES // 2))
    return joined, {"crop_start": start, "source_pair": [pair["first"].name, pair["second"].name],
                    "pause_start": pause["clip_relative_start_sample"],
                    "pause_end": pause["second_speech_start_sample"],
                    "pause_duration_s": pause["duration_s"]}


def crop_after_precompute(x: np.ndarray, start: int) -> np.ndarray:
    result = np.asarray(x[start:start + CROP_SAMPLES], dtype=np.float32)
    if len(result) != CROP_SAMPLES:
        raise RuntimeError("selected crop falls outside precomputed utterance")
    return result


def cap_target_speech_ratio(clean: np.ndarray, target: np.ndarray) -> float:
    """Reject pairs where the frozen denoiser erased the reference speech."""
    frame, hop = 320, 160
    n = min(len(clean), len(target))
    c, t = clean[:n], target[:n]
    if n < frame:
        return float(np.sqrt(np.mean(t * t) + 1e-20) /
                     max(np.sqrt(np.mean(c * c) + 1e-20), 1e-10))
    c_frames = np.lib.stride_tricks.sliding_window_view(c, frame)[::hop]
    t_frames = np.lib.stride_tricks.sliding_window_view(t, frame)[::hop]
    c_rms = np.sqrt(np.mean(c_frames.astype(np.float64) ** 2, axis=1) + 1e-20)
    t_rms = np.sqrt(np.mean(t_frames.astype(np.float64) ** 2, axis=1) + 1e-20)
    active = c_rms >= max(float(c_rms.max()) * .02, 1e-5)
    if not active.any():
        return 0.0
    return float(np.sqrt(np.mean(t_rms[active] ** 2)) /
                 max(np.sqrt(np.mean(c_rms[active] ** 2)), 1e-10))


def paired_clean_scale_ratio(original_clean: np.ndarray, cap_input_clean: np.ndarray) -> float:
    """RMS scale between clean and the exact paired clean signal sent to Cap60."""
    n = min(len(original_clean), len(cap_input_clean))
    original = np.asarray(original_clean[:n], dtype=np.float64)
    paired = np.asarray(cap_input_clean[:n], dtype=np.float64)
    mask = active_indices(np.asarray(original_clean, dtype=np.float32))
    mask = mask[mask < n]
    if not len(mask):
        return 0.0
    reference_rms = float(np.sqrt(np.mean(original[mask] ** 2) + 1e-24))
    paired_rms = float(np.sqrt(np.mean(paired[mask] ** 2) + 1e-24))
    return paired_rms / max(reference_rms, 1e-12)


def ratio_db(ratio: float) -> float:
    return float(20.0 * np.log10(max(float(ratio), 1e-12)))


def write_exclusion_summary(output_dir: Path, filename: str, rows: list[dict],
                            group_keys: tuple[str, ...]) -> dict:
    path = output_dir / filename
    path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                                for row in rows))
    groups: dict[tuple, dict[str, int]] = {}
    for row in rows:
        key = tuple(str(row.get(name)) for name in group_keys)
        groups.setdefault(key, {"attempted": 0, "excluded": 0})
        groups[key]["attempted"] += 1
        groups[key]["excluded"] += int(bool(row.get("excluded")))
    summary_rows = []
    for key, counts in sorted(groups.items()):
        summary_rows.append({**dict(zip(group_keys, key)), **counts,
                             "exclusion_percent": (100.0 * counts["excluded"] /
                                                   counts["attempted"]
                                                   if counts["attempted"] else 0.0)})
    summary_path = output_dir / filename.replace(".jsonl", "-summary.json")
    summary_path.write_text(json.dumps({"group_keys": group_keys,
        "attempted_candidates": len(rows), "excluded_candidates": sum(
            bool(row.get("excluded")) for row in rows),
        "excluded_percent": (100.0 * sum(bool(row.get("excluded")) for row in rows) /
                             len(rows) if rows else 0.0),
        "groups": summary_rows}, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return {"exclusion_manifest": str(path), "exclusion_summary": str(summary_path),
            "exclusion_manifest_sha256": sha256(path),
            "exclusion_summary_sha256": sha256(summary_path),
            "attempted_candidates": len(rows),
            "excluded_candidates": sum(bool(row.get("excluded")) for row in rows),
            "excluded_percent": (100.0 * sum(bool(row.get("excluded")) for row in rows) /
                                 len(rows) if rows else 0.0)}


def make_examples(rows: list, train_rirs: list[dict], output_dir: Path,
                  executable: Path, max_files: int) -> dict:
    rng = np.random.default_rng(SEED)
    pairs = utterance_pairs(rows, max_files)
    if len(pairs) < 100:
        raise RuntimeError(f"too few same-speaker utterance pairs: {len(pairs)}")
    pairs_by_speaker: dict[str, list[dict]] = {}
    for pair in pairs:
        pairs_by_speaker.setdefault(pair["speaker"], []).append(pair)
    speaker_ids = sorted(pairs_by_speaker)
    rng.shuffle(speaker_ids)
    pairs = [pairs_by_speaker[speaker][int(rng.integers(0, len(pairs_by_speaker[speaker])))]
             for speaker in speaker_ids]
    by_room: dict[str, list[dict]] = {}
    for item in train_rirs:
        by_room.setdefault(item["room"], []).append(item)
    cache_dir = output_dir / "cap60-cache"
    waveform_dir = output_dir / "arrays"
    waveform_dir.mkdir(parents=True, exist_ok=True)
    rows_out = []
    target_checks: list[dict] = []
    # One pair per speaker: avoid letting speakers with many utterances dominate.
    examples = min(128 if not max_files else 16, len(pairs))
    schedule = (["identity"] * (examples // 4) + ["rir_only"] * (examples * 3 // 8) +
                ["fan20"] * (examples * 3 // 16) + ["fan10"] * (examples * 3 // 16))
    rng.shuffle(schedule)
    rejected_targets = 0
    used_speakers = set()
    for index in range(examples):
        kind = schedule[index]
        accepted = None
        for attempt in range(min(len(pairs), 40)):
            pair = pairs[(index + attempt) % len(pairs)]
            if pair["speaker"] in used_speakers:
                continue
            clean, source = make_clean(pair, rng)
            gain = float(10 ** (rng.uniform(-15, -3) / 20))
            crop_start = int(source["crop_start"])
            key = f"{index:06d}-{attempt:02d}"
            if kind == "identity":
                clean_for_ratio = clean
                input_full, meta = cap60_one(executable, clean, SR, output_dir,
                                             f"input-{key}", cache_dir)
                target_full = input_full
                rir_kind, rir_room, rir_recipe, noise_seed = "identity", None, {"identity": True}, None
                branch_scale = 1.0
            else:
                rir, rir_kind, rir_room, rir_recipe = choose_rir(rng, train_rirs, by_room)
                pair_data = measured_pair(clean, rir)
                clean_target = pair_data["clean"]
                clean_for_ratio = clean_target
                branch_scale = 1.0
                mixture_raw = pair_data["reverberant"]
                noise_seed = None
                if kind in ("fan20", "fan10"):
                    noise_seed = int(rng.integers(2**32))
                    mixture_raw = make_fan_mix(clean, mixture_raw, noise_seed,
                                               20 if kind == "fan20" else 10)
                    shared_peak = max(float(np.max(np.abs(clean_target))),
                                      float(np.max(np.abs(mixture_raw))), 1e-8)
                    if shared_peak > .95:
                        branch_scale = .95 / shared_peak
                        mixture_raw, clean_target = mixture_raw * branch_scale, clean_target * branch_scale
                        clean_for_ratio = clean_target
                input_full, meta = cap60_one(executable, mixture_raw, SR, output_dir,
                                             f"input-{key}", cache_dir)
                target_full, target_meta = cap60_one(executable, clean_target, SR, output_dir,
                                                     f"target-{key}", cache_dir)
            c_crop = crop_after_precompute(clean, crop_start)
            input_arr = crop_after_precompute(input_full, crop_start) * gain
            target_arr = crop_after_precompute(target_full, crop_start) * gain
            ratio = cap_target_speech_ratio(crop_after_precompute(clean_for_ratio, crop_start),
                    crop_after_precompute(target_full, crop_start))
            paired_scale_ratio = paired_clean_scale_ratio(clean, clean_for_ratio)
            target_check = {"speaker": pair["speaker"], "condition": kind,
                "rir_kind": rir_kind, "rir_room": rir_room, "rir_recipe": rir_recipe,
                "noise_seed": noise_seed, "crop_start_sample": crop_start,
                "cap_target_speech_to_clean_rms_ratio": ratio,
                "cap_target_speech_to_clean_db": ratio_db(ratio),
                "cap_input_clean_to_original_rms_ratio": paired_scale_ratio,
                "cap_input_clean_to_original_db": ratio_db(paired_scale_ratio),
                "shared_branch_scale": branch_scale,
                "source_pair": source["source_pair"], "excluded": ratio < .05}
            target_checks.append(target_check)
            if ratio < .05:
                rejected_targets += 1
                continue
            accepted = (pair, clean, source, gain, crop_start, rir_kind, rir_room,
                        rir_recipe, noise_seed, meta, c_crop, input_arr, target_arr, ratio)
            break
        if accepted is None:
            raise RuntimeError(f"could not find a Cap60-preserved clean target for {kind} example {index}")
        (pair, clean, source, gain, crop_start, rir_kind, rir_room, rir_recipe,
         noise_seed, meta, c_crop, input_arr, target_arr, ratio) = accepted
        used_speakers.add(pair["speaker"])
        np.save(waveform_dir / f"x-{index:06d}.npy", input_arr, allow_pickle=False)
        np.save(waveform_dir / f"t-{index:06d}.npy", target_arr, allow_pickle=False)
        np.save(waveform_dir / f"c-{index:06d}.npy", c_crop * gain, allow_pickle=False)
        rows_out.append({"index": index, "kind": kind, "speaker": pair["speaker"],
                         "source": source, "rir_kind": rir_kind, "rir_room": rir_room,
                         "rir_recipe": rir_recipe, "noise_seed": noise_seed,
                         "source_sha256": [sha256(pair["first"]), sha256(pair["second"])],
                         "post_cap_gain": gain,
                         "cap_target_speech_to_clean_rms_ratio": ratio,
                         "cap_target_speech_to_clean_db": ratio_db(ratio),
                         "cap_input_clean_to_original_rms_ratio": paired_scale_ratio,
                         "cap_input_clean_to_original_db": ratio_db(paired_scale_ratio),
                         "shared_branch_scale": branch_scale,
                         "crop_start_sample": crop_start,
                         "x": f"arrays/x-{index:06d}.npy", "c": f"arrays/c-{index:06d}.npy",
                         "t": f"arrays/t-{index:06d}.npy",
                         "input_sha256": sha256(waveform_dir / f"x-{index:06d}.npy"),
                         "target_sha256": sha256(waveform_dir / f"t-{index:06d}.npy"),
                         "preprocessor": meta})
    manifest_path = output_dir / "pairs.jsonl"
    manifest_path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows_out))
    exclusion_stats = write_exclusion_summary(output_dir, "target-exclusions.jsonl",
                                               target_checks, ("speaker", "condition"))
    return {"examples": len(rows_out), "pair_manifest": str(manifest_path),
            "kind_counts": {kind: sum(row["kind"] == kind for row in rows_out)
                            for kind in ("identity", "rir_only", "fan20", "fan10")},
            "speaker_count": len({row["speaker"] for row in rows_out}),
            "rejected_cap60_targets_below_0_05_speech_rms_ratio": rejected_targets,
            **exclusion_stats,
            "cache_files": len(list(cache_dir.glob("*.npy"))),
            "cap60_inference_seconds": float(sum(
                json.loads(meta.read_text()).get("elapsed_s", 0.0)
                for meta in cache_dir.glob("*.json"))),
            "training_manifest_sha256": sha256(manifest_path)}


def make_dev_examples(rows: list, dev_rirs: list[dict], output_dir: Path,
                      executable: Path, max_files: int) -> dict:
    """Fixed BUT dev set with deterministic speech+fan and noise-only controls."""
    from train_measured_mix import measured_pair
    rng = np.random.default_rng(SEED + 91)
    pairs = utterance_pairs(rows, max_files)
    if len(pairs) < 6:
        raise RuntimeError("too few dev utterance pairs")
    rng.shuffle(pairs)
    by_room = {room: [x for x in dev_rirs if x["room"] == room]
               for room in sorted({x["room"] for x in dev_rirs})}
    room_order = sorted(by_room)
    candidate_count = min(18, len(pairs))
    candidates = []
    for i, pair in enumerate(pairs[:candidate_count]):
        room = room_order[i % len(room_order)]
        rir_item = by_room[room][i % len(by_room[room])]
        candidates.append({"index": i, "pair": pair, "room": room, "rir_item": rir_item})
    candidate_counts = {room: sum(candidate["room"] == room for candidate in candidates)
                        for room in room_order}
    if candidate_count != 18 or any(count != 6 for count in candidate_counts.values()):
        raise RuntimeError(f"expected 18 fixed RIR candidates (6 per room): {candidate_counts}")
    identity_indices = set()
    for room in room_order:
        selected = sorted((candidate for candidate in candidates if candidate["room"] == room),
            key=lambda candidate: (candidate["pair"]["speaker"],
                                   candidate["rir_item"]["sha256"]))[:2]
        identity_indices.update(candidate["index"] for candidate in selected)
    identity_candidate_counts = {room: sum(
        candidate["room"] == room and candidate["index"] in identity_indices
        for candidate in candidates) for room in room_order}
    if len(identity_indices) != 6 or any(count != 2 for count in identity_candidate_counts.values()):
        raise RuntimeError(f"expected six fixed identity candidates (two per room): {identity_candidate_counts}")
    candidate_manifest_path = output_dir / "dev-candidates.json"
    candidate_manifest = {"selection_policy": "first 18 seeded dev-clean speaker pairs, six RIR candidates per BUT dev room; choose two identity candidates per room by canonical (speaker_id, RIR-SHA256) order before target filtering",
        "rir_candidates": [{"candidate_index": candidate["index"],
            "speaker": candidate["pair"]["speaker"], "room": candidate["room"],
            "rir_configuration": candidate["rir_item"]["configuration"],
            "rir_sha256": candidate["rir_item"]["sha256"],
            "source_files": [candidate["pair"]["first"].name,
                             candidate["pair"]["second"].name],
            "source_sha256": [sha256(candidate["pair"]["first"]),
                              sha256(candidate["pair"]["second"])],
            "identity_candidate": candidate["index"] in identity_indices}
            for candidate in candidates],
        "rir_candidates_per_room": candidate_counts,
        "identity_candidates_per_room": identity_candidate_counts,
        "target_filter_applied_after_selection": True,
        "replacement_after_exclusion": False}
    candidate_manifest_path.write_text(json.dumps(candidate_manifest, indent=2,
        ensure_ascii=False, allow_nan=False) + "\n")
    rows_out = []
    evaluation_rows = []
    target_checks: list[dict] = []
    preprocessing_failures: list[dict] = []
    evaluation_arrays = output_dir / "arrays"
    evaluation_arrays.mkdir(parents=True, exist_ok=True)

    def save_array(name: str, audio: np.ndarray) -> str:
        rel = f"arrays/{name}.npy"
        np.save(output_dir / rel, np.asarray(audio, dtype=np.float32), allow_pickle=False)
        return rel

    for candidate in candidates:
        i, pair, room, rir_item = (candidate["index"], candidate["pair"],
                                   candidate["room"], candidate["rir_item"])
        try:
            clean, source = make_clean(pair, rng)
        except Exception as exc:
            failure = {"speaker": pair["speaker"], "room": room, "candidate_index": i,
                "reason": "clean_audio_or_inserted_pause_preprocessing_failure",
                "error": f"{type(exc).__name__}: {exc}"}
            preprocessing_failures.append({**failure, "condition": "rir_only"})
            if i in identity_indices:
                preprocessing_failures.append({**failure, "condition": "identity"})
            continue
        gain = float(10 ** (rng.uniform(-12, -3) / 20))
        global_pause_start = int(source["pause_start"])
        global_pause_end = int(source["pause_end"])
        max_start = max(0, len(clean) - CROP_SAMPLES)
        crop_low = max(0, global_pause_end - CROP_SAMPLES)
        crop_high = min(max_start, global_pause_start - int(.35 * SR))
        if crop_low > crop_high:
            failure = {"speaker": pair["speaker"], "room": room, "candidate_index": i,
                "reason": "controlled_pause_does_not_fit_2s_crop_with_350ms_speech_reference"}
            preprocessing_failures.append({**failure, "condition": "rir_only"})
            if i in identity_indices:
                preprocessing_failures.append({**failure, "condition": "identity"})
            continue
        crop_start = min(max(global_pause_start - int(.6 * SR), crop_low), crop_high)
        source = {**source, "crop_start": crop_start,
                  "pause_start": global_pause_start - crop_start,
                  "pause_end": global_pause_end - crop_start}
        start = int(source["crop_start"])
        c_base = crop_after_precompute(clean, start)

        if i in identity_indices:
            try:
                identity_full, identity_meta = cap60_one(executable, clean, SR, output_dir,
                    f"dev-identity-target-{i:04d}", output_dir / "cap60-cache")
                identity_base = crop_after_precompute(identity_full, start)
                identity_ratio = cap_target_speech_ratio(c_base, identity_base)
            except Exception as exc:
                preprocessing_failures.append({"speaker": pair["speaker"], "room": room,
                    "candidate_index": i, "condition": "identity",
                    "reason": "identity_clean_cap60_preprocessing_failure",
                    "error": f"{type(exc).__name__}: {exc}"})
            else:
                identity_check = {"candidate_index": i,
                    "speaker": pair["speaker"], "condition": "identity",
                    "rir_kind": "identity", "rir_room": room,
                    "rir_configuration": rir_item["configuration"], "rir_sha256": rir_item["sha256"],
                    "crop_start_sample": start,
                    "cap_target_speech_to_clean_rms_ratio": identity_ratio,
                    "cap_target_speech_to_clean_db": ratio_db(identity_ratio),
                    "cap_input_clean_to_original_rms_ratio": 1.0,
                    "cap_input_clean_to_original_db": 0.0,
                    "source_pair": source["source_pair"], "excluded": identity_ratio < .05}
                target_checks.append(identity_check)
                if identity_ratio < .05:
                    identity_check["reason"] = "cap60_identity_target_speech_rms_ratio_below_0_05"
                else:
                    identity = identity_base * gain
                    identity_stem = f"dev-identity-{i:04d}"
                    for suffix in ("x", "t"):
                        np.save(output_dir / "arrays" / f"{suffix}-{identity_stem}.npy",
                                identity, allow_pickle=False)
                    np.save(output_dir / "arrays" / f"c-{identity_stem}.npy",
                            c_base * gain, allow_pickle=False)
                    rows_out.append({"index": len(rows_out), "kind": "identity",
                        "speaker": pair["speaker"], "rir_room": None,
                        "selection_room": room, "selection_rir_sha256": rir_item["sha256"],
                        "cap_target_speech_to_clean_rms_ratio": identity_ratio,
                        "cap_target_speech_to_clean_db": ratio_db(identity_ratio),
                        "x": f"arrays/x-{identity_stem}.npy",
                        "t": f"arrays/t-{identity_stem}.npy",
                        "c": f"arrays/c-{identity_stem}.npy", "source": source,
                        "source_sha256": [sha256(pair["first"]), sha256(pair["second"])],
                        "input_sha256": sha256(output_dir / "arrays" / f"x-{identity_stem}.npy"),
                        "target_sha256": sha256(output_dir / "arrays" / f"t-{identity_stem}.npy"),
                        "cap60_target": identity_meta})
                    eval_stem = f"dev-eval-identity-{i:04d}"
                    evaluation_rows.append({"kind": "identity", "candidate_index": i,
                        "speaker": pair["speaker"], "selection_room": room,
                        "selection_rir_sha256": rir_item["sha256"],
                        "source": source, "post_cap_gain": gain,
                        "cap_target_speech_to_clean_rms_ratio": identity_ratio,
                        "arrays": {"clean": save_array(f"c-{eval_stem}", c_base * gain),
                                   "target": save_array(f"t-{eval_stem}", identity)},
                        "cap60": {"target": identity_meta},
                        "pause": {"clip_relative_start_sample": source["pause_start"],
                                  "second_speech_start_sample": source["pause_end"],
                                  "duration_s": source["pause_duration_s"]}})

        try:
            rir, _ = prepare_rir(rir_item)
            paired = measured_pair(clean, rir)
            mix20, noise20 = make_fan_components(paired["clean"], paired["reverberant"],
                int(hash_text(f"{pair['speaker']}:{room}:{rir_item['sha256']}:fan20")[:8], 16), 20)
            mix10, noise10 = make_fan_components(paired["clean"], paired["reverberant"],
                int(hash_text(f"{pair['speaker']}:{room}:{rir_item['sha256']}:fan10")[:8], 16), 10)
            shared_peak = max(float(np.max(np.abs(paired["clean"]))),
                              float(np.max(np.abs(paired["reverberant"]))),
                              float(np.max(np.abs(mix20))), float(np.max(np.abs(mix10))),
                              float(np.max(np.abs(noise20))), float(np.max(np.abs(noise10))), 1e-8)
            branch_scale = min(1.0, .95 / shared_peak) if shared_peak > .95 else 1.0
            clean_target = (paired["clean"] * branch_scale).astype(np.float32)
            wet_input = (paired["reverberant"] * branch_scale).astype(np.float32)
            mix20 = (mix20 * branch_scale).astype(np.float32)
            mix10 = (mix10 * branch_scale).astype(np.float32)
            noise20 = (noise20 * branch_scale).astype(np.float32)
            noise10 = (noise10 * branch_scale).astype(np.float32)
            target_full, target_meta = cap60_one(executable, clean_target, SR, output_dir,
                                                 f"dev-target-{i:04d}", output_dir / "cap60-cache")
            target_crop = crop_after_precompute(target_full, start)
            ratio = cap_target_speech_ratio(crop_after_precompute(clean_target, start), target_crop)
        except Exception as exc:
            preprocessing_failures.append({"speaker": pair["speaker"], "room": room,
                "candidate_index": i, "condition": "rir_only",
                "reason": "rir_or_cap60_target_preprocessing_failure",
                "error": f"{type(exc).__name__}: {exc}"})
            continue
        t_base = target_crop
        paired_scale_ratio = paired_clean_scale_ratio(clean, clean_target)
        check = {"candidate_index": i, "speaker": pair["speaker"], "condition": "rir_only",
                 "rir_kind": "but_measured", "rir_room": room,
                 "rir_configuration": rir_item["configuration"], "rir_sha256": rir_item["sha256"],
                 "crop_start_sample": start,
                 "cap_target_speech_to_clean_rms_ratio": ratio,
                 "cap_target_speech_to_clean_db": ratio_db(ratio),
                 "cap_input_clean_to_original_rms_ratio": paired_scale_ratio,
                 "cap_input_clean_to_original_db": ratio_db(paired_scale_ratio),
                 "source_pair": source["source_pair"], "excluded": ratio < .05}
        target_checks.append(check)
        if ratio < .05:
            check["reason"] = "cap60_target_speech_rms_ratio_below_0_05"
            continue
        try:
            condition_sources = {"rir_only": wet_input, "fan20": mix20, "fan10": mix10}
            noise_sources = {"fan20": noise20, "fan10": noise10}
            input_full, input_meta = cap60_one(executable, wet_input, SR, output_dir,
                f"dev-input-{i:04d}", output_dir / "cap60-cache")
            processed_inputs = {"rir_only": (input_full, input_meta)}
            processed_noises = {}
            for condition in ("fan20", "fan10"):
                processed_inputs[condition] = cap60_one(executable, condition_sources[condition], SR,
                    output_dir, f"dev-input-{condition}-{i:04d}", output_dir / "cap60-cache")
                processed_noises[condition] = cap60_one(executable, noise_sources[condition], SR,
                    output_dir, f"dev-noise-{condition}-{i:04d}", output_dir / "cap60-cache")
            input_crop = crop_after_precompute(input_full, start)
        except Exception as exc:
            preprocessing_failures.append({"speaker": pair["speaker"], "room": room,
                "candidate_index": i, "condition": "rir_only",
                "reason": "rir_input_or_fan_cap60_preprocessing_failure",
                "error": f"{type(exc).__name__}: {exc}"})
            continue
        x_base = input_crop
        x, t, c = x_base * gain, t_base * gain, c_base * gain
        stem = f"dev-{i:04d}"
        for suffix, audio in (("x", x), ("t", t), ("c", c)):
            np.save(output_dir / "arrays" / f"{suffix}-{stem}.npy", audio, allow_pickle=False)
        rows_out.append({"index": len(rows_out), "kind": "rir_only", "speaker": pair["speaker"],
                         "rir_room": room, "rir_configuration": rir_item["configuration"],
                         "rir_sha256": rir_item["sha256"],
                         "cap_target_speech_to_clean_rms_ratio": ratio,
                         "cap_target_speech_to_clean_db": ratio_db(ratio),
                         "cap_input_clean_to_original_rms_ratio": paired_scale_ratio,
                         "cap_input_clean_to_original_db": ratio_db(paired_scale_ratio),
                         "x": f"arrays/x-{stem}.npy", "t": f"arrays/t-{stem}.npy",
                         "c": f"arrays/c-{stem}.npy", "source": source,
                         "source_sha256": [sha256(pair["first"]), sha256(pair["second"])],
                         "input_sha256": sha256(output_dir / "arrays" / f"x-{stem}.npy"),
                         "target_sha256": sha256(output_dir / "arrays" / f"t-{stem}.npy"),
                         "cap60_input": input_meta, "cap60_target": target_meta})

        stem = f"dev-eval-{i:04d}"
        c_audio = c_base * gain
        target_audio = t_base * gain
        eval_paths = {"clean": save_array(f"c-{stem}", c_audio),
                      "target": save_array(f"t-{stem}", target_audio),
                      "wet_rir_only": save_array(f"w-rir-{stem}",
                          crop_after_precompute(wet_input, start) * gain),
                      "x_rir_only": save_array(f"x-rir-{stem}", x_base * gain)}
        for condition in ("fan20", "fan10"):
            mix_full, mix_meta = processed_inputs[condition]
            noise_full, noise_meta = processed_noises[condition]
            eval_paths[f"wet_{condition}"] = save_array(f"w-{condition}-{stem}",
                crop_after_precompute(condition_sources[condition], start) * gain)
            eval_paths[f"x_{condition}"] = save_array(f"x-{condition}-{stem}",
                crop_after_precompute(mix_full, start) * gain)
            eval_paths[f"noise_{condition}"] = save_array(f"noise-{condition}-{stem}",
                crop_after_precompute(noise_full, start) * gain)
        evaluation_rows.append({"kind": "rir", "candidate_index": i,
            "speaker": pair["speaker"], "room": room,
            "rir_configuration": rir_item["configuration"], "rir_sha256": rir_item["sha256"],
            "source": source, "source_sha256": [sha256(pair["first"]), sha256(pair["second"])],
            "post_cap_gain": gain, "shared_branch_scale": branch_scale,
            "cap_input_clean_to_original_rms_ratio": paired_scale_ratio,
            "cap_input_clean_to_original_db": ratio_db(paired_scale_ratio),
            "cap_target_speech_to_clean_rms_ratio": ratio,
            "cap_target_speech_to_clean_db": ratio_db(ratio),
            "fan_seeds": {"fan20": int(hash_text(f"{pair['speaker']}:{room}:{rir_item['sha256']}:fan20")[:8], 16),
                          "fan10": int(hash_text(f"{pair['speaker']}:{room}:{rir_item['sha256']}:fan10")[:8], 16)},
            "arrays": eval_paths,
            "cap60": {"target": target_meta,
                "inputs": {condition: in_meta for condition, (_, in_meta)
                           in processed_inputs.items()},
                "noise": {condition: noise_meta for condition, (_, noise_meta)
                          in processed_noises.items()}},
            "pause": {"clip_relative_start_sample": source["pause_start"],
                      "second_speech_start_sample": source["pause_end"],
                      "duration_s": source["pause_duration_s"]}})
    path = output_dir / "dev-pairs.jsonl"
    per_room = {room: sum(row.get("rir_room") == room for row in rows_out
                          if row["kind"] == "rir_only") for room in by_room}
    path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows_out))
    eval_path = output_dir / "dev-evaluation.jsonl"
    eval_path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                                    for row in evaluation_rows))
    dev_exclusion_stats = write_exclusion_summary(output_dir, "dev-target-exclusions.jsonl",
        target_checks, ("speaker", "condition", "rir_room"))
    rir_eval_count = sum(row["kind"] == "rir" for row in evaluation_rows)
    identity_eval_count = sum(row["kind"] == "identity" for row in evaluation_rows)
    rir_target_rejections = sum(row["condition"] == "rir_only" and row["excluded"]
                                for row in target_checks)
    identity_target_rejections = sum(row["condition"] == "identity" and row["excluded"]
                                     for row in target_checks)
    failure_summary = {}
    for condition, denominator in (("rir_only", candidate_count), ("identity", len(identity_indices))):
        failures = [row for row in preprocessing_failures if row["condition"] == condition]
        failure_summary[condition] = {"candidate_count": denominator,
            "preprocessing_failure_count": len(failures),
            "preprocessing_failure_percent": (100.0 * len(failures) / denominator
                                               if denominator else 0.0),
            "by_room": {room: sum(row["room"] == room for row in failures)
                        for room in room_order}}
    return {"dev_pair_manifest": str(path), "dev_pairs": len(rows_out),
            "dev_candidate_manifest": str(candidate_manifest_path),
            "dev_candidate_manifest_sha256": sha256(candidate_manifest_path),
            "dev_evaluation_manifest": str(eval_path),
            "dev_rir_candidate_count": candidate_count,
            "dev_rir_candidate_count_per_room": candidate_counts,
            "dev_identity_candidate_count": len(identity_indices),
            "dev_identity_candidate_count_per_room": identity_candidate_counts,
            "dev_evaluation_rir_cases": rir_eval_count,
            "dev_evaluation_identity_cases": identity_eval_count,
            "dev_evaluation_conditions": ["rir_only", "fan20", "fan10", "noise_only_fan20", "noise_only_fan10"],
            "dev_fan_seed_policy": "SHA256(speaker:room:RIR-SHA256:condition), first 32 bits",
            "dev_preprocessing_failures": preprocessing_failures,
            "dev_speakers": len({row["speaker"] for row in rows_out}),
            "dev_rir_rooms": sorted(room for room, count in per_room.items() if count),
            "dev_rir_valid_pairs_per_room": per_room,
            "dev_rir_coverage_sufficient_for_room_gate": all(count >= 3 for count in per_room.values()),
            "dev_rir_target_rejections_below_0_05": rir_target_rejections,
            "dev_identity_target_rejections_below_0_05": identity_target_rejections,
            "dev_rir_crop_preprocessing_failures": sum(
                row["condition"] == "rir_only" for row in preprocessing_failures),
            "dev_identity_crop_preprocessing_failures": sum(
                row["condition"] == "identity" for row in preprocessing_failures),
            "dev_preprocessing_failure_summary": failure_summary,
            **{f"dev_{key}": value for key, value in dev_exclusion_stats.items()}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=LIBRISPEECH)
    parser.add_argument("--rir-root", type=Path, default=BUT_ROOT)
    parser.add_argument("--deep-filter", type=Path, default=DEEP_FILTER)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--max-files-per-speaker", type=int, default=0,
                        help="small deterministic smoke dataset; 0 builds the frozen full cache")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if not args.deep_filter.exists():
        raise FileNotFoundError(args.deep_filter)
    train_rows, dev_rows, future_ids = read_train_dev(args.data_root,
            args.output_dir / "sealed-future-speakers.txt")
    train_rirs, dev_rirs = selected_rooms(args.rir_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "sealed-future-speakers.txt").write_text("\n".join(future_ids) + "\n")
    result = make_examples(train_rows, train_rirs, args.output_dir,
                           args.deep_filter, args.max_files_per_speaker)
    result.update(make_dev_examples(dev_rows, dev_rirs, args.output_dir,
                                    args.deep_filter, args.max_files_per_speaker))
    manifest = {"seed": SEED, "sample_rate": SR, "crop_samples": CROP_SAMPLES,
        "guard_samples": GUARD_SAMPLES, "deep_filter_sha256": sha256(args.deep_filter),
        "deep_filter_command": [str(args.deep_filter), "--atten-lim-db", "60", "--compensate-delay"],
        "train_split": "train-clean-100", "dev_split": "dev-clean",
        "train_speakers": len({s for _, s in train_rows}), "dev_speakers": len({s for _, s in dev_rows}),
        "speaker_overlap": [], "future_holdout_speakers": future_ids,
        "train_rir_rooms": list(TRAIN_ROOMS), "dev_rir_rooms": list(HOLDOUT_ROOMS),
        "train_rir_hashes": [x["sha256"] for x in train_rirs],
        "dev_rir_hashes": [x["sha256"] for x in dev_rirs],
        "no_air_or_dechorate_training": True, "test_wav_accessed": False, **result}
    (args.output_dir / "training-data-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
