"""Frozen A/B screen with measured BUT ReverbDB room responses."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import fftconvolve, resample_poly

from data import read_librispeech
from model import CompactDereverb16k

SR, FRAME, HOP = 16_000, 320, 160
CLIP_SAMPLES = 6 * SR
INSERTED_PAUSE_SAMPLES = int(.8 * SR)
RIR_PAGE = "https://speech.fit.vut.cz/software/but-speech-fit-reverb-database"
RIR_LICENSE = "CC BY 4.0 (dataset page); accompanying README includes Apache-2.0 notices"
RIR_ATTRIBUTION = "BUT Speech@FIT Reverb Database, Brno University of Technology; Szoke et al. (2018)"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_distance(meta: Path) -> float | None:
    if not meta.exists():
        return None
    match = re.search(r"^\$EnvMic\d+RelDistance\s+([-+0-9.eE]+)",
                      meta.read_text(errors="replace"), re.MULTILINE)
    return float(match.group(1)) if match else None


def select_rirs(root: Path, source_url: str = RIR_PAGE, dataset_version: str = "BUT ReverbDB 2019-06",
                license_text: str = RIR_LICENSE, attribution: str = RIR_ATTRIBUTION) -> list[dict]:
    """Preselect nearest and farthest distinct measured configurations per room."""
    by_room: dict[str, dict[str, dict]] = {}
    for path in sorted(root.rglob("*.wav")):
        if path.parent.name != "RIR":
            continue
        rel = path.relative_to(root)
        mic_setup = next((i for i, part in enumerate(rel.parts) if part.startswith("MicID")), None)
        if mic_setup is None or mic_setup < 1 or len(rel.parts) < mic_setup + 4:
            continue
        room = rel.parts[mic_setup - 1]
        config_dir = path.parent.parent
        config_key = str(config_dir.relative_to(root))
        prior = by_room.setdefault(room, {}).get(config_key)
        if prior is None or path.name < Path(prior["path"]).name:
            by_room[room][config_key] = {
                "path": str(path), "room": room, "configuration": config_key,
                "distance_m": parse_distance(config_dir / "mic_meta.txt")}
    result = []
    for room, configs in sorted(by_room.items()):
        choices = [c for c in configs.values() if c["distance_m"] is not None]
        if len(choices) < 2:
            raise RuntimeError(f"{room}: fewer than two configurations with distance metadata")
        choices.sort(key=lambda c: (c["distance_m"], c["configuration"], c["path"]))
        for role, choice in zip(("short_distance", "long_distance"), (choices[0], choices[-1])):
            item = dict(choice)
            item.update({"distance_role": role, "sha256": sha256(Path(item["path"])),
                         "source_url": source_url, "dataset_version": dataset_version,
                         "license": license_text, "attribution": attribution})
            result.append(item)
    expected = 2 * len(by_room)
    if len(result) != expected:
        raise RuntimeError(f"expected two distance-selected RIRs per room ({expected}), found {len(result)}")
    return result


def prepare_rir(item: dict) -> tuple[np.ndarray, dict]:
    path = Path(item["path"])
    raw, sr = sf.read(path, dtype="float64", always_2d=True)
    if raw.shape[1] != 1:
        raise ValueError(f"expected mono physical RIR, got {raw.shape[1]} channels: {path}")
    rir = raw[:, 0]
    source_sr = int(sr)
    if sr != SR:
        from math import gcd
        common = gcd(int(sr), SR)
        rir = resample_poly(rir, SR // common, int(sr) // common)
    head = rir[:int(.05 * SR)]
    if not head.size or not np.isfinite(rir).all():
        raise ValueError(f"invalid RIR: {path}")
    peak_sample = int(np.argmax(np.abs(head)))
    onset = int(np.flatnonzero(np.abs(head) >= np.max(np.abs(head)) * .1)[0])
    aligned = rir[onset:].copy()
    direct_peak = float(np.max(np.abs(aligned[:min(len(aligned), int(.01 * SR))])))
    if direct_peak <= 1e-12:
        raise ValueError(f"RIR direct/early peak too small: {path}")
    # The training target uses a unit-gain direct impulse. Normalize this
    # measured response to the same convention, retaining all relative early
    # reflections and late decay. Early-RMS normalization would multiply the
    # direct path by roughly sqrt(160), invalidating output/clean level gates.
    aligned /= direct_peak
    return aligned.astype(np.float32), {
        "source_sample_rate": source_sr,
        "resampler": "scipy.signal.resample_poly" if source_sr != SR else "none (native 16 kHz)",
        "selected_channel": 0, "direct_peak_sample_first_50ms": peak_sample,
        "direct_path_onset_sample": onset, "early_path_peak_before_normalization_0_10ms": direct_peak,
        "rir_scalar_applied": 1.0 / direct_peak,
        "normalization_rule": "training-domain early-path peak normalization: peak in first 10 ms after detected onset to unit gain; not an acoustic SPL calibration; retain entire tail"}


def rms_frames(x: np.ndarray) -> np.ndarray:
    if x.size < FRAME:
        return np.empty(0)
    frames = np.lib.stride_tricks.sliding_window_view(
        np.asarray(x, dtype=np.float64), FRAME)[::HOP]
    return np.sqrt(np.mean(frames * frames, axis=-1) + 1e-20)


def _read_mono(path: Path) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if sr != SR:
        raise ValueError(f"expected 16 kHz speech, got {sr}: {path}")
    if audio.ndim == 2:
        audio = audio.mean(axis=1, dtype=np.float32)
    if not np.isfinite(audio).all() or not audio.size:
        raise ValueError(f"invalid speech audio: {path}")
    return audio


def speech_bounds(x: np.ndarray) -> tuple[int, int]:
    """Clean-only 20/10 ms activity bounds with a fixed 2% peak threshold."""
    levels = rms_frames(x)
    active = levels >= max(float(levels.max()) * .02, 1e-5)
    ids = np.flatnonzero(active)
    if not ids.size:
        raise ValueError("utterance has no active speech under the fixed clean-only threshold")
    start = int(ids[0] * HOP)
    end = min(len(x), int(ids[-1] * HOP + FRAME))
    return start, end


def make_inserted_pause_pair(first_path: Path, second_path: Path) -> tuple[np.ndarray, dict]:
    """Join full adjacent same-speaker utterances with a controlled 800 ms gap."""
    first, second = _read_mono(first_path), _read_mono(second_path)
    if min(first.size, second.size) < SR:
        raise ValueError("both utterances must be at least one second long")
    _, first_end = speech_bounds(first)
    second_start, _ = speech_bounds(second)
    joined = np.concatenate((first, np.zeros(INSERTED_PAUSE_SAMPLES, np.float32), second))
    speech_offset = first_end
    next_speech_start = first.size + INSERTED_PAUSE_SAMPLES + second_start
    if speech_offset + int(.60 * SR) > next_speech_start - int(.20 * SR):
        raise ValueError("insufficient clean-only pause for the 300-600 ms score and 200 ms guard")
    pause = {"start_sample": speech_offset, "end_sample": next_speech_start,
             "clip_relative_start_sample": speech_offset,
             "first_speech_start_sample": int(speech_bounds(first)[0]),
             "first_speech_end_sample": speech_offset,
             "second_speech_start_sample": next_speech_start,
             "inserted_gap_start_sample": first.size,
             "duration_s": (next_speech_start - speech_offset) / SR,
             "kind": "inserted_controlled_pause", "first_utterance": first_path.name,
             "second_utterance": second_path.name}
    return joined, pause


def convolve_pair(clean: np.ndarray, rir: np.ndarray) -> dict:
    edge = int(.05 * SR)
    early_rir = rir[:edge]
    late_rir = np.pad(rir[edge:], (edge, 0))
    if late_rir.size < rir.size:
        late_rir = np.pad(late_rir, (0, rir.size - late_rir.size))
    wet = fftconvolve(clean, rir, mode="full")
    early = fftconvolve(clean, early_rir, mode="full")
    late = fftconvolve(clean, late_rir, mode="full")
    early = np.pad(early, (0, max(0, wet.size - early.size)))[:wet.size]
    late = np.pad(late, (0, max(0, wet.size - late.size)))[:wet.size]
    scale = .80 / max(float(np.max(np.abs(clean))), float(np.max(np.abs(wet))), 1e-8)
    return {"clean": (clean * scale).astype(np.float32),
            "reverberant": (wet * scale).astype(np.float32),
            "early": (early * scale).astype(np.float32),
            "late": (late * scale).astype(np.float32), "shared_scale": scale}


def db_ratio(a: float, b: float) -> float:
    return float(10 * np.log10(max(float(a), 1e-30) / max(float(b), 1e-30)))


def multiband_log_spectral_error(clean: np.ndarray, signal: np.ndarray) -> dict[str, float | None]:
    n_fft, hop = 512, 160
    count = 1 + (min(clean.size, signal.size) - n_fft) // hop
    if count <= 0:
        return {}
    window = np.hanning(n_fft)
    limit = (count - 1) * hop + n_fft
    c_frames = np.lib.stride_tricks.sliding_window_view(clean[:limit], n_fft)[::hop]
    s_frames = np.lib.stride_tricks.sliding_window_view(signal[:limit], n_fft)[::hop]
    c_spec = np.abs(np.fft.rfft(c_frames * window, axis=-1))
    s_spec = np.abs(np.fft.rfft(s_frames * window, axis=-1))
    error = np.abs(20 * np.log10(np.maximum(s_spec, 1e-8)) -
                   20 * np.log10(np.maximum(c_spec, 1e-8)))
    frame_rms = rms_frames(clean)[:count]
    active = frame_rms >= max(float(frame_rms.max()) * .02, 1e-5)
    freqs = np.fft.rfftfreq(n_fft, 1 / SR)
    bands = {"0_500hz": (0, 500), "500_2khz": (500, 2000),
             "2_4khz": (2000, 4000), "4_8khz": (4000, 8000)}
    out = {}
    for name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs < high)
        out[name] = (float(np.mean(error[active][:, mask]))
                     if active.any() and mask.any() else None)
    return out


def pause_metrics(clean: np.ndarray, wet: np.ndarray, out: np.ndarray,
                  dry_out: np.ndarray, pauses: list[dict],
                  late: np.ndarray | None = None) -> list[dict]:
    rows = []
    for pause in pauses:
        start = int(pause["clip_relative_start_sample"])
        next_speech = int(pause["second_speech_start_sample"])
        # Fixed input reference: reverberant-input power over the first
        # utterance's clean-derived active mask (same denominator for input
        # and both processed outputs). Keep a separate dry reference for floor.
        clean_levels = rms_frames(clean)
        active_threshold = max(float(clean_levels.max()) * .02, 1e-5)
        frame_starts = np.arange(clean_levels.size) * HOP
        ref_ids = np.flatnonzero((frame_starts >= max(0, start - int(.30 * SR))) &
                                 (frame_starts + FRAME <= start) &
                                 (clean_levels >= active_threshold))
        if not ref_ids.size:
            continue
        ref_indices = np.unique(np.concatenate([
            np.arange(i * HOP, i * HOP + FRAME) for i in ref_ids]))
        ref_e = float(np.mean(wet[ref_indices].astype(np.float64) ** 2))
        dry_ref_e = float(np.mean(clean[ref_indices].astype(np.float64) ** 2))
        for band, lo, hi in (("50_150", 50, 150), ("150_300", 150, 300), ("300_600", 300, 600)):
            a, b = start + int(lo * SR / 1000), start + int(hi * SR / 1000)
            if b > next_speech - int(.20 * SR) or b > out.size or b > wet.size:
                continue
            in_e = float(np.mean(wet[a:b].astype(np.float64) ** 2))
            out_e = float(np.mean(out[a:b].astype(np.float64) ** 2))
            floor_e = float(np.mean(dry_out[a:b].astype(np.float64) ** 2))
            late_e = (float(np.mean(late[a:b].astype(np.float64) ** 2))
                      if late is not None else 0.0)
            input_level = db_ratio(in_e, ref_e)
            output_level = db_ratio(out_e, ref_e)
            # Censoring must compare output and dry-control floor against the
            # same fixed denominator. Keep the dry-relative floor too as a
            # separate interpretable diagnostic.
            floor_level_dry_ref = db_ratio(floor_e, dry_ref_e)
            floor_level_input_ref = db_ratio(floor_e, ref_e)
            rows.append({"pause_start_sample": start, "pause_duration_s": pause["duration_s"],
                "band_ms": band, "input_tail_vs_fixed_input_speech_db": input_level,
                "output_tail_vs_same_fixed_input_speech_db": output_level,
                "known_late_vs_fixed_input_speech_db": db_ratio(late_e, ref_e) if late is not None else None,
                "tail_reduction_input_to_output_db": db_ratio(in_e, out_e),
                "dry_model_floor_vs_fixed_dry_speech_db": floor_level_dry_ref,
                "dry_model_floor_vs_fixed_input_speech_db": floor_level_input_ref,
                "output_margin_above_dry_floor_db": output_level - floor_level_input_ref,
                "output_floor_censored": bool(output_level <= floor_level_input_ref + 3.0)})
    return rows


def load_model(checkpoint: Path, device: torch.device) -> CompactDereverb16k:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = CompactDereverb16k(base_channels=int(state["base_channels"])).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model


def infer(model: CompactDereverb16k, x: np.ndarray, device: torch.device) -> tuple[np.ndarray, float]:
    tensor = torch.from_numpy(x).unsqueeze(0).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        y = model(tensor)[0].float().cpu().numpy()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return y, time.perf_counter() - start


def speech_metrics(clean: np.ndarray, wet: np.ndarray, out: np.ndarray,
                   pauses: list[dict] | None = None) -> dict:
    c, x, y = rms_frames(clean), rms_frames(wet), rms_frames(out)
    active = c >= max(float(c.max()) * .02, 1e-5)
    if pauses:
        # Do not classify deliberately inserted digital silence as weak speech.
        for pause in pauses:
            start = int(pause["clip_relative_start_sample"])
            end = start + int(pause["duration_s"] * SR)
            first_frame = max(0, (start - FRAME) // HOP)
            last_frame = min(active.size, (end + HOP - 1) // HOP)
            active[first_frame:last_frame] = False
    ids = np.flatnonzero(active)
    weak = ids[np.argsort(c[ids])[:max(1, int(np.ceil(.2 * ids.size)))]]
    gain = 20 * np.log10(np.maximum(y, 1e-10) / np.maximum(c, 1e-10))
    rising = np.zeros_like(active)
    rising[1:] = active[1:] & (c[1:] > np.maximum(c[:-1] * 1.5, 1e-5))
    return {"active_frame_count": int(active.sum()), "weak_frame_count": int(weak.size),
        "weak_output_vs_clean_db_p50_p90_p99": [float(v) for v in np.percentile(gain[weak], [50, 90, 99])],
        "weak_fraction_over_6db": float(np.mean(gain[weak] > 6)),
        "weak_fraction_over_10db": float(np.mean(gain[weak] > 10)),
        "active_output_vs_clean_db_p10_p50_p90": [float(v) for v in np.percentile(gain[ids], [10, 50, 90])],
        "rising_output_vs_clean_db_p10_p50_p90": ([float(v) for v in np.percentile(gain[rising], [10, 50, 90])]
            if rising.any() else [None] * 3),
        "low_energy_output_vs_input_db_p50": (float(np.median(20 * np.log10(
            np.maximum(y[~active], 1e-10) / np.maximum(x[~active], 1e-10)))) if (~active).any() else None),
        "input_vs_clean_multiband_log_spectral_mae_db": multiband_log_spectral_error(clean, wet),
        "output_vs_clean_multiband_log_spectral_mae_db": multiband_log_spectral_error(clean, out)}


def screen(args: argparse.Namespace) -> dict:
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    rirs = select_rirs(args.rir_root, args.rir_page, args.rir_dataset_version,
                       args.rir_license, args.rir_attribution)
    if args.rooms:
        requested = set(args.rooms)
        rirs = [item for item in rirs if item["room"] in requested]
        if {item["room"] for item in rirs} != requested:
            raise ValueError(f"requested rooms not found: {sorted(requested - {item['room'] for item in rirs})}")
    excluded = set()
    for split in ("train-clean-100", "dev-clean", "dev-other", "test-clean", "test-other"):
        excluded.update(s for _, s in read_librispeech(args.data_root, split))
    if args.exclude_speakers:
        excluded.update(line.strip() for line in args.exclude_speakers.read_text().splitlines()
                        if line.strip() and not line.lstrip().startswith("#"))
    speech_root = args.speech_root / "train-clean-360"
    if not speech_root.is_dir():
        raise FileNotFoundError(f"missing fresh speech split: {speech_root}")
    by_speaker: dict[str, list[Path]] = {}
    for path in sorted(speech_root.glob("*/*/*.flac")):
        by_speaker.setdefault(path.parent.parent.name, []).append(path)
    eligible = []
    for speaker_index, speaker in enumerate(sorted(set(by_speaker) - excluded)):
        selected = None
        paths = by_speaker[speaker]
        for utterance_index, (path, next_path) in enumerate(zip(paths, paths[1:])):
            try:
                clip, pause = make_inserted_pause_pair(path, next_path)
                selected = (speaker, (path, next_path), clip, [pause])
                break
            except ValueError:
                continue
        if selected is not None:
            eligible.append(selected)
        if len(eligible) >= args.speaker_limit:
            break
    if len(eligible) < args.speaker_limit:
        raise RuntimeError(f"requested {args.speaker_limit} eligible speakers; found {len(eligible)}")
    speakers = [row[0] for row in eligible]
    models = {args.baseline_name: load_model(args.baseline, device),
              args.challenger_name: load_model(args.challenger, device)}
    rows, speech_manifest = [], []
    timings = {name: [] for name in models}
    for speaker, paths, raw_clean, pauses in eligible:
        path = paths[0]
        speech_manifest.append({"speaker": speaker, "paths": [str(p) for p in paths],
                                "sha256": [sha256(p) for p in paths],
                                "pause_count": len(pauses), "pause_kind": "inserted_controlled_pause"})
        for rir_item in rirs:
            rir, rir_meta = prepare_rir(rir_item)
            pair = convolve_pair(raw_clean, rir)
            for name, model in models.items():
                dry_output, dry_elapsed = infer(model, pair["clean"], device)
                output, elapsed = infer(model, pair["reverberant"], device)
                timings[name].append((elapsed, pair["clean"].size / SR))
                rows.append({"model": name, "condition": "dry", "speaker": speaker,
                    "utterance": [p.name for p in paths], "room": rir_item["room"], "rir_file": rir_item["path"],
                    "configuration": rir_item["configuration"], "distance_m": rir_item["distance_m"],
                    "distance_role": rir_item["distance_role"], "rir_sha256": rir_item["sha256"],
                    "speech_sha256": speech_manifest[-1]["sha256"], "runtime_s": dry_elapsed,
                    "speech_metrics": speech_metrics(pair["clean"], pair["clean"], dry_output[:pair["clean"].size], pauses),
                    "inserted_pause_tail_metrics": pause_metrics(pair["clean"], pair["clean"],
                                                                  dry_output[:pair["clean"].size], dry_output[:pair["clean"].size], pauses)})
                rows.append({"model": name, "condition": "measured_reverb",
                    "speaker": speaker, "utterance": [p.name for p in paths],
                    "room": rir_item["room"], "rir_file": rir_item["path"],
                    "configuration": rir_item["configuration"], "distance_m": rir_item["distance_m"],
                    "distance_role": rir_item["distance_role"], "rir_sha256": rir_item["sha256"],
                    "speech_sha256": speech_manifest[-1]["sha256"], "rir_processing": rir_meta,
                    "shared_convolution_scale": pair["shared_scale"], "runtime_s": elapsed,
                    "speech_metrics": speech_metrics(pair["clean"], pair["reverberant"][:pair["clean"].size], output[:pair["clean"].size], pauses),
                    "inserted_pause_tail_metrics": pause_metrics(pair["clean"], pair["reverberant"],
                                                                 output[:pair["clean"].size], dry_output[:pair["clean"].size], pauses, pair["late"][:pair["clean"].size])})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "cases.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row, allow_nan=False) + "\n")
    manifest = {"rir_dataset": args.rir_dataset_version,
        "rir_page": args.rir_page, "rir_license": args.rir_license, "rir_attribution": args.rir_attribution,
        "speech_dataset": "LibriSpeech train-clean-360", "speech_license": "CC BY 4.0",
        "speech_license_url": "https://www.openslr.org/12/", "speech_speakers": speech_manifest,
        "excluded_speaker_count": len(excluded), "speaker_overlap": sorted(set(speakers) & excluded),
        "rir_selection": rirs, "room_count": len({r["room"] for r in rirs}), "rir_count": len(rirs),
        "case_count": len(rows), "device": str(device),
        "models": {name: {"checkpoint": str(path), "sha256": sha256(path),
            "parameters": sum(p.numel() for p in models[name].parameters()),
            "median_forward_seconds_per_case": float(np.median([t[0] for t in timings[name]])),
            "median_input_duration_seconds": float(np.median([t[1] for t in timings[name]])),
            "median_rtf": float(np.median([t[0] / t[1] for t in timings[name]]))}
            for name, path in ((args.baseline_name, args.baseline), (args.challenger_name, args.challenger))},
        "dry_control_models": list(models),
        "protocol": f"12 fresh train-clean-360 speakers; one deterministic pair of successive sorted same-speaker utterance files used in full with an inserted controlled 800 ms digital silence; clean-only activity threshold is 2% of each utterance peak using 20 ms RMS/10 ms hop; t0 is end of last active frame in utterance 1; score 50-150/150-300/300-600 ms after t0 and keep a 200 ms guard before utterance 2; convolve the complete sequence and full RIR tail; selected measured rooms {sorted({r['room'] for r in rirs})} × nearest/farthest source-mic distance configurations; mono 16 kHz RIR; training-domain early-path peak normalization: peak in the first 10 ms after detected onset scaled to 1 to match the model's unit-impulse training convention. This is not acoustic SPL calibration. Paired dry/reverb, no added noise, no local gain matching; frozen checkpoints. Tail levels use one fixed input-reverberant speech-active mask reference for input and model outputs. The dry-model floor is recorded relative to dry speech and converted to the same fixed wet-speech reference for censoring; floors are never subtracted. Paired A/B analysis uses the more conservative of both model floors plus 3 dB.",
        "limitations": ["This is an external measured-RIR evaluation; training provenance is recorded per checkpoint.",
            "This inserted-pause screen does not measure natural pauses or conversational timing.",
            "Linear convolution does not model real microphone noise, nonlinear playback/capture, or motion.",
            "Audio and derived renders stay local; the report contains metrics only."]}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return {"speakers": speakers, "room_count": manifest["room_count"], "rir_count": len(rirs),
            "case_count": len(rows), "models": manifest["models"]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--challenger", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--speech-root", type=Path, required=True)
    p.add_argument("--rir-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--speaker-limit", type=int, default=12)
    p.add_argument("--seed", type=int, default=20261011)
    p.add_argument("--rooms", nargs="+", default=None, help="Optional explicit room subset")
    p.add_argument("--exclude-speakers", type=Path, default=None,
                   help="Optional text file of speaker IDs to keep out of this evaluation")
    p.add_argument("--baseline-name", default="baseline", help="Name in the output case manifest")
    p.add_argument("--challenger-name", default="weak_over", help="Name in the output case manifest")
    p.add_argument("--rir-dataset-version", default="BUT Speech@FIT Reverb Database RIR-only release 2019-06")
    p.add_argument("--rir-page", default=RIR_PAGE)
    p.add_argument("--rir-license", default=RIR_LICENSE)
    p.add_argument("--rir-attribution", default=RIR_ATTRIBUTION)
    print(json.dumps(screen(p.parse_args()), indent=2))


if __name__ == "__main__":
    main()
