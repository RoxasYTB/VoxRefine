"""Frozen speaker/RIR holdout screen for the compact dereverb proof."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from data import make_validation_pair, read_librispeech
from model import CompactDereverb16k
from train import _read_crop


SR = 16_000
CLIP_SAMPLES = 40_000  # 2.5 s
EVENT_START = 8_000    # 0.5 s leading silence
EVENT_SAMPLES = 8_000  # 0.5 s speech event


def db_ratio(numerator: float, denominator: float) -> float:
    return float(10.0 * np.log10(max(numerator, 1e-20) / max(denominator, 1e-20)))


def energy(x: np.ndarray) -> float:
    return float(np.mean(np.square(np.asarray(x, dtype=np.float64))))


def percentile_summary(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return {"median": float("nan"), "p10": float("nan"),
                "p90": float("nan"), "worst": float("nan")}
    return {"median": float(np.median(a)), "p10": float(np.percentile(a, 10)),
            "p90": float(np.percentile(a, 90)), "worst": float(np.min(a))}


def embed_event(clean: np.ndarray) -> np.ndarray:
    """Put a speech fragment in a 2.5 s quiet clip without changing its level."""
    clean = np.asarray(clean, dtype=np.float32)
    if clean.size < EVENT_SAMPLES:
        clean = np.tile(clean, int(np.ceil(EVENT_SAMPLES / clean.size)))
    # A fixed middle crop avoids selection based on model outputs.
    offset = max(0, (clean.size - EVENT_SAMPLES) // 2)
    event = clean[offset:offset + EVENT_SAMPLES]
    fade = min(int(0.005 * SR), EVENT_SAMPLES // 4)
    if fade:
        ramp = np.linspace(0.0, 1.0, fade, endpoint=False, dtype=np.float32)
        event[:fade] *= ramp
        event[-fade:] *= ramp[::-1]
    x = np.zeros(CLIP_SAMPLES, dtype=np.float32)
    x[EVENT_START:EVENT_START + EVENT_SAMPLES] = event
    return x


def measure_case(inp: np.ndarray, out: np.ndarray, target: np.ndarray, late: np.ndarray,
                 dry_control: bool = False) -> dict[str, float]:
    event = slice(EVENT_START, EVENT_START + EVENT_SAMPLES)
    event_in, event_out, event_target = energy(inp[event]), energy(out[event]), energy(target[event])
    result = {
        "event_level_change_db": db_ratio(event_out, event_in),
        "event_target_error_db": db_ratio(event_out, event_target),
        "peak_dbfs": float(20 * np.log10(max(float(np.max(np.abs(out))), 1e-8))),
        "clipped_samples": int(np.count_nonzero(np.abs(out) >= 0.999)),
    }
    # Locate a weak speech frame using the fixed clean target, never the estimate.
    frame, hop = 320, 160
    tgt = target[event]
    frames = np.lib.stride_tricks.sliding_window_view(tgt, frame)[::hop]
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=-1) + 1e-20)
    active = np.flatnonzero(rms > max(float(rms.max()) * .03, 1e-5))
    if active.size:
        weak = active[np.argmin(rms[active])]
        sl = slice(EVENT_START + int(weak * hop), EVENT_START + int(weak * hop) + frame)
        result["weak_level_change_db"] = db_ratio(energy(out[sl]), energy(target[sl]))
        first = int(active[0])
        sl = slice(EVENT_START + first * hop, EVENT_START + first * hop + min(frame, 640))
        result["onset_level_change_db"] = db_ratio(energy(out[sl]), energy(target[sl]))
    else:
        result["weak_level_change_db"] = float("nan")
        result["onset_level_change_db"] = float("nan")
    if not dry_control:
        tail_metrics = {}
        for name, start_ms, end_ms in (("tail_150_300", 150, 300), ("tail_300_600", 300, 600)):
            a = EVENT_START + EVENT_SAMPLES + int(start_ms * SR / 1000)
            b = min(CLIP_SAMPLES, EVENT_START + EVENT_SAMPLES + int(end_ms * SR / 1000))
            tail_in, tail_out = energy(inp[a:b]), energy(out[a:b])
            target_late = energy(late[a:b])
            tail_metrics[name + "_input_dbfs"] = db_ratio(tail_in, 1.0)
            tail_metrics[name + "_output_dbfs"] = db_ratio(tail_out, 1.0)
            tail_metrics[name + "_target_late_dryref_db"] = db_ratio(target_late, event_target)
            # Don't grade a filter for relative changes to numerical/noise-floor
            # tails where the known late stem is already effectively absent.
            if tail_metrics[name + "_target_late_dryref_db"] > -55.0:
                tail_metrics[name + "_dryref_reduction_db"] = db_ratio(tail_in, event_target) - db_ratio(tail_out, event_target)
            else:
                tail_metrics[name + "_dryref_reduction_db"] = float("nan")
        result.update(tail_metrics)
        # Compare model residual in the post-event window against known generated late stem.
        tail_sl = slice(EVENT_START + EVENT_SAMPLES, CLIP_SAMPLES)
        result["late_residual_correlation"] = float(np.corrcoef(out[tail_sl], late[tail_sl])[0, 1]) if energy(out[tail_sl]) and energy(late[tail_sl]) else 0.0
    else:
        post = out[EVENT_START + EVENT_SAMPLES:]
        result["dry_post_event_rms_dbfs"] = float(20 * np.log10(np.sqrt(energy(post)) + 1e-10))
    return result


def _infer(model, x: np.ndarray, device: torch.device) -> tuple[np.ndarray, float]:
    tensor = torch.from_numpy(x).unsqueeze(0).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        y = model(tensor)[0].float().cpu().numpy()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return y, time.perf_counter() - start


def evaluate(checkpoint: Path, data_root: Path, output_dir: Path, device_arg: str = "auto",
             speaker_limit: int = 30, utterances_per_speaker: int = 2,
             rir_count: int = 3, speaker_offset: int = 0,
             rir_seed_start: int = 90_000, split: str = "test-clean") -> dict:
    device = torch.device("cuda" if device_arg == "auto" and torch.cuda.is_available()
                          else "cpu" if device_arg == "auto" else device_arg)
    test_rows = read_librispeech(data_root, split)
    train_speakers = {speaker for _, speaker in read_librispeech(data_root, "train-clean-100")}
    dev_speakers = {speaker for _, speaker in read_librispeech(data_root, "dev-clean")}
    all_test_speakers = sorted({speaker for _, speaker in test_rows})
    test_speakers = all_test_speakers[speaker_offset:speaker_offset + speaker_limit]
    if not test_speakers:
        raise ValueError("speaker offset/limit selected no test speakers")
    overlap = (train_speakers | dev_speakers) & set(test_speakers)
    if overlap:
        raise RuntimeError(f"speaker leakage into final test: {sorted(overlap)}")
    grouped: dict[str, list[Path]] = {s: [] for s in test_speakers}
    for path, speaker in test_rows:
        if speaker in grouped and len(grouped[speaker]) < utterances_per_speaker:
            grouped[speaker].append(path)

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = CompactDereverb16k(base_channels=int(state["base_channels"])).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    timings: list[float] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for speaker_index, speaker in enumerate(test_speakers):
        for utterance_index, path in enumerate(grouped[speaker]):
            speech = _read_crop(path, np.random.default_rng(81_000 + speaker_index * 31 + utterance_index))
            clean = embed_event(speech)
            dry_out, elapsed = _infer(model, clean, device)
            timings.append(elapsed)
            dry_metrics = measure_case(clean, dry_out, clean, np.zeros_like(clean), dry_control=True)
            rows.append({"case_type": "dry", "speaker": speaker, "utterance": path.name,
                         "rir_seed": "", **dry_metrics})
            for rir_index in range(rir_count):
                seed = rir_seed_start + speaker_index * 1_000 + utterance_index * 100 + rir_index
                if seed < 90_000:
                    raise RuntimeError("final test RIR seed must be >=90000")
                pair = make_validation_pair(clean, {"seed": seed,
                    "t60_s": (.35, .65, .95)[rir_index % 3],
                    "direct_to_reverb_db": (12.0, 6.0, 0.0)[rir_index % 3]})
                enhanced, elapsed = _infer(model, pair["reverberant"], device)
                timings.append(elapsed)
                metrics = measure_case(pair["reverberant"], enhanced, pair["clean"], pair["late"])
                rows.append({"case_type": "reverb", "speaker": speaker, "utterance": path.name,
                             "rir_seed": seed, "t60_s": (.35, .65, .95)[rir_index % 3],
                             "drr_db": (12.0, 6.0, 0.0)[rir_index % 3], **metrics})

    csv_path = output_dir / "cases.csv"
    keys = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader(); writer.writerows(rows)
    rev = [row for row in rows if row["case_type"] == "reverb"]
    dry = [row for row in rows if row["case_type"] == "dry"]
    metrics = ("tail_150_300_dryref_reduction_db", "tail_300_600_dryref_reduction_db",
               "event_level_change_db", "event_target_error_db", "weak_level_change_db",
               "onset_level_change_db")
    summary = {name: percentile_summary([float(r[name]) for r in rev if name in r and np.isfinite(float(r[name]))]) for name in metrics}
    summary.update({"dry_event_level_change_db": percentile_summary([r["event_level_change_db"] for r in dry]),
        "dry_post_event_rms_dbfs": percentile_summary([r["dry_post_event_rms_dbfs"] for r in dry]),
        "output_peak_dbfs": percentile_summary([float(r["peak_dbfs"]) for r in rows]),
        "total_clipped_samples": int(sum(int(r["clipped_samples"]) for r in rows)),
        "speaker_count": len(test_speakers), "case_count": len(rev), "dry_case_count": len(dry),
        "test_split": split, "test_speakers": test_speakers, "train_dev_overlap": sorted(overlap),
        "scored_tail_cases": {name: int(sum(np.isfinite(float(r.get(name, "nan"))) for r in rev)) for name in metrics[:2]},
        "checkpoint": str(checkpoint), "device": str(device), "mean_forward_s": float(np.mean(timings)),
        "median_forward_s": float(np.median(timings)), "clip_s": CLIP_SAMPLES / SR,
        "rtf_median": float(np.median(timings) / (CLIP_SAMPLES / SR)),
        "peak_allocated_vram_mib": (torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None),
        "peak_reserved_vram_mib": (torch.cuda.max_memory_reserved(device) / 1024**2 if device.type == "cuda" else None),
        "speaker_offset": speaker_offset, "rir_seed_start": rir_seed_start,
        "protocol": "40 ms RMS frame metrics; no event gain matching; 0.5 s embedded held-out speech event; fixed speaker/RIR seeds"})
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path(".tools/compact-dereverb/evaluation"))
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--speaker-limit", type=int, default=30)
    p.add_argument("--utterances-per-speaker", type=int, default=2)
    p.add_argument("--rir-count", type=int, default=3)
    p.add_argument("--speaker-offset", type=int, default=0)
    p.add_argument("--rir-seed-start", type=int, default=90_000)
    p.add_argument("--split", choices=("test-clean", "test-other"), default="test-clean")
    args = p.parse_args()
    print(json.dumps(evaluate(args.checkpoint, args.data_root, args.output_dir,
        args.device, args.speaker_limit, args.utterances_per_speaker, args.rir_count,
        args.speaker_offset, args.rir_seed_start, args.split), indent=2))


if __name__ == "__main__":
    main()
