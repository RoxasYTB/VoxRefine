#!/usr/bin/env python3
"""Build the frozen G-early-v2 TRAIN set from its new speaker roster."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts/benchmarks/universal_enhancer"))

from data import make_procedural_rir  # noqa: E402
from model_g_early_v2 import CompactAttenuationOnlyDereverbG2  # noqa: E402
from pilot_g2_feasibility import active_dry_levels, hash_seed  # noqa: E402
from prepare_cap60_conditioned_pairs import (  # noqa: E402
    CROP_SAMPLES, DEEP_FILTER, LIBRISPEECH, SR, make_fan_components, sha256,
)
from prepare_f2_tailbank import cap60_cached, crop, tail_input_levels  # noqa: E402
from prepare_g_early import hybrid_mask, save_audio  # noqa: E402
from screen_measured_rirs import make_inserted_pause_pair  # noqa: E402
from train_measured_mix import measured_pair  # noqa: E402

EXPERIMENT = ROOT / ".tools/compact-dereverb/g-early-v2-2026-10-10"
ROSTER = EXPERIMENT / "source-roster.json"
DEFAULT_OUT = EXPERIMENT / "data"
TAIL_LIMIT = 32
TAIL_T60 = (1.65, 1.85)
TAIL_DRR = (-14.5, -12.5)
WINDOWS = ("150_300", "300_600")


def atomic_json(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def load_source_pairs(roster: dict) -> dict[str, tuple[Path, Path]]:
    by_id = {str(item["speaker_id"]): item for item in roster["sources"]
             if item["split"] == "train"}
    result = {}
    for row in roster["train_rows"]:
        speaker = str(row["speaker"])
        source = by_id[speaker]["utterances"][:2]
        paths = tuple(Path(item["path"]) for item in source)
        if len(paths) != 2 or any(not p.is_file() for p in paths):
            raise FileNotFoundError(f"frozen source pair missing for speaker {speaker}")
        if [sha256(p) for p in paths] != [item["sha256"] for item in source]:
            raise RuntimeError("frozen source file hash changed")
        result[speaker] = paths
    return result


def prepare(args) -> dict:
    if not args.deep_filter.is_file():
        raise FileNotFoundError(args.deep_filter)
    if not ROSTER.is_file():
        raise FileNotFoundError("freeze the v2 source roster before preparation")
    roster = json.loads(ROSTER.read_text())
    roster_sha = sha256(ROSTER)
    model_sha = sha256(HERE / "model_g_early_v2.py")
    audit_sha = sha256(HERE / "audit_g_early_v2_mechanics.py")
    preparer_sha = sha256(Path(__file__))
    integrity_path = EXPERIMENT / "source-roster-integrity.json"
    if not integrity_path.is_file():
        raise FileNotFoundError("run the independent source-roster audit before Cap60 inference")
    integrity = json.loads(integrity_path.read_text())
    if (integrity.get("roster_sha256") != roster_sha or
            integrity.get("split_speaker_disjoint") is not True or
            integrity.get("prior_speakers_disjoint") is not True or
            integrity.get("all_train_source_hashes_pass") is not True or
            integrity.get("holdout_opened") is not False or
            integrity.get("test_wav_accessed") is not False):
        raise RuntimeError("independent v2 source-roster audit did not pass")
    if (roster.get("name") != "G-early-v2-real-2026-10-10" or
            roster.get("frozen_before_cap60_inference") is not True or
            roster.get("test_wav_accessed") is not False or
            roster.get("holdout_opened") is not False or
            len(roster.get("train_rows", [])) != 128):
        raise RuntimeError("v2 roster is not a valid pre-inference frozen design")
    expected_receipt_hashes = {
        "f3c275f7c7a5e43252356fd65d10d9b3ad5f31a7fff01faa2a789e9bd9017435",
        "16b84fca29db5fae9a32c84f0bf61e077b5bf42c5d8d3fd2e973c9bc2c91a617",
    }
    if (len(args.audit_receipt) != 2 or
            {sha256(path) for path in args.audit_receipt} != expected_receipt_hashes):
        raise RuntimeError("both exact deterministic synthetic audit receipts are required")
    for receipt_path in args.audit_receipt:
        audit = json.loads(receipt_path.read_text())
        audit_sources = audit.get("source_sha256", {})
        if (audit.get("overall") != "PASS" or
                audit_sources.get("model") != model_sha or
                audit_sources.get("audit") != audit_sha):
            raise RuntimeError("synthetic preflight does not pass against current v2 sources")

    out = args.output_dir
    if out.exists() and not args.resume:
        raise FileExistsError(f"refusing existing output {out}; use --resume for partial data")
    if (out / "pairs.jsonl").exists():
        raise FileExistsError("completed v2 training set is immutable")
    arrays, cache, work = out / "arrays", out / "cap60-cache", out / "cap60-work"
    for path in (arrays, cache, work):
        path.mkdir(parents=True, exist_ok=True)
    source_pairs = load_source_pairs(roster)
    rows = roster["train_rows"]
    progress_path = out / "progress.json"
    progress = (json.loads(progress_path.read_text()) if args.resume and progress_path.is_file()
                else {"name": "G-early-v2-progress", "rows": [], "resolved_tail_slots": [],
                      "source_roster_sha256": roster_sha,
                      "deep_filter_sha256": sha256(args.deep_filter),
                      "model_source_sha256": model_sha,
                      "audit_source_sha256": audit_sha,
                      "preparer_source_sha256": preparer_sha,
                      "test_wav_accessed": False})
    frozen = {"source_roster_sha256": roster_sha,
        "deep_filter_sha256": sha256(args.deep_filter), "model_source_sha256": model_sha,
        "audit_source_sha256": audit_sha, "preparer_source_sha256": preparer_sha}
    if any(progress.get(key) != value for key, value in frozen.items()) or progress.get("test_wav_accessed") is not False:
        raise RuntimeError("partial v2 preparation differs from frozen sources or preflight")
    output_rows = progress["rows"]
    if [int(row["index"]) for row in output_rows] != list(range(len(output_rows))):
        raise RuntimeError("partial TRAIN cache is not a contiguous prefix")
    if len(output_rows) > 128:
        raise RuntimeError("partial TRAIN cache has too many rows")
    cache_index: dict[str, tuple[Path, Path]] = {}
    tail_stats = progress.get("resolved_tail_slots", [])
    if len(tail_stats) != sum(row["kind"] == "rir_only" for row in output_rows):
        raise RuntimeError("partial tail audit does not match completed rows")
    start_time = time.perf_counter()

    for row in rows[len(output_rows):]:
        index, kind = int(row["index"]), str(row["kind"])
        speaker = str(row["speaker"])
        joined, pause_meta = make_inserted_pause_pair(*source_pairs[speaker])
        joined = np.asarray(joined, dtype=np.float32)
        joined *= .80 / max(float(np.max(np.abs(joined))), 1e-8)
        pause_start = int(pause_meta["clip_relative_start_sample"])
        crop_start = pause_start - int(.75 * SR)
        pause_crop = pause_start - crop_start
        clean_crop = crop(joined, crop_start)
        selected = None
        history = []
        noise_branch = None

        if kind == "identity":
            target, target_meta = cap60_cached(args.deep_filter, joined, work,
                f"g2-{index:03d}-clean", cache, cache_index)
            wet = joined
            clean_branch = joined
            cap_input, input_meta = target, target_meta
            recipe = {"kind": "identity"}
            eligible = False
        elif kind == "rir_only":
            for candidate in range(TAIL_LIMIT):
                seed = hash_seed(f"G-early-v2-train|{speaker}|{index}|{candidate}")
                rng = np.random.default_rng(seed)
                t60 = float(rng.uniform(*TAIL_T60))
                drr = float(rng.uniform(*TAIL_DRR))
                rir, _early, _late = make_procedural_rir(SR, seed, t60, drr,
                    max_t60_s=2.0, min_direct_to_reverb_db=-20.0)
                pair = measured_pair(joined, rir)
                cap_input_try, input_meta_try = cap60_cached(args.deep_filter,
                    pair["reverberant"], work, f"g2-{index:03d}-wet-{candidate:02d}",
                    cache, cache_index)
                target_try, target_meta_try = cap60_cached(args.deep_filter,
                    pair["clean"], work, f"g2-{index:03d}-dry-{candidate:02d}",
                    cache, cache_index)
                xq = crop(cap_input_try, crop_start)
                cq = crop(target_try, crop_start)
                measured = tail_input_levels(cq, xq, pause_crop,
                                             activity_clean_crop=clean_crop)
                valid = bool(measured.get("reference_valid"))
                windows = {}
                if valid:
                    dry = active_dry_levels(cq, pause_crop,
                                            float(measured["reference_energy"]))
                    for window in WINDOWS:
                        lx = float(measured["window_db"][window])
                        ldry = float(dry[window])
                        threshold = max(-60.0, ldry + 6.0)
                        windows[window] = {"lx_db": lx, "ldry_db": ldry,
                            "threshold_db": threshold, "passes": lx > threshold}
                else:
                    windows = {w: {"lx_db": None, "ldry_db": None,
                        "threshold_db": None, "passes": False} for w in WINDOWS}
                item = {"candidate_index": candidate, "seed": seed, "t60_s": t60,
                    "direct_to_reverb_db": drr, "wet_cap60_output_sha256": hashlib.sha256(
                        cap_input_try.tobytes()).hexdigest(),
                    "dry_cap60_output_sha256": hashlib.sha256(target_try.tobytes()).hexdigest(),
                    "reference_valid": valid, "windows": windows,
                    "tail_eligible": bool(valid and windows["150_300"]["passes"])}
                history.append(item)
                if candidate == 0:
                    base_control = (pair, cap_input_try, input_meta_try,
                                    target_try, target_meta_try, item)
                if item["tail_eligible"]:
                    selected = (pair, cap_input_try, input_meta_try,
                                target_try, target_meta_try, item)
                    break
            if selected is None:
                if len(history) != TAIL_LIMIT:
                    raise RuntimeError("tail slot unresolved before all candidates")
                selected = base_control
                eligible = False
                classification = "BASE_ONLY_CONTROL"
            else:
                eligible = True
                classification = "TAIL_ELIGIBLE"
            pair, cap_input, input_meta, target, target_meta, chosen = selected
            wet = pair["reverberant"]
            clean_branch = pair["clean"]
            recipe = {**chosen, "tail_slot": len(tail_stats),
                "classification": classification, "input_only_selection": True,
                "tail_loss_enabled": eligible, "candidates_tested": len(history),
                "candidate_history": history}
            tail_stats.append({"index": index, "eligible": eligible,
                "candidate_index": chosen["candidate_index"],
                "classification": classification})
        else:
            seed = int(row["rir_recipe"]["seed"])
            rir, _early, _late = make_procedural_rir(SR, seed,
                float(row["rir_recipe"]["t60_s"]),
                float(row["rir_recipe"]["direct_to_reverb_db"]))
            pair = measured_pair(joined, rir)
            clean_branch = pair["clean"]
            wet, noise_branch = make_fan_components(pair["clean"], pair["reverberant"],
                                                     int(row["noise_seed"]),
                                                     20 if kind == "fan20" else 10)
            cap_input, input_meta = cap60_cached(args.deep_filter, wet, work,
                f"g2-{index:03d}-wet", cache, cache_index)
            target, target_meta = cap60_cached(args.deep_filter, pair["clean"], work,
                f"g2-{index:03d}-dry", cache, cache_index)
            recipe = {"kind": "procedural", **row["rir_recipe"],
                "noise_seed": row["noise_seed"], "nominal_snr_db":
                20 if kind == "fan20" else 10}
            eligible = False

        gain = float(row["post_cap_gain"])
        xq = (crop(cap_input, crop_start) * gain).astype(np.float32)
        cq = (crop(clean_branch, crop_start) * gain).astype(np.float32)
        aq = (crop(target, crop_start) * gain).astype(np.float32)
        mask, mask_meta = hybrid_mask(clean_crop)
        hybrid = (mask * cq + (1.0 - mask) * aq).astype(np.float32)
        stem = f"{index:03d}"
        files = {}
        for name, audio in (("x", xq), ("c", cq), ("a", aq),
                            ("hybrid", hybrid), ("mask", mask), ("tail_ref", aq)):
            rel = f"arrays/{name}-{stem}.npy"
            files[name] = rel
            save_audio(arrays / f"{name}-{stem}.npy", audio)
        if noise_branch is not None:
            noise_crop = (crop(noise_branch, crop_start) * gain).astype(np.float32)
            files["noise_only"] = f"arrays/noise-{stem}.npy"
            save_audio(arrays / f"noise-{stem}.npy", noise_crop)
        out_row = {"index": index, "kind": kind, "speaker": speaker,
            "source_pair": [p.name for p in source_pairs[speaker]],
            "source_sha256": [sha256(p) for p in source_pairs[speaker]],
            "crop_start_sample": crop_start, "pause_start_in_crop": pause_crop,
            "post_cap_gain": gain, "shared_branch_scale": 1.0,
            "rir_kind": row["rir_kind"], "rir_recipe": recipe, **files,
            "x_sha256": sha256(arrays / f"x-{stem}.npy"),
            "clean_target_sha256": sha256(arrays / f"c-{stem}.npy"),
            "cap_target_sha256": sha256(arrays / f"a-{stem}.npy"),
            "hybrid_target_sha256": sha256(arrays / f"hybrid-{stem}.npy"),
            "tail_ref_sha256": sha256(arrays / f"tail_ref-{stem}.npy"),
            "mask_sha256": sha256(arrays / f"mask-{stem}.npy"),
            "noise_only_sha256": (sha256(arrays / f"noise-{stem}.npy")
                                  if noise_branch is not None else None),
            "mask_definition": mask_meta, "cap60_input": input_meta,
            "cap60_clean_reference": target_meta,
            "input_only_tail_eligible": bool(kind == "rir_only" and eligible)}
        output_rows.append(out_row)
        progress.update({"rows": output_rows, "resolved_tail_slots": tail_stats,
                         "test_wav_accessed": False})
        atomic_json(progress_path, progress)
        print(json.dumps({"row": index, "kind": kind,
            "resolved_tail_slots": len(tail_stats),
            "eligible_tail_slots": sum(x["eligible"] for x in tail_stats),
            "elapsed_s": time.perf_counter() - start_time}), flush=True)

    counts = {kind: sum(row["kind"] == kind for row in output_rows)
              for kind in ("identity", "rir_only", "fan20", "fan10")}
    tail_rows = [row for row in output_rows if row["kind"] == "rir_only"]
    eligible_rows = [row for row in tail_rows if row["input_only_tail_eligible"]]
    if len(output_rows) != 128 or counts != {"identity": 32, "rir_only": 48,
                                             "fan20": 24, "fan10": 24}:
        raise RuntimeError(f"unexpected frozen G2 training schedule: {counts}")
    pairs_path = out / "pairs.jsonl"
    pairs_path.write_text("".join(json.dumps(row, allow_nan=False) + "\n"
                                   for row in output_rows))
    coverage = {"pair_manifest_sha256": sha256(pairs_path),
        "resolved_tail_slots": len(tail_rows), "required_tail_slots": 48,
        "eligible_tail_slots": len(eligible_rows), "minimum_eligible_tail_slots": 36,
        "executable": len(eligible_rows) >= 36, "test_wav_accessed": False}
    atomic_json(out / "coverage-audit.json", coverage)
    manifest = {"name": "G-early-v2-shared-training-data",
        "pair_manifest_sha256": coverage["pair_manifest_sha256"],
        **frozen, "source_roster_integrity_sha256": sha256(integrity_path),
        "sample_rate": SR, "crop_samples": CROP_SAMPLES,
        "row_count": len(output_rows), "kind_counts": counts,
        "tailbank": {"slots": len(tail_rows), "eligible_slots": len(eligible_rows),
            "base_only_control_slots": len(tail_rows) - len(eligible_rows),
            "required_slots": 48, "minimum_eligible_slots": 36,
            "candidate_limit": TAIL_LIMIT, "namespace": "G-early-v2-train|speaker|row|candidate",
            "rt60_s_uniform": list(TAIL_T60), "drr_db_uniform": list(TAIL_DRR),
            "selection": "first exact Cap60 candidate j=0..31 with Lx(W1)>max(-60dB,Ldry(W1)+6dB); otherwise j=0 BASE_ONLY_CONTROL",
            "input_only": True},
        "training_started": False, "model_outputs_accessed": False,
        "test_wav_accessed": False}
    atomic_json(out / "training-data-manifest.json", manifest)
    progress_path.unlink(missing_ok=True)
    return {"rows": len(output_rows), "kind_counts": counts,
        "eligible_tail_slots": len(eligible_rows), "executable": coverage["executable"],
        "pair_manifest_sha256": coverage["pair_manifest_sha256"],
        "holdout_opened": False, "test_wav_accessed": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deep-filter", type=Path, default=DEEP_FILTER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--audit-receipt", type=Path, nargs=2, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args), indent=2))


if __name__ == "__main__":
    main()
