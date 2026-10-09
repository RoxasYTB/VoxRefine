#!/usr/bin/env python3
"""Inventory and integrity-check locally stored Adobe v2 comparison assets.

This script reads audio headers/data and existing reports only. It never runs
enhancement inference, changes audio, or treats a spectral metric as quality.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "results/adobe-v2-pair-audit-2026-10-09"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect(path: Path, group: str, role: str) -> dict:
    info = sf.info(path)
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    finite = bool(np.isfinite(audio).all())
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    return {
        "group": group,
        "role": role,
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "sample_rate_hz": int(rate),
        "channels": int(audio.shape[1]),
        "frames": int(audio.shape[0]),
        "duration_s": float(audio.shape[0] / rate),
        "subtype": info.subtype,
        "format": info.format,
        "finite_samples": finite,
        "sample_peak_dbfs": float(20 * np.log10(max(peak, 1e-12))),
        "readable": True,
    }


def collect() -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    group_rows: list[dict] = []
    groups: list[tuple[str, dict[str, Path]]] = []

    for event in ("christiane-event-1", "christiane-event-2", "naf-event-2"):
        for condition in ("dry-control", "rir-only"):
            base = ROOT / "results/noise-rir-truth-01" / f"adobe-pair-{event}-{condition}"
            if not base.exists():
                continue
            pair = ROOT / "results/noise-rir-truth-01/rir-tail-truth-02"
            pair_name = f"{event}-{condition}"
            groups.append((f"{event}/{condition}", {
                "source_input": pair / "inputs" / f"{pair_name}.wav",
                "dry_speech_reference": pair / "inputs" / f"{event}-dry-control.wav",
                "deepfilternet_18db": pair / "renders" / f"{pair_name}-deepfilternet-18db.wav",
                "resemble_denoiser": pair / "renders" / f"{pair_name}-resemble-denoiser.wav",
                "resemble_nfe32": pair / "renders" / f"{pair_name}-resemble-nfe32.wav",
                "resemble_nfe64_c": pair / "renders" / f"{pair_name}-resemble-nfe64-c.wav",
                "adobe_v2": base / "adobe-v2-output.wav",
            }))

    fortune_in = ROOT / "corpus/samples/librivox-fortune"
    fortune_adobe = ROOT / "results/librivox-fortune/adobe-v2"
    fortune_dfn = ROOT / "results/librivox-fortune/deepfilter"
    for number, slug in (("01", "pauses-nombreuses"),
                         ("02", "parole-continue"),
                         ("03", "variations-intensite")):
        groups.append((f"librivox-fortune/{number}-{slug}", {
            "source": fortune_in / f"{number}-{slug}.wav",
            "adobe_v2": fortune_adobe / f"{number}-{slug}-adobe-v2.wav",
            "deepfilternet": fortune_dfn / f"{number}-{slug}-deepfilter.wav",
        }))

    christiane = ROOT / "results/resemble-enhance-01/adobe-christiane-04"
    if christiane.exists():
        groups.append(("christiane/long-form-adobe-pair", {
            "source": christiane / "listening/B-source.wav",
            "adobe_v2": christiane / "listening/A-adobe_v2.wav",
            "nfe16": christiane / "listening/C-nfe16.wav",
            "nfe32": christiane / "listening/D-nfe32.wav",
            "nfe64": christiane / "listening/E-nfe64.wav",
            "nfe64_c": christiane / "listening/F-nfe64_c.wav",
        }))

    controlled = ROOT / "results/resemble-enhance-01/controlled-challenger-03/audio"
    if controlled.exists():
        groups.append(("controlled-noise/snr10-adobe-pair", {
            "source_dry": controlled / "noise-snr10-adobe-dry.wav",
            "noisy_input": controlled / "noise-snr10-adobe-input.wav",
            "adobe_v2": controlled / "adobe-v2-reference.wav",
            "nfe16": controlled / "noise-snr10-adobe-resemble-nfe16.wav",
            "nfe16_c": controlled / "noise-snr10-adobe-resemble-nfe16-C.wav",
        }))

    clear_root = ROOT / "corpus/samples/clear-noisy-mix-01"
    clear_outputs = ROOT / "results/clear-noisy-mix-01/audio"
    clear_adobe = OUTPUT / "exports"
    for sample_id in ("emy_mixed_ambience18", "remi_crowd18", "stephanie_fan18"):
        noise_name = {"emy_mixed_ambience18": "fan+crowd-stem.wav",
                      "remi_crowd18": "crowd-stem.wav",
                      "stephanie_fan18": "fan-stem.wav"}[sample_id]
        groups.append((f"clear-noisy-18db/{sample_id}", {
            "noisy_input": clear_root / "noisy" / f"{sample_id}-noisy.wav",
            "clean_speech": clear_root / "clean" / f"{sample_id}-clean-reference.wav",
            "noise_stem": clear_root / "noise" / f"{sample_id}-{noise_name}",
            "nfe64_c": clear_outputs / f"{sample_id}-resemble-nfe64-C.wav",
            "adobe_v2": clear_adobe / f"{sample_id}-adobe-v2.wav",
        }))

    for group, assets in groups:
        present = []
        for role, path in assets.items():
            if path.is_file():
                try:
                    row = inspect(path, group, role)
                except Exception as exc:  # Keep the failed asset visible in the audit.
                    row = {"group": group, "role": role,
                           "path": str(path.relative_to(ROOT)), "readable": False,
                           "error": f"{type(exc).__name__}: {exc}"}
                rows.append(row)
                if row.get("readable"):
                    present.append(row)
            else:
                rows.append({"group": group, "role": role,
                             "path": str(path.relative_to(ROOT)), "readable": False,
                             "error": "missing"})
        readable = [r for r in present if r.get("readable")]
        durations = [r["duration_s"] for r in readable]
        rates = sorted({r["sample_rate_hz"] for r in readable})
        group_row = {
            "group": group,
            "expected_assets": len(assets),
            "present_readable_assets": len(readable),
            "missing_or_unreadable_assets": len(assets) - len(readable),
            "sample_rates_hz": rates,
            "duration_min_s": min(durations) if durations else None,
            "duration_max_s": max(durations) if durations else None,
            "duration_spread_ms": (max(durations) - min(durations)) * 1000 if durations else None,
        }
        group_rows.append(group_row)

    return rows, group_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows, group_rows = collect()
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (args.out / "assets.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "purpose": "Inventory and file-integrity check only; no inference or acoustic quality ranking.",
        "asset_count": len(rows),
        "readable_asset_count": sum(bool(row.get("readable")) for row in rows),
        "group_count": len(group_rows),
        "groups": group_rows,
        "assets_csv": str((args.out / "assets.csv").relative_to(ROOT)),
        "limitations": [
            "Presence of an Adobe render does not prove its UI settings unless a paired record documents them.",
            "Duration and sample rate do not prove sample-level alignment; use event-specific analysis reports.",
            "Adobe output is a style reference, not clean ground truth.",
            "No missing audio was downloaded or reconstructed by this audit.",
        ],
    }
    (args.out / "audit.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"groups={len(group_rows)} assets={len(rows)} readable={payload['readable_asset_count']}")
    for group in group_rows:
        print(f"{group['group']}: {group['present_readable_assets']}/{group['expected_assets']} readable, "
              f"duration spread={group['duration_spread_ms']:.1f} ms" if group['duration_spread_ms'] is not None
              else f"{group['group']}: {group['present_readable_assets']}/{group['expected_assets']} readable")


if __name__ == "__main__":
    main()
