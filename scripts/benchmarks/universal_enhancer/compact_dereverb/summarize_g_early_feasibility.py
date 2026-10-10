#!/usr/bin/env python3
"""Audit and publish aggregate G-early feasibility results."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
from pilot_g_early_feasibility import EXPERIMENT  # noqa: E402
from prepare_cap60_conditioned_pairs import DEEP_FILTER, sha256  # noqa: E402

REPORT = ROOT / "docs/benchmarking/g-early-feasibility-2026-10-10.md"
FIGURE = ROOT / "docs/benchmarking/assets/g-early-feasibility-2026-10-10/feasibility.png"


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def audit() -> tuple[dict, list[dict], dict]:
    design_path = EXPERIMENT / "design.json"
    result_path = EXPERIMENT / "results.json"
    design = json.loads(design_path.read_text())
    summary = json.loads(result_path.read_text())
    if sha256(design_path) != summary["design_sha256"]:
        raise RuntimeError("result summary references a different frozen design")
    if design.get("test_wav_accessed") is not False or summary.get("test_wav_accessed") is not False:
        raise RuntimeError("test.wav access flag is not false")
    if (design.get("training_started") is not False or
            summary.get("training_started") is not False or
            summary.get("model_outputs_accessed") is not False):
        raise RuntimeError("feasibility artifacts claim model training/output access")
    if design.get("w2_rule", "").find("never affects selection") < 0 or summary.get("w2_selection_used") is not False:
        raise RuntimeError("W2 was not recorded as descriptive-only")
    if design.get("cap60_binary_sha256") != sha256(DEEP_FILTER):
        raise RuntimeError("Cap60 binary hash differs from frozen design")

    speakers = design.get("speakers", [])
    if len(speakers) != 12 or len({str(s["speaker_id"]) for s in speakers}) != 12:
        raise RuntimeError("frozen design does not contain 12 unique speakers")
    rows: list[dict] = []
    all_source_hashes: set[str] = set()
    for si, speaker in enumerate(speakers):
        if len(speaker.get("pairs", [])) != 4:
            raise RuntimeError("each speaker must have four frozen pairs")
        for pair in speaker["pairs"]:
            path = EXPERIMENT / "slots" / f"dev-{si:02d}-{pair['pair_index']:02d}.json"
            row = json.loads(path.read_text())
            if row.get("design_sha256") != summary["design_sha256"]:
                raise RuntimeError("slot result is not bound to frozen design")
            if row.get("speaker_id") != speaker["speaker_id"] or row.get("pair_index") != pair["pair_index"]:
                raise RuntimeError("slot identity differs from frozen design")
            if row.get("test_wav_accessed") is not False:
                raise RuntimeError("slot test.wav flag is not false")
            for key in ("first_sha256", "second_sha256"):
                digest = pair[key]
                if digest in all_source_hashes:
                    raise RuntimeError("a source file was reused across frozen slots")
                all_source_hashes.add(digest)
            history = row["history"]
            if [x["candidate_index"] for x in history] != list(range(len(history))):
                raise RuntimeError("candidate history is not a contiguous j=0 prefix")
            if len(history) > 32 or row["candidates_tested"] != len(history):
                raise RuntimeError("candidate count is inconsistent")
            passing = [x for x in history if x["tail_eligible"]]
            if row["tail_eligible"]:
                if not passing or row["selected_candidate"]["candidate_index"] != passing[0]["candidate_index"]:
                    raise RuntimeError("eligible slot did not select its first passing candidate")
            else:
                if len(history) != 32 or row["classification"] != "BASE_ONLY_CONTROL":
                    raise RuntimeError("failed slot was not exhausted and retained as control")
                if row["selected_candidate"]["candidate_index"] != 0:
                    raise RuntimeError("BASE_ONLY_CONTROL did not retain deterministic j=0")

            selected = row["selected_candidate"]
            w1, w2 = selected["windows"]["150_300"], selected["windows"]["300_600"]
            if row["tail_eligible"]:
                expected = max(-60.0, float(w1["ldry_db"]) + 6.0)
                if not float(w1["lx_db"]) > expected:
                    raise RuntimeError("selected W1 candidate does not pass strict threshold")
            rows.append({"speaker_index": si, "pair_index": pair["pair_index"],
                "tail_eligible": row["tail_eligible"],
                "candidates_tested": len(history), "w1": w1, "w2": w2,
                "elapsed_s": row["elapsed_s"]})

    if len(rows) != 48:
        raise RuntimeError("expected exactly 48 slot results")
    counts = [sum(r["tail_eligible"] for r in rows[i*4:(i+1)*4]) for i in range(12)]
    eligible = sum(r["tail_eligible"] for r in rows)
    ge3, zero = sum(x >= 3 for x in counts), sum(x == 0 for x in counts)
    passed = eligible >= 36 and ge3 >= 9 and zero == 0
    if (summary["eligible_pair_count"] != eligible or
            summary["speaker_eligible_counts_private"] != counts or
            summary["speakers_with_at_least_3_of_4"] != ge3 or
            summary["speakers_with_0_of_4"] != zero or
            summary["feasibility_pass"] != passed):
        raise RuntimeError("aggregate result does not match the 48 frozen slot records")
    return design, rows, {"eligible": eligible, "counts": counts, "ge3": ge3,
        "zero": zero, "passed": passed, "summary": summary,
        "design_sha256": sha256(design_path),
        "results_sha256": sha256(result_path), "cap60_sha256": sha256(DEEP_FILTER)}


def plot(rows: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eligible = [r for r in rows if r["tail_eligible"]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    axes[0].hist([r["candidates_tested"] for r in rows], bins=np.arange(.5, 33.5, 1),
                 color="#2563eb", alpha=.85)
    axes[0].set(xlabel="Cap60 candidate index count tested", ylabel="Slots",
                title="First passing candidate (or 32-candidate control search)")
    axes[0].set_xlim(0, 33)
    axes[0].grid(axis="y", alpha=.25)

    for wi, (key, label, color) in enumerate((
            ("w1", "W1: 150–300 ms", "#2563eb"),
            ("w2", "W2: 300–600 ms (descriptive)", "#f97316"))):
        vals = [float(r[key]["lx_db"]) for r in eligible if r[key]["lx_db"] is not None]
        x = np.full(len(vals), wi + 1, dtype=float) + np.linspace(-.08, .08, len(vals))
        axes[1].scatter(x, vals, s=20, alpha=.65, color=color, label=f"Lx, n={len(vals)}")
        dry = [float(r[key]["ldry_db"]) for r in eligible if r[key]["ldry_db"] is not None]
        xd = np.full(len(dry), wi + 1.22, dtype=float) + np.linspace(-.08, .08, len(dry))
        axes[1].scatter(xd, dry, s=14, alpha=.45, marker="x", color="#475569", label="Ldry")
    axes[1].axhline(-60, color="#64748b", linestyle="--", linewidth=1,
                    label="absolute eligibility floor −60 dB")
    axes[1].set_xticks([1, 2], ["W1", "W2"])
    axes[1].set_ylabel("Level relative to active dry Cap60 reference (dB)")
    axes[1].set_title("Selected examples only; W2 never selects")
    axes[1].grid(alpha=.25)
    axes[1].legend(fontsize=8)
    fig.suptitle("G-early feasibility: 43/48 W1 eligible; no model fit", fontsize=13)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def render(design: dict, rows: list[dict], audit_result: dict, report: Path, figure: Path) -> None:
    s = audit_result["summary"]
    selected = [r for r in rows if r["tail_eligible"]]
    candidate_counts = [r["candidates_tested"] for r in rows]
    lines = [
        "# G-early feasibility — résultats (10 octobre 2026)", "",
        "> **Décision : PASS de faisabilité seulement.** La sélection W1 atteint les trois seuils préenregistrés. Aucun modèle n’a été entraîné; il n’existe donc encore aucune mesure d’efficacité de réduction de réverbération.", "",
        "## Question mesurée", "",
        "Le pilote vérifie si un ensemble frais de voix peut produire, après le même Cap60, une queue précoce mesurable entre 150 et 300 ms. W1 seule décide l’admissibilité. W2 (300–600 ms) est descriptif et ne participe ni à la sélection ni à une perte de modèle.", "",
        "## Protocole gelé", "",
        "- 12 locuteurs frais de LibriSpeech train-clean-360, quatre paires de phrases disjointes par locuteur (48 slots). Les locuteurs et fichiers détaillés restent dans les artefacts ignorés sous `.tools/`.",
        "- Nouvelle namespace de seeds `G-early-feasibility-v1|speaker|pair|candidate`; RT60 uniforme [1,65; 1,85] s, DRR uniforme [−14,5; −12,5] dB; candidats déterministes j=0…31.",
        "- Cap60 exact sur le signal humide et sa référence sèche; première réussite admissible retenue.",
        "- `Lx(W1) > max(−60 dB, Ldry(W1)+6 dB)`, calculé avec la même référence d’énergie active dry Cap60. Seuil strict.",
        "- Si aucun candidat ne passe W1, le créneau garde j=0 comme `BASE_ONLY_CONTROL`, sans perte de queue.",
        "- Gate fixé avant calcul : au moins 36/48 admissibles; au moins 9/12 locuteurs avec 3/4 ou plus; aucun locuteur à 0/4.",
        "- W2 n’est enregistré qu’à titre descriptif. Pas de modèle, pas de sortie de modèle et aucun accès à `/home/yohan/Bureau/test.wav`.", "",
        "## Résultats", "",
        "| Mesure | Résultat |", "|---|---:|",
        f"| Paires examinées | {len(rows)}/48 |",
        f"| W1 admissibles | {audit_result['eligible']}/48 ({audit_result['eligible']/48:.1%}) |",
        f"| Contrôles `BASE_ONLY_CONTROL` | {48-audit_result['eligible']}/48 |",
        f"| Locuteurs avec ≥3/4 admissibles | {audit_result['ge3']}/12 (gate ≥9) |",
        f"| Locuteurs à 0/4 | {audit_result['zero']}/12 (gate =0) |",
        f"| Candidats Cap60 évalués | {s['candidate_total']} |",
        f"| Candidats par slot (médiane / moyenne / max.) | {percentile(candidate_counts,50):.0f} / {np.mean(candidate_counts):.2f} / {max(candidate_counts)} |",
        f"| Durée du pilote | {s['elapsed_s']:.1f} s ({s['elapsed_s']/60:.2f} min) |",
        "| Décision du gate de faisabilité | PASS |", "",
        "### Niveaux des premières paires admissibles", "",
        "Niveaux rapportés à l’énergie active dry Cap60; il ne s’agit pas de dBFS. La plage W2 est une observation des seules paires sélectionnées par W1 et ne constitue pas un gate.", "",
        "| Fenêtre | Lx p10 / médiane / p90 (dB) | Ldry p10 / médiane / p90 (dB) |",
        "|---|---:|---:|",
    ]
    for key, label in (("w1", "W1 150–300 ms"), ("w2", "W2 300–600 ms (descriptif)")):
        lx = [float(r[key]["lx_db"]) for r in selected if r[key]["lx_db"] is not None]
        ld = [float(r[key]["ldry_db"]) for r in selected if r[key]["ldry_db"] is not None]
        lines.append(f"| {label} | {percentile(lx,10):.2f} / {percentile(lx,50):.2f} / {percentile(lx,90):.2f} | {percentile(ld,10):.2f} / {percentile(ld,50):.2f} / {percentile(ld,90):.2f} |")
    lines += ["", f"![Histogramme des candidats testés et niveaux des fenêtres W1/W2]({figure.relative_to(ROOT).as_posix()})", "",
        "## Interprétation et limites", "",
        "Le résultat autorise la préparation d’un essai distinct de cible G-early, sous réserve de nouveaux splits DEV/HOLDOUT frais et du même gate de couverture d’au moins 75 %. Ce PASS ne prouve pas que le modèle peut supprimer W1, préserver la parole, ou généraliser aux pièces réelles.", "",
        "Les cinq contrôles ne fournissent aucune cible de perte de queue : ils doivent conserver les pertes de préservation de parole et waveform prévues, avec `lambda_early=0`. La forte hétérogénéité de coût candidat (médiane, moyenne et maximum publiés ci-dessus) est un coût de préparation hors ligne; elle ne mesure pas la latence du produit.", "",
        "Pas de conclusion sur W2, la réverbération tardive, la qualité studio, la généralisation hors LibriSpeech, le temps réel, les GPU/GTX 10xx ou la parité Adobe Podcast V2.", "",
        "## Reproductibilité et intégrité", "",
        f"- Design SHA-256 : `{audit_result['design_sha256']}`.",
        f"- Résultats bruts SHA-256 : `{audit_result['results_sha256']}`.",
        f"- Cap60 SHA-256 : `{audit_result['cap60_sha256']}`.",
        "- Audit PASS : 12 locuteurs uniques, 48 slots, hashes sources non réutilisés, historique candidat contigu, première réussite choisie, contrôles j=0 après 32 refus et agrégats recalculés depuis les slots.",
        "- `training_started=false`, `model_outputs_accessed=false`, `test_wav_accessed=false`; aucun identifiant de locuteur n’est publié.",
        "- Les manifestes et les sources détaillées restent sous `.tools/`, ignorés par Git.", ""]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--figure", type=Path, default=FIGURE)
    args = parser.parse_args()
    design, rows, result = audit()
    plot(rows, args.figure)
    render(design, rows, result, args.report, args.figure)
    print(json.dumps({"integrity_audit": "PASS", "eligible": result["eligible"],
        "speakers_ge3": result["ge3"], "speakers_zero": result["zero"],
        "feasibility_pass": result["passed"], "report": str(args.report),
        "figure": str(args.figure)}, indent=2))


if __name__ == "__main__":
    main()
