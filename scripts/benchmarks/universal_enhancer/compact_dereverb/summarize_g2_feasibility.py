#!/usr/bin/env python3
"""Audit and publish aggregate-only results for the G2 input-only pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
EXPERIMENT = ROOT / ".tools/compact-dereverb/g2-feasibility-2026-10-10"
REPORT = ROOT / "docs/benchmarking/g2-feasibility-2026-10-10.md"
FIGURE = ROOT / "docs/benchmarking/assets/g2-feasibility-2026-10-10/g2-feasibility.png"
BANDS = ("150_300", "300_600")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_and_audit(experiment: Path) -> tuple[dict, dict, list[dict]]:
    design = json.loads((experiment / "design.json").read_text())
    summary = json.loads((experiment / "results.json").read_text())
    paths = sorted((experiment / "slots").glob("*.json"))
    rows = [json.loads(path.read_text()) for path in paths]
    if len(rows) != 64 or summary.get("pair_count") != 64:
        raise RuntimeError("G2 feasibility pilot must contain exactly 64 persisted pair slots")
    if (design.get("speaker_count") != 16 or design.get("pair_count") != 64 or
            design.get("seed_namespace") != "G2-feasibility-dev-v1|speaker|pair|candidate"):
        raise RuntimeError("G2 feasibility design manifest does not match the preregistered pilot")
    for row in rows:
        history = row["history"]
        if (row.get("test_wav_accessed") is not False or row.get("training_started") or
                row.get("model_outputs_accessed") or len(history) != row["candidates_tested"] or
                [item["candidate_index"] for item in history] != list(range(len(history)))):
            raise RuntimeError("pair slot has unsafe or non-contiguous candidate history")
        eligible = [item for item in history if item["eligible"]]
        if row["eligible"]:
            if len(eligible) != 1 or eligible[0] is not history[-1]:
                raise RuntimeError("eligible slot does not contain exactly the first passing candidate")
        elif len(history) != 32 or eligible:
            raise RuntimeError("unresolved slot did not exhaust the 32-candidate input-only search")
        for item in history:
            if (item["cap60_clean_source_sha256"] != item["clean_scaled_source_sha256"] or
                    item["cap60_wet_source_sha256"] != item["wet_scaled_source_sha256"]):
                raise RuntimeError("Cap60 source hash mismatch in candidate history")
            for band in BANDS:
                values = item["windows"][band]
                if values["lx_db"] is None:
                    if item["eligible"]:
                        raise RuntimeError("passing candidate has invalid reference measurement")
                    continue
                expected_threshold = max(-60.0, float(values["ldry_db"]) + 6.0)
                if abs(expected_threshold - float(values["strict_threshold_db"])) > 1e-7:
                    raise RuntimeError("candidate eligibility threshold differs from frozen rule")
                expected_eligible = float(values["lx_db"]) > expected_threshold
                if bool(values["eligible"]) != expected_eligible:
                    raise RuntimeError("candidate per-window eligibility flag is incorrect")
        if row.get("design_sha256") != sha256(experiment / "design.json"):
            raise RuntimeError("slot result does not match frozen design hash")
    if (summary.get("eligible_pair_count") != sum(row["eligible"] for row in rows) or
            summary.get("two_window_feasibility_pass") != all(row["eligible"] for row in rows) or
            summary.get("training_started") or summary.get("model_outputs_accessed") or
            summary.get("test_wav_accessed") is not False):
        raise RuntimeError("summary violates the persisted pair records or safety flags")
    return design, summary, rows


def effective_margin(candidate: dict, band: str) -> float:
    values = candidate["windows"][band]
    return float(values["lx_db"] - values["strict_threshold_db"])


def diagnostic_candidate(row: dict) -> dict:
    if row["selected_candidate"] is not None:
        return row["selected_candidate"]
    # For visualization only: choose the failed candidate with the best
    # worst-window margin. This never changes eligibility or pair selection.
    return max(row["history"], key=lambda item: min(effective_margin(item, band)
                                                    for band in BANDS))


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def make_figure(rows: list[dict], summary: dict, path: Path) -> None:
    candidates = [diagnostic_candidate(row) for row in rows]
    x = [effective_margin(candidate, BANDS[0]) for candidate in candidates]
    y = [effective_margin(candidate, BANDS[1]) for candidate in candidates]
    passed = np.asarray([row["eligible"] for row in rows], dtype=bool)
    candidate_indices = [row["selected_candidate"]["candidate_index"]
                         for row in rows if row["eligible"]]

    fig, (ax, hist) = plt.subplots(1, 2, figsize=(13, 5.4),
                                   gridspec_kw={"width_ratios": [1.5, 1]})
    ax.scatter(np.asarray(x)[passed], np.asarray(y)[passed], s=44, alpha=.8,
               color="#15803d", edgecolors="white", linewidths=.45,
               label=f"Admissibles ({int(passed.sum())})")
    ax.scatter(np.asarray(x)[~passed], np.asarray(y)[~passed], s=58, alpha=.9,
               color="#dc2626", marker="x", linewidths=1.7,
               label=f"Non admissibles ({int((~passed).sum())})")
    ax.axvline(0, color="#334155", linestyle="--", linewidth=1)
    ax.axhline(0, color="#334155", linestyle="--", linewidth=1)
    ax.fill_between([0, max(6, max(x) * 1.04)], 0, max(6, max(y) * 1.04),
                    color="#16a34a", alpha=.07, zorder=0)
    ax.set_xlabel("W1 150–300 ms : Lx − seuil d’admissibilité (dB)")
    ax.set_ylabel("W2 300–600 ms : Lx − seuil d’admissibilité (dB)")
    ax.set_title("Marge aux deux fenêtres")
    ax.grid(alpha=.18)
    ax.legend(frameon=False, loc="lower right")
    ax.text(.02, .98, "Croix rouges : meilleur candidat conjoint parmi 32, diagnostic seulement",
            transform=ax.transAxes, va="top", fontsize=8.5, color="#475569")

    bins = np.arange(-.5, 33.5, 1)
    hist.hist(candidate_indices, bins=bins, color="#2563eb", alpha=.86,
              edgecolor="white", linewidth=.45)
    hist.set_xlim(-.6, 31.6)
    hist.set_xlabel("Premier candidat valide j")
    hist.set_ylabel("Nombre de paires admissibles")
    hist.set_title("RIRs évaluées avant admissibilité")
    hist.set_xticks([0, 5, 10, 15, 20, 25, 30])
    hist.grid(axis="y", alpha=.18)
    fig.suptitle(
        f"G2 feasibility : {summary['eligible_pair_count']}/64 paires admissibles "
        f"({100*summary['eligible_pair_count']/64:.1f} %) — aucun entraînement",
        fontsize=13, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, .93))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_report(design: dict, summary: dict, rows: list[dict],
                 report_path: Path, figure_path: Path, experiment: Path) -> str:
    passed = [row for row in rows if row["eligible"]]
    failed = [row for row in rows if not row["eligible"]]
    selected = [row["selected_candidate"] for row in passed]
    candidate_indices = [int(item["candidate_index"]) for item in selected]
    accepted_first = sum(index == 0 for index in candidate_indices)
    all_candidates = sum(row["candidates_tested"] for row in rows)
    best_failures = [diagnostic_candidate(row) for row in failed]

    def vals(collection, band: str, key: str) -> list[float]:
        return [float(item["windows"][band][key]) for item in collection]

    lines = [
        "# G2 feasibility pilot — résultats (10 octobre 2026)",
        "",
        "> **Décision : NO-GO pour l’entraînement G2 à deux fenêtres.** "
        "Le gate préenregistré exige 64/64 paires admissibles; le pilote en trouve "
        f"{len(passed)}/64. Aucun modèle n’a été entraîné.",
        "",
        "## Question mesurée",
        "",
        "Vérifier si une nouvelle génération de RIR procédurales peut produire, après "
        "Cap60, une queue de réverbération mesurable à la fois en W1 (150–300 ms) et "
        "W2 (300–600 ms), au-dessus du résidu propre Cap60 de la même source. Ce pilote "
        "de faisabilité n’évalue pas la qualité d’un modèle G et ne compare pas une sortie "
        "à Adobe Podcast.",
        "",
        "## Protocole gelé",
        "",
        "- 16 speakers diagnostic déjà consommés par le DEV G-v1, 4 paires par speaker (64 slots); "
        "aucun speaker n’est réutilisé pour un éventuel train/dev/holdout G2.",
        "- Sources LibriSpeech `train-clean-360`; nouvelles RIR procédurales, RT60 uniforme "
        "[1,65; 1,85] s, DRR uniforme [−14,5; −12,5] dB.",
        "- Namespace nouveau `G2-feasibility-dev-v1|speaker|pair|candidate`; candidats "
        "j=0…31, Cap60 exact sur signal humide et référence sèche, première paire de fenêtres valide.",
        "- Pour chaque fenêtre W, niveaux d’énergie rapportés au même RMS actif du dry Cap60 :",
        "  - `Lx(W) = 10 log10((E_x(W)+ε)/(E_ref+ε))` ;",
        "  - `Ldry(W) = 10 log10((E_dry(W)+ε)/(E_ref+ε))` ;",
        "  - admission stricte si `Lx(W) > max(−60 dB, Ldry(W)+6 dB)` en W1 **et** W2.",
        "- Aucun output G consulté; seuils, fenêtres, sources et paramètres RIR non ajustés après mesure.",
        "",
        "## Résultats",
        "",
        "| Mesure | Résultat |",
        "|---|---:|",
        f"| Slots examinés | {len(rows)}/64 |",
        f"| Admissibles aux deux fenêtres | {len(passed)}/64 ({100*len(passed)/64:.2f} %) |",
        f"| Non admissibles après 32 candidats | {len(failed)}/64 |",
        f"| Passent dès j=0 | {accepted_first}/64 |",
        f"| Candidats par slot (médiane / moyenne / maximum) | {percentile([r['candidates_tested'] for r in rows],50):.0f} / {all_candidates/64:.2f} / 32 |",
        f"| Candidats Cap60 examinés au total | {all_candidates} |",
        f"| Temps total du pilote | {summary['elapsed_s']/60:.2f} min ({summary['elapsed_s']:.1f} s) |",
        "",
        "### Niveaux des 55 paires admissibles",
        "",
        "Tous les niveaux ci-dessous sont relatifs à l’énergie active dry Cap60, pas dBFS. "
        "Le plancher −120 dB provient de l’epsilon de mesure et décrit une référence sous le plancher numérique du calcul.",
        "",
        "| Fenêtre | Lx p10 / médiane / p90 (dB) | Ldry p10 / médiane / p90 (dB) | Marge minimale au seuil (dB) |",
        "|---|---:|---:|---:|",
    ]
    for band, label in (("150_300", "W1 150–300 ms"), ("300_600", "W2 300–600 ms")):
        lx = vals(selected, band, "lx_db")
        dry = vals(selected, band, "ldry_db")
        margin = [effective_margin(item, band) for item in selected]
        lines.append(
            f"| {label} | {percentile(lx,10):.2f} / {percentile(lx,50):.2f} / {percentile(lx,90):.2f} "
            f"| {percentile(dry,10):.2f} / {percentile(dry,50):.2f} / {percentile(dry,90):.2f} "
            f"| {min(margin):.2f} |")
    fail_w1 = sum(not c["windows"]["150_300"]["eligible"] for c in best_failures)
    fail_w2 = sum(not c["windows"]["300_600"]["eligible"] for c in best_failures)
    fail_both = sum(not c["windows"]["150_300"]["eligible"] and
                    not c["windows"]["300_600"]["eligible"] for c in best_failures)
    fail_w1_dry = vals(best_failures, "150_300", "ldry_db")
    fail_w1_wet = vals(best_failures, "150_300", "lx_db")
    fail_w1_margin = [effective_margin(item, "150_300") for item in best_failures]
    lines += [
        "",
        "### Diagnostic des neuf slots non admissibles",
        "",
        "Pour expliquer les refus uniquement, chaque slot non admissible est représenté par le candidat "
        "parmi ses 32 essais qui maximise la plus faible marge des deux fenêtres. Cette règle de "
        "visualisation n’affecte ni la sélection, ni l’admissibilité.",
        "",
        f"- W1 reste sous son seuil pour {fail_w1}/9 slots; W2 pour {fail_w2}/9; les deux pour {fail_both}/9.",
        f"- Au meilleur candidat diagnostic, W1 wet médian = {percentile(fail_w1_wet,50):.2f} dB, "
        f"W1 dry médian = {percentile(fail_w1_dry,50):.2f} dB, marge W1 au seuil médiane = "
        f"{percentile(fail_w1_margin,50):.2f} dB.",
        "- Les neuf refus sont répartis sur plusieurs speakers (sept sur seize ont au moins un slot refusé); "
        "aucun identifiant n’est publié.",
        "",
        "![Marge aux deux fenêtres et distribution du premier candidat valide](/" +
        figure_path.resolve().as_posix().lstrip("/") + ")",
        "",
        "## Interprétation et suite",
        "",
        "Le seuil absolu de G-v1 (W1 et W2 au-dessus de −50 dB) était trop strict pour la "
        "queue tardive de Cap60 sur certaines entrées. Le pilote relatif G2 rend 55 slots "
        "mesurables, mais neuf restent bloqués surtout par W1 : leur référence source Cap60 "
        "contient déjà un résidu précoce comparable à la queue humide ajoutée. Ce résultat "
        "écarte un fit G2 à deux fenêtres sur ce jeu et cette règle; il ne démontre pas qu’un "
        "modèle échouerait, car aucun entraînement n’a eu lieu.",
        "",
        "Suite selon la revue GPT Web : définir un protocole distinct G-early, avec W1 primaire "
        "et W2 comme non-régression, seulement après avoir figé la règle input-only, les seuils "
        "de protection et des splits nouveaux. Les 16 speakers du pilote restent diagnostiques "
        "et sont exclus de futurs fits/évaluations G2. Les neuf slots rejetés ne sont pas remplacés "
        "dans ce pilote.",
        "",
        "### Limites des conclusions",
        "",
        "Ce résultat mesure uniquement la faisabilité d’une sélection d’exemples synthétiques "
        "Cap60 sous deux fenêtres. Il ne mesure pas la réduction de réverbération en sortie, "
        "l’intelligibilité, le son perçu, le temps réel, la qualité universelle, ni l’écart à Adobe V2. "
        "Il ne valide pas des pièces réelles et ne justifie aucune promesse de parité studio.",
        "",
        "## Reproductibilité et intégrité",
        "",
        f"- Design SHA-256 : `{sha256(experiment / 'design.json')}`.",
        f"- Résultats SHA-256 : `{sha256(experiment / 'results.json')}`.",
        f"- Binaire Cap60 SHA-256 : `{design['cap60_binary_sha256']}`.",
        "- Chaque paire et chaque candidat ont une trace de source Cap60, niveau, RIR, seed et décision; "
        "le résumé public ne contient pas les identifiants.",
        "- `training_started=false`, `model_outputs_accessed=false`, `test_wav_accessed=false`.",
        "- Toute l’information détaillée et les identifiants restent sous `.tools/` et sont ignorés par Git.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, default=EXPERIMENT)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--figure", type=Path, default=FIGURE)
    args = parser.parse_args()
    design, summary, rows = load_and_audit(args.experiment_dir)
    make_figure(rows, summary, args.figure)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(build_report(design, summary, rows,
        args.report, args.figure, args.experiment_dir))
    print(json.dumps({
        "report": str(args.report), "figure": str(args.figure),
        "pairs": len(rows), "eligible": sum(row["eligible"] for row in rows),
        "training_started": summary["training_started"],
        "test_wav_accessed": summary["test_wav_accessed"],
        "integrity_audit": "PASS",
    }, indent=2))


if __name__ == "__main__":
    main()
