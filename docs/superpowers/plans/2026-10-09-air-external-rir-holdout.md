# Holdout externe AIR pour la déréverbération — 2026-10-09

## Question et statut

Le checkpoint `measured-RIR-v1`, entraîné avec un mélange de RIR BUT mesurées et de RIR procédurales, transfère-t-il son effet face au checkpoint weak-over synthétique sur des réponses mesurées provenant d’un corpus indépendant ?

Le test est terminé. Les deux checkpoints sont restés gelés. Il s’agit d’un diagnostic externe à BUT, pas d’une comparaison Adobe ni d’une validation de produit.

## Données scellées avant inférence

- Source officielle : [Aachen Impulse Response Database (AIR) v1.4](https://www.iks.rwth-aachen.de/en/research/tools-downloads/databases/aachen-impulse-response-database), archive téléchargée depuis RWTH Aachen.
- L’archive contient un `readme.txt` indiquant que la base est sous MIT et un `license.txt` avec la notice MIT RWTH. L’archive locale SHA-256 est `d2fd52767505c402e8aed9299dcd046493c8254cfab2f769fcf480dc7c616a5f`.
- Sélection déterministe, indépendante des sorties des modèles : réponses les plus proches et les plus éloignées dans quatre lieux présents dans AIR v1.4 : booth, office, meeting et lecture. Huit RIR au total.
- Un seul canal droit du jeu binaural est extrait (`channel=0`), sans mannequin de tête (`head=0`). Les RIR sont converties de 48 kHz vers mono 16 kHz pour l’entrée du modèle. Il s’agit d’un trajet d’oreille mesuré sans mannequin, pas d’une mesure omnidirectionnelle mono typique de microphone.
- Les deux checkpoints et 12 voix `train-clean-360` nouvelles par rapport aux écrans BUT ont été fixés avant le scoring. Deux énoncés successifs par voix, avec une pause numérique insérée de 800 ms; aucune sortie n’a été écoutée ou utilisée pour choisir les RIR.
- Même normalisation de pic early-path vers 1, convolution complète, référence wet fixe, contrôles secs et seuil commun `Cᵢ=max(F_Aᵢ,F_Bᵢ)+3 dB` que pour l’analyse BUT. Les planchers restent censurés, jamais soustraits.

## Gates avant ouverture du résultat

- La borne basse de la médiane censor-aware est supérieure à 0 dB dans les deux fenêtres.
- Pas de salle avec médiane clairement négative.
- Perte supplémentaire sur les contrôles secs : pas plus de 1 dB en activité et en onset p10.
- Pas d’effondrement manifeste de la voix sèche.

Un gain médian garanti supérieur à 3 dB dans une fenêtre serait un signal fort; ce n’est pas un prérequis de ce premier test externe.

## Sources et reproduction

- AIR v1.4 : page officielle RWTH ci-dessus; l’archive et les SHA des huit extraits se trouvent dans `.tools/compact-dereverb/data/air/` (données ignorées par Git).
- Extraction : [`extract_air_rirs.py`](../../scripts/benchmarks/universal_enhancer/compact_dereverb/extract_air_rirs.py).
- Runner : [`screen_measured_rirs.py`](../../scripts/benchmarks/universal_enhancer/compact_dereverb/screen_measured_rirs.py).
- Analyse censor-aware : [`analyze_censored_pairs.py`](../../scripts/benchmarks/universal_enhancer/compact_dereverb/analyze_censored_pairs.py).
- Mesures détaillées, rendus et manifeste restent dans `.tools/compact-dereverb/air-external-2026-10-09/`.
