# G-early-v1 — décision DEV NO-GO (10 octobre 2026)

## Décision

Les deux variantes G-early-v1 échouent sur DEV16. Aucun gagnant n'est sélectionné et le HOLDOUT12 reste fermé définitivement. Ces modèles ne doivent pas être présentés comme une alternative studio ni comme une reproduction d'Adobe Podcast.

| Variante | W1 150–300 ms, Mlow | LCB bootstrap locuteur 95 % | Speech | Décision |
|---|---:|---:|---|---|
| G-early-clean | +0,113 dB | +0,045 dB | gates passés | NO-GO: gate W1 ≥ +2 dB échoué |
| G-early-hybrid | +0,314 dB | +0,165 dB | plusieurs gates échoués | NO-GO: gate W1 ≥ +2 dB et gates speech échoués |

Les deux candidats satisfont les critères W2, pente W1 et floors bruit, mais ces résultats ne compensent pas l'absence de suppression précoce. L'hybride gagne environ 0,20 dB W1 médian sur clean, au prix d'une préservation de parole moins bonne : perte active p50 médiane −0,543 dB, p10 −3,176 dB, perte weak sous −3 dB médiane 10,13 % et sous −6 dB 5,15 %.

L'évaluation corrigée a vérifié 64/64 références gelées avec une différence maximale de 0,000 dB, tolérance 0,01 dB. Les mesures W1 ont n=63 à cause des références communes au plancher; la couverture W2 informative est 28/64 (43,75 %). Les hashes des données, modèle, évaluateur, checkpoints et reçu sont conservés dans les artefacts ignorés `.tools/compact-dereverb/g-early-2026-10-10/` et `dev-evaluation/summary.json`.

## Diagnostic de l'apprentissage

Le code de `tail_loss` utilise directement le tenseur prédit et des opérations différentiables; il n'y a pas de rupture de gradient dans ce chemin. La loss est moyennée uniquement sur les lignes W1 éligibles. Avec batch size 1, les contrôles à loss nulle ne diluent donc pas la loss. Les journaux montrent toutefois de grands gradients totaux avant clipping global à 3,0 (jusqu'à 6 953 pour clean et 41 184 pour hybrid dans les lignes enregistrées). Ils ne contiennent pas de normes séparées `grad_base` et `grad_early`; on ne peut donc pas attribuer avec certitude le NO-GO à une compétition de gradients.

La tête de E démarre à `sigmoid(a)` avec `a=+5`: gain initial `0,9933`, dérivée `0,00665`. La perte early observée autour de 0,9 correspond mathématiquement à une baisse W1 d'environ 0,15 dB, cohérente avec le faible résultat clean. C'est une hypothèse mécaniste plausible de saturation/gradient trop faible, pas une cause prouvée sans audit séparé.

## Suite autorisée

Le protocole G-early-v2 est séparé et preregistré dans [le plan v2](../superpowers/plans/2026-10-10-g-early-v2.md). Il commence par un audit synthétique déterministe, sans lire les données v1. La seule modification de modèle envisagée est une tête bornée moins saturée, `g = 1 − 0.5 sigmoid(a)`, biais `−3`. Aucun nouveau vrai jeu de données ne sera préparé avant réussite du pré-audit. Tout éventuel TRAIN/DEV/HOLDOUT v2 devra utiliser des locuteurs nouveaux, tous disjoints des locuteurs déjà consommés.

## Limites

Cette décision porte uniquement sur les voix LibriSpeech et les réverbérations procédurales fortes de ce benchmark. Elle ne mesure ni les pièces réelles, ni l'audio de l'utilisateur, ni Adobe Podcast. Le fichier personnel `test.wav` n'a pas été lu. La machine disponible dans la session courante n'expose pas PyTorch dans ses runtimes Python détectés; aucun nouveau calcul de gradient ni entraînement n'est revendiqué ici.
