# G-early-v2 — pré-audit synthétique (10 octobre 2026)

## Résultat et portée

**PASS mécanique uniquement.** Le modèle G2 sait transmettre les gradients, apprendre une baisse de queue W1 sur quatre signaux synthétiques et respecter la borne de gain. Cela autorise la préparation d'un nouveau corpus à locuteurs disjoints. Ce résultat n'évalue aucun vrai locuteur, aucune pièce réelle, ni Adobe Podcast. Il ne prouve pas une qualité de production.

## Modèle gelé

- Même U-Net STFT/ISTFT et 555 922 paramètres que E.
- Gain borné `g = 1 - 0.5 sigmoid(a)`, soit `0.5 < g < 1`; le modèle ne peut pas amplifier.
- Biais initial gain `a=-3`: `g=0,976287`, équivalent à 0,208 dB d'atténuation amplitude. La sensibilité `|dg/da|=0,022588`, soit 3,398 fois `sigmoid(+5)` de E.
- Poids du canal gain tirés sur CPU de `N(0, 1e-4)` avec la seed dédiée `2026101002`; canal phase exactement nul à l'initialisation.
- Pour réduire l'énergie de 3 dB par un gain uniforme, `g≈0,707946`, atteignable vers `a≈0,340`.

## Fixtures et protocole figés

- Seed fixtures `20261010`; quatre signaux harmoniques de 3 secondes à 16 kHz, f0 103/127/149/181 Hz, enveloppe de voix synthétique, pause connue et 23 réflexions décroissantes déterministes. Aucun corpus, fichier vocal, split ou checkpoint v1 n'a été lu.
- À `a=-3,0,+3`, l'autograd correspond à la dérivée analytique avec erreur relative `<1e-4`; 10 001 logits de `[-20,+20]` respectent `(0.5,1)`.
- `y=x` donne `ΔW1=0 dB`, `L_early=1`.
- Les normes de gradients `L_base` et `L_early` vers tête, trunk et paramètres complets sont finies et non nulles avant update; les quatre fixtures passent.
- Chaque bras part des mêmes poids initiaux et parcourt exactement 400 updates batch-one selon `index=step mod 4`, `AdamW(lr=2e-4, weight_decay=1e-4)`, clipping global 3.
- Gates gelés: chaque fixture tail-only améliore W1 d'au moins 2,5 dB; bras full-loss médiane ≥0,5 dB et aucune fixture <−0,25 dB; gradients finis; sortie silence <1e-7.
- Le bras full-loss applique exactement `L_base + L_early`; aucune loss ni seuil n'a été choisi après résultat.

## Résultats

Deux exécutions avant activation des kernels déterministes ont toutes deux passé les gates, mais n'étaient pas identiques. Les deux reçus sont conservés plutôt que de sélectionner le meilleur.

| Mesure W1, dB | Run initial A | Rejeu de décomposition B |
|---|---:|---:|
| Tail-only, gains des 4 fixtures | 3,868 / 2,967 / 4,843 / 3,474 | 3,860 / 2,928 / 4,844 / 3,436 |
| Full-loss, améliorations vs état initial | 4,092 / 3,403 / 4,140 / 3,655 | 3,813 / 3,315 / 3,291 / 3,262 |
| Médiane full-loss | 3,874 | 3,304 |

Décomposition descriptive du Run B — mêmes poids entraînés, reconstruits avec chaque composant isolé. Les effets ne s'additionnent pas après ISTFT.

| Fixture | Masque appris complet | Gain seul (`φ=0`) | Phase seule (`g=1`) |
|---:|---:|---:|---:|
| 1 | 4,021 dB | 2,988 dB | 1,466 dB |
| 2 | 3,523 dB | 3,138 dB | 0,535 dB |
| 3 | 3,500 dB | 2,658 dB | 1,227 dB |
| 4 | 3,470 dB | 3,196 dB | 0,374 dB |

Sous full-loss, le gain seul dépasse la phase seule sur 4/4 fixtures; le masque combiné dépasse le gain seul de 0,274 à 1,034 dB dans W1. La médiane brute de gain sur toutes les cases TF bougeait de seulement `−0,00003 à −0,00022`: elle masque l'atténuation concentrée dans les bins porteurs de queue.

La conservation d'énergie active cible/sortie du bras full-loss est −0,23 / −0,43 / −0,80 / −0,77 dB dans le rejeu B. Ce n'est qu'un indicateur synthétique de mécanisme.

## Audit de reproductibilité

Après avoir gelé les réglages déterministes (algorithmes PyTorch déterministes, cuDNN déterministe, benchmarking et TF32 coupés, `CUBLAS_WORKSPACE_CONFIG=:4096:8`), deux nouvelles exécutions ont passé les gates et leurs valeurs non temporelles sont identiques à `1e-5` près (aucune différence détectée). Les durées ne sont pas comparées. Le pré-audit prend environ 140 secondes par répétition sur GTX 1050 Ti; ce coût concerne seulement l'entraînement de diagnostic hors-ligne.

Les quatre reçus ignorés restent sous `.tools/compact-dereverb/g-early-v2-*/mechanism-audit.json`. Les hashes des deux reçus reproductibles sont `f3c275f7c7a5e43252356fd65d10d9b3ad5f31a7fff01faa2a789e9bd9017435` et `16b84fca29db5fae9a32c84f0bf61e077b5bf42c5d8d3fd2e973c9bc2c91a617`. Code modèle SHA-256 `14741400052b76d90f1dd4a8e1ef6fbb90fb884d8c063b6305a4262e59b52a6a`; audit déterministe SHA-256 `bdf8bd521f6ea467f7556ade2e73b135061d2192ffcbb7da590ba83ad4d30d18`.

Une première tentative a été interrompue avant update et avant écriture de métriques par une erreur de dimensions dans le masque d'activité; une correction mécanique et la relance sont incluses dans les sources gelées. Aucun résultat de cette tentative n'a été consulté.

## Décision et limites

Le protocole ouvre uniquement la voie à des données TRAIN/DEV/HOLDOUT v2 avec locuteurs frais disjoints. Les gates déterministes et la décomposition gain/phase resteront des diagnostics dans l'évaluation future; ils ne changent pas les critères produit gelés. G-early-v1 reste NO-GO, son HOLDOUT reste scellé et `test.wav` n'a pas été lu.
