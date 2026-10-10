# G2 feasibility pilot — résultats (10 octobre 2026)

> **Décision : NO-GO pour l’entraînement G2 à deux fenêtres.** Le gate préenregistré exige 64/64 paires admissibles; le pilote en trouve 55/64. Aucun modèle n’a été entraîné.

## Question mesurée

Vérifier si une nouvelle génération de RIR procédurales peut produire, après Cap60, une queue de réverbération mesurable à la fois en W1 (150–300 ms) et W2 (300–600 ms), au-dessus du résidu propre Cap60 de la même source. Ce pilote de faisabilité n’évalue pas la qualité d’un modèle G et ne compare pas une sortie à Adobe Podcast.

## Protocole gelé

- 16 speakers diagnostic déjà consommés par le DEV G-v1, 4 paires par speaker (64 slots); aucun speaker n’est réutilisé pour un éventuel train/dev/holdout G2.
- Sources LibriSpeech `train-clean-360`; nouvelles RIR procédurales, RT60 uniforme [1,65; 1,85] s, DRR uniforme [−14,5; −12,5] dB.
- Namespace nouveau `G2-feasibility-dev-v1|speaker|pair|candidate`; candidats j=0…31, Cap60 exact sur signal humide et référence sèche, première paire de fenêtres valide.
- Pour chaque fenêtre W, niveaux d’énergie rapportés au même RMS actif du dry Cap60 :
  - `Lx(W) = 10 log10((E_x(W)+ε)/(E_ref+ε))` ;
  - `Ldry(W) = 10 log10((E_dry(W)+ε)/(E_ref+ε))` ;
  - admission stricte si `Lx(W) > max(−60 dB, Ldry(W)+6 dB)` en W1 **et** W2.
- Aucun output G consulté; seuils, fenêtres, sources et paramètres RIR non ajustés après mesure.

## Résultats

| Mesure | Résultat |
|---|---:|
| Slots examinés | 64/64 |
| Admissibles aux deux fenêtres | 55/64 (85.94 %) |
| Non admissibles après 32 candidats | 9/64 |
| Passent dès j=0 | 49/64 |
| Candidats par slot (médiane / moyenne / maximum) | 1 / 6.28 / 32 |
| Candidats Cap60 examinés au total | 402 |
| Temps total du pilote | 68.59 min (4115.6 s) |

### Niveaux des 55 paires admissibles

Tous les niveaux ci-dessous sont relatifs à l’énergie active dry Cap60, pas dBFS. Le plancher −120 dB provient de l’epsilon de mesure et décrit une référence sous le plancher numérique du calcul.

| Fenêtre | Lx p10 / médiane / p90 (dB) | Ldry p10 / médiane / p90 (dB) | Marge minimale au seuil (dB) |
|---|---:|---:|---:|
| W1 150–300 ms | -46.79 / -34.50 / -15.66 | -120.00 / -120.00 / -40.20 | 1.01 |
| W2 300–600 ms | -52.00 / -45.15 / -32.42 | -120.00 / -120.00 / -120.00 | 3.92 |

### Diagnostic des neuf slots non admissibles

Pour expliquer les refus uniquement, chaque slot non admissible est représenté par le candidat parmi ses 32 essais qui maximise la plus faible marge des deux fenêtres. Cette règle de visualisation n’affecte ni la sélection, ni l’admissibilité.

- W1 reste sous son seuil pour 8/9 slots; W2 pour 3/9; les deux pour 2/9.
- Au meilleur candidat diagnostic, W1 wet médian = -32.07 dB, W1 dry médian = -32.60 dB, marge W1 au seuil médiane = -12.00 dB.
- Les neuf refus sont répartis sur plusieurs speakers (sept sur seize ont au moins un slot refusé); aucun identifiant n’est publié.

![Marge aux deux fenêtres et distribution du premier candidat valide](/mnt/windata/C/Users/Yohan/Desktop/Projets/projects/VoxRefine/docs/benchmarking/assets/g2-feasibility-2026-10-10/g2-feasibility.png)

## Interprétation et suite

Le seuil absolu de G-v1 (W1 et W2 au-dessus de −50 dB) était trop strict pour la queue tardive de Cap60 sur certaines entrées. Le pilote relatif G2 rend 55 slots mesurables, mais neuf restent bloqués surtout par W1 : leur référence source Cap60 contient déjà un résidu précoce comparable à la queue humide ajoutée. Ce résultat écarte un fit G2 à deux fenêtres sur ce jeu et cette règle; il ne démontre pas qu’un modèle échouerait, car aucun entraînement n’a eu lieu.

Suite selon la revue GPT Web : définir un protocole distinct G-early, avec W1 primaire et W2 comme non-régression, seulement après avoir figé la règle input-only, les seuils de protection et des splits nouveaux. Les 16 speakers du pilote restent diagnostiques et sont exclus de futurs fits/évaluations G2. Les neuf slots rejetés ne sont pas remplacés dans ce pilote.

### Limites des conclusions

Ce résultat mesure uniquement la faisabilité d’une sélection d’exemples synthétiques Cap60 sous deux fenêtres. Il ne mesure pas la réduction de réverbération en sortie, l’intelligibilité, le son perçu, le temps réel, la qualité universelle, ni l’écart à Adobe V2. Il ne valide pas des pièces réelles et ne justifie aucune promesse de parité studio.

## Reproductibilité et intégrité

- Design SHA-256 : `3c7fbf71731f536f1ae15010e111b6bbf19bcdc3f6ef9223ecaa69d7ad38ce5d`.
- Résultats SHA-256 : `244234a2cbf3f4120a82731fab56e33326529a26792097ebe4412f86999084da`.
- Binaire Cap60 SHA-256 : `70775e251eee44c0f2451a1e833326cf8bcbbe304d3e7cd12851e6fce72ef7da`.
- Chaque paire et chaque candidat ont une trace de source Cap60, niveau, RIR, seed et décision; le résumé public ne contient pas les identifiants.
- `training_started=false`, `model_outputs_accessed=false`, `test_wav_accessed=false`.
- Toute l’information détaillée et les identifiants restent sous `.tools/` et sont ignorés par Git.
