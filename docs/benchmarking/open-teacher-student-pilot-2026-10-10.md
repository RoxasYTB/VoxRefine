# Préentraînement student sur teacher open — 2026-10-10

## Pourquoi ce run

La demande d’autorisation Adobe est en cours. En attendant, ce pilote met en place
et mesure l’infrastructure student sans utiliser de sorties, métriques ou traces
Adobe. Il ne constitue ni une distillation Adobe, ni une preuve de parité avec
Adobe Podcast.

Le teacher local est DeepFilterNet 0.5.6 (`.tools/deepfilternet/deep-filter`), dont
le dépôt upstream publie le code et les poids sous double licence MIT ou Apache-2.0
([dépôt officiel](https://github.com/Rikorose/DeepFilterNet)). Les cibles parole
proviennent de VCTK 0.92 mic1, CC BY 4.0, sur des extraits speaker-disjoint du
corpus préparé pour le pilote précédent. Deux textures de bruit locales CC0 sont
déclarées dans `corpus/raw/heldout-noise-sources.json`. Les entrées sont générées
avec du bruit, des RIR synthétiques et une branche simulée 16→48 kHz.

## Données préparées

- 212 paires de 3 secondes : 180 apprentissage, 32 validation.
- 24 locuteurs train et 6 locuteurs validation, strictement séparés.
- Conditions aléatoires : bruit seul, pièce synthétique seule, pièce+bruit et
  bande étroite simulée+bruit.
- DeepFilterNet a été invoqué localement en mode `--compensate-delay --atten-lim-db 30`.
- Le dataset et les checkpoints sont ignorés par Git. Manifest et hashes :
  `corpus/samples/open-teacher-student-01/manifest.json`.
- Aucun audio, modèle, mesure ou output Adobe n’a été utilisé dans ce run.

## Architecture et matériel

- Student : U-Net complexe STFT 48 kHz, largeur 32, **1 011 810 paramètres**.
- Carte utilisée : NVIDIA GTX 1050 Ti, 4 GiB.
- Batch 1, crops 1 s, AMP activé.
- Pic d’allocation PyTorch mesuré pendant l’entraînement : **236 090 880 octets**.
- Inférence mesurée sur 5 s : **354 196 480 octets**, longueur de sortie identique,
  0,360 s sur cette machine (RTF **0,072**, hors chargement du modèle).
- Un passage brut sur les WAV de validation a produit des crêtes de 1,10–1,21 pour
  un input de crête 1.0 : le checkpoint d’imitation nécessite encore un garde-fou
  anti-écrêtage et une meilleure préservation du niveau.

## Deux essais, deux compromis

La validation mesure l’imitation du teacher open et le contrôle identité clean→clean.
La valeur initiale est le student à tête résiduelle nulle (identité exacte).

| Run/checkpoint | Teacher waveform L1 | Teacher STFT L1 | Clean identité SDR | Interprétation |
|---|---:|---:|---:|---|
| Initial, identité | 0,525 | 3,413 | 100 dB | Pas d’imitation, sortie identique à l’entrée |
| Sans garde clean, pas 500 | 0,469 | 2,571 | 21,2 dB | Apprend le teacher, mais colore beaucoup les voix propres |
| Garde clean x4, pas 500 | 0,489 | 3,087 | 100 dB | Préserve le clean, mais reste presque à l’identité et n’apprend pas le teacher |

Ces résultats indiquent un compromis réel à traiter dans la loss, l’architecture
et la diversité des données. **Aucun checkpoint n’est qualifié pour le produit.**
Le run sans garde clean montre une petite imitation sur un domaine limité; il n’a
pas encore la stabilité de niveau/écrêtage ni la préservation clean nécessaires.

## Décision

Le budget matériel passe largement : objectif demandé ≤2 GiB train et ≤1 GiB
inference respecté sur ce prototype. Le but qualité n’est pas atteint : aucune
comparaison contre Adobe n’est autorisée ou faite dans cette expérience, et la
préservation du clean ainsi que les crêtes de sortie échouent encore sur le
checkpoint le plus proche du teacher.

On garde le meilleur artefact uniquement comme checkpoint de recherche. La
prochaine étape immédiate est de traiter la réponse écrite d’Adobe et d’en vérifier
le périmètre exact : collecte des outputs, stockage des paires, entraînement,
évaluation, publication/openweights et redistribution. Si elle ne couvre pas
explicitement un de ces usages, aucun output Adobe ne sera utilisé pour celui-ci.
Si l’autorisation couvre l’entraînement, un run Adobe séparé, avec namespaces et
manifests propres, pourra reprendre le même socle, construire une vraie validation
speaker-disjoint, et sélectionner une loss qui équilibre fidélité teacher, identité
clean et absence d’écrêtage.

## Reproduction

Préparer des paires teacher open :

```bash
python scripts/benchmarks/universal_enhancer/prepare_open_teacher_pairs.py \
  --train-per-speaker 4 --validation-per-speaker 3 --variants 2 \
  --seconds 3.0 --seed 20261010
```

Imitation teacher open, sans garde clean renforcée :

```bash
python scripts/benchmarks/universal_enhancer/train_open_teacher_student.py \
  --steps 500 --validate-every 50 --validation-crops 8 --seconds 1.0 --width 32
```

Essai avec garde clean :

```bash
python scripts/benchmarks/universal_enhancer/train_open_teacher_student.py \
  --steps 500 --validate-every 50 --validation-crops 12 --seconds 1.0 \
  --width 32 --identity-weight 4.0 \
  --output results/open-teacher-student-identity-2026-10-10
```

Artefacts : `results/open-teacher-student-2026-10-10/` et
`results/open-teacher-student-identity-2026-10-10/`.
