# Modèle étudiant multilingue FLEURS — plan d’implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Entraîner un enhancer vocal compact sur des paires propres/dégradées synthétiquement, avec couverture multilingue et une piste distincte pour la restauration large bande.

**Architecture:** Utiliser FLEURS pour le nettoyage et la déréverbération à 16 kHz, avec les splits officiels train/dev/test et une séparation par locuteur/langue documentée. Garder l’apprentissage 48 kHz sur les prises VCTK réellement large bande ; ne pas présenter une interpolation d’une cible 16 kHz comme reconstruction supervisée de hautes fréquences.

**Tech Stack:** Python 3, PyTorch CUDA, SoundFile, NumPy/SciPy, API Hugging Face Dataset Hub et archives FLEURS originales.

## Contraintes globales

- FLEURS est identifié en CC BY 4.0 ; conserver attribution, source, licence, hashes et transformations dans chaque manifeste.
- Utiliser uniquement les archives officielles publiques FLEURS ; exclure toutes les sorties Adobe et tout audio privé de ce corpus étudiant.
- Le modèle d’usage doit rester compatible avec l’entraînement sous 2 GiB VRAM et l’inférence sous 1 GiB sur GTX 1050 Ti.
- FLEURS est à 16 kHz ; le modèle multilingue 16 kHz n’est pas une preuve de reconstruction de fréquences au-delà de 8 kHz.
- Garder les fichiers audio téléchargés, corpus transformés et checkpoints dans des chemins ignorés par Git ; ne pas pousser de données audio.
- Conserver intactes les modifications locales existantes.

---

### Task 1: Préparation reproductible du corpus FLEURS

**Files:**
- Create: `scripts/benchmarks/universal_enhancer/prepare_fleurs_multilingual.py`
- Runtime data only: `corpus/samples/fleurs-multilingual-16k-01/`

**Interfaces:**
- CLI : `--locales all|en_us,fr_fr,...`, `--splits train,dev`, `--max-clips-per-locale`, `--output`, `--resume`, `--dry-run`.
- Produire un `manifest.json` atomique avec URI, licence, locale, split, identifiant de locuteur si fourni, SHA-256 source/cible, fréquence d’échantillonnage, durée et statut d’extraction.
- Refuser l’écrasement des données déjà préparées sans `--force`.

- [ ] Énumérer les 102 configurations FLEURS et les tailles des archives train/dev depuis l’API Hub ; calculer l’espace temporaire requis avant tout téléchargement.
- [ ] Télécharger les archives dans un cache temporaire avec reprise HTTP Range, vérifier taille et SHA-256 LFS, extraire les WAV 16 kHz source sans rééchantillonnage ni réencodage.
- [ ] Lire les TSV officiels pour maintenir les identifiants et métadonnées, séparer train de dev et rejeter les enregistrements non mono, non 16 kHz ou illisibles.
- [ ] Écrire le manifeste avec attribution FLEURS CC BY 4.0 et citation Conneau et al. (SLT 2022), puis supprimer l’archive temporaire vérifiée après extraction.
- [ ] Exécuter un dry-run et un petit lot anglais/français avant l’extraction complète ; enregistrer tailles, durées, langues et locuteurs obtenus.

### Task 2: Student 16 kHz multilingue avec dégradations reproductibles

**Files:**
- Create: `scripts/benchmarks/universal_enhancer/multilingual_student.py`
- Create: `scripts/benchmarks/universal_enhancer/train_multilingual_student.py`
- Create: `scripts/benchmarks/universal_enhancer/evaluate_multilingual_student.py`
- Runtime checkpoints only: `results/multilingual-student-fleurs-16k-01/`

**Interfaces:**
- Dataset item : `clean: Tensor[T]`, `locale: str`, `speaker_id: str`; le loader retourne `input, clean_target, condition_id`.
- Modèle : mono 16 kHz, mêmes longueur et fréquence en entrée/sortie, avec chemin d’identité sur parole propre.
- Train CLI : `--steps`, `--crop-seconds`, `--seed`, `--max-vram-gib`, `--resume`, `--output`.

- [ ] Adapter le petit student STFT existant à 16 kHz en faisant dépendre les pertes et masques de fréquence du taux d’échantillonnage au lieu des constantes 48 kHz.
- [ ] Générer à la volée du bruit CC0, du bruit coloré, des RIR mesurées/synthétiques, des échos discrets, du clipping modéré, des codecs et des bandes 8 kHz ; stocker les paramètres et seed pour chaque validation.
- [ ] Échantillonner les langues de façon équilibrée plutôt qu’en proportion brute, et éviter qu’une voix/phrase répétée traverse les splits.
- [ ] Combiner perte multi-résolution, waveform bornée, contrôle d’identité propre, préservation des faibles trames/onsets et pénalité de silence ; choisir checkpoint seulement sur dev multilingue.
- [x] Lancer un smoke-test GPU 100 pas (≈0,11 Gio alloué) : pipeline exécutable, mais le score room bouge peu.
- [x] Pilote initial stoppé à 1 000/2 000 pas : sur 11 langues, L1 du clean augmente de ~0 à 0,103 ; le checkpoint est rejeté.
- [ ] Corriger les pertes pour garantir le chemin identité propre et vérifier le jeu de RIR avant tout nouveau pilote ; ne prolonger que si les critères clean et room s’améliorent ensemble.

### Task 3: Validation indépendante et décision de capacité

**Files:**
- Create: `docs/benchmarking/multilingual-student-fleurs-16k-01.md`
- Figures and metrics: `results/multilingual-student-fleurs-16k-01/`

**Interfaces:**
- Rapporter métriques globales et par langue/condition : SI-SDR, MR-STFT, STOI si dépendance/licence confirmées, identité clean, dérive de niveau, clipping, temps, RTF et VRAM.

- [ ] Figer les clips dev avant le tuning et garder les tests officiels scellés pour une mesure finale.
- [ ] Comparer entrée, student et références propres sous bruit seul, RIR seule, bruit+RIR, bande limitée et parole propre.
- [ ] Comparer à un baseline open licencié séparément sans mettre ses poids dans la distribution finale sauf validation de leur licence.
- [ ] Décider GO uniquement si les scores sont reproduits sur des locuteurs et langues tenus à part, si la parole propre reste quasi identique et si aucun défaut audible automatique/mesuré n’augmente.
- [ ] Documenter que FLEURS entraîne un enhancer 16 kHz ; garder le modèle 48 kHz et l’extension de bande supervisée sur des corpus 48 kHz indépendants.

### Task 4: Documentation d’attribution et versionnement

**Files:**
- Create: `docs/models/fleurs-multilingual-data-card.md`
- Modify only after validation: `README.md`

- [ ] Écrire attribution, citation, CC BY 4.0, contenu inclus/exclu, transformations et répartition locale des langues.
- [ ] Ne pas inclure les enregistrements FLEURS, les sorties Adobe, le NDA ou ses clauses dans le dépôt Git.
- [ ] Ne publier un checkpoint que si la carte de modèle, provenance des données, licence du code/poids et validation sont toutes renseignées.

## Résultat intermédiaire du 10 octobre 2026

Le pipeline d’acquisition est lancé et reprend via `--resume`. Les premières 11 locales train/dev totalisent 27 005 clips / 91,92 h train et 3 941 clips / 12,70 h dev. Ces données restent dans des chemins ignorés par Git. Le pilote de 1 000 pas a dégradé la validation « clean » (wave L1 ≈0,103) et n’a que faiblement réduit l’erreur de réverbération synthétique ; aucun poids n’est retenu comme candidat de publication. Le problème doit être corrigé avant d’augmenter le nombre de pas.
