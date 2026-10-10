# Modèle étudiant de restauration vocale — voie open source

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Concevoir un enhancer vocal local compact, mesurable et redistribuable, avec entraînement sous 2 GiB de VRAM et inférence sous 1 GiB.

**Architecture:** Ne pas distiller les sorties Adobe sans autorisation écrite. Sous les conditions Adobe actuellement publiées, les sorties et informations dérivées ne peuvent pas servir à créer, entraîner, tester ou améliorer un modèle, et le suivi des E/S pour recréer le produit est explicitement visé par l’interdiction de reverse engineering. La voie exécutable sans permission est un student entraîné sur un teacher open source dont les licences code et poids sont vérifiées, complété par des paires parole propre/dégradations synthétiques et des RIR/bruits redistribuables. Ce modèle peut viser une restauration vocale de qualité, mais ne peut pas être présenté comme une imitation Adobe ni être évalué contre Adobe.

**Tech Stack:** Python, PyTorch CUDA, SoundFile, NumPy/SciPy; GTX 1050 Ti 4 GiB; modèle mono 48 kHz avec blocs 1D depthwise/dilatés ou U-Net STFT compact, AMP pour entraînement.

## Global Constraints

- Cible VRAM entraînement : maximum 2 GiB alloués; cible inférence : maximum 1 GiB.
- Ne pas utiliser les sorties, rendus, mesures dérivées ou traces Adobe pour entraîner, tester, régler ou améliorer VoxRefine sans permission écrite explicite d’Adobe.
- Ne pas scraper, automatiser les uploads, contourner authentification, quotas, CAPTCHA ou autres restrictions Adobe.
- Les sorties Adobe déjà présentes dans `results/` ne font pas partie du dataset student; ne pas les inclure dans les manifests ni dans les scripts de benchmark de cette voie.
- Ne pas utiliser d’enregistrement privé sans choix et autorisation explicites de la personne concernée.
- Exclure des splits d’entraînement les locuteurs déjà réservés dans les manifests VoxRefine.
- Auditer les licences du code ET des poids de tout teacher avant de l’inclure ou de redistribuer un checkpoint.
- Ne pas modifier le pipeline produit avant qu’une validation prospective et speaker-disjoint montre une amélioration robuste et une bonne préservation des voix propres.
- Ne pas promettre égalité Adobe ni reconstruction exacte des fréquences qui n’existent plus dans une source 8/16 kHz.

## Sources de décision et porte juridique

Les [Adobe General Terms of Use, sections 6 et 17](https://www.adobe.com/legal/terms.html) interdisent notamment l’accès hors interface autorisée, le contournement des restrictions, le data-mining/scraping, le reverse engineering défini comme suivi des entrées/sorties pour recréer le service, et l’utilisation des services ou de contenus/données/sorties/informations dérivées pour créer, entraîner, tester ou améliorer un système ML/IA. Les [Adobe Generative AI User Guidelines](https://www.adobe.com/legal/licenses-terms/adobe-gen-ai-user-guidelines.html), révisées le 15 mai 2026, proscrivent les processus automatisés ou scripts non autorisés, tels que les uploads en masse automatisés. Cette lecture opérationnelle n’est pas un avis juridique.

**Porte A — autorisation écrite Adobe.** Si Adobe autorise explicitement par écrit l’usage des sorties Enhance Speech comme données d’entraînement/évaluation pour VoxRefine, créer un plan séparé limité exactement au périmètre autorisé et vérifier l’autorisation avant toute collecte. Le présent plan ne suppose pas cette autorisation.

**Porte B — aucune autorisation écrite.** Continuer uniquement avec les modèles, données et rendus ayant des licences qui autorisent l’entraînement et l’usage prévus. La similitude avec Adobe ne sera pas une métrique de ce travail.

## État d’exécution pendant l’attente Adobe

- Vérification du teacher open : le dépôt officiel DeepFilterNet déclare code et poids sous MIT ou Apache-2.0; teacher local `deep_filter 0.5.6`.
- Dataset de pilote prêt : VCTK mic1 CC BY 4.0, 180 paires train / 32 validation, 24/6 locuteurs disjoints; bruit CC0; aucune sortie Adobe incluse.
- Student 1 011 810 paramètres entraîné sur deux runs de 500 pas. Budget GPU passe (236 MiB train / 354 MiB inference), mais aucun checkpoint ne passe la qualité clean+teacher+peak; détails dans `docs/benchmarking/open-teacher-student-pilot-2026-10-10.md`.
- Le travail conditionnel Adobe n’a pas commencé. Toute future collecte demeure suspendue à l’autorisation écrite et au périmètre qu’elle accorde.

---

### Task 1: Verrouiller provenance, licences et jeu de données

**Files:**
- Create: `docs/models/open-teacher-license-audit-2026-10-10.md`
- Create: `scripts/benchmarks/universal_enhancer/prepare_open_student_data.py`
- Runtime only, ignored: `corpus/samples/open-student-01/`

**Interfaces:**
- CLI: `--seed`, `--train-speakers`, `--validation-speakers`, `--clips-per-speaker`, `--dry-run`.
- Manifest JSON contient par source : URL/citation, licence, speaker ID, split, SHA-256, SR, canaux, durée, corruption seed et paramètres.

- [ ] Vérifier dans les sources amont les licences des poids et du code du teacher candidat; consigner la licence de chaque dépendance et métrique. Si les poids interdisent entraînement, redistribution ou usage commercial prévu, ne pas les utiliser comme teacher.
- [ ] Sélectionner un corpus propre redistribuable à 48 kHz avec locuteurs séparables; préserver les fichiers d’origine en lecture seule et exclure les speaker IDs des manifests holdout existants.
- [ ] Préparer des dégradations déterministes et paramétrées : bruit stationnaire/non-stationnaire, RIR mesurées et synthétiques, EQ/micro, codec, clipping léger, bandes 8/16 kHz et combinaisons; conserver les cibles clean non modifiées.
- [ ] Valider le split strict par locuteur et par RIR/bruit; calculer hashes et durée totale; rendre le manifeste atomique et refuser de remplacer un corpus existant sans `--force`.
- [ ] Ne pas inclure les WAV/mesures Adobe ni les outputs Adobe historiques dans le manifeste, le loader ou les objectifs du student.

### Task 2: Établir le teacher open et le budget matériel

**Files:**
- Create: `scripts/benchmarks/universal_enhancer/open_teacher_adapter.py`
- Create: `scripts/benchmarks/universal_enhancer/benchmark_student_memory.py`
- Create: `docs/architecture/open-student-budget-2026-10-10.md`

**Interfaces:**
- Adapter: `enhance(audio: Tensor[B,T], sample_rate: int) -> Tensor[B,T]` avec longueur, délai et SR explicités.
- Benchmark: peak VRAM train et inference, temps/audio, temps de démarrage, RTF, modèle/poids/hash et version logicielle.

- [ ] Comparer des teachers disponibles localement uniquement avec leurs licences upstream archivées; retenir un teacher open dont les poids permettent l’usage visé ou un teacher supervisé que VoxRefine entraîne depuis ses cibles clean.
- [ ] Mesurer l’empreinte réelle du teacher. Si le teacher dépasse 2 GiB, l’utiliser uniquement pour pré-générer les pseudo-labels hors entraînement; cela ne réduit pas la mémoire GPU nécessaire à l’inférence student.
- [ ] Définir explicitement le comportement de l’adapter pour entrées mono/stéréo, SR 8/16/44.1/48 kHz, longueurs courtes, silence et dernière frame.
- [ ] Ne pas invoquer de service Adobe, API privée ou endpoint non documenté dans l’adapter.

### Task 3: Construire un student compact à long contexte

**Files:**
- Create: `scripts/benchmarks/universal_enhancer/open_student.py`
- Create: `scripts/benchmarks/universal_enhancer/train_open_student.py`
- Test: `tests/test_open_student.py`

**Interfaces:**
- `OpenStudent.forward(audio: Tensor[B,T], sample_rate=48000) -> Tensor[B,T]` préserve longueur et accepte des chunks avec état documenté.
- CLI train expose `--steps`, `--batch-size`, `--crop-seconds`, `--amp`, `--seed`, `--output`, `--resume`.

- [ ] Démarrer par 0,5–2 M paramètres en mono 48 kHz, blocs 1D multi-échelle/depthwise-dilatés avec un champ réceptif de plusieurs centaines de ms; garder l’architecture et l’empreinte observables dans un `model-card.json`.
- [ ] Entraîner sur crops de 0,5–2 s, batch 1, AMP, accumulation facultative, gradient clipping; vérifier la limite de 2 GiB sur la machine réelle avant de lancer une passe longue.
- [ ] Pour la supervision clean, utiliser une combinaison bornée de L1/Charbonnier waveform et MR-STFT; pour le teacher open licencié, ajouter une loss d’imitation (log-magnitude et waveform alignés) avec poids faible au début.
- [ ] Ajouter une proportion explicite de clean→clean et une loss d’identité pour que la sortie tende vers l’identité sur les voix déjà propres; entraîner la BWE comme tête/tâche distincte avec indicateur de bande d’entrée.
- [ ] Sauvegarder checkpoints atomiques et reprendre le meilleur checkpoint de validation sans écraser les runs historiques.

### Task 4: Évaluation prospective et critères de décision

**Files:**
- Create: `scripts/benchmarks/universal_enhancer/evaluate_open_student.py`
- Create: `docs/benchmarking/open-student-pilot-2026-10-10.md`

**Interfaces:**
- Sorties JSON/CSV par speaker, condition, checkpoint et baseline; aucun Adobe score ou artefact Adobe.

- [ ] Figer un challenge set speaker-disjoint avec clean untouched, bruit seul, RIR seule, bruit+RIR, bandes 16/8 kHz, codec et clipping; garder des bruits/RIR jamais vus pour validation.
- [ ] Comparer entrée, pipeline actuel, teacher open et student; rapporter SI-SDR/SDR, MR-STFT, STOI si licence/domaine adaptés, dérive d’identité, faibles trames, niveaux/clipping et RTF/VRAM.
- [ ] Mesurer une scaling curve sur une même architecture avec 1 h, 5 h et 20 h de paires autorisées pour vérifier que le gain se poursuit avec les données.
- [ ] GO vers un pilote plus long seulement si les voix clean sont quasi identiques, les gains de restauration se répètent sur des speakers/bruits/RIR inconnus, les artefacts sont absents et le pic reste ≤2 GiB train / ≤1 GiB inference.
- [ ] STOP ou réduire la portée si l’amélioration plafonne, si la voix propre est colorée, ou si le teacher/data ne couvre pas les pièces réelles. Décrire l’outil comme enhancer indépendant, pas comme équivalent Adobe.

### Task 5: Décider le chemin conditionnel Adobe

**Files:**
- Modify only if Adobe grants written permission: a separate plan and a separate dataset namespace under `corpus/samples/adobe-authorized-student-*/`.

- [ ] Archive the written authorization and verify it explicitly covers collecting outputs, creating training/validation pairs, model training, and any planned open-source distribution.
- [ ] If any right is absent or unclear, do not collect or use Adobe outputs; remain on Task 1–4 only.
- [ ] If permission covers only private evaluation, keep Adobe data out of training and student selection/tuning unless evaluation use is also explicitly authorized.
