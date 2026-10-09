# Adobe Podcast Enhance v2 Convergence Implementation Plan

> **For agentic workers:** Execute inline, one evidence-gated task at a time. Steps use `- [ ]` checkboxes; preserve independent audio problem scopes.

**Goal:** Improve VoxRefine’s local speech enhancement toward matched Adobe Podcast Enhance v2 outputs without damaging speech, and keep every quality claim within the measured data.

**Architecture:** Keep the existing Python offline path and treat Adobe as a perceptual style reference. Track noise, room reverb, loudspeaker echo/AEC, tone, and bandwidth extension as separate capabilities; no candidate enters the default chain without passing speech-preservation and held-out gates.

**Tech Stack:** Python, NumPy/SciPy/SoundFile for analysis, existing pinned DPDFNet/Resemble backends, JSON/CSV and static plots for evidence.

## Principes et contraintes

1. Garder une séparation explicite entre bruit stationnaire, réverbération de pièce, écho de haut-parleur (AEC), coloration tonale et extension de bande passante. Un seul modèle/slider ne doit pas être crédité de fonctions qui ne sont pas mesurées.
2. Ne jamais ajuster une égalisation pour réduire la distance à Adobe sans mesurer en parallèle le résidu bruit/tail, le niveau vocal, l’enveloppe et l’intelligibilité. Comparaisons d’écoute à loudness égalisée; fichiers natifs conservés.
3. Pré-enregistrer les seuils par expérience. Les extraits déjà utilisés servent au diagnostic et au choix de paramètres seulement; toute revendication de généralisation exige un jeu tenu à l’écart.
4. Garder toutes les branches modèles facultatives et locales. Le profil stable actuel ne change pas tant qu’un challenger n’a pas franchi les portes décrites ci-dessous.
5. Ne pas réécrire le cœur en Rust/C++ pour la qualité audio. Cette étape est justifiée seulement si le profilage d’une chaîne gagnante révèle un coût logiciel important. Le vrai obstacle actuel est la qualité généralisable et les mesures, pas l’orchestration Python hors ligne.

---

## Fichiers et responsabilités

- `scripts/benchmarks/universal_enhancer/audit_adobe_v2_pairs.py`: inventory and integrity audit for existing source/Adobe/candidate audio assets; outputs ignored CSV/JSON only.
- `docs/benchmarking/adobe-v2-pair-audit-2026-10-09.md`: protocol, inventory, missing-pair table, and limits.
- `docs/benchmarking/adobe-paired-rir-screen-2026-10-08.md` and `docs/benchmarking/adobe-spectral-proximity-tradeoff-2026-10-08.md`: existing paired acoustic and spectral evidence.
- `docs/benchmarking/product-evidence-matrix-2026-10-08.md`: product capability gates and current evidence status.
- `voxrefine/` and `tests/`: untouched until one processing candidate passes the offline audio gates.

## Plan d’exécution

### Étape 1 — Figer le scorecard Adobe v2 reproductible

- Regrouper les paires source/Adobe déjà présentes et vérifier les hashes, durée, fréquence, alignement, paramètres UI et versions d’export.
- Pour chaque paire, recalculer par événement : LUFS / niveau actif, true peak, enveloppe RMS 20 ms (corrélation et erreur), spectre ERB/1/6 octave après gain fixe documenté, bruit en pauses, baisse de tail si la vérité de tail existe, et latence/RTF des candidats.
- Distinguer les métriques qui ont une vérité connue (stems de bruit, dry + RIR) de celles seulement descriptives (spectre Adobe, sorties Adobe sur parole naturelle).
- Produire une matrice de disponibilité des paires et marquer explicitement les cas non comparables. Ne pas créer de nouveaux rendus Adobe si le navigateur est déconnecté; les fichiers existants suffisent à l’analyse.

**Porte :** chaque chiffre est recalculable depuis un script versionné et les hashes appariés; aucune métrique spectrale n’est présentée comme métrique de qualité perceptive.

**Livrable d'exécution :** le script d'audit produit pour chaque WAV le SHA-256, le format, le nombre de frames, la durée et l'état de lecture. Il ne traite pas l'audio et ne prétend pas recalculer la qualité acoustique.

### Étape 2 — Établir le meilleur candidat actuel sur les paires Adobe

- Réutiliser en premier les sorties existantes de DPDFNet, Resemble denoiser/NFE et les essais doux de tonalité. Ne pas refaire un balayage aveugle des paramètres.
- Sur les paires exactes, comparer trois chaînes maximum :
  - traitement conservateur DPDFNet + niveau natif;
  - Resemble denoiser/NFE déjà rendu, sans post-EQ;
  - chaîne hybride uniquement si ses fichiers et paramètres exacts existent déjà.
- Ajouter au plus une variante de tone shaping doux, appliquée après nettoyage, avec filtres stables et headroom. Ce test ne prétend pas reconstruire le traitement interne Adobe.
- Calculer une fiche par paire et une fiche agrégée; conserver chaque échec et chaque métrique par locuteur.

**Porte :** ne garder que les candidats qui réduisent le bruit/tail attendu sans échec de suppression de phonèmes, baisse anormale de l’enveloppe ou crête/clipping. Un candidat n’est « plus proche » que sur un ensemble de critères, jamais selon le seul MAE spectral.

### Étape 3 — Diagnostiquer les différences restantes par famille de dégradation

- **Bruit :** évaluer les stems de bruit connus et des clips réalistes déjà disponibles; vérifier le résidu et les consonnes faibles.
- **Réverbération :** comparer la queue et l’enveloppe uniquement sur les RIR à vérité connue; ne pas traiter la fin naturelle d’une phrase comme un tail artificiel.
- **Tonalité :** examiner les écarts Adobe par bande après correction du loudness; tester les étagères douces déjà implémentées seulement lorsque le biais se répète entre paires.
- **Écho/AEC :** garder LocalVQE dans un benchmark duplex isolé. Aucun traitement de fichier mono ne peut annuler un retour haut-parleur sans référence far-end.
- **Bande passante 8 kHz :** maintien comme branche expérimentale distincte; n’annoncer aucune reconstruction de détail absent avant une validation avec référence haute fréquence.

**Porte :** chaque changement doit améliorer sa dégradation cible sans régression sur les contrôles propres/voix faibles. Sinon le chemin reste optionnel ou est abandonné.

### Étape 4 — Validation croisée et test hors échantillon

- Figer les réglages sur les paires Adobe et le corpus de développement existant.
- Avant d’évaluer une revendication universelle, réserver des fichiers encore inutilisés du corpus avec provenance et conditions acoustiques différentes; grouper par locuteur afin d’éviter les fuites entre train/choix et validation.
- Ajouter les exemples courts, longs, faibles, bruités et réverbérants. Documenter qu’un audiobook lu ne représente pas les micros conversationnels.
- Faire une écoute A/B à niveau égal lorsque l’utilisateur sera disponible; jusqu’à cette écoute, les conclusions restent objectives et limitées.

**Porte :** pas de promotion de profil si le gain médian cache un cas de destruction vocale ou si les résultats tenus à l’écart s’inversent.

### Étape 5 — Produit, seulement après portes qualité

- Exposer des réglages distincts uniquement lorsque les fonctions sont réellement isolables et validées : réduction de bruit, réverbération/AEC, tonalité, voix/musique.
- Rendre le profil non destructif, niveaux secs/mouillés alignés, export WAV/MP3 vérifié et rapport de provenance.
- Garder le mode fichier hors ligne comme voie de qualité maximale; le mode temps réel aura son propre budget de latence et ses modèles causaux.
- Cibler GTX 10xx si GPU sélectionné, avec fallback CPU explicite; établir RAM/VRAM, RTF, latence de bout en bout et limites des formats.

## Tableau de progression

| Étape | État | Critère de clôture |
|---|---|---|
| 1. Scorecard apparié v2 | Inventaire et premier cohort propre/bruité terminés; couverture encore limitée | Scripts versionnés vérifient les paires et publient mesures descriptives + figures; diversifier les locuteurs et conditions |
| 2. Challenger local sur paires exactes | En cours; NFE64-C → DPDFNet 6/18 dB échoue sur voix faibles | Candidat sans trames actives écrasées, validé sur plusieurs voix; rapport [itération](../../benchmarking/adobe-v2-next-iteration-2026-10-09.md) |
| 3. Diagnostic bruit / RIR / ton / AEC / 8 kHz | Partiel; RIR tenu à l’écart, AEC virtuel seulement | Mesures par tâche avec références correctes; ne pas assimiler mono fichier et AEC |
| 4. Validation hors échantillon | Partiel; VCTK/LibriVox contrôlés, pas d’Adobe apparié sur les nouveaux domaines | Exports v2 sur voix conversationnelles, appareils et bruits non stationnaires; holdout locuteur |
| 5. Produit stable et contrôles | DPDFNet live virtuel, non validé sur périphérique; NFE hors temps réel sur GTX 1050 Ti mesurée | Profilage end-to-end réel CPU/GPU, underruns, mémoire; qualité et live comme voies séparées |

### Progression factuelle — 2026-10-09

- **Adobe v2, trois voix propres + bruit contrôlé :** les trois entrées ont un stem propre et un bruit connu; SNR actif mesuré 17,29–18,00 dB. Les exports Adobe v2 et NFE64-C sont appariés, 15 s, 48 kHz. Détails, hashes et figures : [rapport de cohorte](../../benchmarking/adobe-v2-clear-noisy-cohort-2026-10-09.md).
- **Préservation :** corrélation médiane d’enveloppe 0,976 pour Adobe contre 0,962 pour NFE64-C. Le p10 des trames vocales faibles NFE64-C sur Emy chute à −37,41 dB contre −30,34 dB pour Adobe; ce candidat ne passe donc pas la porte faible-voix sur ce cas. Une MAE spectrale médiane de 2,40 dB ne compense pas cette régression et ne mesure pas la qualité perçue.
- **Suite de l’étape 2 :** diagnostiquer la suppression des trames faibles (détection d’activité, masque de gain, plancher de réduction / transition) sur un jeu de développement, ajouter des cas hors réglage, puis réévaluer sans promouvoir le profil en défaut.
- **Cascade DPDFNet après NFE64-C :** sur les trois mixes identiques, les variantes 6 et 18 dB font tomber la corrélation médiane d’enveloppe à 0,657 / 0,664 et écrasent au moins 12,5 % des faibles trames à plus de 80 dB sous la référence. Le résultat est décrit dans [le rapport de prochaine itération](../../benchmarking/adobe-v2-next-iteration-2026-10-09.md); ne pas l’activer en défaut.
- **Temps réel / GTX 1050 Ti :** l’inférence NFE64-C observée est à ~2,85 RTF, donc trop lente pour le direct sur cette carte; DPDFNet CPU a un coût moyen de ~0,413 RTF mais des dépassements ponctuels et aucun test matériel réel. La simulation n’est pas une garantie de live.

- **Tonalité fixe :** le profil `adobe_curve` réduit la MAE active 20 Hz–12 kHz de 4,34 dB (plat) à 1,77 dB sur un clip bruité contrôlé; les MP3 d'écoute ont été nivelés en LUFS. Cela valide uniquement la distance de courbe sur cet extrait, pas le son.
- **Contrôle par cellules appariées :** le profil réduit la MAE dans cinq paires source/Adobe déjà traitées, mais empire le contrôle sec Naf de 1,55 dB. La médiane des six cellules baisse de 3,60 à 3,00 dB; l'effectif reste trop petit pour conclure à une généralisation.
- **Préservation vocale :** aucune mesure objective de préservation phonétique/intelligibilité n'a été calculée sur toutes ces variantes d'EQ. Cette porte reste ouverte.
- **AEC LocalVQE :** son rapport duplex est indépendant; cette branche ne doit pas être mélangée aux comparaisons Adobe de nettoyage de fichiers.
- **Préservation après EQ (diagnostic in-sample) :** sur les trois paires RIR utilisées pour son ajustement, la MAE spectrale médiane baisse de 4,49 à 3,67 dB, mais le p10 de niveau des fenêtres vocales faibles se dégrade de 3,64 dB et l'enveloppe médiane ne gagne que 0,010. La courbe reste hors du profil par défaut; la panne d'enveloppe Naf persiste.

## Résultat réaliste attendu

La prochaine livraison doit réduire une différence mesurée et reproductible envers Adobe v2, pas promettre la parité parfaite. Si les sorties Adobe ne sont pas disponibles pour de nouveaux exemples, la comparaison reste limitée aux exports déjà stockés. L’objectif de travail est de maximiser la qualité locale et la cohérence sans masquer les limites de généralisation.

### Gates proposées pour les prochains challengers (revue GPT Web, 2026-10-09)

Ces seuils servent de portes de présélection, pas de définition complète de la qualité perçue. Les appliquer à des manifests figés, par locuteur et condition; publier aussi les distributions, les échecs et les valeurs brutes. Ne pas composer un score unique.

- **Voix faible d'abord :** masque d'événement dérivé du stem propre uniquement pour l'analyse; rapporter p10/p1, parts sous −20/−40/−80 dB, plancher numérique exact et corrélation d'enveloppe. Aucun effondrement exact ou sous −80 dB; p10 faible ≥−24 dB par locuteur, aucun locuteur <−30 dB; perte p90 événement ≤6 dB et aucun événement >10 dB.
- **Parole normale :** perte événement médiane <1 dB, p90 par locuteur <3 dB, corrélation médiane d'enveloppe ≥0,90.
- **Bruit / pièce :** viser ≥8 dB médian sur babble à 10 dB SNR et ≥12 dB sur bruit stationnaire à 10 dB, avec amélioration dans ≥75 % des cas. Pour les RIR à vérité connue, perte p90 du précoce <3 dB; réduire le tail mesurable d'au moins 8 dB (RIR moyen) / 10 dB (long), sans effacer l'événement vocal.
- **Adobe v2 secondaire :** sur les seules paires exactes disponibles, corrélation d'enveloppe du candidat ≥Adobe−0,03, p10 faible à moins de 6 dB d'Adobe et aucun effondrement >40 dB si Adobe n'en a aucun. Adobe reste un style de référence, pas la vérité propre.
- **Diagnostics seulement :** local SNR, énergie 2–4 / 4–8 kHz, voix voisée/non voisée et recouvrement temps-fréquence peuvent expliquer un échec; ils ne peuvent pas piloter un routeur entraîné sur le holdout.

### État candidat temps réel — 2026-10-09

- DPDFNet2 causal 48 kHz CPU a été rendu sur 126 mixes gelés (7 locuteurs × 3 événements faibles × 6 conditions, 378 s audio). RTF médian par fichier 0,34; durée d’inférence agrégée 131,63 s (RTF global 0,348), hors chargement modèle qui n’a pas été chronométré dans ce rendu. Aucun événement ne dépasse 6 dB de perte de voix selon la projection sur le stem de 250 ms. Sur les conditions 10 dB, la perte médiane par locuteur est 0,13–0,18 dB pour babble et ventilateur; la réduction de bruit projetée est 9,1–25,1 dB selon texture. Au stress 5 dB, perte médiane 0,16–0,31 dB et réduction projetée 13,6–18,8 dB. Le diagnostic de fenêtres faibles ne montre aucune trame DPDFNet2 sous −80 dB ni au plancher, mais le p10 atteint −6,26 dB et l’enveloppe faible 0,787 sur babble à 5 dB; les portes d’enveloppe ne sont donc pas encore toutes franchies. Détails et prudence d'interprétation : [screen faible-voix](../../benchmarking/realtime-weak-preservation-2026-10-09.md).
- Ce résultat est prometteur pour la **présélection** de DPDFNet2 live, mais ne prouve ni intelligibilité ni préférence à l'écoute : les événements sont courts, atténués de −18 dB et mélangés à des bruits contrôlés; la projection peut surestimer la suppression si la sortie synthétisée reste corrélée au stem. Même domaine LibriVox français, aucune sortie Adobe pour ces 126 entrées. Le calcul de survie conserve les mesures p1/p10, fractions sous −20/−40/−80 dB, trames au plancher et corrélation d’enveloppe, ventilées dans le rapport.
- DeepFilterNet cap6 n'est pas disponible dans le runtime local; ne pas prétendre l'avoir comparé à cap18. Résembles denoiser-only est une référence offline de préservation, tandis que cap100 reste un contrôle agressif historique.
- **Décision :** aucun changement du preset produit sur ces seules mesures. Réutiliser DPDFNet2 comme candidat live pour le prochain holdout indépendant et l'évaluation RIR séparée; garder NFE64 hors live sur la GTX 1050 Ti mesurée (2,85 RTF inférence, ~4,40 RTF bout en bout).

### Prochain ordre d'exécution

1. Installer/activer un environnement isolé DeepFilterNet cap6 reproductible ou documenter le blocage, puis rerendre les mêmes 126 entrées avec hashes et durées invariants.
2. Compléter sur ces mêmes inputs l'analyse de préservation faible (dont exact-floor, p1/p10, seuils −20/−40/−80 dB) pour les candidats admissibles; ne pas changer les paramètres après examen du holdout.
3. Tester DPDFNet2 seul sur le holdout RIR moyen/long avec événements précoces et tails connus, rapports séparés de bruit et de voix; aucune seconde passe denoiser automatique.
4. Si les paires Adobe et les fichiers sources sont présents, comparer uniquement les exact-pairs v2 par voix, à niveau aligné, comme contrôle secondaire. Ne pas uploader ou collecter de nouveaux audios utilisateur sans demande.
5. Avant de toucher au chemin live, utiliser une capture/lecture virtuelle documentée; le test matériel sur micro réel reste à faire quand il sera possible et autorisé. Mesurer service p50/p95/p99, underruns, démarrage et mémoire, distinctement du RTF.
