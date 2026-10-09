# Plan de convergence vers Adobe Podcast Enhance v2

**Objectif.** Faire converger VoxRefine vers le rendu Adobe v2 sur des entrées appariées, tout en gardant une voix intacte et un traitement local compatible avec le matériel modeste du projet. Adobe est une référence de rendu, jamais une vérité propre. Une proximité de spectre seule ne vaut pas une amélioration audible.

**État de départ.** Les paires Adobe v2 disponibles comprennent trois événements RIR synthétiques (deux voix) et quelques extraits avec bruit contrôlé. Les mesures existantes montrent une réponse Adobe dépendante de l’événement : tail mesuré de +2,5 à +26,2 dB sur les trois événements RIR. La proximité spectrale peut récompenser le sous-traitement. NFE est parfois proche en spectre mais instable en enveloppe et en queue; DPDFNet est plus constant sur la réduction du tail, mais plus éloigné en balance spectrale. Une EQ ajustée sur ces paires réduit la MAE spectrale dans cinq cellules, mais la dégrade de 1,55 dB sur le contrôle sec Naf : une courbe statique ne doit pas devenir le réglage universel. LocalVQE duplex est rapide sur le CPU testé, mais la préservation vocale est trop variable pour l’intégrer.

## Principes et contraintes

1. Garder une séparation explicite entre bruit stationnaire, réverbération de pièce, écho de haut-parleur (AEC), coloration tonale et extension de bande passante. Un seul modèle/slider ne doit pas être crédité de fonctions qui ne sont pas mesurées.
2. Ne jamais ajuster une égalisation pour réduire la distance à Adobe sans mesurer en parallèle le résidu bruit/tail, le niveau vocal, l’enveloppe et l’intelligibilité. Comparaisons d’écoute à loudness égalisée; fichiers natifs conservés.
3. Pré-enregistrer les seuils par expérience. Les extraits déjà utilisés servent au diagnostic et au choix de paramètres seulement; toute revendication de généralisation exige un jeu tenu à l’écart.
4. Garder toutes les branches modèles facultatives et locales. Le profil stable actuel ne change pas tant qu’un challenger n’a pas franchi les portes décrites ci-dessous.
5. Ne pas réécrire le cœur en Rust/C++ pour la qualité audio. Cette étape est justifiée seulement si le profilage d’une chaîne gagnante révèle un coût logiciel important. Le vrai obstacle actuel est la qualité généralisable et les mesures, pas l’orchestration Python hors ligne.

## Plan d’exécution

### Étape 1 — Figer le scorecard Adobe v2 reproductible

- Regrouper les paires source/Adobe déjà présentes et vérifier les hashes, durée, fréquence, alignement, paramètres UI et versions d’export.
- Pour chaque paire, recalculer par événement : LUFS / niveau actif, true peak, enveloppe RMS 20 ms (corrélation et erreur), spectre ERB/1/6 octave après gain fixe documenté, bruit en pauses, baisse de tail si la vérité de tail existe, et latence/RTF des candidats.
- Distinguer les métriques qui ont une vérité connue (stems de bruit, dry + RIR) de celles seulement descriptives (spectre Adobe, sorties Adobe sur parole naturelle).
- Produire une matrice de disponibilité des paires et marquer explicitement les cas non comparables. Ne pas créer de nouveaux rendus Adobe si le navigateur est déconnecté; les fichiers existants suffisent à l’analyse.

**Porte :** chaque chiffre est recalculable depuis un script versionné et les hashes appariés; aucune métrique spectrale n’est présentée comme métrique de qualité perceptive.

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
| 1. Scorecard apparié v2 | En cours | Script et manifeste vérifient toutes les paires Adobe disponibles |
| 2. Challenger local sur paires exactes | À faire | Fiche par paire + critères conjoints préenregistrés |
| 3. Diagnostic bruit / RIR / ton / AEC / 8 kHz | Partiel | Amélioration séparée sans confusion entre tâches |
| 4. Validation hors échantillon | Bloqué par corpus indépendant limité | Résultats tenus à l’écart et écoute A/B |
| 5. Produit stable et contrôles | Après qualité | Pas de promotion avant les portes précédentes |

### Progression factuelle — 2026-10-09

- **Tonalité fixe :** le profil `adobe_curve` réduit la MAE active 20 Hz–12 kHz de 4,34 dB (plat) à 1,77 dB sur un clip bruité contrôlé; les MP3 d'écoute ont été nivelés en LUFS. Cela valide uniquement la distance de courbe sur cet extrait, pas le son.
- **Contrôle par cellules appariées :** le profil réduit la MAE dans cinq paires source/Adobe déjà traitées, mais empire le contrôle sec Naf de 1,55 dB. La médiane des six cellules baisse de 3,60 à 3,00 dB; l'effectif reste trop petit pour conclure à une généralisation.
- **Préservation vocale :** aucune mesure objective de préservation phonétique/intelligibilité n'a été calculée sur toutes ces variantes d'EQ. Cette porte reste ouverte.
- **AEC LocalVQE :** son rapport duplex est indépendant; cette branche ne doit pas être mélangée aux comparaisons Adobe de nettoyage de fichiers.

## Résultat réaliste attendu

La prochaine livraison doit réduire une différence mesurée et reproductible envers Adobe v2, pas promettre la parité parfaite. Si les sorties Adobe ne sont pas disponibles pour de nouveaux exemples, la comparaison reste limitée aux exports déjà stockés. L’objectif de travail est de maximiser la qualité locale et la cohérence sans masquer les limites de généralisation.
