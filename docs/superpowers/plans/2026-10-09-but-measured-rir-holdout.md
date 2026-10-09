# Holdout de déréverbération sur RIR mesurées — 2026-10-09

## Question

Le candidat weak-over, entraîné uniquement sur des réponses de pièce procédurales, réduit-il les queues de réverbération de pièces réellement mesurées sans dégrader les voix, comparé au checkpoint de base ?

## Hypothèses et périmètre gelés

- Les checkpoints baseline et weak-over ne changent pas pendant cette expérience.
- Corpus RIR : BUT Speech@FIT Reverb Database, version 2019-06, uniquement le paquet RIR-only; neuf salles, CC BY 4.0 selon la page du jeu de données.
- Par salle, choisir avant toute inférence les deux configurations source/micro aux distances mesurées minimale et maximale.
- Corpus vocal : LibriSpeech train-clean-360, CC BY 4.0; retenir 12 speakers absents de train-clean-100, dev-clean, dev-other, test-clean et test-other.
- Choisir déterministement deux énoncés adjacents d’un même speaker dans train-clean-360 et insérer 800 ms de silence numérique. Cette pause contrôlée n’est pas une pause naturelle; convoluer la séquence complète et garder 200 ms de marge avant la seconde phrase.
- Un canal micro physique, une RIR mono, trajet direct détecté selon une règle déterministe, totalité de la queue conservée, pic des 10 premières millisecondes après l’onset normalisé à un gain unitaire pour correspondre à la convention d’entraînement. Cette « normalisation du pic early-path alignée sur le domaine d’entraînement » n’est pas une normalisation du trajet direct garanti ni une calibration acoustique SPL; un unique facteur partagé ajuste sec et convolué sans gain local.
- Aucun bruit, DPDFNet, Adobe output, alignement de niveau local, apprentissage ou réglage sur ces voix/pièces.
- Le rendu et les données dérivées restent dans `.tools/`; le dépôt ne contient que scripts, manifeste de provenance sans audio, métriques agrégées et graphiques.

## Métriques séparées

1. Parole active : niveau output/clean, niveau weak-frame output/clean, fractions weak au-dessus de +6 et +10 dB, niveau onset/rising-frame p10.
2. Pause insérée contrôlée : énergie de queue input/output rapportée à la même énergie d’entrée réverbérée sur le masque actif de la première phrase (masque dérivé du dry), fenêtres 50–150, 150–300 et 300–600 ms; plancher du contrôle sec rapporté séparément et sorties à moins de 3 dB du plancher censurées, jamais soustraites.
3. Coloration : erreur absolue log-spectrale multi-bande output-vers-clean, comparée à l’entrée-vers-clean, sans normalisation fréquentielle.
4. Coût : temps d’inférence 6 s, RTF, paramètres et mémoire si disponibles.

Ne pas compter les 216 rendus réverbérés comme 216 observations indépendantes. Publier les médianes par salle et par speaker, puis leur agrégat équilibré.

## Critères de poursuite préenregistrés

- Amélioration médiane de queue ≥3 dB dans 150–300 et 300–600 ms.
- Amélioration dans au moins 7 des 9 salles.
- Weak p50 proche de −1 à +1,5 dB et fraction >+6 dB nettement réduite par rapport au baseline.
- Onset/rising p10 au plus 1 dB moins bon que le baseline.
- Contrôle sec ±1 dB; pas de hausse notable du plancher artificiel.
- Une baisse de queue accompagnée d’une forte coloration ou d’une perte de parole ne passe pas.

## Étapes et état

- [x] Vérifier les pages officielles de licence et la structure métadonnée du jeu de données.
- [x] Écrire le runner de sélection et d’évaluation; ajouter des tests unitaires de sélection, normalisation, convolution, pause et métriques.
- [x] Télécharger et vérifier le MD5 officiel de LibriSpeech train-clean-360.
- [x] Télécharger le paquet RIR-only BUT, calculer son SHA-256 et extraire le contenu sous `.tools/`.
- [x] Inventorier les neuf salles, distances et 18 configurations sélectionnées; retenir un protocole à pause insérée car les énoncés ont peu de pauses internes longues.
- [x] Exécuter les deux checkpoints gelés sur le même plan apparié sec/RIR après correction de l’échelle RIR et de la référence métrique.
- [x] Produire résultats par salle/speaker, graphiques, limites et décision.

## Sources

- [BUT Speech@FIT Reverb Database — description, salles, mesures et licence](https://speech.fit.vut.cz/software/but-speech-fit-reverb-database)
- [BUT ReverbDB README — structure et métadonnées RIR](https://merlin.fit.vutbr.cz/ReverbDB/read_me.txt)
- [OpenSLR SLR12 — LibriSpeech, CC BY 4.0, archives et MD5 officiels](https://www.openslr.org/12/)

## Résultat du holdout mesuré

Rapport complet : [`docs/benchmarking/but-measured-rir-dereverb-2026-10-09.md`](../../benchmarking/but-measured-rir-dereverb-2026-10-09.md). Après correction du dénominateur des planchers et comparaison au seuil commun `max(F_A,F_B)+3 dB`, le gain faible-over contre baseline a une médiane possible de **[1,15; 1,21] dB** sur 150–300 ms et **0,61 dB** sur 300–600 ms. Le gate préenregistré de +3 dB dans les deux fenêtres échoue. Les résultats varient selon les salles; E112 et L212 régressent. Ce résultat ne valide ni une qualité studio ni une équivalence Adobe.

L’expérience A/B de domaine a ensuite gardé architecture, perte, optimiseur, budget et voix d’entraînement identiques, en mélangeant 50 % de RIR procédurales et 50 % de RIR BUT mesurées dans six salles. Trois salles BUT ont été mises à part pour le diagnostic cross-room. Face au weak-over synthétique, la médiane possible du gain mesuré-mix vaut **[6,79; 6,95] dB** sur 150–300 ms et **[6,29; 6,38] dB** sur 300–600 ms; l’amélioration est de même signe dans les trois salles. Les contrôles secs restent dans le garde-fou d’environ 1 dB. Ce signal encourageant ne constitue pas un test externe : les neuf salles BUT avaient déjà été explorées. Le candidat mesuré-mix est gelé en version expérimentale; aucune revendication universelle n’en découle.

La prochaine étape est une validation sur une source RIR externe qui n’a pas été utilisée dans l’entraînement, la sélection ou les essais précédents. Avant tout téléchargement ou entraînement, vérifier la provenance, les conditions de licence pour les mesures et les dérivés, et l’adéquation des métadonnées permettant d’apparier les conditions. Conserver le protocole à plancher commun, publier les bornes censurées et agréger les résultats au niveau des pièces, sans p-values naïves sur speaker × RIR.
