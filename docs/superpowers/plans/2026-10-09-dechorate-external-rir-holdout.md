# Holdout externe dEchorate — 2026-10-09

## Question et état

Le candidat `measured-RIR-v1`, adapté sur BUT, transfère-t-il son effet par rapport au weak-over synthétique à des réponses mesurées provenant d’un autre dépôt, d’une configuration de salle reconfigurée et d’une géométrie de source/micro différente ?

Le sous-ensemble de six RIR est choisi depuis les métadonnées SOFA et les coordonnées de l’annotation, sans utiliser les sorties de modèle. Aucun checkpoint, seuil ou traitement n’est ajusté sur dEchorate.

## Source et licence

- Index de fichiers SOFA : https://sofacoustics.org/data/database/dechorate/
- Page du dépôt de recherche : https://zenodo.org/records/6576203
- Un SOFA d’exemple de 1,7 Mo est inspecté avant scoring. Son attribut `License` embarque le texte MIT de Diego Di Carlo. L’extracteur vérifie cet attribut dans chaque fichier de sonde et chaque fichier sélectionné. Le grand archive HDF5 de RIR ne sert pas à ce test.
- Les RIR sont des réponses `SingleRoomSRIR` 48 kHz avec cinq récepteurs par fichier. L’échantillon inspecté contient positions source/récepteur, délai de canal, descriptions du matériel et licence. L’extrait mono conserve le retard indiqué par `Data.Delay`; le runner commun réalise ensuite la conversion à 16 kHz et la normalisation early-path vers 1.

## Sélection scellée

1. Énumérer les codes de configuration présents dans l’index SOFA officiel.
2. Télécharger uniquement le fichier source 1 / array 1 pour chaque configuration (11 sondes, environ 19 Mo).
3. Sur les cinq récepteurs de chaque sonde, calculer la médiane du T20 par courbe de décroissance énergétique de Schroeder : ajustement linéaire de −5 à −25 dB puis extrapolation à 60 dB.
4. Sélectionner les configurations au T20 médian minimal, médian et maximal : `room000100`, `room020002`, `room011111`.
5. Les fichiers source 1 / array 1 ne servent qu’à classer les configurations et sont exclus du test. Pour chaque configuration sélectionnée, énumérer les couples source/micro disponibles dans l’index SOFA et calculer leur distance 3D depuis l’annotation officielle. Choisir les extrêmes avec égalités départagées par source, array puis microphone. Le résultat actuel est source 9 / array 1 / micro 1 à 0,899 m et source 4 / array 2 / micro 10 à 4,258 m.
6. Télécharger seulement les trois fichiers SOFA contenant ces six réponses. Enregistrer chaque réponse et son identifiant, son SHA-256, sa configuration, le T20 de classement, sa fréquence, son délai et sa licence.

Les extrêmes géométriques changent aussi de source et d’array. Near/far est donc un axe descriptif, pas une comparaison causale isolant uniquement la distance; l’effet du modèle est estimé par les paires A/B sur chaque RIR fixe.

## Modèles, voix et métriques

- Référence A : weak-over synthétique gelé, SHA-256 `31e0b7b01224103c58bd5e4453005ad5065a710f71a3d78106c0976662434a40`.
- Candidat B : adaptation mesuré-mix gelée, SHA-256 `c49ccf72b4874c233548d14e74a02c6ac66b1f5f653762a18986e60499aa31b3`.
- 12 voix `train-clean-360` inédites par rapport aux holdouts BUT et AIR; un couple déterministe d’énoncés successifs par voix et une pause numérique contrôlée de 800 ms. On utilise une paire par voix afin de garder le runner et la comparaison appariée inchangés.
- Comparaison d’abord sur 150–300 et 300–600 ms, avec la même référence wet fixe et le plancher commun `Cᵢ=max(F_Aᵢ,F_Bᵢ)+3 dB`. Les observations censurées restent des intervalles. Pas de p-value sur les paires speaker×RIR.
- Contrôles secs et onset p10 restent séparés de la mesure de queue. Pas de gain de sortie par fichier et pas d’EQ.

## Gates de diagnostic

- Borne basse de médiane censor-aware >0 dB dans les deux fenêtres.
- Rapport par configuration de mur et distance; aucune médiane de configuration clairement négative.
- Perte sèche supplémentaire ne dépassant pas 1 dB en activité et en onset p10 face à A.
- Toute borne basse de médiane >3 dB est signalée séparément comme résultat fort; aucun résultat ne constitue une preuve de parité Adobe ou de généralisation universelle.

## Résultat et reproduction

Les sorties sont locales dans `.tools/compact-dereverb/dechorate-external-2026-10-09/`. L’extracteur SOFA est [`extract_dechorate_rirs.py`](../../scripts/benchmarks/universal_enhancer/compact_dereverb/extract_dechorate_rirs.py); l’inférence commune et l’analyse censor-aware utilisent les scripts du dossier `compact_dereverb`.
