# Changelog

## 0.2.0

- Conversion locale automatique des WAV, MP3 et M4A avec FFmpeg.
- Sortie WAV mono PCM16 à 48 kHz ; original inchangé.
- Préparation unique des entrées du benchmark et conservation des références.
- Option `--ffmpeg` pour choisir le binaire.
- Limite d'atténuation DeepFilterNet réglable avec `--attenuation-limit-db`,
  dans le nettoyage et les comparaisons ; valeur par défaut inchangée.

## 0.1.0

- Nettoyage local WAV mono PCM16 à 48 kHz avec DeepFilterNet 0.5.6 et RNNoise v0.1.
- Comparaison par manifeste avec sorties audio et rapport technique JSON.
- Conservation de la durée, compensation du délai et refus d'écraser les fichiers.
- Tests unitaires, essais natifs sur signaux synthétiques et CI.

Premier retour positif lors d'un essai local sur une voix avec un bruit de
voiture en arrière-plan. Pas encore d'évaluation comparative sur un corpus vocal.
