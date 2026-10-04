# VoxRefine

Nettoyage de voix en local, depuis une commande Python.

Cette première version permet de traiter un fichier avec DeepFilterNet ou
RNNoise, puis de comparer les deux sur les mêmes enregistrements. Aucun audio
n'est envoyé à un serveur. Les moteurs s'installent séparément ; VoxRefine ne
télécharge rien pendant le traitement.

## État du projet

La version `0.2.0` accepte **WAV, MP3 et M4A**. FFmpeg convertit
les entrées qui le nécessitent en WAV PCM 16 bits, mono, 48 kHz.
La sortie reste dans ce format, sans écraser de fichier existant.
Le nettoyage conserve la durée exacte de l'audio décodé ; les codecs compressés
peuvent ajouter du padding selon leur encodage.
Pas de normalisation automatique, de transcription ou d'interface graphique.

Les deux moteurs ont été exécutés sur du silence et un signal synthétique.
Un premier essai local sur une voix a également donné un retour positif sur
la suppression d'un bruit de voiture en arrière-plan. Ce retour ponctuel ne
remplace pas une comparaison à l'écoute sur un corpus varié. Le choix du moteur
reste ouvert jusqu'à cette évaluation.
Le projet ne revendique pas une qualité équivalente à Adobe Podcast.

## Installation

Python 3.10 ou plus récent. Le code Python n'a pas de dépendance d'exécution.

```sh
git clone https://github.com/pf1s/VoxRefine.git
cd VoxRefine
python3 -m venv .venv
. .venv/bin/activate
python -m pip install .
voxrefine --version
```

Sur Debian/Ubuntu, `python3-venv` peut être nécessaire. Installer également
FFmpeg pour les formats compressés et les WAV qui doivent être convertis :

```sh
sudo apt install ffmpeg
```

Un WAV mono PCM16 à 48 kHz fonctionne toujours sans FFmpeg.
Sans installation du package, les mêmes commandes fonctionnent avec
`python3 -m voxrefine` depuis le dépôt.

### Moteurs — Linux x86_64

Ces commandes installent les versions utilisées pour les premiers essais.
Le binaire DeepFilterNet intègre son modèle. RNNoise est compilé avec les poids
présents dans sa source ; ni GPU ni PyTorch ne sont nécessaires.

DeepFilterNet 0.5.6 :

```sh
mkdir -p .tools
curl -fL https://github.com/Rikorose/DeepFilterNet/releases/download/v0.5.6/deep-filter-0.5.6-x86_64-unknown-linux-musl \
  -o .tools/deep-filter
echo '70775e251eee44c0f2451a1e833326cf8bcbbe304d3e7cd12851e6fce72ef7da  .tools/deep-filter' \
  | sha256sum -c -
chmod u+x .tools/deep-filter
```

RNNoise v0.1, avec Git et un compilateur C :

```sh
git clone --branch v0.1 --depth 1 https://github.com/xiph/rnnoise.git .tools/rnnoise
test "$(git -C .tools/rnnoise rev-parse HEAD)" = cdf196b1e9de2f8ff1003328ebf9a4316477429d
cc -O2 -shared -fPIC \
  -I.tools/rnnoise/include -I.tools/rnnoise/src \
  .tools/rnnoise/src/denoise.c .tools/rnnoise/src/rnn.c \
  .tools/rnnoise/src/rnn_data.c .tools/rnnoise/src/pitch.c \
  .tools/rnnoise/src/celt_lpc.c .tools/rnnoise/src/kiss_fft.c \
  -lm -o .tools/librnnoise.so
```

L'adaptateur RNNoise cible l'ancienne ABI de cette version. Les bibliothèques
récentes avec chargement de modèle sont refusées. Ce choix sert de référence
reproductible, pas de recommandation définitive du meilleur modèle.

Les autres plateformes ne sont pas encore validées. Pour DeepFilterNet, consulter
les [binaires amont](https://github.com/Rikorose/DeepFilterNet/releases/tag/v0.5.6).

## Nettoyer un fichier

```sh
python3 -m voxrefine clean corpus/voice.m4a results/voice-df.wav \
  --engine deepfilter --deep-filter .tools/deep-filter

python3 -m voxrefine clean corpus/voice.wav results/voice-rnnoise.wav \
  --engine rnnoise --rnnoise-library .tools/librnnoise.so
```

La préparation est automatique et temporaire. Un message indique la conversion
en mono : les canaux sont mélangés, pas nettoyés séparément. L'original reste
inchangé. La première piste audio est utilisée si le fichier en contient plusieurs.
Utiliser `--ffmpeg /chemin/ffmpeg` si le binaire n'est pas dans le PATH.
Les erreurs de décodage sont signalées, sans produire de résultat de substitution.
Les moteurs
visent surtout la réduction de bruit : la réverbération forte, la saturation et
les voix qui se chevauchent restent des cas difficiles.

RNNoise traite des blocs de 10 ms. DeepFilterNet utilise son CLI native, qui peut
charger tout le fichier en mémoire ; sa consommation mémoire n'est pas encore
mesurée. VoxRefine ajoute une fin silencieuse puis retire le délai et l'excédent
pour ne pas couper la fin de l'enregistrement. Aucun moteur n'est découpé en
segments indépendants.

### Régler le nettoyage DeepFilterNet

```sh
python3 -m voxrefine clean corpus/voice.wav results/voice-doux.wav \
  --engine deepfilter --deep-filter .tools/deep-filter \
  --attenuation-limit-db 12
```

`--attenuation-limit-db` expose le réglage natif de DeepFilterNet, entre 0 et
100 dB. Essayer 12 dB pour un nettoyage doux ou 20 dB pour davantage de réduction.
Une valeur plus basse conserve davantage du signal original, mais aussi du bruit.
Ce n'est pas une garantie de préserver chaque son faible : écouter le résultat.
0 désactive la réduction de bruit du moteur, sans garantir une copie bit à bit.
100 correspond au traitement actuel sans limite d'atténuation ; c'est toujours
la valeur par défaut. RNNoise ne propose pas ce réglage dans VoxRefine.

## Comparer les moteurs

Placer les enregistrements dans `corpus/`, puis adapter `examples/corpus.json`.
Les chemins audio sont relatifs au manifeste. Chaque entrée indique un
identifiant, une catégorie et les droits d'utilisation. Le champ `rights` est
une déclaration, pas une vérification juridique.

```sh
python3 -m voxrefine benchmark examples/corpus.json \
  --output results/comparison-01 \
  --deep-filter .tools/deep-filter \
  --rnnoise-library .tools/librnnoise.so
```

Le répertoire de sortie doit être nouveau. Chaque entrée est préparée une seule
fois pour tous les moteurs. Il contient les références `ID-input.wav`, les
fichiers nettoyés et `report.json` : empreintes des entrées originales et
préparées, sorties et moteurs, versions,
durées, temps de traitement, facteur temps réel, niveau RMS, crête et nombre
d'échantillons à pleine échelle. Un facteur temps réel inférieur à 1 signifie
un traitement plus rapide que la durée de l'audio sur la machine utilisée.
Le temps inclut l'initialisation et les entrées/sorties, mais pas la conversion
initiale. Les niveaux et durées d'entrée sont mesurés sur la référence préparée.
L'option `--attenuation-limit-db` fonctionne aussi pour le benchmark : elle
s'applique uniquement à DeepFilterNet et est enregistrée dans son identité
dans le rapport. Pour comparer plusieurs valeurs, lancer des benchmarks dans
des répertoires de sortie distincts. Une sélection RNNoise seule refuse l'option.

Le rapport reste marqué `running` en cas d'interruption, `failed` lors d'une
erreur de traitement et `complete` uniquement à la fin. Une erreur arrête la
comparaison : aucune sortie manquante n'est remplacée par l'original.

Ces mesures **ne classent pas la qualité**. RMS n'est pas une mesure LUFS et
une baisse de volume n'est pas une preuve de nettoyage réussi.

Pour l'écoute, constituer 30 à 50 extraits autorisés : voix françaises variées,
ventilateur, rue, pièce réverbérante, téléphone et voix déjà propre. Comparer
original et sorties à volume comparable, dans un ordre aléatoire. Noter la
préservation des mots et du timbre, le bruit restant et les artefacts ; conserver
ces observations séparément du rapport technique.

`corpus/`, `results/`, les fichiers audio et `.tools/` sont ignorés par Git.
Ne pas publier d'enregistrements privés ou sans autorisation.

## Tests

```sh
python3 -m unittest discover -v

VOXREFINE_DEEP_FILTER="$PWD/.tools/deep-filter" \
VOXREFINE_RNNOISE="$PWD/.tools/librnnoise.so" \
python3 -m unittest discover -v
```

La seconde commande active les essais réels des moteurs, notamment les fichiers
plus courts qu'un bloc et les longueurs non multiples de 480 échantillons.
La CI exécute aussi ces essais.

## Versions et licences

Les essais restent locaux. Un jalon exploitable et vérifié fait l'objet d'un
commit et d'un tag `vMAJOR.MINOR.PATCH`. Le push attend les essais locaux et
l'accord du mainteneur. Pendant la série `0.x`, l'API et les commandes peuvent
encore changer.

VoxRefine est sous licence MIT. Les moteurs gardent leurs propres licences :
[DeepFilterNet](https://github.com/Rikorose/DeepFilterNet/tree/v0.5.6)
(MIT ou Apache-2.0 pour le code) et
[RNNoise](https://github.com/xiph/rnnoise/blob/v0.1/COPYING)
(BSD-3-Clause). Aucun binaire, poids ou enregistrement tiers n'est redistribué
dans ce dépôt. Avant de les embarquer dans une application, vérifier également
les conditions des poids retenus et joindre les notices requises.
