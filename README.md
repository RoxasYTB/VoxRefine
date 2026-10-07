# VoxRefine

Nettoyage de voix en local, depuis une commande Python.

VoxRefine nettoie les fichiers localement avec DeepFilterNet et sa protection
de voix, ou permet de comparer explicitement avec RNNoise. Aucun audio n'est
envoyé à un serveur. Les moteurs s'installent séparément ; VoxRefine ne
télécharge rien pendant le traitement.

## État du projet

La version locale `0.3.0.dev0` ajoute des profils de nettoyage DeepFilterNet
et simplifie la validation audio. Elle n'est pas encore publiée.

La version `0.2.0` accepte **WAV, MP3 et M4A**. FFmpeg convertit
les entrées qui le nécessitent en WAV PCM 16 bits, stéréo, 48 kHz.
La sortie est un WAV PCM16 mono ou stéréo à 48 kHz, sans écraser de fichier
existant.
Le nettoyage conserve la durée exacte de l'audio décodé ; les codecs compressés
peuvent ajouter du padding selon leur encodage.
Pas de normalisation automatique, de transcription ou d'interface graphique.

Les deux moteurs ont été exécutés sur du silence et un signal synthétique.
Un premier essai local sur une voix a également donné un retour positif sur
la suppression d'un bruit de voiture en arrière-plan. Ce retour ponctuel ne
remplace pas une comparaison à l'écoute sur un corpus varié. DeepFilterNet
avec protection de voix est le mode par défaut ; cette préférence ne garantit
pas une qualité identique sur tous les enregistrements.
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

Un WAV mono ou stéréo PCM16 à 48 kHz fonctionne sans FFmpeg.
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
python3 -m voxrefine clean corpus/voice.m4a results/voice.wav

python3 -m voxrefine clean corpus/voice.wav results/voice-rnnoise.wav \
  --engine rnnoise --rnnoise-library .tools/librnnoise.so
```

DeepFilterNet utilise toujours la protection de voix : le fichier de sortie
contient un seul rendu protégé, sans sortie standard supplémentaire. C'est le
mode par défaut, avec DeepFilterNet comme moteur. RNNoise sert à détecter la
parole et doit être installé dans `.tools/librnnoise.so`. Les moteurs installés
suivant les instructions ci-dessus sont détectés automatiquement ; les options
`--deep-filter` et `--rnnoise-library` permettent de préciser d'autres chemins.
Pour utiliser le DeepFilterNet standard, passer `--no-protect-voice`.
`--engine rnnoise` sélectionne explicitement RNNoise.

La préparation est automatique et temporaire. Les WAV mono/stéréo PCM16 à 48 kHz
gardent leur format. Pour les autres entrées, VoxRefine convertit en stéréo ;
les sources multicanales sont ramenées à deux canaux. Chaque canal est nettoyé
séparément par le moteur mono, puis recombiné en conservant la stéréo. L'original
reste inchangé. La première piste audio est utilisée si le fichier en contient
plusieurs.
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

Pour commencer avec un traitement moins agressif :

```sh
python3 -m voxrefine clean corpus/voice.wav results/voice-natural.wav \
  --profile natural --no-protect-voice
```

Les profils sont des raccourcis vers le réglage natif :

| Profil | Limite | Usage |
|---|---:|---|
| `natural` | 12 dB | Atténuation plus limitée, davantage de bruit résiduel possible |
| `balanced` | 20 dB | Réduction intermédiaire |
| `strong` | 100 dB | Comportement historique, nettoyage sans limite d'atténuation |

Sans option, le profil reste `strong`. Le profil `natural` (12 dB) requiert
`--no-protect-voice`, car la protection compare avec le rendu à 12 dB. Les profils
ne modifient pas le
modèle et ne garantissent pas une meilleure qualité sur tous les enregistrements.
Ils fonctionnent aussi avec `benchmark`, et sont réservés à DeepFilterNet.
`--profile` et `--attenuation-limit-db` ne peuvent pas être utilisés ensemble.

Pour choisir une limite précise :

```sh
python3 -m voxrefine clean corpus/voice.wav results/voice-doux.wav \
  --attenuation-limit-db 12 --no-protect-voice
```

`--attenuation-limit-db` expose le réglage natif de DeepFilterNet, entre 0 et
100 dB. Essayer 12 dB pour un nettoyage doux ou 20 dB pour davantage de réduction.
Une valeur plus basse conserve davantage du signal original, mais aussi du bruit.
Ce n'est pas une garantie de préserver chaque son faible : écouter le résultat.
0 désactive la réduction de bruit du moteur, sans garantir une copie bit à bit.
100 correspond au traitement actuel sans limite d'atténuation ; c'est toujours
la valeur par défaut. RNNoise ne propose pas ce réglage dans VoxRefine.

### Égaliser les basses et les aigus

L'égalisation est facultative et désactivée par défaut. Ajuster séparément les
étagères de basses (centrées à 150 Hz) et d'aigus (4 kHz) entre -12 et +12 dB :

```sh
python3 -m voxrefine clean corpus/voice.wav results/voice-plus-claire.wav \
  --bass-db -2 --treble-db 3
```

Une valeur positive renforce la bande, une valeur négative l'atténue. Il n'y a
pas de réglage idéal pour toutes les voix : commencer par de petits ajustements
et écouter. Si l'égalisation ferait dépasser le niveau maximal du WAV, VoxRefine
atténue automatiquement le signal pour éviter l'écrêtage. Ces options marchent
aussi avec `benchmark` et leurs valeurs apparaissent dans le rapport.

## Comparer les moteurs

### Protection de la voix

```sh
python3 -m voxrefine clean corpus/voice.wav results/voice-protected.wav \
  --deep-filter /path/to/deep-filter \
  --rnnoise-library /path/to/librnnoise.so
```

Par défaut, le traitement ajoute une seconde passe DeepFilterNet à 12 dB et utilise les
probabilités de parole de RNNoise sur l'entrée. Lorsque la confiance atteint
0,8 et que le rendu principal est plus de 6 dB sous le rendu doux, VoxRefine
introduit progressivement jusqu'à 75 % du rendu doux. L'attaque dure 20 ms
et le retour au rendu principal 100 ms. Les deux rendus ont le même délai
compensé ; aucun signal brut n'est mélangé directement.

Le réglage principal doit être supérieur à 12 dB. La protection s'applique
systématiquement à DeepFilterNet et fonctionne aussi dans `benchmark`, qui
enregistre les paramètres et l'empreinte du détecteur. Elle nécessite la
bibliothèque RNNoise même si seul DeepFilterNet est sélectionné. Pour comparer
avec le rendu DeepFilterNet standard, utiliser `--no-protect-voice`.

Ce détecteur peut manquer une voix ou confondre du bruit avec de la parole.
L'option peut réintroduire du bruit et ne corrige pas tous les artefacts
robotiques. Elle ne reconstruit pas les mots. Le coût comprend deux rendus
DeepFilterNet et une passe RNNoise ; comparer à l'écoute avant de la retenir.

### Corpus et rapports

Placer les enregistrements dans `corpus/`, puis adapter `examples/corpus.json`.
Les chemins audio sont relatifs au manifeste. Chaque entrée indique un
identifiant, une catégorie et les droits d'utilisation. Le champ `rights` est
une déclaration, pas une vérification juridique.

```sh
python3 -m voxrefine benchmark examples/corpus.json \
  --output results/comparison-01 \
  --deep-filter /path/to/deep-filter \
  --rnnoise-library /path/to/librnnoise.so
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
dans le rapport. Depuis `0.3.0.dev0`, `attenuation_limit_db` est un nombre JSON,
et non une chaîne de caractères. Pour comparer plusieurs valeurs, lancer des benchmarks dans
des répertoires de sortie distincts. Une sélection RNNoise seule refuse l'option.

Le rapport reste marqué `running` en cas d'interruption, `failed` lors d'une
erreur de traitement et `complete` uniquement à la fin. Une erreur arrête la
comparaison : aucune sortie manquante n'est remplacée par l'original.

Ces mesures **ne classent pas la qualité**. RMS n'est pas une mesure LUFS et
une baisse de volume n'est pas une preuve de nettoyage réussi.

Pour l'écoute, constituer 30 à 50 extraits autorisés : voix françaises variées,
ventilateur, rue, pièce réverbérante, téléphone et voix déjà propre. Comparer
original et sortie protégée à volume comparable. Noter la
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
