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

Le pipeline expérimental **Studio** ajoute AP-BWE (restauration de bande
passante 16 → 48 kHz) avant DeepFilterNet 0.5.6. Il fonctionne hors ligne et
ne revendique ni traitement micro temps réel ni dé-réverbération dédiée. Un
batch contrôlé sur 11 lecteurs de quatre œuvres montre que DeepFilterNet seul
préserve mieux la référence vocale connue dans les conditions testées; AP-BWE
reste donc une étape optionnelle de restauration, pas un traitement universel.
Protocole, limites et résultats : `docs/benchmarking/controlled-noise-v1.md`.

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

### Essayer GTCRN (CPU, expérimental)

GTCRN nécessite les extras `numpy` et ONNX Runtime ainsi qu'un modèle ONNX
streaming GTCRN fourni localement. VoxRefine ne télécharge aucun poids :

```sh
python3 -m pip install '.[gtcrn]'
python3 -m voxrefine clean corpus/voice.wav results/voice-gtcrn.wav \
  --engine gtcrn --model /chemin/vers/gtcrn_simple.onnx
```

Le modèle traite du mono 16 kHz float32. Un fichier stéréo est refusé; pour
autoriser son downmix, ajouter `--channel-policy downmix`. Par défaut, le WAV
reprend la fréquence source (`--output-rate source`); `--output-rate 16000`
garde le taux du modèle. L'inférence utilise exclusivement `CPUExecutionProvider`,
avec une session et des caches par fichier. Le frontend streaming utilise FFT
512, hop 256 et sqrt-Hann comme l'exemple officiel. La parité sur la fixture
upstream et les seuils de régression sont documentés dans
`docs/benchmarking/gtcrn-upstream-validation.md`. La sortie garde
une bande utile d'environ 8 kHz même si le fichier final est rééchantillonné à
44,1 ou 48 kHz. Ce backend n'ajoute pas encore de séparation ni de curseurs
voix/musique/bruit.

Pour créer un rapport technique GTCRN indépendant des anciens rapports :

```sh
python3 -m pip install '.[gtcrn,bench]'
python3 -m voxrefine benchmark examples/corpus.json \
  --output results/gtcrn-benchmark-01 --engines gtcrn \
  --model /chemin/vers/gtcrn_simple.onnx --output-rate source
```

Ce rapport versionné inclut hash du modèle, runtime/provider CPU, p50/p95/p99
par frame, RTF, RSS et CPU quand `psutil` est disponible. Les résultats sont
techniques; ils ne prédisent pas la préférence d’écoute ni la latence complète
d’un futur chemin microphone.

### Préparer une écoute à l’aveugle

Une fois les candidats générés, décris-les dans un manifeste JSON avec un id,
un nom de moteur et un chemin relatif pour chaque sortie, puis lance :

```sh
python3 -m voxrefine blind-listen results/experiment/candidates.json \
  --output results/experiment/listening-01 --seed 17
```

Les copies d’écoute sont randomisées, recadrées à la durée commune par clip,
normalisées à -20 LUFS / -1,5 dBTP, en mono 48 kHz PCM 24 bits. Les fichiers
bruts ne sont pas modifiés. Écoute uniquement le dossier `blind/`; garde
`key/reveal.json` séparé jusqu’à ce que tu aies noté tes préférences. Détails et
format du manifeste : `docs/benchmarking/blind-listening.md`.

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

### Pipeline Studio AP-BWE → DeepFilterNet (expérimental)

AP-BWE et ses poids ne sont pas téléchargés par VoxRefine. Préparer un
environnement Python isolé avec PyTorch et Torchaudio **de versions et de
builds compatibles** (CPU ou CUDA selon la machine), puis récupérer le dépôt
amont à la révision évaluée et le checkpoint 16 → 48 kHz depuis une source
autorisée. Les empreintes attendues par ce prototype sont consignées dans le
rapport; le checkpoint fait environ 119 MB. Le code et la licence des poids
amont sont MIT. Garder le checkout et les poids hors du dépôt VoxRefine.

```sh
git clone https://github.com/yxlu-0102/AP-BWE.git .tools/ap-bwe-src
git -C .tools/ap-bwe-src checkout 751710f22404c27e5bcc983248f8b856a04b8422
python3 -m pip install .
python3 -m voxrefine studio input.wav results/studio.wav \
  --ap-bwe-source .tools/ap-bwe-src \
  --checkpoint .tools/ap-bwe/g_01000000.pt \
  --deep-filter .tools/deepfilternet/deep-filter \
  --device cpu --threads 4 --channel-policy reject \
  --report results/studio-report.json
```

Pour plusieurs sources, utiliser un manifeste au format de `examples/corpus.json`
et un nouveau dossier de sortie :

```sh
python3 -m voxrefine studio-batch examples/corpus.json \
  --output results/studio-batch-01 \
  --ap-bwe-source .tools/ap-bwe-src \
  --checkpoint .tools/ap-bwe/g_01000000.pt \
  --deep-filter .tools/deepfilternet/deep-filter \
  --device cpu --threads 4 --channel-policy reject
```

Le batch charge les deux moteurs une fois, écrit `audio/` et un `report.json`
avec les hashes des entrées/poids, versions, durée, temps par étape et facteur
temps réel (RTF < 1 signifie plus rapide que la durée du fichier). `reject`
refuse les entrées stéréo; `downmix` mélange les canaux et `first` conserve le
canal gauche. Ces options sont explicites, car elles sonnent différemment.
`--chunk-frames` limite les activations AP-BWE; le signal STFT complet reste en
mémoire. Le prototype est hors ligne, pas un flux causal. Vérifier le RTF, la
RAM/VRAM et les rendus avant d'envisager un usage temps réel, en particulier
sur une GTX 10xx.

Les commandes exigent le code source et le checkpoint présents localement;
elles n'installent pas les dépendances PyTorch à la volée et ne récupèrent pas
de poids. Installer PyTorch/Torchaudio ensemble selon le backend choisi. Aucun
score technique contre Adobe n'est calculé par ce runner.

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

### Façonnage tonal après Resemble (expérimental)

Le preset `--tone soft-edges` est une option locale pour atténuer le sous-grave
et les aigus après le rendu Resemble : −3 dB sous 100 Hz et −2,5 dB au-dessus
de 3,5 kHz, en plus du ton C. Il n'applique pas de gain de compensation. Le
défaut reste `C`, car les mesures sur trois voix améliorent le spectre de deux
références propres et le dégradent sur une. Sur un extrait bruité apparié à
Adobe v2, l'erreur de courbe spectrale lissée passe de 3,92 à 3,02 dB; une
coupe large sous 200 Hz l'aggrave. Ces valeurs sont descriptives et ne prouvent
pas une qualité perceptuelle égale à Adobe.

```sh
python3 -m voxrefine studio-resemble input.wav output-soft-edges.wav \
  --model-python .tools/resemble-venv/bin/python \
  --upstream .tools/resemble-enhance-src \
  --model-dir results/resemble-enhance-01/model/enhancer_stage2 \
  --device auto --nfe 64 --chunk-seconds 3 --tone soft-edges
```

Protocole, définitions des mesures, empreintes, figures et CSV :
[`adobe-tone-shaping-study-2026-10-09.md`](docs/benchmarking/adobe-tone-shaping-study-2026-10-09.md).
Le sample source LibriVox est marqué Public Domain Mark (vérifier la juridiction);
le rendu Adobe v2 et tous les fichiers audio restent locaux et ne sont pas
publiés dans le dépôt.

### Adoucissement des pics après traitement audio

Le preset `studio-resemble` utilise maintenant un compresseur doux (seuil
−16 dBFS, ratio 1,5:1, genou 6 dB), puis une atténuation de sortie de −2 dB.
Sur trois voix bruitées, les crêtes vocales p99 baissent de 2,0 à 4,8 dB; le
changement est surtout marqué sur l'extrait dont les pics étaient les plus
forts. Ce traitement régularise le niveau, mais ne répare pas à lui seul les
artefacts du modèle. Le mode `--dynamics off` et `--output-gain-db 0` permettent
de comparer sans ces étapes. Mesures, graphiques, commande reproductible et
limites : [`post-denoise-finish-2026-10-09.md`](docs/benchmarking/post-denoise-finish-2026-10-09.md).
