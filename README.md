# kyone

Utilitaire en ligne de commande pour **améliorer un enregistrement vocal** : il retire le bruit de fond et la réverbération de la pièce, rééquilibre le timbre, atténue les sifflantes, uniformise le volume et amène le fichier au niveau sonore d'un podcast. C'est une réécriture en Go de [VoxRefine](https://github.com/clem-pnt/VoxRefine), avec pour cible l'expérience « un clic » d'[Adobe Podcast Enhance](https://podcast.adobe.com/fr/enhance).

```bash
kyone entretien.mp3                       # écrit entretien.enhanced.wav à côté
kyone clean entretien.mp3 --profile studio # enhancement complet : bruit, timbre « studio », voix présente
kyone live --profile studio               # le micro, en direct, vers un micro virtuel (Discord, Teams…)
```

## Ce qui change par rapport à VoxRefine

|  | VoxRefine (Python) | kyone (Go) |
|---|---|---|
| Débruitage | DeepFilterNet ou RNNoise, binaires externes à installer | Moteur DSP intégré, **un seul binaire, aucune dépendance à installer** |
| Dé-réverbération | non | oui |
| Tonalité, de-esser, compresseur | EQ optionnel | EQ + de-esser + compresseur |
| Niveau sonore | pas de normalisation | normalisation EBU R128 (LUFS) + limiteur true-peak |
| Formats d'entrée | WAV/MP3/M4A (via FFmpeg) | WAV et MP3 en natif ; M4A, AAC, FLAC, OGG… via FFmpeg s'il est installé |
| Sortie | WAV 48 kHz PCM16, jamais d'écrasement | idem (+ 24 bits en option) |
| Plateforme | Linux x86_64 testé | Go pur, portable (testé uniquement sur macOS arm64) |

**Ce que kyone n'est pas** : ce n'est pas un réseau de neurones. Adobe Enhance et DeepFilterNet reconstruisent la voix avec un modèle appris ; kyone utilise du traitement du signal classique (estimateur spectral OM-LSA), très bon pour le bruit stationnaire, le souffle, le bourdonnement du secteur et l'écho de pièce, moins pour les bruits brefs (porte, clavier) ou une voix noyée dans le bruit. Les résultats ont été vérifiés sur des signaux synthétiques et de la synthèse vocale bruitée, **pas encore sur de vraies voix enregistrées** : écoutez le résultat avant de publier.

## Installation

Go 1.24 ou plus récent.

```bash
go build -o kyone ./cmd/kyone      # ou : go install ./cmd/kyone
```

**Windows et macOS.** Le traitement de fichiers est du Go pur : `CGO_ENABLED=0 GOOS=windows go build ./cmd/kyone` (ou `GOOS=darwin`) fonctionne depuis n'importe quel système, mais le mode direct y est alors désactivé (message explicite). Le **mode direct** utilise miniaudio (WASAPI sous Windows, Core Audio sous macOS) via cgo : il faut compiler sur chaque système avec un compilateur C (outils Xcode sur macOS, [MinGW-w64](https://www.mingw-w64.org/) ou MSYS2 sur Windows). `.github/workflows/build.yml` fait ces compilations (macOS Apple Silicon et Intel, Windows) à chaque push.

## Utilisation

```bash
kyone clean voix.wav                       # voix.enhanced.wav
kyone clean voix.m4a propre.wav            # nom de sortie explicite
kyone clean voix.wav --profile strong      # débruitage plus agressif
kyone clean voix.wav --profile natural     # plus doux, garde l'ambiance de la pièce
kyone clean voix.wav --attenuation-limit-db 30 --dereverb 70
kyone clean voix.wav --bass-db 2 --treble-db -1 --target-lufs -19
kyone clean voix.wav --no-compress --no-deess --no-normalize   # débruitage seul
kyone analyze voix.wav                     # niveaux, pics, bruit de fond, SNR estimé
kyone analyze voix.wav --json
kyone profiles
```

Le fichier de sortie n'écrase jamais un fichier existant sans `--force`. Les options peuvent être placées avant ou après les fichiers.

### Profils de débruitage

| Profil | Réduction max. (entre les mots) | Plancher dans la parole | Dé-réverb. | Correction de timbre | Clarté | Usage |
|---|---|---|---|---|---|---|
| `natural` | 12 dB | — | 15 | 0 | 0 | doux, garde l'ambiance et le timbre d'origine |
| `balanced` (défaut) | 24 dB | 8 dB | 35 | 50 | 30 | voix claire, fond calme, légère correction de timbre |
| `strong` | 40 dB | 10 dB | 60 | 70 | 40 | voix « studio sec », peut sonner légèrement traité |
| `studio` | 40 dB | 8 dB | 20 | 100 | 60 | enhancement complet, le plus proche d'Adobe Enhance |

Le **plancher dans la parole** est la protection de la voix : entre les mots le bruit est ramené jusqu'à la réduction maximale, mais tant que quelqu'un parle les détails faibles (souffle, consonnes douces, énergie entre les harmoniques) ne sont atténués que de cette valeur. Sans cela, un débruiteur spectral laisse des « trous » dans la voix et un rendu aquatique.

`--attenuation-limit-db N` (0–100) remplace `--profile` : plus la valeur est basse, plus on conserve de signal d'origine (et de bruit) ; `0` désactive le débruitage. Contrairement à VoxRefine, la valeur 100 dB n'est pas le profil par défaut : sans plancher raisonnable, un débruiteur spectral coupe le fond à zéro entre les mots, ce qui sonne artificiel.

### Toutes les options de `clean`

| Option | Défaut | Rôle |
|---|---|---|
| `--profile` | `balanced` | `natural`, `balanced`, `strong`, `studio` |
| `--attenuation-limit-db` | — | réduction max. personnalisée (exclusif avec `--profile`) |
| `--speech-floor-db` | selon profil | réduction max. *pendant* la parole, 0–100 dB (protection de la voix) |
| `--tone-match` | selon profil | correction **automatique** du timbre vers une courbe « studio », 0–100 (0 = off) |
| `--clarity` | selon profil | voix plus présente et ouverte (présence + aigus, actifs seulement pendant la parole), 0–100 |
| `--dereverb` | selon profil | suppression de la réverbération, 0–100 (0 = off) |
| `--highpass` | 80 | filtre passe-haut anti-grondement, en Hz (0 = off) |
| `--bass-db` / `--treble-db` | 0 | shelf graves à 150 Hz / aigus à 4 kHz, ±12 dB |
| `--presence-db` | 1.5 | cloche de présence à 3,2 kHz, ±12 dB |
| `--eq` | — | bande d'EQ libre `type:Hz:gain[:Q]` (`low`, `high`, `peak`, ±24 dB), répétable : `--eq low:200:+9 --eq high:6300:+3` |
| `--no-deess` | | désactive l'atténuation des sifflantes |
| `--no-compress` | | désactive le compresseur de niveau |
| `--no-normalize` | | conserve le volume, protège seulement contre l'écrêtage |
| `--target-lufs` | −16 | niveau cible (−16 pour un podcast, −19 pour du mono « broadcast » doux) |
| `--ceiling-db` | −1 | plafond de true-peak en dBTP |
| `--mono` | | mixe le résultat en mono |
| `--bits` | 16 | profondeur de sortie : 16 ou 24 |
| `--engine` | `native` | `native` (intégré) ou `deepfilter` (neuronal, à installer, voir plus haut) |
| `--deepfilter-bin` / `--deepfilter-lib` / `--deepfilter-model` | — | chemins de `deep-filter`, de `libdf` et du modèle |
| `--deepfilter-postfilter` | | post-filtre de DeepFilterNet |
| `--ffmpeg` | `PATH` | chemin de ffmpeg pour les formats hors WAV/MP3 |
| `--force` | | autorise l'écrasement du fichier de sortie |
| `-q` | | n'affiche que les erreurs |

## Mode direct : améliorer son micro en partie vocale

`kyone live` traite le micro en temps réel et envoie le résultat vers un **micro virtuel** ; Discord, Teams, Zoom, etc. choisissent ce micro virtuel à la place du vrai. Même moteur que pour les fichiers, adapté au direct (voir plus bas).

**1. Installer un micro virtuel** (kyone ne peut pas en créer un : ce sont des pilotes système) :

| Système | Micro virtuel | Installation |
|---|---|---|
| macOS | [BlackHole](https://existential.audio/blackhole/) (libre) | `brew install blackhole-2ch` |
| Windows | [VB-CABLE](https://vb-audio.com/Cable/) (gratuit) | installeur du site, puis redémarrer |

**2. Lancer kyone** — il trouve tout seul le micro virtuel :

```bash
kyone devices                      # liste micros, sorties et micros virtuels détectés
kyone live --profile studio        # micro par défaut → micro virtuel
kyone live --input "Blue Yeti" --output "CABLE Input" --profile balanced
```

**3. Dans l'application de chat**, choisir comme micro : *BlackHole 2ch* (macOS) ou *CABLE Output* (Windows). Dans Discord, couper la suppression de bruit **Krisp** et le **contrôle automatique du gain** (kyone s'en charge) ; garder l'annulation d'écho.

Pour s'écouter soi-même (au casque !) : `kyone live --monitor`.

Autorisations : macOS demande l'accès au micro pour l'application qui lance `kyone` (Terminal, iTerm…) à la première exécution ; sous Windows, vérifier *Paramètres → Confidentialité → Microphone → applications de bureau*.

Ce que fait le direct, par rapport au traitement de fichiers : le bruit est suivi sur les 2 dernières secondes seulement (au lieu de ±2 s autour de chaque instant) ; la correction de timbre et le niveau s'apprennent pendant que vous parlez, **il faut compter une dizaine de secondes avant que le rendu soit à son meilleur** (la première seconde laisse passer le bruit) ; le niveau est réglé par un contrôle de gain lent (cible `--level-db`, −20 dBFS par défaut, gain limité à ±15 dB, jamais appliqué au silence) au lieu de la normalisation LUFS ; le limiteur a 3 ms d'anticipation. Les clics isolés ne sont pas filtrés (il faudrait de l'anticipation).

| Mesure | Valeur |
|---|---|
| latence de l'algorithme | ≈ 24 ms (+ ≈ 60 ms de tampons audio avec `--block-ms 10`, réglable) |
| charge CPU | ≈ 3 % d'un cœur (Apple Silicon) |
| rendu après apprentissage, sur `test.wav` répété (écart à la référence, parole) | timbre 2,5 dB, spectrogramme 4,3 dB, contre 1,9 et 4,1 dB hors ligne |

`kyone live --file voix.wav` fait passer un fichier par le moteur direct, par blocs de la taille du micro, et écrit `voix.live.wav` : on peut ainsi écouter et mesurer le rendu du direct sans matériel audio.

## Moteur neuronal DeepFilterNet (optionnel)

`--engine deepfilter` remplace le débruiteur intégré de kyone par [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) (réseau de neurones, MIT ou Apache-2.0), pour les fichiers (`clean`) comme en direct (`live`). Le reste de la chaîne (correction de timbre, de-esser, clarté, compresseur, niveau, limiteur) est inchangé. kyone ne l'embarque pas : il faut le fournir.

```bash
kyone clean bruyant.wav --engine deepfilter --profile studio
kyone live --engine deepfilter --profile studio
```

**Quand l'utiliser.** Les mesures sur une voix avec des clics de clavier (≈ 4 par seconde, 8 dB sous la voix) montrent l'écart avec le moteur intégré, qui ne peut pas suivre des bruits aussi brefs :

| Distance à la voix propre (spectrogramme mel, parole ; plus bas = mieux) | moteur intégré | DeepFilterNet |
|---|---|---|
| fichier | 13,8 dB | **5,9 dB** |
| direct | 13,9 dB | **7,4 dB** |

À l'inverse, sur un bruit stationnaire (souffle, ventilateur), comme celui de `test.wav`, le moteur intégré fait aussi bien ou mieux : 4,1 dB contre 5,5 dB d'écart à la référence Adobe, et un fond plus profond. D'où une règle simple : bruit constant → moteur intégré ; clavier, voix de fond, bruits irréguliers → DeepFilterNet.

**Direct : latence et charge** (mesurées, macOS Apple Silicon, blocs de 10 ms) :

| | moteur intégré | DeepFilterNet « low latency » | DeepFilterNet standard |
|---|---|---|---|
| latence de l'algorithme | ≈ 24 ms | **≈ 13 ms** (10 ms + limiteur) | ≈ 33 ms |
| charge CPU | ≈ 3 % d'un cœur | ≈ 18 % | ≈ 10 % |

La latence est mesurée par kyone au démarrage sur une voix synthétique et affichée par `kyone live`. Les blocs de l'audio sont alignés sur les trames du réseau (10 ms), ce qui évite toute latence d'alignement. Avec DeepFilterNet en direct, la dé-réverbération de kyone est désactivée (elle demanderait sa propre analyse spectrale et son retard) ; la correction de timbre reste active, sous forme d'un filtre à phase minimale qui n'ajoute aucun délai.

**Installation** (une fois ; on peut mettre les fichiers à côté de `kyone`, dans un dossier `deepfilter/` à côté, ou donner leurs chemins) :

| Pour | Il faut | Où |
|---|---|---|
| les fichiers | l'outil `deep-filter` | [releases](https://github.com/Rikorose/DeepFilterNet/releases) : `deep-filter-0.5.6-aarch64-apple-darwin` (macOS Apple Silicon), `…-x86_64-apple-darwin` (macOS Intel), `…-x86_64-pc-windows-msvc.exe` (Windows). Renommer en `deep-filter` / `deep-filter.exe`. Option `--deepfilter-bin`, variable `KYONE_DEEP_FILTER`, ou dans le `PATH`. |
| le direct | la bibliothèque C `libdf` | à compiler (Rust requis) : `git clone --depth 1 https://github.com/Rikorose/DeepFilterNet` puis `cargo build --release -p deep_filter --lib --features capi` → `target/release/libdf.dylib` (macOS) ou `df.dll` (Windows). Option `--deepfilter-lib`, variable `KYONE_DF_LIB`. |
| le direct | un modèle | `models/DeepFilterNet3_ll_onnx.tar.gz` du dépôt (36 Mo, « low latency »). Option `--deepfilter-model`, variable `KYONE_DF_MODEL`. `DeepFilterNet3_onnx.tar.gz` (8 Mo) est plus léger mais ajoute 20 ms de latence. |

Le chargement de la bibliothèque se fait à l'exécution, sans cgo (via [purego](https://github.com/ebitengine/purego)), sur macOS et Windows ; seul `kyone live` a besoin de cgo, pour l'audio. `--deepfilter-postfilter` active le post-filtre du modèle (bruit plus réduit, un peu plus d'artefacts). `--attenuation-limit-db` fixe la réduction maximale transmise au réseau (profil `studio` : 40 dB).

Validé ici : l'outil `deep-filter` 0.5.6 et la bibliothèque (compilée depuis le dépôt) sur macOS Apple Silicon, en fichier, en direct simulé (`kyone live --file`) et par les tests (`KYONE_DF_BIN`, `KYONE_DF_LIB`, `KYONE_DF_MODEL` les activent). Non validé : la compilation de la bibliothèque et l'exécution sous Windows, et le branchement sur un vrai micro.

## Chaîne de traitement

Tout est ramené à 48 kHz (rééchantillonneur sinc/Kaiser), mono ou stéréo ; la stéréo est traitée avec des gains **liés** pour ne pas déplacer l'image.

1. **Passe-haut** Butterworth 4ᵉ ordre (80 Hz) : grondements, souffle de manipulation.
2. **Débruitage + dé-réverbération** (`internal/enhance/spectral.go`), STFT 1024/256 :
   - le bruit est suivi dans le temps en moyennant les ~10 % de trames les plus calmes dans une fenêtre de ±2 s autour de chaque point (le fichier entier étant disponible, la fenêtre est centrée : pas de retard ni de biais comme avec un suiveur causal) ; ça capte aussi bien le souffle que le 50 Hz et ses harmoniques ;
   - gain **OM-LSA** (log-spectral amplitude optimisé) avec rapport signal/bruit « decision-directed » et probabilité de présence de parole : la voix est protégée, le bruit entre les mots est ramené au plancher choisi ;
   - **protection de la voix** : un détecteur de parole à l'échelle de la trame (rapport signal/bruit large bande, attaque rapide, relâchement ≈ 175 ms) fait glisser le plancher du gain de la réduction maximale (entre les mots) vers `--speech-floor-db` (pendant la parole) ;
   - sous 330 Hz le plancher dans la parole est plus profond de 12 dB : la parole y est forte et harmonique, ce qui s'y trouve entre les harmoniques est surtout du grondement, que la correction de timbre relèverait sinon ;
   - les **clics isolés** (bruits de bouche, clic de souris : 1 à 3 trames fortes sans rien de relevé 40 à 130 ms avant et après) sont ramenés au plancher le plus profond ; un mot, même court, n'est jamais pris pour un clic ;
   - la réverbération tardive est modélisée comme une copie retardée (40 ms) et décroissante (RT60 ≈ 0,45 s) de la puissance déjà nettoyée, soustraite dans le même estimateur.
3. **Correction de timbre automatique** (`tonematch.go`) : mesure le spectre de la parole par tiers d'octave (avant et après débruitage), le compare à une courbe « studio » lissée et applique la différence, bornée (+15 dB graves, +9 dB présence, +4 dB aigus, −12 dB), sans jamais relever une bande que la voix n'occupe pas ou qui se noyait dans le bruit. Corrige un micro pauvre en basses ou trop sourd sans réglage manuel.
4. **De-esser** : croisement Linkwitz-Riley à 5,5 kHz, bande haute atténuée (jusqu'à 7 dB) quand elle domine l'énergie à court terme.
5. **Tonalité manuelle** : shelf graves, cloche de présence, shelf aigus, bandes `--eq`.
6. **Clarté** (`clarity.go`) : cloche de présence à 3,2 kHz (jusqu'à +5 dB) et relèvement au-dessus de 6,5 kHz, tous deux commandés par l'activité de la parole ; les aigus ne sont relevés que sur les passages voisés pour ne pas rendre les « s » plus durs.
7. **Compresseur de niveau** : 2,5:1, genou doux, seuil placé automatiquement à partir du niveau de parole mesuré, détection liée entre canaux.
8. **Normalisation EBU R128 / ITU-R BS.1770-4** vers `--target-lufs` (loudness intégré avec double gating), puis **limiteur true-peak** (sur-échantillonnage ×4, look-ahead 10 ms). Le gain est ajusté itérativement pour tenir compte de ce que le limiteur retire ; si les pics empêchent d'atteindre la cible sans plus de 12 dB de limitation, le résultat le plus fort acceptable est conservé.
9. **Écriture** WAV PCM avec dither TPDF (déterministe), via un fichier temporaire renommé : pas de fichier tronqué en cas d'erreur.

Le programme n'introduit aucune latence : le fichier de sortie a exactement la durée de l'entrée.

À titre indicatif (Apple Silicon, un cœur sauf le suivi de bruit) : environ 30× le temps réel, et de l'ordre de 4 × 4 octets par échantillon en mémoire (≈ 2,7 Go pour une heure de mono 48 kHz).

## Architecture

```
cmd/kyone/            CLI : clean, live, devices, analyze, profiles
internal/audio/       Buffer, lecture/écriture WAV, décodage MP3/FFmpeg, rééchantillonnage
internal/dsp/         FFT, biquads (RBJ), Butterworth, true-peak ×4
internal/loudness/    mesures BS.1770 : LUFS, LRA, true-peak, bruit de fond, SNR
internal/deepfilter/  DeepFilterNet : outil `deep-filter` (fichiers) et bibliothèque `libdf` chargée sans cgo (direct)
internal/device/      micro → traitement → sortie (miniaudio via cgo ; sans cgo, un stub)
internal/enhance/     options/profils, estimateur spectral, de-esser, tonalité, dynamique, pipeline, moteur direct (`live*.go`)
tools/                aides de développement (voir ci-dessous)
```

## Développement

```bash
go vet ./... && go test ./...
```

Les tests couvrent la FFT, les filtres, le WAV (y compris flux et fichiers tronqués), le rééchantillonnage, les mesures LUFS/true-peak (valeurs de référence BS.1770), chaque étage du pipeline sur des signaux synthétiques de référence et la CLI de bout en bout.

Quatre utilitaires dans `tools/` servent à évaluer la qualité sans écoute :

```bash
# dégrader une voix propre : bruit à 14 dB de SNR, ronflement −50 dBFS, réverbération 0,6 s
go run ./tools/degrade -snr 14 -hum -50 -rt60 0.6 propre.wav bruite.wav
kyone clean bruite.wav sortie.wav --no-normalize --no-compress --no-deess --presence-db 0 --highpass 0
# SI-SDR, niveau résiduel dans les pauses
go run ./tools/compare propre.wav bruite.wav sortie.wav
# spectrogrammes empilés (PNG) pour comparer avant/après
go run ./tools/spectrogram comparaison.png propre.wav bruite.wav sortie.wav
# comparer des sorties à un fichier de référence (voir la section suivante)
go run ./tools/match -fit entree.wav reference.wav sortie.wav
```

### Se rapprocher d'un fichier de référence

`tools/match` sert quand on dispose d'un fichier déjà traité (par exemple par Adobe Podcast Enhance) à partir du même enregistrement et qu'on veut s'en approcher. Il affiche, par tiers d'octave, le niveau dans la parole et dans les pauses de l'entrée, de la référence et des candidats, puis deux distances à la référence : l'écart des niveaux par bande (timbre) et l'écart de spectrogrammes mel trame par trame (insensible à la phase, donc valable face à un résultat de réseau de neurones). Avec `-fit`, il cherche l'EQ (shelf grave, shelf aigu, cloche) qui rapproche le premier candidat de la référence et l'affiche sous forme d'options `--eq`.

Exemple sur le fichier `test.wav` du dépôt de travail, dont la référence est `test-optimisé-v2.wav`. Une seule commande suffit, sans réglage manuel :

```bash
kyone clean test.wav test-kyone-studio.wav --profile studio --target-lufs -28.3
```

(`--target-lufs -28.3` ne sert qu'à régler le volume sur celui de la référence ; sans lui, le fichier est amené à −16 LUFS, le niveau d'un podcast.)

| Écart à la référence | `strong` seul | recette manuelle (EQ ajustée à la main) | `--profile studio` |
|---|---|---|---|
| timbre dans la parole (dB, bandes ⅓ d'octave) | 6,6 | 1,3 | 1,9 |
| timbre dans les pauses (dB) | 11,6 | 8,4 | **4,0** |
| spectrogramme mel, parole (dB) | 10,5 | 4,0 | 4,1 |
| spectrogramme mel, pauses (dB) | 9,1 | 7,3 | **6,1** |
| dynamique des graves (pente référence/candidat, idéal 1,00) | — | 1,31 | **1,04** |
| bruit de fond (dBFS, référence : −66,4) | −63,9 | −66,4 | −70,4 |

La courbe « studio » vers laquelle `--tone-match` corrige le timbre vient de la référence de cet exemple (rendu Adobe Podcast Enhance d'un enregistrement fait avec un micro d'ordinateur portable) : elle donne donc, sur ce fichier, un résultat très proche de la référence par construction. Sur d'autres voix, les corrections sont bornées et plus modestes (quelques dB) ; l'option `--tone-match` règle leur force, `--tone-match 0` les désactive. Pour s'approcher d'une autre référence, `tools/match -fit` calcule l'EQ à appliquer avec `--eq`.

## Limites connues

- Le moteur intégré n'est pas un modèle appris : les bruits non stationnaires (voix de fond, claviers, portes) ne sont pas supprimés, hormis les clics isolés. Pour ceux-là, `--engine deepfilter` (voir plus haut).
- La correction de timbre automatique vise une courbe tirée d'un seul exemple (une voix, un traitement) ; elle est bornée et progressive, mais sur une voix très différente elle peut ne pas convenir : réglez `--tone-match` ou coupez-la.
- La clarté (`--clarity`) n'a été validée que par des mesures (+1 à +2 dB de présence en moyenne sur la parole, rien dans les pauses), pas à l'écoute.
- Le dé-réverbérateur suppose une réverbération tardive diffuse et réduit un peu le niveau des voyelles tenues ; mettez `--dereverb 0` sur un enregistrement déjà sec.
- Les métriques sont des repères, pas un classement de qualité perceptive : validez à l'écoute.
- MP3 décodé avec `go-mp3` (testé sur un fichier d'exemple) ; les autres formats passent par FFmpeg, non testé ici faute d'installation.
- Le mode direct n'a été testé ici que sans matériel : moteur (blocs de toute taille, latence, débit, apprentissage), liste des périphériques et choix par nom sur macOS, simulation par fichier. **La capture micro → micro virtuel n'a pas été essayée en réel, ni sous Windows** (aucune machine Windows, pas de micro virtuel installé) ; la compilation Windows avec cgo n'a pas été vérifiée.
- En direct, le débruitage intégré (causal) est moins bon que hors ligne et met une dizaine de secondes à s'installer ; sur les bruits irréguliers, utiliser `--engine deepfilter`.
- DeepFilterNet supprime le bruit, il ne « recrée » pas la voix comme Adobe : sur un bruit constant, il laisse un fond plus haut que le moteur intégré. La recréation générative (type Resemble Enhance) n'est pas intégrée.
- Le fichier est traité en mémoire (pas de streaming) ; la sortie WAV est limitée à 4 Gio.
