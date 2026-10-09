# Binding PipeWire natif: compile-check 1.4.2

**Statut :** binding candidat compilé et lié contre les headers PipeWire 1.4.2 et
le runtime PipeWire local 1.4.2. Aucun contexte, stream, périphérique ou
microphone n'a été créé. Ce jalon ne prouve ni le comportement des callbacks sur
un serveur réel, ni la sûreté hard-real-time, ni la latence, ni la qualité audio.

## Périmètre

`native/pipewire_adapter.{h,c}` déclare les callbacks avec les types réels de
`pw_stream_events`, parse `SPA_PARAM_Format` avec
`spa_format_audio_raw_parse` et adapte le contrat interne de framing C11. Le
contexte est fourni et possédé par un futur intégrateur : ses fonctions init
reçoivent un `pw_stream *` déjà créé et ne créent/connectent/détruisent rien.
Le harness ne construit pas de stream. Il appelle la callback réelle `param_changed` avec des POD synthétiques, mais n'invoque pas `process` ou `state_changed`.

Le candidat n'accepte actuellement que :

- format négocié F32, 48 kHz, mono ;
- un seul `spa_data` de type `SPA_DATA_MemPtr` ;
- des données lisibles en capture et inscriptibles en lecture ;
- un stride d'échantillon de 4 octets en capture ;
- des régions contiguës dans `maxsize`, sans wrap-around du chunk ;
- un pointeur aligné sur `float` et une taille multiple de `sizeof(float)`.

Les formats PCM entiers, les buffers MemFd/DMABuf, plusieurs régions, le
resampling, le downmix, les statistiques PipeWire xrun/horodatage, la politique
de séquence/stale/failure et le cycle de vie des streams restent hors de ce
checkpoint. Les buffers vides sont valides. Un hop capture refusé par le ring
est compté; la lecture remplit de zéros les frames manquantes et compte le
shortfall. Les buffers sont soit remis à PipeWire sans consommation via
`pw_stream_return_buffer`, soit recyclés/émis via `pw_stream_queue_buffer`, une
seule fois sur chaque chemin qui a réussi le dequeue.

Les compteurs sont séparés : `format_not_ready`, `unsupported_buffers`,
`malformed_buffers`, `capture_dropped_hops`, `playback_shortfall_frames`,
`no_buffer_callbacks` et `queue_errors` (échecs de queue ou de return). Ils sont
destinés au diagnostic de ce binding, pas à être fusionnés dans les métriques du moteur. Leur lecture/reset
concurrents n'est pas encore une API synchronisée.

Le binding ne publie qu'un indicateur atomique « format exact prêt »; il ne
transporte pas de snapshot de paramètres multiples, car le seul format accepté
est fixe. Une renégociation pendant `RUNNING` n'est pas supportée : le futur
propriétaire doit désactiver/quiescer le stream et les callbacks, refaire la
négociation, puis initialiser un nouveau contexte avant de reprendre. Ce
quiesce reste un contrat d'intégration à vérifier avec le vrai cycle de vie
PipeWire; le booléen atomique seul ne constitue pas une barrière contre un
callback déjà en cours.

La capture emprunte la région délimitée par le chunk, puis copie ses frames
dans l'accumulateur et le ring VoxRefine avant de rendre le buffer. La lecture
borne les frames à `maxsize - offset`, consomme le splitter, remplit le suffixe
de silence et publie la taille/stride résultants dans le chunk. Aucun pointeur
vers `pw_buffer`, `spa_buffer`, `spa_data` ou `spa_chunk` n'est conservé.

## SDK temporaire utilisé sur cette machine

Le runtime PipeWire 1.4.2 était présent; les headers ne l'étaient pas. Les deux
paquets de développement Debian `libpipewire-0.3-dev` et `libspa-0.2-dev` ont
été téléchargés puis extraits dans
`/tmp/voxrefine-pipewire-sdk-1.4.2/sysroot`. Aucun paquet système n'a été
installé. Pour reproduire sur une machine compatible Debian/amd64 :

```sh
mkdir -p /tmp/voxrefine-pipewire-sdk-1.4.2/packages
cd /tmp/voxrefine-pipewire-sdk-1.4.2/packages
apt-get download libpipewire-0.3-dev libspa-0.2-dev
mkdir -p ../sysroot
for package in ./*.deb; do dpkg-deb -x "$package" ../sysroot; done
```

Le test Python résout `pkg-config` exclusivement depuis le sysroot, exige la
version 1.4.2, compile en C11 avec `-Wall -Wextra -Werror -pedantic`, puis lie
les symboles du runtime hôte et vérifie `ldd`. Les includes PipeWire/SPA sont
traités en `-isystem` car leurs headers utilisent des extensions GNU internes;
les diagnostics stricts restent actifs pour le code VoxRefine. Le test requiert
le chemin de runtime Linux amd64 actuellement codé dans le harness; le
packaging multiplateforme n'est pas inclus dans cette tranche.

```sh
VOXREFINE_PIPEWIRE_SYSROOT=/tmp/voxrefine-pipewire-sdk-1.4.2/sysroot \
  .venv/bin/python -m unittest tests.test_pipewire_native -v

VOXREFINE_PIPEWIRE_SYSROOT=/tmp/voxrefine-pipewire-sdk-1.4.2/sysroot \
VOXREFINE_PIPEWIRE_SANITIZERS=address,undefined \
  .venv/bin/python -m unittest tests.test_pipewire_native -v
```

## Ce que vérifie le harness

`native/test_pipewire_adapter.c` utilise des scalaires et un POD audio construit
par SPA pour vérifier le parseur officiel, F32/48k/mono accepté, S16 rejeté,
les callback slots/version, les spans valides/vides, les pointeurs nuls ou mal
alignés, les strides/tailles incorrects, les bornes offset/size, les formats et
les channels non supportés, puis 50 000 combinaisons déterministes de métadonnées
avec vérification des propriétés de bounds. Il appelle aussi
`pw_stream_state_as_string` afin que le lien vers le runtime soit réel.

Il ne crée pas de connexion PipeWire. Le harness exerce la callback réelle `param_changed` sur une renégociation synthétique F32/48 kHz/mono → S16 → F32 et vérifie les lectures acquire du booléen publié. Un audit structurel de `process` vérifie un seul dequeue, une seule mise en file sur le chemin normal, et une restitution immédiate sur chaque retour anticipé post-dequeue; les types mémoire non supportés sont distincts des métadonnées mal formées. Cette vérification n'exécute pas `process` contre un vrai buffer PipeWire. La publication playback efface également les flags SPA précédents.

Il ne crée pas de connexion PipeWire. Les callbacks sont compilés et liés,
mais l'ownership réel des buffers, les quanta fournis en vrai, les flags des ports,
la synchronisation du lifecycle, les xruns et la mesure de service restent à
vérifier avant toute activation. Le callback `process` peut être appelé depuis
un thread temps réel selon la configuration du futur stream; ce code n'est pas
encore qualifié de hard-real-time. Le noyau C ne fait pas d'allocation dans son
hot path, mais cela ne prouve pas le comportement complet de PipeWire, de
l'ordonnanceur ou du système.

## Vérification de reprise — 2026-10-09

- Le `return_buffer` immédiat des branches de rejet et le `queue_buffer` du chemin normal sont chacun traités comme une seule restitution du buffer détenu. Les échecs des deux opérations incrémentent maintenant `queue_errors`.
- Les tests vérifient format supporté → non supporté → supporté; un format non supporté désactive le booléen au lieu de garder l’ancien statut actif. Ce booléen unique n’a pas de champs associés; le changement pendant un stream actif reste interdit par le contrat quiescent.
- `vr_pw_publish_playback_chunk` efface les flags hérités du buffer réutilisé; le harness commence avec `SPA_CHUNK_FLAG_CORRUPTED` et vérifie leur remise à zéro.
- Vérifications exécutées : test PipeWire 1.4.2 avec ASan/UBSan (2 tests); tests natifs/live ciblés (15 tests); suite complète `unittest discover` (127 tests, 12 ignorés optionnels).
- Aucun stream, serveur, appareil ou microphone n’a été ouvert. La propriété/rythme des buffers avec le vrai serveur et la qualité live restent inconnus.

## Prochaine étape

Avant un smoke périphérique séparé : extraction des fonctions pures de
préparation/publication de régions si elles évoluent, vérification de chaque
branche dequeue→queue/return, prise en charge éventuelle de types mémoire
supplémentaires, compilation/sanitizers sur la frontière, et décisions
explicites de lifecycle/ownership. Un essai réel doit mesurer séparément les
formats réellement négociés, les quanta, erreurs de buffer, underruns/xruns et
temps de callback. Il ne répondra pas à la comparaison acoustique Adobe : cette
tranche est uniquement une validation d'intégration native.
