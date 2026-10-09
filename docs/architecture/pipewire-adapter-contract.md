# Contrat du futur adaptateur PipeWire

**Statut :** primitives de framing/ring C11 et un binding PipeWire C candidat
compilé/lié contre l'API locale 1.4.2. Aucun stream matériel n'a été ouvert;
voir [`native-pipewire-binding-2026-10-09.md`](native-pipewire-binding-2026-10-09.md)
pour les limites précises. Ce jalon ne valide pas encore le comportement d'un
callback face à un serveur ou périphérique réel. La callback playback efface les flags SPA hérités sur un buffer réutilisé; les échecs de `queue_buffer` et `return_buffer` sont comptés dans `queue_errors`. La callback `param_changed` a été exercée sur une séquence synthétique valide → invalide → valide; cela ne teste pas le lifecycle lors d’une renégociation active.

## But et frontière

PipeWire doit fournir/consommer des buffers audio avec des échéances courtes;
le traitement DPDFNet doit rester dans un worker séparé. Le callback natif ne
doit appeler ni Python, NumPy, ONNX, attente de worker/condition, allocation,
journalisation ni fonction d'I/O. Une copie mémoire bornée, des compteurs
atomiques et des `try_push`/`try_pop` sont permis; ce contrat ne prétend pas
prouver un lock-free absolu. Les adaptateurs Python existants sont des références de
sémantique et des outils de simulation, pas du code callback-safe. Le futur
callback et ses buffers doivent être en C/C++ (ou Rust avec garanties de
temps réel vérifiables); le worker peut rester Python tant que sa charge est
mesurée et ses queues restent bornées.

```text
PipeWire capture callback
  -> conversion/copie PCM bornée -> SPSC capture ring (frames mono float32)
  -> worker: 480 frames -> backend -> SPSC playback ring (480-frame blocks)
  -> playback callback: splitter hop->quantum -> PipeWire
```

Les rings SPSC VoxRefine ont un seul producteur et un seul consommateur. Cette
propriété concerne uniquement les buffers internes préalloués; elle ne suppose
ni ownership, ni contiguïté, ni interleaving, ni type particulier des buffers
fournis par PipeWire. Chaque callback reçoit une vue et un nombre de frames,
copie vers/depuis ses buffers internes, puis ne conserve aucun pointeur après
son retour. La capture ne doit jamais attendre le worker; la lecture ne doit
jamais attendre le worker.
Si la capacité est épuisée, le callback compte l'incident et applique une
politique explicite, sans allocation ni boucle de rattrapage non bornée.

## Format initial et conversion

Le domaine interne du premier prototype est **48 kHz, mono, float32**. Il ne
préjuge pas du format que PipeWire négociera : l'adaptateur doit lire et
rapporter le format effectivement négocié, puis accepter uniquement les
formats explicitement supportés ou refuser `open()` avant `RUNNING`. Toute
conversion éventuelle est distincte et déclarée. Le callback accepte des
quanta positifs variables, sans supposer un quantum nominal par appel;
256 frames est la valeur observée sur l'hôte, 64–1 024 la plage annoncée, pas
une constante du contrat. Si la source est
stéréo, le downmix doit être une étape déclarée et testée avant activation; ne
pas sélectionner silencieusement un canal. Un taux différent de 48 kHz exige
un resampler explicite avec état et compteurs de délai; l'adaptateur de frames
ne change que le framing et ne fait pas de resampling.

Conversion PCM entier vers float32, clamp/saturation et conversion de sortie
doivent être définis sans NaN/Inf. Le premier chemin validé doit éviter
l'upmix implicite. Toute conversion de format sera testée sur les valeurs
min/max, zéro, faible amplitude, buffers vides/interdits et plusieurs quanta.

## Sémantique du callback

Capture copie exactement les frames disponibles dans le ring capture. Le
framing mono accumule les frames en hops de 480; un reliquat inférieur à 480
reste dans l'adaptateur, puis est compté et jeté à l'arrêt gracieux. Si le ring
capture déborde, la session échoue explicitement ou se réinitialise par une
procédure de discontinuité définie ultérieurement; aucune insertion de zéros
ni réinjection désordonnée n'est permise.

Lecture remplit exactement le quantum demandé. Le splitter sert le suffixe du
hop courant avant d'obtenir le hop logique suivant. Si aucune sortie n'est
disponible à sa frontière, il rend du silence pour la partie manquante et
compte l'underrun; une sortie arrivée après la séquence attendue est stale et
doit être comptée puis jetée selon `fixed_latency`. La fin de stream doit
vider les hops terminaux valides et le suffixe local avant la fermeture du
port. Un arrêt abortif compte les frames abandonnées séparément.

Le callback utilise uniquement mémoire préallouée et copies de taille bornée.
Il ne logue pas, ne lève pas d'exception à travers l'ABI, ne consulte pas
l'heure murale et ne fait aucun calcul DSP. Les timestamps et flags de
discontinuité fournis par PipeWire sont transférés vers des compteurs ou un
canal de diagnostic sans bloquer.

## Cycle de vie et erreurs

```text
CLOSED -> OPEN -> RUNNING -> DRAINING -> STOPPED
                       \----------------> FAILED
```

- `open`: négocier les ports/formats/quanta et allouer toutes les ressources;
  refuser les formats non pris en charge avec le détail de négociation.
- `start`: connecter les ports et lancer le worker avant d'autoriser la
  production; attendre le prébuffer uniquement hors callback.
- `callback`: convertir/copier dans les rings préalloués, mettre à jour des
  compteurs atomiques; aucun appel Python ou ONNX.
- `stop` gracieux: désactiver la capture, signaler EOF au worker, attendre le
  drain avec délai borné et laisser le callback jouer la sortie déjà valide
  tant que le sink reste actif.
- `abort`: arrêter et compter les reliquats/sorties abandonnés explicitement.
- déconnexion: état terminal immédiat et comptage des frames non jouées; ne pas
  promettre le drain après perte du sink.
- `close`: déconnecter et libérer après arrêt confirmé; idempotent.
- toute erreur terminale publie un code stable et des compteurs; pas de
  récupération silencieuse vers une autre source ou un autre canal.

Ne pas détruire les rings tant qu'un callback peut encore les lire. Les
transitions de propriétaire doivent suivre l'API de cycle de vie PipeWire et
être vérifiées sous start/stop/error concurrents avant un test matériel.

## Mesures et critères d'acceptation

Rapporter séparément: format/rate/quanta négociés, frames par callback,
quanta min/max, changements et histogrammes pour capture et lecture,
discontinuités PipeWire, xruns (si l'API les expose), capture ring
high-water/overruns, playback ring high-water/underruns, hops stale, silence
rendu, durées de service worker et cause/état terminal. Les timestamps de
callback et les temps de service ne doivent pas être confondus avec la latence
microphone-vers-sortie; cette dernière requiert une boucle physique connue.

Avant le premier smoke matériel: tests de conversion et quanta synthétiques,
parité des séquences contre `_LiveCore`, fuzz de quanta bornés, test arrêt et
erreur, run virtuel plusieurs minutes à quanta variables et vérification de
l'absence d'allocations/verrous bloquants dans le callback natif. Le smoke
matériel doit être une étape distincte et explicitement activée. Ces critères
ne prouvent rien sur la similitude acoustique avec Adobe Podcast.

## Décisions laissées ouvertes

Le binding exact (libpipewire C ABI, wrapper C++ ou crate Rust), la source/sink
choisis, le mode duplex et la politique de récupération après xrun restent à
trancher après un prototype isolé. Le probe actuel est en lecture seule et
n'offre pas encore la mesure de callback nécessaire à ces décisions.

## Modèle fonctionnel simulé

`PipeWireAdapterModel` dans `voxrefine/live/pipewire_model.py` exerce
négociation, appels de capture/lecture, quanta changeants, xruns et
déconnexion. Il reçoit des tableaux mono float32 **déjà normalisés** dans les
tests; ce format de test ne décrit pas l'ABI ni les buffers fournis par
PipeWire. Le modèle est du Python/NumPy et peut allouer/verrouiller : il vérifie
les transitions et compteurs fonctionnels uniquement, sans mesure de temps
réel, fuzz ABI ou ouverture de périphérique.

`native/live_primitives.{h,c}` ajoute une couche séparée, compilable sans
PipeWire : accumulateur mono 480 frames, splitter hop-vers-quanta arbitraires
et ring SPSC préalloué de blocs `{sequence, float[480]}`. La capacité déclarée
du ring est entièrement utilisable, même à 1; elle ne sacrifie aucun slot. Le
ring a exactement un producteur et un consommateur; `try_push` et `try_pop`
échouent sans attente sur full ou empty. Initialisation, reset et libération du
stockage ne sont permis qu'en l'absence d'accès concurrent. L'initialisation
reçoit la mémoire du propriétaire et les opérations ne l'allouent pas. Le
ring utilise les compteurs monotones pour full/empty et des curseurs physiques
séparés pour les slots; cela garde le FIFO correct au wrap uint64, y compris
avec une capacité non puissance de deux. La capacité est strictement inférieure
à `2^63`, ce qui borne l'écart non signé des compteurs. Le
splitter conserve au plus un hop local: le tampon de sortie effectif peut donc
contenir `ring capacity + 1 hop`. `vr_split_buffered_frames` rapporte son
reliquat local. Avant un reset de session, l'appelant doit terminer/compter le
reliquat capture avec `vr_acc_finish` et celui de playback avec
`vr_split_discard`, puis lire les compteurs avant que le reset les efface. Le
splitter ne décide pas quoi faire des frames manquantes;
il retourne le nombre de frames audio copiées et laisse le suffixe destination
au pilote. Les politiques d'état, de silence, de séquence, de stale/drop, de
prébuffer et d'EOF restent définies hors de ces primitives. `atomic_is_lock_free`
peut être interrogé au runtime, car C11 ne garantit pas que les atomics 64-bit
soient lock-free sur toutes les architectures. Le harness concurrent ne prouve
pas la sûreté hard-real-time, l'absence d'allocations dans un futur binding, les
performances de l'ordonnanceur, le comportement PipeWire ou la latence physique.

Les frames de capture sont réconciliables en tout instant comme
`frames reçues = hops complets émis + frames encore dans l'accumulateur +
reliquat jeté`, puis
`hops émis = frames acceptées par le moteur + refusées par le moteur + hops
non soumis après erreur`. Les frames requises au playback se répartissent en
audio moteur rendu, silence d'underrun, silence de prébuffer et silence
terminal; reliquats et hops jetés lors d'un abort sont rapportés à part. Les
compteurs device simulés et ceux de `_LiveCore` restent dans des groupes
distincts. En cas de `FAILED`, aucune nouvelle sortie logique/backend n'est
produite; un suffixe déjà validé et détenu par le splitter peut encore être
joué durant un arrêt gracieux. Un abort ou une déconnexion le jette et le
compte. `reset()` ouvre une nouvelle session sur le même format et remet à
zéro état, buffers et compteurs; les vues précédentes ne sont jamais reprises.
