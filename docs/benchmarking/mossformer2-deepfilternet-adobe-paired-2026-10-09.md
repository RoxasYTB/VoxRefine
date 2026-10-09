# MossFormer2 et DeepFilterNet contre Adobe v2 — screen apparié du 2026-10-09

## Protocole

Trois entrées françaises mono 48 kHz de 15 s ont été traitées sans changer le contenu source : Emy (ventilateur + foule), Rémi (foule), Stéphanie (ventilateur). Comparateurs : Adobe Podcast Enhance v2, Resemble NFE64-C existant, DeepFilterNet avec cap d'atténuation 6 dB et 18 dB, et MossFormer2 SE 48 kHz. Les stems clean et noise sont conservés comme références. Les entrées et exports Adobe/MossFormer2 sont mono PCM16 48 kHz / 720000 frames. DeepFilterNet `--compensate-delay` enlève son délai fixe de 30 ms et rend 718560 frames (14,97 s). Pour que toutes les mesures et auditions soient comparées sur le même intervalle strict, chaque fichier a été limité au recouvrement commun de 718560 frames; aucune fin silencieuse n’a été ajoutée au candidat.

DeepFilterNet a nécessité une conversion PCM24→PCM16 sans changement de fréquence/canaux, car le binaire Rust panique sur ces WAV PCM24. Modèle/binaire DeepFilterNet v0.5.6; option `--compensate-delay`; aucune post-cascade. La conversion 24→16 bits change la quantification, mais laisse les mêmes échantillons à la résolution PCM16. La sortie Dfn compensée est raccourcie de 30 ms; le common span exclut donc les 30 dernières millisecondes de tous les candidats, y compris Adobe et MossFormer2. Un test complémentaire de vidange n’a pas permis d’obtenir une fin alignée complète fiable avec le binaire CLI; ce point est documenté plutôt que masqué par du silence ajouté.

MossFormer2 utilise ClearerVoice commit `6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61`; checkpoint Hugging Face `alibabasglab/MossFormer2_SE_48K` revision `eff8c97925c8bec812af707814b3e5d777fd4503`, métadonnée de licence Apache-2.0. Il tourne sur la GTX 1050 Ti. Modèle chargé en 19,0 s; inférence chaude du lot 45 s en 5,57 s (RTF 0,124); pic VRAM alloué 356 MB / réservé 380 MB. Ceci mesure le chemin ClearVoice en lot sur Linux/GTX1050 Ti, pas un chemin streaming ni une mesure portable d'un autre OS.

DeepFilterNet cap 6 : 3 fichiers 15 s en 10,37 s (RTF total 0,230); cap 18 : 10,21 s (RTF 0,227). Ce temps inclut démarrage du binaire et I/O sur cette machine. Les deux sont des exécutions hors-ligne; l'adéquation à des callbacks temps réel requiert encore la voie streaming séparée.

## Résultats principaux par sample

Les chiffres suivants sont calculés sur 14,97 s communs. `weak p10` est le 10e percentile du rapport de niveau par trame 20 ms entre la sortie et la parole propre, dans le quartile de trames propres actives les plus faibles. `Gap` est le RMS médian de la sortie dans les trames hors parole définies par le stem propre; plus négatif indique moins de résiduel, mais les gaps peuvent contenir souffle/tail. Emy ne contient aucune trame hors masque, donc son gap est non défini. L'écart spectral est une MAE de courbe 1/6 octave 80 Hz–12 kHz après appariement du niveau vocal actif au stem clean; ce nombre n'est pas un score de qualité.

| Voix / méthode | Corrélation enveloppe | Faibles p10 (dB) | Faibles <-40 dB | Gap médian (dBFS) | MAE spectrale Adobe (dB) |
|---|---:|---:|---:|---:|---:|
| Emy — Adobe v2 | 0,958 | −30,3 | 0 % | n.d. | référence |
| Emy — DeepFilter cap 6 | 0,995 | −3,1 | 0 % | n.d. | 0,89 |
| Emy — DeepFilter cap 18 | 0,969 | −12,2 | 0 % | n.d. | 1,22 |
| Emy — MossFormer2 | 0,951 | −23,4 | 0 % | n.d. | 1,14 |
| Rémi — Adobe v2 | 0,975 | −16,9 | 0 % | −72,4 | référence |
| Rémi — DeepFilter cap 6 | 0,952 | −0,7 | 0 % | −46,0 | 4,94 |
| Rémi — DeepFilter cap 18 | 0,979 | −5,2 | 0 % | −57,8 | 5,08 |
| Rémi — MossFormer2 | 0,993 | −8,2 | 0 % | **−75,6** | 5,24 |
| Stéphanie — Adobe v2 | 0,983 | −10,2 | 0 % | −72,1 | référence |
| Stéphanie — DeepFilter cap 6 | 0,957 | −0,4 | 0 % | −46,6 | 3,39 |
| Stéphanie — DeepFilter cap 18 | 0,983 | −2,7 | 0 % | −57,0 | 3,39 |
| Stéphanie — MossFormer2 | 0,995 | −9,8 | 0 % | **−72,4** | 3,43 |

## Lecture des samples

- **MossFormer2** donne sur Rémi et Stéphanie des pauses d'un niveau très proche d'Adobe (respectivement 3,3 dB et 0,2 dB plus bas). Sa corrélation d'enveloppe est élevée sur les trois voix. Sa préservation de trames faibles est proche d'Adobe pour Stéphanie et meilleure pour Rémi; Emy reste plus fragile au p10. Sa courbe de parole est assez proche d'Adobe pour Emy/Stéphanie, mais plus éloignée sur Rémi (MAE 5,24 dB). C'est le meilleur nouveau candidat HQ de ce petit screen, pas un gagnant universel.
- **DeepFilterNet cap 6** préserve très bien le niveau des trames faibles et l'enveloppe, mais laisse 25,6 dB et 25,5 dB de plus dans les gaps que l'export Adobe sur Rémi/Stéphanie. C'est un profil conservateur utile, mais son nettoyage est insuffisant pour imiter Adobe sur ces pauses.
- **DeepFilterNet cap 18** réduit davantage les gaps que cap 6, tout en gardant les trames au-dessus de −12,3 dB de weak p10 dans ce trio. Il laisse toutefois environ 14 dB de plus qu'Adobe dans les gaps des deux cas mesurables. C'est un candidat live/conservateur à auditionner, pas une équivalence Adobe.
- Les faibles p10, MAE spectrale et niveaux natifs ne racontent pas à eux seuls le son perçu. Les versions `audition-levelmatched/` ont un gain fixe par fichier vers la même médiane de parole; les versions natives demeurent dans `listening/`.

## Graphique et audio

![Courbes de spectre, enveloppes natives et niveaux de pause](../../results/two-alternatives-adobe-v2-2026-10-09/comparaison-candidats.png)

Les copies WAV brutes, les auditions à niveau fixe, les métriques et les manifests sont sous `results/two-alternatives-adobe-v2-2026-10-09/` (ignoré par Git, incluant les poids téléchargés). `metrics.csv` contient les valeurs par candidat, `audition-manifest.json` décrit les 18 fichiers égalisés, `mossformer2/runtime.json` et `deepfilter-runtime.json` gardent les temps et VRAM.

## Conclusion limitée

MossFormer2 est la piste à écouter en premier côté HQ. Pour une voie rapide, DeepFilterNet cap 18 est plus propre que cap 6 mais reste sensiblement moins efficace qu'Adobe sur les gaps; cap 6 protège mieux la voix faible avec le compromis d'un fond plus présent. Pas d'EQ, cascade, ni normalisation dynamique dans ce screen. Trois lectures françaises à SNR contrôlé ne permettent pas de démontrer l'universalité, la qualité studio équivalente ni la latence réelle en direct.

## Micro-balaye tonal après MossFormer2 (2026-10-09)

Pour répondre à l'hypothèse « un peu plus de basses, un peu moins de trebles », six variantes fixes ont été appliquées aux mêmes trois sorties MossFormer2 : référence sans EQ, shelves +0,5/+1/+1,5 dB sous 150 Hz et −0,5/−1/−1,5 dB au-dessus de 5 kHz, ainsi que deux variantes asymétriques. La forme est un shelf causal de premier ordre. Aucun compresseur, déesseur, denoiser ou normaliseur n'a été ajouté. Des copies `*_levelmatched.wav` permettent l'écoute avec un gain fixe commun par voix dérivé du niveau de base; les WAV non suffixés conservent le niveau natif.

| Voix | Variante | Delta énergie 40–180 Hz | Delta énergie 4–12 kHz | Crête native |
|---|---|---:|---:|---:|
| Emy | +0,5 / −0,5 dB | +0,24 dB | −0,28 dB | −1,21 dBFS |
| Emy | +1 / −1 dB | +0,50 dB | −0,55 dB | −1,06 dBFS |
| Emy | +1,5 / −1,5 dB | +0,77 dB | −0,81 dB | −0,87 dBFS |
| Rémi | +0,5 / −0,5 dB | +0,28 dB | −0,34 dB | −2,98 dBFS |
| Rémi | +1 / −1 dB | +0,57 dB | −0,67 dB | −3,01 dBFS |
| Rémi | +1,5 / −1,5 dB | +0,88 dB | −0,98 dB | −3,03 dBFS |
| Stéphanie | +0,5 / −0,5 dB | +0,24 dB | −0,31 dB | −3,81 dBFS |
| Stéphanie | +1 / −1 dB | +0,49 dB | −0,61 dB | −3,70 dBFS |
| Stéphanie | +1,5 / −1,5 dB | +0,76 dB | −0,89 dB | −3,58 dBFS |

Les niveaux de bande sont des sommes d'énergie STFT, pas une mesure psychoacoustique de timbre. Sur ces sorties, le shelf ne modifie que modestement l'énergie de bande. L'écart plus large à Adobe visible sur les courbes est déjà présent dans le rendu MossFormer2 et varie selon la voix; un shelf fixe ne peut pas le corriger universellement. Les crêtes ne clippent pas ces trois fichiers, mais Emy approche −1 dBFS avec la variante +1,5/−1,5 : un flux live doit conserver headroom et gérer les crêtes après EQ.

La passe EQ seule, préchauffée et en mémoire sur CPU, a pris 27–34 ms (médiane 29 ms) pour 45 s de mono float32 48 kHz, soit RTF médian 0,00065. Un fichier de 14,97 s s'égalise en ~9–16 ms après import chaud dans cette passe; le premier appel de scipy/JIT d'import a pris 1,12 s pour Emy et n'est pas le coût audio régulier. Ce micro-benchmark n'inclut ni ouverture du modèle, réduction de bruit, lecture/écriture disque ni périphériques. Dans la mesure antérieure sur la GTX 1050 Ti, MossFormer2 a chargé en 19,0 s, puis traité 45 s en 5,57 s; cela prouve une conversion hors ligne rapide après chargement, pas une chaîne micro temps réel.

Artifacts : `results/tone-balance-moss-2026-10-09/metrics.csv`, `eq-runtime.json`, `tone-sweep-adobe.png`, plus les WAV base et variants; résultats ignorés par Git pour garder poids et exports locaux hors des changements versionnés.

## Faisabilité micro virtuel / Discord

Le post-EQ shelf est causal et n'a pas de look-ahead : c'est une étape compatible avec un chemin bloc faible latence si l'état du filtre est conservé entre callbacks et si on réserve le headroom. Elle est négligeable devant le modèle. La passe mesurée de MossFormer2 reste toutefois un appel ClearVoice en lot, donc elle ne prouve pas que MossFormer2 peut fournir sans interruption des blocs microphone en direct.

Le dépôt contient un prototype de moteur DPDFNet2 CPU, un worker et des rings, ainsi qu'un replay WAV cadencé; il n'ouvre pas encore de périphérique capture/lecture physique, et le code Python autour des rings n'est pas callback-safe. Il existe des primitives natives et des sondes PipeWire, mais pas encore un vrai stream audio. Pour Discord, la forme produit habituelle serait capture microphone -> moteur à blocs -> périphérique d'entrée virtuel (« micro VoxRefine ») sélectionné comme source dans Discord. C'est techniquement faisable sur Linux/Windows/macOS au moyen de l'API audio et d'un périphérique loopback/virtuel propre à chaque OS; les permissions et l'installation du périphérique font partie du produit. Il reste à développer et valider la capture duplex, la sortie virtuelle, la latence aller-retour, la dérive d'horloge, les underruns et la qualité d'écoute en communication. Aucune promesse de direct MossFormer2 universel ne découle du timing offline.

## Recommandation de coordination GPT Web

Le retour GPT Web converge avec la limite mesurée : ne pas promouvoir de shelf fixe comme correction universelle, et ne pas appliquer d'EQ frame-by-frame. Pour une expérimentation future, estimer seulement la coloration introduite par le moteur sur des trames de parole suffisamment actives, médiane robuste sur 5–15 s, correction lissée très lentement (slew indicatif 0,05–0,1 dB/s), bornée à ±0,75–1 dB, avec gel en silence/bruit dominant et option de rester à 0 dB. Cette stratégie demeure une hypothèse et requiert un nouveau screen contre les stems propres; elle n'est pas activée dans le pipeline.

Pour le micro virtuel, l'ordre proposé est : (1) capture vers source virtuelle sans DSP; (2) même transport avec backend identité et stalls simulés; (3) DPDFNet2 réel avec mesures par hop p50/p95/p99, underruns, backlog et drift; (4) sélection/test de la source dans Discord. Un premier gate d'ingénierie suggéré est <100 ms médiane VoxRefine capture→source virtuelle et p95 <120 ms, sans drift, overrun ou série d'underruns sur 10 minutes; ensuite tenter 60–80 ms. Ces valeurs sont des objectifs proposés, pas des résultats du dépôt. Le test actuel n'a toujours pas ouvert de stream physique.
