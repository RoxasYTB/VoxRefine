# Adobe Podcast Enhance v2 contre NFE64-C — trois voix avec bruit contrôlé

## Objectif et paire exacte

Trois extraits de parole français de 15 s ont été mélangés à 18 dB de SNR actif mesuré, puis traités avec Adobe Podcast Enhance **v2** à ses réglages gratuits visibles par défaut (Voix 50 %, Musique 10 %, Bruit de fond 10 %) et avec VoxRefine Resemble NFE64-C. Chaque entrée a un stem de parole propre connu; le bruit est un ventilateur, une foule, ou les deux. Les voix sont issues du corpus localement documenté LibriVox, les sons de fond sont CC0. Le script du test contrôle les SHA-256 des cinq fichiers de chaque paire.

Les trois rendus Adobe v2 ont été téléchargés en WAV mono 48 kHz PCM16, 15 s / 720000 échantillons chacun. Les entrées source sont en WAV mono 48 kHz PCM24 avec les mêmes durées. Les rendus NFE64-C ont aussi 15 s / 48 kHz. Les comparaisons utilisent le même point de départ, sans time-warp ni recalage par candidat.

## Résultats par voix

Les niveaux de parole sont comparés au stem propre, par trames de 20 ms. Le « p10 faibles » est le 10e percentile de la différence de niveau sur le quartile de trames actives les plus faibles du stem propre. L'enveloppe est une corrélation de log-RMS sur les seules trames actives. La MAE spectrale est calculée entre 80 Hz et 12 kHz après gain fixe de chaque candidat pour faire correspondre le RMS actif à la voix propre; Adobe est la référence de courbe, pas la vérité propre.

| Voix / bruit | SNR mesuré | Candidat | Enveloppe vs propre | Delta niveau vocal médian (dB) | p10 trames faibles (dB) | Écart spectral à Adobe (dB) | Intervalles NFE / Adobe (dBFS) |
|---|---:|---|---:|---:|---:|---:|---:|
| Emy · ventilateur + foule | 18,00 dB | NFE64-C | 0,896 | −6,13 | **−37,41** | 1,86 | n.d. / n.d. |
|  |  | Adobe v2 | **0,958** | −9,71 | −30,34 | référence | n.d. |
| Rémi · foule | 17,29 dB | NFE64-C | 0,962 | −1,36 | −15,83 | 2,40 | −83,76 / −72,36 |
|  |  | Adobe v2 | **0,976** | −3,09 | −16,88 | référence | −72,36 |
| Stéphanie · ventilateur | 17,32 dB | NFE64-C | 0,979 | −0,99 | −10,68 | 3,81 | −75,57 / −72,14 |
|  |  | Adobe v2 | **0,983** | −1,03 | −10,18 | référence | −72,14 |

**Lecture prudente :** Adobe a la meilleure corrélation d'enveloppe sur ces trois exemples, mais il ressort en moyenne à un niveau vocal plus faible que le stem propre. Pour Emy, les deux traitements ont des fenêtres faibles très atténuées; NFE64-C est environ 7,1 dB plus bas que l'export Adobe au p10. Le seuil de parole propre ne trouve aucune fenêtre d'inactivité chez Emy, donc sa métrique d'intervalle est non définie, pas nulle.

Les médianes simples des trois voix sont : enveloppe 0,962 pour NFE64-C contre 0,976 pour Adobe; niveau vocal −1,36 contre −3,09 dB; p10 faible −15,83 contre −16,88 dB; MAE spectrale NFE64-C vers Adobe 2,40 dB. Cette proximité moyenne du spectre ne signifie pas que la parole est plus naturelle : le cas faible d'Emy diverge fortement et les niveaux de pause diffèrent.

Sur Rémi et Stéphanie, le niveau médian des trames hors masque est plus bas de 11,4 dB et 3,4 dB avec NFE64-C par rapport à Adobe. C'est compatible avec davantage de gating/silence, mais ces trames peuvent aussi contenir souffles ou activité vocale manquée par le masque; ce n'est pas un score pur de bruit.

## Graphiques

![Courbes de puissance par voix avec niveau de parole apparié](../../results/adobe-v2-pair-audit-2026-10-09/clear-noisy-adobe-spectra.png)

![Spectrogrammes des entrées bruitées et des deux résultats](../../results/adobe-v2-pair-audit-2026-10-09/clear-noisy-adobe-spectrograms.png)

## Décision pour le rapprochement v2

Le profil Adobe v2 réduit plus régulièrement le fond entre mots et suit un peu mieux l'enveloppe sur ce petit cohort. NFE64-C obtient une courbe active assez proche d'Adobe sur deux des trois voix, mais son comportement de niveau varie beaucoup, surtout dans les faibles trames d'Emy. Cela valide le besoin d'un contrôle de préservation des sons faibles en plus de l'atténuation de fond; une EQ ou un volume global ne répare pas les trames perdues.

**Ne pas conclure à une équivalence ou une supériorité universelle.** Il n'y a que trois voix de lecture avec des bruits synthétiques stationnaires à un seul SNR. Aucun STOI/PESQ ni test de reconnaissance des mots n'a été utilisé, et aucune écoute comparative n'a été effectuée dans cette passe. Les métriques de trames et de spectre restent descriptives.

## Reproduction et données locales

```bash
.venv/bin/python scripts/benchmarks/universal_enhancer/compare_clear_noisy_to_adobe_v2.py
.venv/bin/python scripts/benchmarks/universal_enhancer/audit_adobe_v2_pairs.py
```

Les WAV téléchargés, hashes détaillés, CSV de chaque candidat et figures restent dans le dossier local ignoré `results/adobe-v2-pair-audit-2026-10-09/`. Le rapport n'ajoute pas ces enregistrements à GitHub.
