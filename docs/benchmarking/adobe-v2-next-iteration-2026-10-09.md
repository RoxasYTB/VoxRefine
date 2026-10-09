# Adobe v2 — résultats de la prochaine itération et portes de qualité

## Résumé

Cette passe réévalue les exports déjà disponibles au lieu de modifier les paramètres du modèle sur les mêmes trois voix. Elle compare Resemble NFE64-C, ses rendus déjà post-traités par DPDFNet à 6 et 18 dB, et trois exports Adobe Podcast Enhance v2. Les entrées sont les mêmes mixes 48 kHz mono de 15 s avec stems propres et bruits connus.

Le résultat est négatif mais actionnable : **la seconde passe DPDFNet coupe des trames faibles entières**. La corrélation médiane de l'enveloppe tombe de 0,962 avec NFE64-C seul à 0,657 / 0,664 avec les cascades; le p10 médian des trames faibles tombe à −134,85 dB (plancher numérique), et 12,5 % de ces trames sont plus de 80 dB sous la référence propre. Adobe v2 n'a aucune trame sous ce seuil dans ce cohort. La cascade baisse les niveaux entre mots, mais cette apparente propreté vient en partie d'un gating destructeur.

**Décision immédiate :** ne pas enchaîner DPDFNet sur NFE64-C dans le chemin voix par défaut. Conserver le rendu NFE64-C seul comme challenger, sans le déclarer équivalent à Adobe : il reste nettement plus faible sur les passages faibles d'Emy et son enveloppe médiane est moins fidèle à Adobe.

## Données et méthode

- Trois lecteurs LibriVox français, trois ambiances CC0 contrôlées, SNR mesuré de 17,29 à 18,00 dB; stems propres connus.
- Chaque variante se décode en WAV mono 48 kHz, 720 000 échantillons; les traitements locaux analysés sont les fichiers déjà présents dans `results/post-denoise-finish-01/deepfilter-finished/`.
- Enveloppe et niveaux : RMS en fenêtres de 20 ms, activité déterminée par le stem propre; « p10 faible » porte sur le quartile actif le plus faible.
- Une trame compte comme quasi effacée si son RMS de sortie est inférieur de plus de 80 dB à celui de la trame propre alignée. C'est un détecteur de quasi-silence, pas une mesure de reconnaissance phonétique.
- Aucun nouvel appel au modèle, EQ, time-warp ou recalage par candidat n'a été effectué. Adobe est une référence de rendu, et la voix propre la référence de niveau.

| Variante | Corrélation d'enveloppe médiane | p10 faible médian vs stem propre | Fraction médiane des faibles trames sous −80 dB | RMS médian en intervalles mesurables |
|---|---:|---:|---:|---:|
| NFE64-C | 0,962 | −15,83 dB | 0 % | −79,66 dBFS |
| NFE64-C → DPDFNet 6 dB | 0,657 | −134,85 dB | 12,5 % | plancher numérique |
| NFE64-C → DPDFNet 18 dB | 0,664 | −134,85 dB | 12,5 % | plancher numérique |
| Adobe Podcast v2 | 0,976 | −16,88 dB | 0 % | −72,25 dBFS |

Les médianes agrégées cachent Emy : NFE64-C a un p10 faible de −37,41 dB, contre −30,34 dB pour Adobe. Chez Rémi et Stéphanie, les cascades ont respectivement 18,0 % et 12,5 % de trames faibles quasi effacées. La variante à 6 dB n'évite donc pas la destruction; la variante à 18 dB ne l'améliore pas.

![Comparaison des trames faibles, du quasi-silence et de l'enveloppe](../../results/adobe-v2-pair-audit-2026-10-09/cascade-threevoice-comparison.png)

## Les quatre axes

### 1. Préserver les consonnes et fins de mots faibles

La mesure sur trois voix identifie un mode de panne net de la cascade DPDFNet : des trames sont ramenées au plancher numérique. Une baisse de bruit en pause ne peut pas compenser cette perte. Garder les cascades désactivées dans le preset voix. Pour toute nouvelle protection, mesurer conjointement les fenêtres faibles et les stems de bruit; les anciens essais de réinjection oracle montrent qu'une restauration pleine bande peut ramener du bruit et ne doit pas devenir une règle produit.

**Porte avant promotion :** aucun bloc de parole active au plancher numérique, pas de régression du p10 faible sur le holdout voix, et baisse du résidu bruit mesurée avec stem connu. Toute moyenne doit être accompagnée du pire locuteur.

### 2. Réverbération, écho acoustique et AEC

Le holdout VCTK à 76 mélanges suggère que Resemble denoiser → DPDFNet réduit fortement les queues de réverbération par rapport à DPDFNet seul (médiane 87,9 % à 150–300 ms et 97,1 % à 300–600 ms), mais échoue au seuil de perte de voix au p90 : 8,53 dB, avec 18/76 événements au-dessus de 6 dB. Ce n'est donc pas une chaîne universelle validée.

L'annulation de retour haut-parleur est une autre fonction : elle exige une référence far-end synchronisée au signal lu. Le traitement mono de fichier ne peut pas reconstituer cette référence. LocalVQE reste une voie duplex expérimentale; le simulateur virtuel ne valide ni un périphérique PipeWire/ALSA réel ni la qualité Adobe.

**Porte :** garder la désréverbération en challenger distinct avec stems/RIR connus; ne pas la confondre avec l'AEC. L'AEC attend un test de boucle far-end/micro réel ou un banc matériel équivalent.

### 3. Généralisation à d'autres voix, bruits et appareils

Le cohort Adobe v2 disponible ne contient que trois lectures françaises et des bruits contrôlés à un seul SNR. Les rapports tenus à l'écart couvrent davantage de locuteurs, bruits et RIR synthétiques/mesurés, mais ils n'ont pas tous un export Adobe v2 apparié. Les enregistrements DNS réels n'ont pas non plus de vérité propre exploitable pour l'instant.

**Porte :** étendre le jeu Adobe apparié à des voix conversationnelles, micros/téléphones, bruit non stationnaire et réverbération réelle; geler les critères et réserver des locuteurs/conditions non utilisés pour le réglage. Tant que ces exports n'existent pas, les conclusions restent limitées aux trois voix.

### 4. Latence et matériel ancien

Sur GTX 1050 Ti, la fiche d'inférence NFE64-C déjà mesurée rapporte environ **2,85 RTF d'inférence** (42,75 s pour 15 s audio), **4,40 RTF de bout en bout** en incluant démarrage et orchestration, et environ 2,13 GiB de VRAM allouée. Ce NFE64-C n'est donc pas temps réel sur cette machine dans cette configuration.

DPDFNet2 CPU a une moyenne de service d'environ 0,413 RTF sur un benchmark de 5 minutes; les appels modèle sont sous 10 ms pour 99 % des hops mesurés, avec un maximum de 33,4 ms. La simulation fixed-latency de 30 s n'a pas eu d'underrun sur un run, mais d'autres politiques/runs ont eu des underruns. Cela ne constitue ni une garantie temps réel stable, ni une validation audio réelle.

**Porte :** séparer le preset qualité hors ligne (NFE) du chemin live causal (DPDFNet), publier démarrage/RTF/latence p50/p95/p99/VRAM/underruns et tester sur périphérique réel avant d'annoncer le temps réel. Garder un chemin CPU et tester les limites mémoire des GPU GTX 10xx; aucun modèle chargé ne doit être obligatoire pour ouvrir l'application.

## Plan immédiat

1. Fermer la cascade NFE64-C → DPDFNet comme défaut et garder les artefacts/mesures comme contre-exemple reproductible.
2. Chercher une protection de parole faible qui estime l'activité sans accès au stem propre; ne jamais recopier la vérité oracle dans le produit.
3. Ne régler qu'un challenger à la fois, sur un jeu de développement; rejouer ensuite le scorecard Adobe v2 et un holdout par locuteur.
4. Garder les branches RIR, AEC duplex, extension 8 kHz et latence dans des rapports séparés; aucune fonction ne doit être créditée sans son entrée de référence correspondante.
5. Promouvoir un profil uniquement après absence de phonèmes effacés, baisse de bruit mesurée, maintien de l'intelligibilité et budgets de latence publiés.

## Reproduction

```bash
.venv/bin/python scripts/benchmarks/universal_enhancer/compare_existing_cascades_adobe.py
.venv/bin/python scripts/benchmarks/universal_enhancer/audit_adobe_v2_pairs.py
```

Mesures et figures restent dans le dossier local ignoré `results/adobe-v2-pair-audit-2026-10-09/`; les exports audio ne sont pas commités.
