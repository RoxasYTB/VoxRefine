# DPDFNet2 contre STSubNet sur exemples de pièces réelles (2026-10-09)

## Résumé

DPDFNet2 traite les dix exemples disponibles à environ **0.34 RTF CPU** sur l'hôte de benchmark, en 48 kHz et sans GPU. Les sorties STSubNet publiées sont davantage atténuées dans toutes les bandes mesurées et comportent plus de trames faibles entre les segments vocaux. Ce signal correspond à une réduction plus forte de l'énergie inter-phonèmes, mais ne prouve pas une meilleure suppression de réverbération : nous n'avons ni cible anéchoïque appariée ni écoute subjective équilibrée.

**Décision produit :** garder DPDFNet2 comme chemin actuel. Ne pas chaîner ou copier le comportement STSubNet automatiquement. La piste qui mérite la suite est un dereverb explicite, avec bypass et détection de prise sèche, évalué contre des références sèches et des RIR connus.

## Données et méthode

- Dix exemples AMI du dossier REVERB Challenge publiés dans le dépôt [STSubNet](https://github.com/ffxiong/stsubnet), cinq `near` et cinq `far`.
- Entrées mono à 16 kHz et sorties STSubNet publiées en 48 kHz. L'entrée a été rééchantillonnée à 48 kHz avant DPDFNet2; toutes les sorties ont ensuite été ramenées à la bande commune originale de 16 kHz pour les mesures spectrales.
- DPDFNet2 a été appelé à son taux natif de 48 kHz avec l'adaptateur de benchmark existant et son alignement mesuré de 40 ms. Durée exacte conservée sur les dix clips. Chemin ONNX CPU; aucun GPU n'a été utilisé.
- Mesures : RMS global, énergie par bande, écart-type de l'enveloppe RMS par trames de 20 ms, proportion de trames faibles selon un seuil relatif à chaque fichier. Ce sont des descriptions de sortie, pas des métriques de qualité.

Les entrées et sorties STSubNet se trouvent dans un dépôt d'exemples sans licence de modèle séparée clairement documentée. Elles sont utilisées localement comme référence comparative uniquement; le projet ne les redistribue pas.

## Résultats appariés

Médiane sur dix exemples :

| Mesure | Entrée | STSubNet publié | DPDFNet2 |
|---|---:|---:|---:|
| RMS global (dBFS) | −47.87 | −50.52 | −49.32 |
| Énergie 80–300 Hz (dB, spectre moyen) | −8.83 | −11.67 | −10.60 |
| Énergie 300 Hz–2 kHz | −21.59 | −23.63 | −22.41 |
| Énergie 2–4 kHz | −37.48 | −40.92 | −38.53 |
| Énergie 4–8 kHz | −38.01 | −42.01 | −39.17 |
| Enveloppe RMS, écart-type en dB / 20 ms | 5.87 | 16.36 | 30.98 |
| Trames faibles sous seuil relatif | 1.0 % | 36.9 % | 29.0 % |

Le résultat d'enveloppe dépend de l'activité vocale et de la dynamique de sortie; il ne peut pas être assimilé à la quantité d'écho. Les chiffres d'énergie par bande comprennent la voix, la salle, les variations de niveau et les changements de phase.

### Challenger DPDFNet8

Le même lot a ensuite été traité par DPDFNet8, sans réglage par fichier. Médianes : RMS −50.61 dBFS; écart-type d'enveloppe à 20 ms 31.47 dB; 30.0 % de trames faibles; RTF CPU 0.79 (min 0.78, max 0.82). La réduction médiane par rapport à l'entrée est −2.45 dB en 80–300 Hz, −1.94 dB en 300 Hz–2 kHz, −2.28 dB en 2–4 kHz et −2.93 dB en 4–8 kHz. Le niveau RMS final ressemble à STSubNet (−50.52 dBFS), mais DPDFNet8 ne reproduit pas son profil temporel de manière stable et son enveloppe varie davantage.

![Spectrogrammes entrée, STSubNet et DPDFNet8](../../results/stsubnet-real-reverb-dpdfnet8-compare-2026-10-09/real-reverb-spectrograms-dpdfnet8.png)

Le niveau moyen égal ne signifie pas qualité égale : DPDFNet8 est déjà connu pour perdre certains évènements de parole faible dans les tests contrôlés du projet. Il ne devient pas le choix par défaut sur la base de cette comparaison sans cible sèche.

![Spectrogrammes entrée, STSubNet et DPDFNet2](../../results/stsubnet-real-reverb-dpdfnet-compare-2026-10-09/real-reverb-spectrograms.png)

![Statistiques de niveau et d'enveloppe sur dix exemples](../../results/stsubnet-real-reverb-dpdfnet-compare-2026-10-09/real-reverb-summary.png)

## Ce que ces résultats autorisent à conclure

1. STSubNet produit ici un rendu plus atténué que DPDFNet2, notamment dans les bandes 2–8 kHz et dans les trames faibles. C'est compatible avec un gate/dereverb plus fort.
2. DPDFNet2 garde plus d'énergie entre les phonèmes sur ces enregistrements; cela peut expliquer pourquoi la pièce reste audible. Cela peut aussi conserver une sensation plus continue/naturelle.
3. La variabilité d'enveloppe de DPDFNet2 est plus élevée dans ce lot. Avant toute réduction de niveau, il faut inspecter les consonnes, pauses et syllabes faibles pour séparer dynamique normale, pompage et queue de pièce.
4. La vitesse mesurée est compatible avec le temps réel sur l'hôte actuel : RTF médian 0.34, min 0.325, max 0.367; les temps complets figurent dans `metrics.csv`.

## Ce que ces résultats ne disent pas

- Il n'y a pas de référence anéchoïque synchronisée pour ces dix prises; une estimation SDR ou STOI ici serait trompeuse.
- Les sorties STSubNet sont des exemples publiés, sans métadonnées détaillées sur la configuration ou la normalisation exacte de chaque wav. Elles ne constituent pas une reproduction du modèle.
- Le dépôt ne contient pas de poids STSubNet ni son code d'inférence; le candidat n'est pas directement intégrable.
- Une ressemblance de spectrogramme avec STSubNet ou Adobe ne démontre ni égalité de timbre, ni intelligibilité, ni artefacts absents.

## Prochaine itération

Utiliser cette observation pour fixer la cible du futur module dereverb : réduire l'énergie de queue après les transitoires vocaux sans transformer chaque pause en gate dur. Le prochain écran devra inclure des prises sèches de contrôle, plusieurs T60 contrôlés, des voix faibles et une évaluation de parole; il ne doit jamais apprendre à maximiser uniquement le nombre de trames faibles.

## Artéfacts

- `results/stsubnet-real-reverb-dpdfnet-compare-2026-10-09/metrics.csv` : métriques par clip et empreintes des entrées.
- `results/stsubnet-real-reverb-dpdfnet-compare-2026-10-09/report.json` : protocole machine lisible.
- `results/stsubnet-real-reverb-dpdfnet-compare-2026-10-09/dpdfnet2/` : rendus DPDFNet2 PCM16 temporaires pour écoute/inspection locale.
- `results/stsubnet-real-reverb-dpdfnet8-compare-2026-10-09/metrics.csv` et `report.json` : challenger DPDFNet8 sur le même lot.
- `scripts/benchmarks/universal_enhancer/compare_dpdfnet_stsubnet_real_reverb.py` : génération des rendus et métriques.
- `scripts/benchmarks/universal_enhancer/plot_dpdfnet_stsubnet_real_reverb.py` : graphiques.
