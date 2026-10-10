# G-early-v2: résultat DEV (2026-10-10)

## Verdict

**NO-GO.** Le candidat G-early-v2-clean ne franchit pas les seuils préenregistrés sur le DEV. Aucun gagnant n'est sélectionné et le HOLDOUT reste scellé. Ce résultat mesure un transfert vers des locuteurs jamais vus dans le domaine de réverbération artificielle forte utilisé ici; il ne démontre ni une parité avec Adobe Podcast, ni une généralisation aux pièces mesurées.

Le modèle est un U-Net complexe à masque borné, 555 922 paramètres, ajustant un gain `g = 1 - 0.5 sigmoid(a)` et une phase `φ = π tanh(p)`. Le fit était fixé à 3 000 mises à jour sur 128 locuteurs TRAIN. Le checkpoint est lié par SHA-256 `03799711e97b661d9b86e73ad2f573bf8e99b3ef02a4a2525ad81ba1e8ec2885`.

## Résultats décisionnels

Évaluation complète: 64 paires, 16 locuteurs DEV, exécution déterministe sur NVIDIA GTX 1050 Ti (PyTorch 2.7.1+cu118). Aucun audio utilisateur (`test.wav`) n'a été lu.

| Mesure | Résultat | Seuil gelé | Verdict |
|---|---:|---:|---|
| Réduction W1 150–300 Hz, Mlow | +0.371 dB (62 fenêtres) | ≥ +2 dB | Échec |
| LCB 95% bootstrap par locuteur, W1 | +0.117 dB | information descriptive | — |
| Préservation parole active, médiane p50 | −0.547 dB | ≥ −0.5 dB | Échec |
| Préservation parole active, médiane p10 | −3.940 dB | ≥ −3 dB | Échec |
| Préservation parole faible, médiane p10 | −4.529 dB | ≥ −3 dB | Échec |
| Préservation onset, médiane p10 | −3.911 dB | ≥ −3 dB | Échec |
| Trames faibles atténuées de plus de 3 dB | 15.85% | ≤ 10% | Échec |
| Trames faibles atténuées de plus de 6 dB | 6.71% | ≤ 3% | Échec |
| W2 informatif | 24/64 (37.5%) | ≥ 25% | Passe |
| Régressions certaines W2 > 1 dB | 0/24 | ≤ 10% | Passe |
| Contrôle bruit fan20 / fan10, Δ énergie AC médian | −0.522 / −0.615 dB | floor/DC guards | Passe |
| Invariant entrée nulle | 100/100 | 100/100 | Passe |

La pente de queue W1 satisfait son garde-fou (+ aucun changement matériel de la décision), et les autres gardes W2, plancher bruit et DC passent. Le défaut déterminant est le compromis: réduction trop faible de la queue grave, en même temps qu'une atténuation excessive de parole faible et de certaines attaques. Les distributions de préservation sont calculées par locuteur; la fraction de trames faibles est la médiane par locuteur.

## Décomposition descriptive du masque

Ces valeurs sont des diagnostics exploratoires, hors portes décisionnelles. Sur W1, la réduction médiane reconstruite était de +0.547 dB en chemin réel, +0.105 dB en gain seul, et +0.395 dB en phase seule. Cela suggère que la tête de phase contribue davantage que le gain à la réduction W1 observée. La même flexibilité de phase peut toutefois déplacer/canceler des composantes dans les zones de parole; ces statistiques seules ne prouvent pas que la phase cause les pertes sur les attaques ou la parole faible. Une future expérience devrait isoler ce mécanisme avec cible sèche explicite et mesures segmentées préenregistrées.

## Coût d'exécution

| Charge | Temps | Débit indicatif |
|---|---:|---:|
| Évaluation G2 complète (64 paires) | 1 841.74 s (30 min 42 s) | — |
| Référence exacte Cap60 correspondante | 1 789.96 s (29 min 50 s) | — |
| Forward modèle moyen, une branche | 0.1691 s | 5.91× temps réel à 16 kHz, hors E/S et pipeline |
| Fit 3 000 étapes | 667.67 s (11 min 08 s) | 453 MiB VRAM max. |

Le forward isolé indique une marge pour du temps réel sur cette GTX 1050 Ti, mais ne constitue pas un benchmark de streaming bout-en-bout: STFT/ISTFT, buffers, transfert, post-traitement et latence de bloc doivent aussi être mesurés.

## Protocole et provenance

- Split DEV: 16 locuteurs / 64 paires; 62 fenêtres W1 éligibles; au moins 3/4 paires pour 16 locuteurs et aucune couverture 0/4.
- Hash index des splits: `42f021647012cb7bf4926770b561914ead40dbc93b95d50617e894ef5ff1981a`.
- Hash manifeste DEV: `14a1c1ff06e02c1e5f0edf1532a0f1443087a4af5deba1dc14516e815e51745b`.
- Hash manifeste TRAIN: `04070f88a42de90db2a4354af493862c389aaf06cb27d996adece92154c72875`.
- Hash résumé DEV: `793ba9c03f8ec6bc865dca8c62086e58eb04df9ecd1cba0b012ea8a1ff3a9b82`.
- Hash évaluateur officiel amendé: `8eeaa8ded3fd85d74c899992f682790eb4c0b968ade5fc2079f7475a892c921d`; hash amendment: `864873e33b91311f50bf512e50c12c512d7fd339f2cd37360d019ea6d336c4b7`.
- Une première tentative technique a échoué après un forward DEV, avant résumé ni métrique inspectée. Le correctif descriptif a été validé sur entrée synthétique et la passe officielle a repris les 64 paires depuis zéro, sans changement de poids, gates ni chemin décisionnel. La tentative est enregistrée séparément sous `.tools/`.
- Holdout: non ouvert, non inféré, non utilisé pour sélectionner/tuner.
- Un W1 en dessous du seuil préenregistré bloque l'ouverture du holdout; aucun chiffre Adobe direct n'a été calculé dans cette expérience.

## Suite recommandée

Ne pas ajuster G2 sur ce DEV. Le prochain candidat doit employer un nouveau jeu de locuteurs entièrement disjoint, des poids de loss et seuils fixés avant le fit, et une cible sèche explicite dans la loss de parole. Les attaques et segments faibles doivent avoir des métriques dédiées. Comparer phase libre à phase contrainte comme ablation préenregistrée; ne poursuivre vers un holdout qu'après un DEV PASS complet.
