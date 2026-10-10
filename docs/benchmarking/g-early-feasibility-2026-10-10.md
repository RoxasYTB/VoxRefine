# G-early feasibility — résultats (10 octobre 2026)

> **Décision : PASS de faisabilité seulement.** La sélection W1 atteint les trois seuils préenregistrés. Aucun modèle n’a été entraîné; il n’existe donc encore aucune mesure d’efficacité de réduction de réverbération.

## Question mesurée

Le pilote vérifie si un ensemble frais de voix peut produire, après le même Cap60, une queue précoce mesurable entre 150 et 300 ms. W1 seule décide l’admissibilité. W2 (300–600 ms) est descriptif et ne participe ni à la sélection ni à une perte de modèle.

## Protocole gelé

- 12 locuteurs frais de LibriSpeech train-clean-360, quatre paires de phrases disjointes par locuteur (48 slots). Les locuteurs et fichiers détaillés restent dans les artefacts ignorés sous `.tools/`.
- Nouvelle namespace de seeds `G-early-feasibility-v1|speaker|pair|candidate`; RT60 uniforme [1,65; 1,85] s, DRR uniforme [−14,5; −12,5] dB; candidats déterministes j=0…31.
- Cap60 exact sur le signal humide et sa référence sèche; première réussite admissible retenue.
- `Lx(W1) > max(−60 dB, Ldry(W1)+6 dB)`, calculé avec la même référence d’énergie active dry Cap60. Seuil strict.
- Si aucun candidat ne passe W1, le créneau garde j=0 comme `BASE_ONLY_CONTROL`, sans perte de queue.
- Gate fixé avant calcul : au moins 36/48 admissibles; au moins 9/12 locuteurs avec 3/4 ou plus; aucun locuteur à 0/4.
- W2 n’est enregistré qu’à titre descriptif. Pas de modèle, pas de sortie de modèle et aucun accès à `/home/yohan/Bureau/test.wav`.

## Résultats

| Mesure | Résultat |
|---|---:|
| Paires examinées | 48/48 |
| W1 admissibles | 43/48 (89.6%) |
| Contrôles `BASE_ONLY_CONTROL` | 5/48 |
| Locuteurs avec ≥3/4 admissibles | 10/12 (gate ≥9) |
| Locuteurs à 0/4 | 0/12 (gate =0) |
| Candidats Cap60 évalués | 222 |
| Candidats par slot (médiane / moyenne / max.) | 1 / 4.62 / 32 |
| Durée du pilote | 2356.8 s (39.28 min) |
| Décision du gate de faisabilité | PASS |

### Niveaux des premières paires admissibles

Niveaux rapportés à l’énergie active dry Cap60; il ne s’agit pas de dBFS. La plage W2 est une observation des seules paires sélectionnées par W1 et ne constitue pas un gate.

| Fenêtre | Lx p10 / médiane / p90 (dB) | Ldry p10 / médiane / p90 (dB) |
|---|---:|---:|
| W1 150–300 ms | -39.10 / -28.11 / -20.29 | -120.00 / -120.00 / -38.40 |
| W2 300–600 ms (descriptif) | -47.19 / -42.45 / -34.76 | -120.00 / -120.00 / -50.87 |

![Histogramme des candidats testés et niveaux des fenêtres W1/W2](docs/benchmarking/assets/g-early-feasibility-2026-10-10/feasibility.png)

## Interprétation et limites

Le résultat autorise la préparation d’un essai distinct de cible G-early, sous réserve de nouveaux splits DEV/HOLDOUT frais et du même gate de couverture d’au moins 75 %. Ce PASS ne prouve pas que le modèle peut supprimer W1, préserver la parole, ou généraliser aux pièces réelles.

Les cinq contrôles ne fournissent aucune cible de perte de queue : ils doivent conserver les pertes de préservation de parole et waveform prévues, avec `lambda_early=0`. La forte hétérogénéité de coût candidat (médiane, moyenne et maximum publiés ci-dessus) est un coût de préparation hors ligne; elle ne mesure pas la latence du produit.

Pas de conclusion sur W2, la réverbération tardive, la qualité studio, la généralisation hors LibriSpeech, le temps réel, les GPU/GTX 10xx ou la parité Adobe Podcast V2.

## Reproductibilité et intégrité

- Design SHA-256 : `a48f47c13f12e9aa6c710c79101640037d2c441c72c2f49abf33d06b3a859db7`.
- Résultats bruts SHA-256 : `4d0b3b0f455b4a0932f23d58c6314003ff690b1550396ae96f215a752bafeaad`.
- Cap60 SHA-256 : `70775e251eee44c0f2451a1e833326cf8bcbbe304d3e7cd12851e6fce72ef7da`.
- Audit PASS : 12 locuteurs uniques, 48 slots, hashes sources non réutilisés, historique candidat contigu, première réussite choisie, contrôles j=0 après 32 refus et agrégats recalculés depuis les slots.
- `training_started=false`, `model_outputs_accessed=false`, `test_wav_accessed=false`; aucun identifiant de locuteur n’est publié.
- Les manifestes et les sources détaillées restent sous `.tools/`, ignorés par Git.

## Gate du train et splits frais

Un second audit a préparé les 128 locuteurs d’entraînement et gelé des splits DEV/HOLDOUT distincts, toujours sans entraîner ni évaluer un modèle.

| Split | Paires | W1 admissibles | Locuteurs ≥3/4 | Locuteurs 0/4 | Gate |
|---|---:|---:|---:|---:|---|
| Train | 48 RIR | 45 (93,8%) | — | — | PASS |
| DEV | 64 | 63 (98,4%) | 16/16 | 0/16 | PASS |
| HOLDOUT | 48 | 45 (93,8%) | 11/12 | 0/12 | PASS, encore scellé |

Le train contient aussi 32 exemples identité, 24 contrôles fan20 et 24 fan10. Ses trois contrôles RIR utilisent bien `BASE_ONLY_CONTROL` sans perte de queue. Les hashes et cibles ont été revérifiés; `tail_ref` est bit-identique à `a_q`. Une analyse diagnostique montre que la cible sèche dépasse l’entrée en magnitude sur 38,64% des bins temps-fréquence éligibles et la cible hybride sur 35,66%. C’est une limite structurelle à mesurer pour le modèle E à atténuation seule, pas un critère de gate.

Le gel DEV/HOLDOUT a utilisé Cap60 sur CPU pendant environ 56 minutes cumulées. Les manifestes déclarent `opened_for_metrics=false`, `holdout_opened=false` et `test_wav_accessed=false`. Hashes: DEV `71ea060ef0bd3fc4f026f35035197c4a9366690917a3d09db637a786b7fcdbc1`; HOLDOUT `28e423bda251360796107e0f3bc477009862dc66ef678955f80b66a32cd5bc97`; index `9842c9dd1f205d245beff1ee331d0425ff595652d15c3a1487c498dc7843a90b`.

GPT Web a revu le gel et les gates. Avant entraînement, son audit de protocole a conduit à rendre W2 explicite: la couverture est informative/total des paires RIR, tandis que Mlow et le taux de régressions certaines utilisent seulement les paires W2 informatives, controls inclus. Le seuil n’a pas changé. Le futur HOLDOUT restera verrouillé jusqu’à la création d’un reçu de décision DEV contenant les hashes du manifest, du résumé et du checkpoint sélectionné.

Les deux fits appariés ont ensuite atteint exactement 3 000 updates chacun sur la GTX 1050 Ti: 206,0 s pour `G-early-clean`, 205,6 s pour `G-early-hybrid`; pic mémoire observé 381,9/383,2 MiB. L’ordre de batch, l’initialisation et les références partagées ont été comparés bit-à-bit. Les checkpoints restent ignorés sous `.tools/`.

Le premier passage d’évaluation DEV a parcouru 64 paires, puis a échoué avant d’écrire résumé ou décision: le code exigeait W2 sur chaque paire, alors que certaines fenêtres 300–600 ms n’étaient pas disponibles. Aucun résultat n’a été consulté pour choisir ou modifier un modèle; HOLDOUT n’a pas été passé au modèle. Correction mécanique: W2 absent compte comme non informatif dans le dénominateur de couverture total RIR; Mlow/régressions n’incluent que les intervalles W2 informatifs; W1 manquant sur une paire éligible reste bloquant. Les deux checkpoints sont conservés, sans réentraînement.

Amendement de code d’évaluation enregistré sous `.tools/compact-dereverb/g-early-2026-10-10/evaluation-code-amendment.json`: ancien SHA évaluation `84a0d7437801990a56fe82f9132f347c031b639d4aaa6f7354c3dd696eee52dd`; version corrigée `1956f17299177e4962340ce5e14f8974f6d5e09d72724c00fc8cf809be9fa28d`; SHA de l’amendement `32677d51cedb7028778ee89c81c77c5841070ca19808c7db4860e809cc6c734c`. Il lie ces versions aux hashes de train, poids et données. L’évaluation DEV est relancée avec la correction; aucun gate ne change.

Ces PASS valident seulement la disponibilité statistique des exemples pour les fits. Ils ne démontrent encore aucune amélioration audio.
