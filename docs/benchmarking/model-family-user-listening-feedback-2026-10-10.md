# Écoute utilisateur : comparaison des familles de moteurs — 2026-10-10

## Résultat humain

Retour d'écoute sur le pack mono 48 kHz du dossier local `results/user-recording-test-2026-10-09/model-family-comparison-02/` :

| Rendu | Retour d'écoute |
|---|---|
| DeepFilterNet Cap60 | Réverbération de la pièce encore audible |
| StuPASE | Voix trop compressée |
| ROSE-CD brut | Réverbération encore audible |
| ROSE-CD après Cap60 | Réverbération encore audible |
| WPE | Réverbération encore audible |
| DPDFNet2 après Cap60 | Réverbération encore plus audible |

Conclusion produit : **aucun de ces candidats ne passe le critère perceptif de réduction de la pièce**. StuPASE est aussi rejeté pour son timbre comprimé. Ce test ne justifie aucun changement du preset de production.

## Révision de la mesure

Le précédent score `RMS 5,79–5,94 s` n'était qu'une mesure d'énergie dans un segment terminal de 150 ms. Il mélange bruit, réponse de pièce, parole faible et artefacts; ce n'est ni une mesure d'RT60, ni un score fiable de sensation de pièce. Sur la nouvelle segmentation de ce clip, l'événement final se termine vers 5,64 s. Il ne reste aucun segment complet et non recouvert pour la fenêtre 300–600 ms; ce résultat doit être censuré, pas remplacé par zéro.

Une lecture diagnostique multi-offsets a été générée depuis les rendus existants avec une activité détectée sur l'entrée brute commune (20 ms RMS > 3,5 % du pic de l'entrée), un gain constant par rendu pour égaliser la parole au niveau actif de Cap60, des fenêtres post-parole censurées dès qu'une autre parole survient et quatre bandes. Les médianes descriptives, en dB relativement au niveau de parole commun, sont :

| Fenêtre après événement | n | Cap60 | Adobe V2 | StuPASE | ROSE brut | ROSE après Cap60 | WPE | DPDFNet2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 50–150 ms | 1 | −26,77 | −42,39 | −40,64 | −20,45 | −27,16 | −39,00 | −43,64 |
| 150–300 ms | 1 | −26,49 | −42,48 | −42,78 | −19,00 | −26,64 | −39,45 | −58,51 |
| 300–600 ms | 0 | censuré | censuré | censuré | censuré | censuré | censuré | censuré |

## SRMR, vérifié sur des paires à vérité connue

J'ai ajouté le SRMR (Speech-to-Reverberation Modulation Energy Ratio) depuis l'implémentation de référence [SRMRpy](https://github.com/jfsantos/SRMRpy), commit `fee009779cef96bed34db3a7e31d10f3ad1ea133`, qui indique des essais à 8 et 16 kHz. Les signaux 48 kHz ont donc été convertis à 16 kHz avant calcul. Pour vérifier le sens du score dans cet environnement, j'ai d'abord comparé deux paires déjà présentes dans le corpus Clarity local : chaque paire comprend un stem sec et sa version réverbérée alignée, de même locuteur et de même durée.

| Paire connue (3 s) | SRMR sec | SRMR réverbéré | Différence |
|---|---:|---:|---:|
| T005_CH2_00450, room12 | 5,374 | 3,175 | −2,200 |
| T005_CH2_00450, room8 | 5,374 | 2,368 | −3,007 |

Sur le clip utilisateur, calculé après l'unique mise à niveau constante déjà décrite :

| Rendu | SRMR original | SRMR normalisé |
|---|---:|---:|
| Cap60 | 7,507 | 5,521 |
| Adobe V2 | 6,901 | 5,593 |
| StuPASE | 8,228 | 5,922 |
| ROSE brut | 6,735 | 5,122 |
| ROSE après Cap60 | 7,476 | 5,690 |
| WPE | 7,610 | 5,581 |
| DPDFNet2 | 7,519 | 5,531 |

Le score confirme qu'il peut répondre à une différence de réverbération connue, mais il ne reproduit pas le jugement d'écoute sur ce clip : le SRMR original favorise StuPASE malgré sa compression audible, tandis que la version normalisée varie d'environ 0,8 point seulement et place Adobe au-dessus de DPDFNet2 de 0,06. Ce n'est pas un classement perceptif fiable, ni une mesure directe de RT60 ou de profondeur de pièce. Bruit, réduction de bruit, compression, bande passante et artefacts changent aussi le spectre de modulation. Les versions sont donc conservées comme axes de diagnostic séparés; aucun réglage de traitement n'est choisi en les maximisant.

La contradiction la plus instructive est conservée : DPDFNet2 est environ 32 dB plus bas que Cap60 sur l'unique offset mesurable entre 150 et 300 ms, mais il est perçu comme plus réverbérant. **La mesure ne classe donc pas la sensation entendue.** Une réduction d'énergie de pause ne prouve pas une suppression des réflexions précoces ou du filtrage en peigne pendant les phonèmes; un timbre artificiel peut même être perçu comme une pièce plus présente. Ces explications restent des hypothèses à départager, pas une attribution causale sur ce clip mono.

Les bandes au-dessus de 8 kHz sont marquées comme non prises en charge pour StuPASE et ROSE-CD, exécutés à 16 kHz puis rééchantillonnés pour le pack. Le rééchantillonnage n'ajoute pas de contenu au-dessus de leur bande native; leurs niveaux HF ne sont donc pas rapportés comme des mesures comparables.

![Résidus multi-bandes après événements de parole](assets/model-family-room-residual-2026-10-10.png)

## Prochaine étape

1. Garder ces sept rendus comme comparaison historique, sans retoucher les gains ou la chaîne.
2. Séparer les trois questions dans les prochains rapports : énergie résiduelle sur plusieurs pauses, coloration/modulation pendant les voyelles, et préservation des attaques/consonnes.
3. Garder SRMR comme indicateur auxiliaire sur plusieurs voix/pièces, avec les paires alignées sèches/réverbérées pour contrôler la direction et les mesures intrusives (ERLE/DRR/RT60 proxy, préservation de parole) là où les stems existent.
4. Ne faire une inférence avec un nouveau moteur qu'après avoir vérifié poids, provenance et licence. VoiceFixer est déjà cloné localement, mais son rapport antérieur note que le checkpoint téléchargé n'a pas satisfait le checksum annoncé; il ne doit pas être rechargé. StoRM est cloné, mais aucun checkpoint n'est présent localement et ses poids distribués séparément n'ont pas été audités. Aucun des deux n'est actuellement un candidat exécutable/promouvable.
5. Pour `test.wav`, sans stem sec Adobe ni réponse impulsionnelle, les métriques restent des indicateurs partiels. Le fichier n'a pas été téléversé ni partagé.

GPT Web a recommandé d'abandonner le RMS terminal comme score principal et de faire évoluer le protocole vers fenêtres multi-offsets, bandes fréquentielles, modulation temporelle (p. ex. SRMR) et écoute anonymisée. Cette passe ajoute SRMR, validé en direction sur deux paires locales à référence sèche, mais ne calcule pas d'indice de filtrage en peigne. La réverbération présente dans l'entrée brute rend plusieurs pauses non mesurables selon la règle conservatrice; il reste seulement 1/1/0 offsets complets selon la fenêtre. Le nombre d'offsets mesurables est trop faible pour une conclusion statistique.

Le script reproductible est [`analyze_model_family_room_perception.py`](../../scripts/benchmarks/universal_enhancer/analyze_model_family_room_perception.py). Les CSV/JSON/PNG et WAV liés à l'enregistrement restent en local sous `results/` et ne sont pas ajoutés au dépôt.

## Limites

- Une personne, une voix, une pièce, un clip d'environ 6,2 s; offsets corrélés, pas d'intervalle de confiance.
- Le niveau de référence est le RMS actif de Cap60, appliqué par un seul gain constant à chaque sortie; chaque sortie garde son propre traitement.
- Les bandes calculées ne séparent pas le champ direct de la réverbération.
- Pas de score universel de qualité ni de revendication de parité Adobe v2.
- Les avis d'écoute sont la cible fonctionnelle; les valeurs spectrales ne les remplacent pas.
