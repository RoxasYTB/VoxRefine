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

Une lecture diagnostique multi-offsets a été générée depuis les rendus existants avec un masque d'activité Cap60 commun, un gain constant par rendu pour égaliser la parole, des fenêtres post-parole censurées dès qu'une autre parole survient et quatre bandes. Les médianes descriptives, en dB relativement au niveau de parole commun, sont :

| Fenêtre après événement | n | Cap60 | Adobe V2 | StuPASE | ROSE brut | ROSE après Cap60 | WPE | DPDFNet2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 50–150 ms | 1 | −26,77 | −42,39 | −40,64 | −20,45 | −27,16 | −39,00 | −43,64 |
| 150–300 ms | 1 | −26,49 | −42,48 | −42,78 | −19,00 | −26,64 | −39,45 | −58,51 |
| 300–600 ms | 0 | censuré | censuré | censuré | censuré | censuré | censuré | censuré |

La contradiction la plus instructive est conservée : DPDFNet2 est environ 32 dB plus bas que Cap60 sur l'unique offset mesurable entre 150 et 300 ms, mais il est perçu comme plus réverbérant. **La mesure ne classe donc pas la sensation entendue.** Une réduction d'énergie de pause ne prouve pas une suppression des réflexions précoces ou du filtrage en peigne pendant les phonèmes; un timbre artificiel peut même être perçu comme une pièce plus présente. Ces explications restent des hypothèses à départager, pas une attribution causale sur ce clip mono.

Les bandes au-dessus de 8 kHz sont marquées comme non prises en charge pour StuPASE et ROSE-CD, exécutés à 16 kHz puis rééchantillonnés pour le pack. Le rééchantillonnage n'ajoute pas de contenu au-dessus de leur bande native; leurs niveaux HF ne sont donc pas rapportés comme des mesures comparables.

![Résidus multi-bandes après événements de parole](assets/model-family-room-residual-2026-10-10.png)

## Prochaine étape

1. Garder ces sept rendus comme comparaison historique, sans retoucher les gains ou la chaîne.
2. Séparer les trois questions dans les prochains rapports : énergie résiduelle sur plusieurs pauses, coloration/modulation pendant les voyelles, et préservation des attaques/consonnes.
3. Ne faire une inférence avec un nouveau moteur qu'après avoir vérifié poids, provenance et licence. VoiceFixer est déjà cloné localement, mais son rapport antérieur note que le checkpoint téléchargé n'a pas satisfait le checksum annoncé; il ne doit pas être rechargé. StoRM est cloné, mais aucun checkpoint n'est présent localement et ses poids distribués séparément n'ont pas été audités. Aucun des deux n'est actuellement un candidat exécutable/promouvable.
4. Pour établir une mesure causale des réflexions précoces et du traînage, prioriser des paires sèches/réverbérées connues et plusieurs réponses impulsionnelles, puis utiliser `test.wav` uniquement comme contrôle d'écoute déjà consommé. Sans stem sec, les métriques sans référence restent des indicateurs partiels.

GPT Web a recommandé d'abandonner le RMS terminal comme score principal et de faire évoluer le protocole vers fenêtres multi-offsets, bandes fréquentielles, modulation temporelle (p. ex. SRMR) et écoute anonymisée. SRMR et l'indice de filtrage en peigne ne sont pas calculés dans cette première lecture. Les seuils de parole (20 ms RMS supérieur à 3,5 % du pic de l'entrée brute, événements séparés après fermeture de gaps de 80 ms) et le masque partagé rendent les sorties comparables ici. La réverbération présente dans l'entrée brute rend plusieurs pauses non mesurables selon la règle conservatrice; il reste seulement 1/1/0 offsets complets selon la fenêtre. Le nombre d'événements mesurables est trop faible pour une conclusion statistique.

Le script reproductible est [`analyze_model_family_room_perception.py`](../../scripts/benchmarks/universal_enhancer/analyze_model_family_room_perception.py). Les CSV/JSON/PNG et WAV liés à l'enregistrement restent en local sous `results/` et ne sont pas ajoutés au dépôt.

## Limites

- Une personne, une voix, une pièce, un clip d'environ 6,2 s; offsets corrélés, pas d'intervalle de confiance.
- La parole de référence utilisée pour aligner les niveaux est Cap60; chaque sortie garde son propre traitement.
- Les bandes calculées ne séparent pas le champ direct de la réverbération.
- Pas de score universel de qualité ni de revendication de parité Adobe v2.
- Les avis d'écoute sont la cible fonctionnelle; les valeurs spectrales ne les remplacent pas.
