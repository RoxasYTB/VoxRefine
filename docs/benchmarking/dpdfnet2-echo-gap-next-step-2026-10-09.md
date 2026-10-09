# DPDFNet2 : écho résiduel et prochaine étape (2026-10-09)

## Décision

DPDFNet2 reste la meilleure base actuelle pour nettoyer la voix. Son rendu apprécié à l'écoute ne démontre pas une annulation d'écho : les mesures disponibles montrent une réduction de queue tardive, avec des résultats variables sur la parole faible et des tests surtout synthétiques. Ne pas ajouter un égaliseur, un fondu de blocs ou une seconde suppression de bruit pour « coller » à Adobe : ces changements peuvent réduire une métrique spectrale tout en causant sifflement, pompage, consonnes manquantes ou effet wah-wah.

Le prochain composant doit être un **dereverberateur mono dédié**, conçu pour raccourcir la décroissance de pièce tout en gardant intacte la réponse précoce de la voix. Ce n'est pas l'AEC téléphonique classique : l'AEC a besoin d'un signal de référence de lecture (audio distant / haut-parleur) synchronisé avec le microphone. Pour un fichier mono déjà enregistré sans cette référence, le problème pertinent est la dereverberation aveugle.

## Ce que les mesures disent aujourd'hui

Les résultats suivants proviennent de corpus et protocoles distincts ; ils ne doivent pas être comparés comme une compétition contrôlée entre modèles.

| Méthode | Mesure utile | Réserve constatée | Décision |
|---|---|---|---|
| DPDFNet2 | Sur l'écran VCTK à RIR connu, suppression médiane de queue tardive d'environ 9 dB; le sous-écran naturel Clarity trouve aussi une baisse de queue. | Jeu simulé; les cas de parole faible peuvent perdre plusieurs dB; ce n'est pas une AEC ni une preuve de rendu studio. | Garder comme denoiser de référence, dereverb modérée uniquement. |
| DPDFNet8 | Queue parfois plus réduite. | Perte des évènements faibles et compromis de préservation raté sur la cohorte gelée. | Ne pas remplacer DPDFNet2. |
| WPE | Baseline classique, CPU modéré. | Environ 1 dB de queue supprimée en médiane sur l'écran retenu, insuffisant. | Écarté pour ce pipeline. |
| DeepVQE DNS3 | Forte queue supprimée sur 4 clips (médiane 17.42 dB, 150–300 ms). | Projection précoce médiane −5.97 dB, pire −14.21 dB; seulement quatre clips et licence du checkpoint non établie. | Ne pas intégrer/distribuer. |
| Denoiser Resemble seul | Préserve généralement bien la cible de parole. | Suppression de queue proche de 0 dB dans le test RIR apparié. | Bon candidat denoise, pas dereverb. |

Sources locales : [écran VCTK RIR](vctk-rir-dereverb-truth-2026-10-08.md), [écran WPE / DeepVQE](vctk-wpe-deepvqe-dereverb-screen-2026-10-08.md), [écran naturel DPDFNet2](dpdfnet-naturallevel-screen-2026-10-08.md), [matrice d'évidence produit](product-evidence-matrix-2026-10-08.md).

## Pourquoi une simple courbe Adobe ne suffit pas

Une égalisation décrit un changement moyen de niveau selon la fréquence. Une réverbération ajoute des copies retardées qui changent dans le temps et se superposent aux phonèmes. La supprimer demande une estimation temporelle / spectrale de la queue; une égalisation statique ne peut pas identifier ces copies. Un modèle peut aussi reconstruire la phase et la texture de la voix différemment d'Adobe, donc la similarité des spectres moyens n'est pas une preuve de même intelligibilité ou de même naturalité.

La recette de recherche la plus cohérente avec notre contrainte est une cible **reverberation-time-shortening** (RTS) : produire une cible à décroissance plus courte plutôt que couper brutalement tout le signal après un seuil. Ce cadre a été étudié pour la dereverberation mono aveugle; il faut encore choisir et valider une implémentation open source compacte. Référence primaire : [Speech Dereverberation with a Reverberation Time Shortening Target](https://arxiv.org/abs/2210.11089).

## Protocole de décision pour le prochain candidat

Évaluer séparément, sur des voix tenues à l'écart et fichiers 48 kHz mono:

1. entrée sèche; entrée avec RIR connu à plusieurs T60; RIR + bruit; et prises naturelles uniquement si les fichiers ont les droits d'usage documentés.
2. sorties : DPDFNet2 seul, dereverb seul, DPDFNet2 puis dereverb, avec une intensité de dereverb fixe annoncée avant le score.
3. métriques appariées : réduction de queue 50–150 / 150–300 / 300–600 ms; conservation de cible précoce; écart d'énergie des événements de parole faible; intelligibilité (STOI/PESQ seulement comme proxys); durée, crêtes/clipping, RTF CPU et usage mémoire.
4. contrôles anti-artéfacts : dry controls, consonnes non voisées / sifflantes, enveloppe à 20 ms, spectrogramme et test de continuité de phase/blocs. Pas d'optimisation des paramètres sur le set de validation.

Gate conservateur pour une option automatique : queue médiane réduite d'au moins 6 dB, baisse d'énergie de la cible précoce p90 inférieure à 2 dB sur sec et bruité, aucun cas d'évènement faible au-delà de −6 dB, pas de claquement/warble, et CPU RTF inférieur à 1 sur le matériel cible. Si le gate échoue, laisser le module en curseur expérimental avec le bypass immédiatement disponible.

## Contraintes produit et matériel

- DPDFNet2 est le chemin par défaut local aujourd'hui; DPDFNet8 / Resemble NFE ne doivent pas être ajoutés automatiquement juste pour réduire davantage une queue.
- LocalVQE est un candidat à examiner pour le **mode microphone duplex** seulement : sa documentation exige la référence far-end pour l'AEC. Sans cette référence, il ne résout pas le cas d'un fichier mono existant. Vérifier la compatibilité de licence des poids avant de redistribuer.
- BUDDy est une piste de recherche mono pertinente (dereverb aveugle par diffusion, checkpoint pré-entraîné annoncé sur VCTK anéchoïque), mais son dépôt public ne précise pas de licence dans les fichiers visibles et ne documente pas de budget temps réel compact. Ne pas intégrer ni redistribuer sans clarification de licence; même après clarification, mesurer CPU/VRAM et risques de texture avant de le retenir.
- NVIDIA RE-USE est un candidat technique intéressant (3.7 M paramètres, 8–48 kHz, flux en ligne, paramètres de look-ahead), mais le dépôt modèle indique la licence NVIDIA One-Way Noncommercial et documente une cible Ampere/A100. Cette licence ne convient pas au livrable open source redistribuable envisagé et la cible matérielle ne couvre pas notre GTX 1050 Ti; candidat rejeté pour intégration, utile seulement comme référence de recherche sous ses conditions.
- Sidon-CoreML annonce du 48 kHz et MIT, mais sa variante int8 pèse environ 407 MB et utilise jusqu'à 1.3 GB de RAM; son runtime publié est centré Apple Neural Engine/Core ML. Ce n'est pas un bon candidat à porter sur notre chemin CPU/GTX ancien sans preuve supplémentaire de portabilité et de coût.
- La GTX 1050 Ti est une contrainte de capacité réelle; le chemin CPU et un mode 8 kHz/16 kHz restent nécessaires. L'upsampling à 44.1/48 kHz ne recrée pas les fréquences que le micro 8 kHz n'a jamais capturées.
- Le spectre d'Adobe est une référence descriptive, pas une cible à recopier bin par bin : égaliser un signal vers cette courbe pourrait empirer la voix et ne révèle pas les opérations internes d'Adobe.

## Étape suivante

Faire une recherche ciblée d'un modèle RTS mono compact, reproductible, dont code **et poids** ont une licence claire. Puis conduire un petit essai apparié sans réglage sur les sorties, sous le gate ci-dessus. Ne pas modifier le profil de production tant que ce test n'a pas franchi le gate; demander ensuite un retour d'écoute sur deux rendus nivelés (DPDFNet2, DPDFNet2 + dereverb candidate).
