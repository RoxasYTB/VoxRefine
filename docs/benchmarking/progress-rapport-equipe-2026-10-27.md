# Rapport d’avancement VoxRefine — 27 octobre 2026

## Reprise Git pour Clem

- **Fork à cloner :** `https://github.com/RoxasYTB/VoxRefine.git`
- **Branche de travail à utiliser :** `codex/gtcrn-evaluation-tools`
- **Commit d’ancrage vérifié :** `ce38bcf` — `research: add measured room dereverb pilot`.
- **Branche `main` du fork au moment de ce rapport :** `e88173e`. La branche de travail a divergé de ce point et contient 95 commits qui ne sont pas dans cette version de `main`; `main` contient un commit qui n’est pas dans la branche de travail. Clem doit donc checkout la branche de travail directement, pas commencer par `main`.
- **Attention à la branche homonyme :** `codex/gtcrcn-evaluation-tools` (un `c` supplémentaire dans `gtcrcn`) est une autre branche plus ancienne, en retard de 61 commits par rapport à la branche recommandée. Ne pas la choisir.

Clonage propre :

```bash
git clone --branch codex/gtcrn-evaluation-tools https://github.com/RoxasYTB/VoxRefine.git
cd VoxRefine
git rev-parse --short HEAD  # attendu : ce38bcf ou un commit ultérieur de cette branche
```

Le commit `ce38bcf` contient les scripts et le rapport de l’essai VCTK/RIR décrit ci-dessous. Il **ne contient pas** les poids/checkpoints, corpus téléchargés, fichiers d’écoute de `test.wav` ni les sorties d’évaluation sous `.tools/` : ces éléments sont ignorés par Git et restent sur la machine ayant réalisé l’expérience. Au moment de la rédaction, le checkout de l’auteur comporte également 542 chemins modifiés ou non suivis, qui ne sont pas inclus dans la branche distante. Pour reprendre à partir du code, la branche suffit; pour reproduire les chiffres ou écouter les sorties, Clem devra récupérer/configurer les données et régénérer les artefacts.

## Résumé exécutif

Nous avons produit un candidat compact de dereverberation entraîné sur des voix VCTK proches du sec et des réponses impulsionnelles de pièces mesurées. Sur un petit jeu contrôlé de 16 mélanges, il réduit la composante tardive estimée tout en préservant mieux la voix que les premières variantes à cible sèche.

**Mais le test demandé sur `test.wav` est négatif : la sortie du modèle est presque identique à l’entrée.** Ce candidat ne doit donc pas être présenté comme un progrès audible sur une vraie prise, comme une alternative à Adobe Podcast, ni intégré aux réglages produit.

## Ce qui a été entraîné

- Modèle existant `CompactDereverb16k`, 555 922 paramètres, entrée mono à 16 kHz.
- 900 extraits VCTK mic1 : 24 locuteurs d’entraînement et 6 de validation. Mic1 est enregistré en chambre hémi-anéchoïque : c’est une référence proche du sec, pas une vérité parfaitement anéchoïque.
- 26 réponses impulsionnelles mesurées réparties sur 13 salles de trois corpus libres : BUT ReverbDB, AIR et dEchorate. Les salles de validation sont séparées par identifiant de salle.
- Cible sur les exemples réverbérés : chemin direct et réflexions des 50 premières millisecondes, plus 20 % de contrôles secs. Cette cible évite le sur-effacement observé avec les versions qui forçaient tous les exemples vers un signal entièrement sec.
- 8 000 mises à jour sur GTX 1050 Ti : 345,8 secondes d’entraînement, pic VRAM alloué 117,9 MiB.

Les artefacts du modèle et des mesures sont locaux et ignorés par Git. Les scripts et le rapport méthodologique sont dans le commit `ce38bcf` de la branche `codex/gtcrn-evaluation-tools`.

## Résultat contrôlé — indicatif seulement

Sur 16 mélanges RIR-only de trois secondes, quatre voix et deux salles de Rochester réservées à l’évaluation pour ce candidat :

| Mesure | Résultat médian |
| --- | ---: |
| Variation SI-SDR vers cible directe + premières réflexions | +2,49 dB |
| Variation du coefficient projeté de réverbération tardive | −6,46 dB |
| Variation du coefficient direct/précoce | −1,10 dB |
| Variation RMS de sortie | −2,82 dB |
| Variation RMS sur contrôles secs | +0,01 dB |

Les deux salles testées s’améliorent selon le SI-SDR, mais un cas sur 16 régresse. Les coefficients de projection sont des diagnostics des stems connus, pas une mesure perceptuelle de la réverbération. Les voix et salles ont aussi été utilisées dans des expériences antérieures du projet : il s’agit d’un holdout par rapport à ce fit, pas d’un corpus totalement inédit pour le projet.

## Résultat sur `test.wav` — échec de transfert

Le fichier utilisateur (`44,1 kHz`, stéréo, 6,23 s) a été converti en mono 16 kHz, puis traité avec le checkpoint au pas 8 000. Aucune égalisation ou normalisation n’a été ajoutée à la sortie.

| Mesure entrée/sortie | Original | Modèle | Changement |
| --- | ---: | ---: | ---: |
| RMS | −27,58 dBFS | −27,76 dBFS | −0,17 dB |
| Crête | −7,99 dBFS | −8,09 dBFS | −0,10 dB |
| Puissance 80–250 Hz | 13,58 dB | 13,20 dB | −0,38 dB |
| Puissance 250–4000 Hz | 24,29 dB | 24,13 dB | −0,16 dB |
| Puissance 4–7,6 kHz | 21,87 dB | 21,73 dB | −0,14 dB |

Ces bandes sont des puissances spectrales relatives calculées avec la même méthode, pas des niveaux absolus calibrés. Leur faible variation confirme que le réseau laisse presque intact ce signal. L’inférence CPU a pris environ 0,36 s pour les 6,23 s du fichier ; cela ne comprend pas le temps d’ouverture/rééchantillonnage et ne constitue pas un benchmark temps réel d’entrée/sortie audio.

### Fichiers d’écoute

- [Original converti mono 16 kHz](../../.tools/compact-dereverb/testwav-vctk-candidate-2026-10-11/test-original-16k-mono.wav)
- [Sortie du checkpoint](../../.tools/compact-dereverb/testwav-vctk-candidate-2026-10-11/test-enhanced-vctk-room-diverse.wav)
- [JSON avec mesures et paramètres](../../.tools/compact-dereverb/testwav-vctk-candidate-2026-10-11/comparison.json)

### Graphiques

![Spectrogrammes entrée et sortie](../../.tools/compact-dereverb/testwav-vctk-candidate-2026-10-11/testwav-spectrograms.png)

![Différence spectrale sortie moins entrée](../../.tools/compact-dereverb/testwav-vctk-candidate-2026-10-11/testwav-spectrogram-difference.png)

![Enveloppe RMS court-terme](../../.tools/compact-dereverb/testwav-vctk-candidate-2026-10-11/testwav-level-envelope.png)

## Interprétation

Les mesures contrôlées répondent à la question « le réseau peut-il apprendre à atténuer la queue d’une réverbération produite avec ces RIR ? » par un résultat encourageant mais étroit. Elles ne démontrent pas qu’il détecte ou enlève la réverbération d’une pièce réelle, d’un microphone ou d’un signal comprenant bruit, compression et réverbération combinés.

Le comportement quasi-identité sur `test.wav` montre que le gain contrôlé ne se transfère pas à cet enregistrement. Les causes possibles incluent un écart entre les convolutions d’entraînement et la prise réelle, une réverbération/faible rapport direct-réverbéré non représentée, ou un résiduel appris trop prudent. Nous ne pouvons pas départager ces causes avec un seul fichier et les métriques actuelles. C’est une hypothèse à vérifier, pas une conclusion établie.

## Décision et suites proposées

1. **Ne pas intégrer ce checkpoint au produit** et ne pas annoncer d’amélioration de `test.wav`.
2. Conserver ce candidat comme preuve d’apprentissage sur paires simulées/mélangées seulement.
3. Avant un nouvel entraînement, analyser la prise de test et son transfert spectral, puis faire un essai borné avec des traitements de dereverberation déjà disponibles et leurs protections de parole. Comparer le signal de fin de phrase avec le signal actif pour vérifier qu’une baisse de queue n’est pas simplement une baisse de voix.
4. Ne poursuivre le fine-tuning que si un jeu d’évaluation réaliste, distinct des données d’ajustement, montre une réduction reproductible de réverbération sans pertes de syllabes ni voix étouffée.
5. Une comparaison avec Adobe V2 exige une référence Adobe appariée et exploitable pour cette même prise. Les statistiques ci-dessus comparent uniquement original et candidat local.

## Sources de données et reproductibilité

- [VCTK 0.92 — University of Edinburgh DataShare](https://datashare.ed.ac.uk/items/30e7453c-9ea8-48b4-8e18-f96d0dc62928)
- [BUT Speech@FIT Reverb Database](https://speech.fit.vut.cz/software/but-speech-fit-reverb-database)
- [dEchorate — SOFA data index](https://sofacoustics.org/data/database/dechorate/)
- [Aachen Impulse Response Database — RWTH Aachen](https://www.iks.rwth-aachen.de/en/research/tools-downloads/databases/aachen-impulse-response-database)
- Protocole, métriques contrôlées et commande d’entraînement : [rapport d’expérience détaillé](vctk-measured-room-dereverb-2026-10-11.md).
