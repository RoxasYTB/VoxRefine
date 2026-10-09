# Audit des comparaisons locales Adobe Podcast v2 — 2026-10-09

## Objet

Vérifier les exports Adobe v2 et leurs entrées/rendus associés avant de décider quelle chaîne VoxRefine mérite un nouvel essai. Le script `scripts/benchmarks/universal_enhancer/audit_adobe_v2_pairs.py` vérifie présence, décodage complet, SHA-256, conteneur, subtype, taux, nombre de canaux, durée, valeurs finies et crête. Il n'exécute aucune inférence et ne classe pas la qualité audio.

## Intégrité des fichiers examinés

| Groupe | Échantillons attendus | Lisibles | Durées |
|---|---:|---:|---:|
| Trois événements RIR synthétiques × sec / avec RIR | 42 | 42 | 2,800 s par cellule, spread 0 ms |
| Trois extraits français avec pauses / parole continue / variations de niveau | 9 | 9 | 39,360 / 39,460 / 39,360 s par cellule, spread 0 ms |
| Christiane, comparaison long-form | 6 | 6 | même durée par cellule |
| Bruit contrôlé SNR 10 dB avec export Adobe | 5 | 5 | même durée par cellule |
| **Total** | **62** | **62** | **aucun écart de durée intra-cellule** |

Le manifeste machine contient les SHA-256 : `results/adobe-v2-pair-audit-2026-10-09/audit.json` et `assets.csv`. Tous les fichiers inventoriés se décodent et ont des échantillons finis. Les WAV des paires Adobe/RIR et Fortune sont à 48 kHz; la table machine détaille aussi les groupes long-form et bruit contrôlé.

Une égalité de durée ne prouve pas que chaque candidat est aligné au même échantillon, que l'export Adobe utilise toujours les mêmes réglages, ou qu'un MP3 intermédiaire n'a pas ajouté de délai. Les rapports événementiels gardent leurs contrôles d'alignement distincts.

## Ce que disent les comparaisons Adobe appariées existantes

Sur les trois événements RIR synthétiques, les médianes résumées sont :

| Chaîne | Réduction médiane du tail | Corrélation médiane de l'enveloppe vocale | MAE spectrale médiane vs Adobe |
|---|---:|---:|---:|
| Adobe Podcast v2 | 17,55 dB | 0,889 | 0 dB (référence) |
| DeepFilterNet −18 dB | 10,99 dB | 0,879 | 6,39 dB |
| Resemble denoiser seul | ~0 dB | 0,842 | 4,29 dB |
| Resemble NFE32 | 16,99 dB | 0,572 | 7,31 dB |
| Resemble NFE64-C | 18,52 dB | 0,623 | 5,71 dB |

Ces médianes cachent une variation critique : Adobe lui-même enlève de +2,50 à +26,19 dB de tail sur ces événements, et NFE64-C va de +25,70 dB à −16,21 dB. La MAE spectrale la plus petite n'identifie donc pas un enhancer plus fiable. DPDFNet garde une enveloppe plus proche d'Adobe sur cet écran mais retire moins de tail; NFE64-C retire souvent beaucoup de tail mais peut modifier l'enveloppe et échouer complètement sur Naf. Les entrées sont des RIR synthétiques et seulement deux voix : ces chiffres ne démontrent pas une performance sur pièce naturelle.

Sources des mesures : [écran RIR apparié](adobe-paired-rir-screen-2026-10-08.md), [analyse du compromis spectral](adobe-spectral-proximity-tradeoff-2026-10-08.md), [décomposition par bandes](adobe-residual-decomposition-2026-10-08.md).

## Test de courbe Adobe et limite de généralisation

Dans une seule entrée bruitée contrôlée, l'EQ `adobe_curve` fait baisser la MAE spectrale 20 Hz–12 kHz de 4,34 dB (profil plat) à 1,77 dB, après comparaison à loudness égalisée. Le profil `C` choisi auparavant atteint 3,40 dB. Sur six cellules dry/RIR déjà analysées, cette EQ réduit la MAE dans cinq et la dégrade de 1,55 dB sur Naf dry. La médiane passe d'environ 3,60 à 3,00 dB.

C'est un diagnostic de coloration, pas une preuve que l'EQ sonne mieux ni qu'elle restaure les détails vocaux. Adobe ne produit pas une simple EQ fixe : les différences mesurées changent selon le contenu. Le profil Adobe-derived doit rester optionnel; aucun changement de défaut n'est justifié.

## Décision et prochaine étape

1. L'audit ferme la porte « fichiers absents / durées mal appariées » sur les 11 groupes inventoriés.
2. Ne pas intégrer LocalVQE à cette voie fichier; son écran streaming duplex est distinct et sa préservation de voix reste problématique.
3. Ne pas promouvoir NFE64-C comme remplaçant général : l'échec sur Naf est un contre-exemple de régression.
4. Utiliser DPDFNet comme baseline conservatrice de réduction du bruit, mais présenter séparément son compromis RIR et spectral.
5. Garder la courbe EQ comme challenger désactivé par défaut. La prochaine étape est de mesurer le profil par rapport aux stems propres/bruit/activité sur les six cellules et les trois clips bruités, et de rechercher une régression de consonnes faibles avant tout changement de produit.
6. Les comparaisons Adobe supplémentaires nécessitent un export Adobe apparié acquis dans le compte utilisateur. En son absence, le scorecard reste borné aux exports déjà sauvegardés.

## Reproduction

```bash
.venv/bin/python scripts/benchmarks/universal_enhancer/audit_adobe_v2_pairs.py
```

Les résultats sont écrits dans `results/adobe-v2-pair-audit-2026-10-09/`, dossier local de résultats ignoré par Git. Les WAV et les corpus sous-jacents ne sont pas copiés ni publiés par le rapport.
