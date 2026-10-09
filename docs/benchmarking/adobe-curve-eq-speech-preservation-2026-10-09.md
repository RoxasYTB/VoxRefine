# EQ dérivée du spectre Adobe v2 : contrôle de parole et de queue — 2026-10-09

## Question

L'EQ statique ajustée à la courbe moyenne Adobe améliore-t-elle seulement la forme spectrale, ou aussi le comportement de la voix et des queues de pièce ? Ce rapport analyse les six paires existantes `dry-control` / `rir-only` pour Christiane 1, Christiane 2 et Naf 2. Il ne lance aucune inférence.

## Méthode

Le script `scripts/benchmarks/universal_enhancer/analyze_adobe_curve_preservation.py` compare les rendus existants NFE64-C, NFE64-C suivi de l'EQ ajustée et Adobe v2. La parole sèche connue fournit le masque d'activité à 20 ms, avec seuil de −35 dB par rapport au pic, avant le point final synthétique de 1,5 s. Les fenêtres faibles sont le quartile inférieur des fenêtres actives de cette cible.

Les métriques comprennent la corrélation de l'enveloppe RMS, le delta de niveau des fenêtres actives et faibles, l'erreur absolue moyenne du spectre lissé entre 80 Hz et 12 kHz après appariement fixe du RMS vocal, le niveau de la queue 300–600 ms après le point final, et le sample peak. La queue est décrite par rapport au niveau vocal de chaque sortie; les cas secs n'ont pas de métrique de queue.

![Distance spectrale, préservation de l'enveloppe et niveau des fenêtres faibles](../../results/adobe-curve-eq-2026-10-09/preservation/weak-speech-tradeoff.png)

## Résumé numérique

Médianes par condition, calculées sur trois événements (deux voix) :

| Condition / candidat | Corrélation enveloppe vs sec | Delta p10 des trames vocales vs sec | MAE spectrale vs Adobe |
|---|---:|---:|---:|
| Sec — Adobe v2 | 0,959 | −6,56 dB | 0 dB |
| Sec — NFE64-C | 0,983 | −2,78 dB | 3,23 dB |
| Sec — NFE64-C + EQ | 0,967 | −3,57 dB | 1,69 dB |
| RIR — Adobe v2 | 0,896 | −7,97 dB | 0 dB |
| RIR — NFE64-C | 0,622 | −10,87 dB | 4,49 dB |
| RIR — NFE64-C + EQ | 0,632 | −14,51 dB | 3,67 dB |

Sur les cas RIR, l'EQ gagne donc 0,82 dB de MAE spectrale médiane, mais le p10 des fenêtres vocales faibles perd 3,64 dB de plus qu'avec la baseline; la corrélation d'enveloppe reste très basse et ne remonte que de 0,010. Sur les cas secs, la MAE médiane gagne 1,54 dB, tandis que le niveau faible médian baisse de 0,79 dB et que la corrélation baisse de 0,016.

La médiane du niveau de queue 300–600 ms vs voix, sur les trois paires RIR, est −54,86 dB pour Adobe, −62,28 dB pour NFE64-C et −62,16 dB pour NFE64-C + EQ. L'EQ ne répare pas le cas Naf, dont le niveau de queue relatif reste autour de −38,7 dB pour ces deux sorties, tandis que son enveloppe vocale reste particulièrement faible (0,481 baseline, 0,436 après EQ).

## Décision

**Ne pas activer l'EQ dérivée d'Adobe par défaut.** Elle rend les courbes plus proches, mais ne règle pas les principales erreurs de comportement temporel dans les cas réverbérés. Les fenêtres vocales faibles méritent un contrôle séparé avant toute autre mise en avant de cette courbe. Ces valeurs signalent un risque d'atténuation locale; elles ne prouvent pas à elles seules des phonèmes supprimés, car il n'y a pas de mesure de reconnaissance de mots dans ce test.

Les événements Christiane/Naf ont aussi servi au réglage / à la sélection de la courbe : ce diagnostic est **in-sample**, et les chiffres ne sont pas une preuve de généralisation. Un jeu indépendant et une écoute à niveau égal restent requis avant de juger le son.

## Reproduction et limites

```bash
.venv/bin/python scripts/benchmarks/universal_enhancer/analyze_adobe_curve_preservation.py
```

Le CSV complet et le JSON descriptif restent dans le dossier local ignoré `results/adobe-curve-eq-2026-10-09/preservation/`. Corpus de deux voix, RIR synthétiques, trois événements, point final artificiel; enveloppe et niveau de fenêtre ne sont pas des scores de qualité perceptive. Adobe est une référence de rendu, pas la vérité propre.
