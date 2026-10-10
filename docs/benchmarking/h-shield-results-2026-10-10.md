# H-shield: préflight synthétique NO-GO (2026-10-10)

## Décision

**Arrêt avant données réelles.** Le mécanisme H-shield ne satisfait pas ses critères synthétiques de confiance après le fit minimal gelé de 400 mises à jour. Aucun fichier du corpus d'entraînement, DEV ou HOLDOUT n'a été lu; aucun forward réel n'a été effectué; aucune source ou split n'a été gelé. Cette expérience ne produit donc aucune affirmation de qualité acoustique réelle ni de parité Adobe.

Le modèle testé est le masque phase-free `g = 1 - 0.5*c*sigmoid(a)`, `c = sigmoid(z)`. Le protocole prévoit `c*=0` sur les trames de parole protégées (activité, voix faible ou onset, avec dilation fixe) et `c*=1` sur W1 éligible, avec `L_conf=1`, `L_guard=1`, `L_early=1`. Tous les coefficients, définitions et seuils du préflight ont été fixés avant l'exécution.

## Résultats synthétiques

Le run a été répété deux fois sur NVIDIA GTX 1050 Ti, PyTorch 2.7.1+cu118, seed 20261010; les deux reçus JSON sont identiques bit à bit, SHA-256 `f33398c699eb43d4fe97bbfd888db5142d9793e3333abbc387447c19603252bb`.

| Critère mécanistique | Résultat (4 fixtures) | Seuil gelé | Verdict |
|---|---:|---:|---|
| Réduction W1 150–300 ms | 5.83–5.94 dB | ≥ 2.5 dB chacune | Passe |
| Gain protégé médian | 0.9898–0.9902 | ≥ 0.995 | Échec |
| Gain protégé p10 | 0.9887–0.9893 | ≥ 0.98 | Passe |
| Confiance médiane sur parole protégée | 0.443–0.450 | ≤ 0.05 | Échec |
| Confiance médiane sur W1 | 0.516–0.587 | ≥ 0.8 | Échec |
| Sortie nulle sur entrée nulle | 0.0 max | < 1e−7 | Passe |
| Normes initiales des gradients base/W1/confidence/guard | 0.0906 / 0.0753 / 0.2735 / 0.00160 | finies et > 0 | Passe |

Le masque réduit fortement l'énergie W1 dans ces fixtures et préserve presque l'amplitude de parole, mais le canal `c` n'apprend pas les labels de confiance préenregistrés. Les mesures synthétiques ne démontrent pas que les seuils auraient prédit la qualité en audio réel; elles montrent seulement que l'architecture/objective telle que spécifiée échoue son verrou de mécanisme, donc le protocole interdit d'aller plus loin avec cette version.

## Provenance et reproductibilité

- Reçu 1 et reçu 2: `.tools/compact-dereverb/h-shield-2026-10-10/synthetic-audit-pass1.json` et `synthetic-audit-pass2.json` (fichiers locaux ignorés).
- Les deux reçus ont le même SHA-256 et les mêmes métriques.
- Modèle: SHA-256 `74b7f22d253b6be9e25ea00976288e32ea89cdce7eb97514384d2b521a86c7dd`.
- Auditeur: SHA-256 `fabfe0d3dca84b92a04cd42f86a3314123c471c521927448f335f2097a5df9cb`.
- Les quatre fixtures synthétiques sont générées en mémoire à partir de signaux déterministes et d'une réponse impulsionnelle synthétique; `training_corpus_read=false`, `dev_or_holdout_read=false`, `test_wav_accessed=false`.
- Temps d'une passe: environ 53 s sur GTX 1050 Ti; aucun fit de corpus et aucune préparation Cap60 n'ont été lancés.

## Conclusion scientifique et suite

La tête de confiance apprend insuffisamment les cibles 0/1 dans le budget synthétique gelé, alors que le gain reste presque neutre sur la parole protégée. Une nouvelle version ne doit pas être faite en ajustant après coup ces seuils ou coefficients. La prochaine hypothèse devra porter un nouvel identifiant d'expérience, réexaminer si une confiance apprise est utile par rapport à un masque directement protégé, définir un préflight synthétique distinct, puis répéter les deux audits avant toute donnée réelle. Le DEV G2 consommé ne sera pas réutilisé; son HOLDOUT reste fermé.
