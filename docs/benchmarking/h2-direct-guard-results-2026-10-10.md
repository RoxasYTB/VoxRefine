# H2 direct guard: synthétique NO-GO (2026-10-10)

## Décision

**Arrêt au préflight, avant toute donnée réelle.** L'expérience H2 `g = 1 - 0.5 sigmoid(a)`, phase-free, avec `L = L_early + L_guard`, protège bien l'énergie de parole dans les fixtures mais ne satisfait pas la gate W1 médiane préenregistrée. Pas de corpus préparé, aucun speaker sélectionné, aucune inférence Cap60, et ni DEV ni HOLDOUT n'ont été lus. `test.wav` n'a pas été accédé. Ce résultat ne démontre ni qualité réelle ni parité avec Adobe.

## Protocole figé

- Quatre voix synthétiques déterministes, quatre queues RIR procédurales distinctes, 16 kHz, 2 secondes.
- Masques de protection issus uniquement de la source propre : activité RMS > 2 % du pic, weak < 35 % du pic, onset > 1.5× le RMS du cadre précédent, dilatation ±20 ms.
- 400 mises à jour AdamW, `lr=2e-4`, `weight_decay=1e-4`, batch cyclique fixe, clipping 3, seed 20261011.
- `L_early` est le hinge W1 existant vers 3 dB, `L_guard = mean_P(((1-g)/0.5)^2)`, poids unitaires. Aucun coefficient ou seuil n'a été changé après exécution.
- Audits waveform : W1 ≥2.5 dB chaque fixture et médiane ≥3 dB; active p50 entre ±0.25 dB et p10 >−0.75 dB; weak p10 >−1 dB et ≤3 % des cadres sous −3 dB; onset p10 >−0.75 dB; sortie nulle exactement nulle.

## Résultats

Les deux passes sur GTX 1050 Ti (PyTorch 2.7.1+cu118) sont identiques bit à bit. La médiane W1 vaut **2.786 dB**, sous le seuil **3.000 dB**. Chaque fixture passe cependant son seuil individuel de 2.5 dB. Les métriques de préservation passent toutes : active p50 autour de −0.095 dB, active p10 autour de −0.098 dB, weak p10 autour de −0.095 dB, aucune trame weak sous −3 dB, onset p10 autour de −0.093 dB. L'invariant zéro est exact (`max abs = 0`). Normes initiales de gradients `L_early` / `L_guard` : 0.12895 / 1.43908, finies et non nulles.

| Fixture | Réduction W1 (dB) | Active p50 (dB) | Active p10 (dB) | Weak p10 (dB) | Weak <−3 dB | Onset p10 (dB) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.700 | −0.096 | −0.099 | −0.097 | 0 % | −0.092 |
| 2 | 2.765 | −0.094 | −0.098 | −0.093 | 0 % | −0.095 |
| 3 | 2.839 | −0.096 | −0.098 | −0.095 | 0 % | −0.091 |
| 4 | 2.807 | −0.096 | −0.098 | −0.093 | 0 % | −0.093 |

## Interprétation limitée

Sur ces fixtures et ce budget précis, le garde-fou de parole domine assez pour préserver la voix, mais la réduction médiane de queue reste 0.214 dB sous la cible. Le hinge W1 à 3 dB peut cesser de fournir un gradient une fois le seuil atteint pour un exemple, tandis que la pénalité de garde continue à pousser le modèle vers l'identité sur la parole protégée; les mesures ne suffisent pas à conclure si cette interaction explique seule l'écart. Il serait invalide de corriger les poids, les seuils ou le nombre d'updates après avoir vu cette sortie et de présenter cela comme le même préflight.

La prochaine expérience devra donc avoir un identifiant séparé et une hypothèse distincte (p. ex. traitement temporel de la queue), avec critères waveform préenregistrés. Aucun test sur audio réel n'est autorisé par cette expérience NO-GO.

## Provenance

- Auditeur : `scripts/benchmarks/universal_enhancer/compact_dereverb/audit_h2_direct_guard.py`
- Modèle : `scripts/benchmarks/universal_enhancer/compact_dereverb/model_h2_direct_guard.py`
- Reçus ignorés sous `.tools/compact-dereverb/h2-direct-guard-2026-10-10/` : `synthetic-audit-pass1.json`, `synthetic-audit-pass2.json`.
- Les deux reçus sont identiques bit à bit : SHA-256 `55e674950c2d66338ad48f3c893022decf2939c1d6ff7ed686f6c73c67ffce78`.
- Modèle SHA-256 : `075e5af38c6ca99a58d3a3d4b54309e43896894d484ea5a875f43a83c82debf5`.
- Auditeur SHA-256 : `625f3f784b93036af795f66bae784657be0a5990fb1559a038a3b2fab6686759`.
- `training_corpus_read=false`, `dev_or_holdout_read=false`, `test_wav_accessed=false`.
