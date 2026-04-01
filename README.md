# oo-model

Modèle natif pour l'écosystème OO.

## But

`oo-model` contient la lignée de modèles spécialisés pour OO :

- petit modèle compatible avec `llm-baremetal`
- dataset orienté boot, journal, policy, recovery, continuity
- pipeline d'entraînement ciblé système plutôt que chatbot généraliste
- évaluation sur tâches réelles OO

## Vision v1

La première cible n'est pas un grand LLM généraliste. C'est un modèle compact, robuste et intégrable.

Cible initiale :

- 15M à 60M paramètres
- vocabulaire technique/système
- contexte court à moyen
- quantisation agressive possible
- comportement déterministe et utile en runtime

## Structure

- `configs/` : configurations de modèles et runs
- `data/` : datasets bruts, nettoyés et évaluations
- `docs/` : spécifications et schémas
- `scripts/` : préparation dataset, entraînement, évaluation
- `src/oo_model/` : code commun de config et helpers

## Démarrage rapide

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python scripts/prepare_dataset.py --input data/raw --output data/processed
python scripts/train_oo_v1.py --config configs/oo_v1_15m.json --dry-run
```

## Priorités

1. définir le modèle `oo-v1-15m`
2. figer le schéma dataset OO
3. extraire le corpus depuis les journaux et handoffs existants
4. faire un premier run dry-run reproductible
5. préparer l'intégration bare-metal

## Premières métriques

- compréhension de commande OO
- résumé de journal
- diagnostic de boot
- proposition d'action sûre
- cohérence de continuité host ↔ sovereign
- taux d'hallucination sur état système
