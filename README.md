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

Pour la piste Mamba prioritaire, le pipeline visé est :

1. `scripts/train_latent.py` — latent SFT
2. `scripts/train_halting_head.py` — apprentissage de `HaltingHead`
3. `scripts/build_tool_dataset.py` — dataset tool-use `[AGENT]/<TOOL>/<RESULT>`
4. `scripts/train_tool_sft.py` — tool-use SFT
5. `scripts/export_ssm_binary.py` — export bare-metal `OOSS`

## Démarrage rapide

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python scripts/prepare_dataset.py --input data/raw --output data/processed
python scripts/train_oo_v1.py --config configs/oo_v1_15m.json --dry-run
```

Le script de préparation extrait maintenant automatiquement les premières sources depuis les repos voisins :

- `llm-baremetal` : logs QEMU, scripts autorun, contrat de handoff, receipt OO
- `oo-host` : journal JSONL, export souverain, états/récupération, handoff packs
- `oo-system` : commandes CLI et contraintes de bridge

Artifacts générés :

- `data/raw/extracted_corpus.jsonl`
- `data/raw/source_manifest.json`
- `data/processed/train.jsonl`
- `data/processed/valid.jsonl`
- `data/processed/test.jsonl`
- `data/processed/eval_oo.jsonl`

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
