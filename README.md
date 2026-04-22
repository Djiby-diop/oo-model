# oo-model

Native model stack for the OO ecosystem.

## Purpose

`oo-model` contains the specialized model lineage for OO:

- compact model compatible with `llm-baremetal`
- datasets focused on boot, journal, policy, recovery, and continuity
- system-targeted training pipeline rather than general chatbot behavior
- evaluation on real OO tasks

## v1 Vision

The first target is not a large general-purpose LLM. It is a compact, robust, integrable model.

Initial target:

- 15M to 60M parameters
- technical/system vocabulary
- short-to-medium context window
- aggressive quantization support
- deterministic and runtime-useful behavior

## Structure

- `configs/`: model and run configurations
- `data/`: raw/processed datasets and evaluations
- `docs/`: specifications and schemas
- `scripts/`: dataset prep, training, evaluation
- `src/oo_model/`: shared config and helper code

For the prioritized Mamba track, the target pipeline is:

1. `scripts/train_latent.py` — latent SFT
2. `scripts/train_halting_head.py` — `HaltingHead` training
3. `scripts/build_tool_dataset.py` — tool-use dataset `[AGENT]/<TOOL>/<RESULT>`
4. `scripts/train_tool_sft.py` — tool-use SFT
5. `scripts/export_mamb_binary.py` — bare-metal runtime export `MAMB`
6. `scripts/export_ssm_binary.py` — enriched OO export `OOSS`

V1 integration rule:

- `MAMB` = loadable format for the current `llm-baremetal` runtime
- `OOSS` = enriched OO-SomaMind extension (`HaltingHead`, OO metadata)

Reference contracts:

- [../oo-system/docs/OO_SOMAMIND_V1_INTEGRATION_CONTRACT.md](../oo-system/docs/OO_SOMAMIND_V1_INTEGRATION_CONTRACT.md)
- [../llm-baremetal/docs/OO_SOMAMIND_RUNTIME_CONTRACT.md](../llm-baremetal/docs/OO_SOMAMIND_RUNTIME_CONTRACT.md)

## Quickstart

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python scripts/prepare_dataset.py --input data/raw --output data/processed
python scripts/train_oo_v1.py --config configs/oo_v1_15m.json --dry-run
```

The preparation script now automatically extracts initial sources from sibling repositories:

- `llm-baremetal`: QEMU logs, autorun scripts, handoff contract, OO receipt
- `oo-host`: JSONL journal, sovereign export, state/recovery data, handoff packs
- `oo-system`: CLI commands and bridge constraints

Generated artifacts:

- `data/raw/extracted_corpus.jsonl`
- `data/raw/source_manifest.json`
- `data/processed/train.jsonl`
- `data/processed/valid.jsonl`
- `data/processed/test.jsonl`
- `data/processed/eval_oo.jsonl`

## Lightweight tools without torch

Some preparation commands do not require the full ML stack yet.
The `oo_model` package exposes `load_config` in a lightweight path, without loading
`torch` until a model is actually imported.

Useful examples for Windows and fast validation:

```bash
python -c "import sys; from pathlib import Path; sys.path.insert(0, str(Path('src').resolve())); from oo_model import load_config; print(load_config('configs/oo_v1_15m.json').keys())"
python scripts/prepare_dataset.py --input data/raw --output data/processed
python -m unittest discover -s tests -v
```

Installing `requirements.txt` remains required for training, evaluation,
and model export.

## Priorities

1. define the `oo-v1-15m` model
2. freeze the OO dataset schema
3. extract corpus from existing journals and handoffs
4. run a first reproducible dry-run
5. prepare bare-metal integration

## First metrics

- OO command understanding
- journal summarization quality
- boot diagnostic accuracy
- safe action proposal quality
- host-to-sovereign continuity coherence
- hallucination rate on system state
