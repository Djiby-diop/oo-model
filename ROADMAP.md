# oo-model Roadmap

## Track A — Transformer (oo-v1-15m)

- [x] Créer le repo `oo-model`
- [x] Définir la spec `oo-v1-15m`
- [x] Définir le schéma dataset OO
- [x] Extraire les premières sources
- [x] Générer corpus `train/valid/test`
- [ ] Fine-tune orienté système
- [ ] Quantiser + export bare-metal

## Track B — Mamba SSM (oo-v1-mamba-130m) ← PRIORITÉ

Architecture : Mamba SSM + dark loops (=) + HaltingHead

Inspiré par batteryphil/llm-baremetal — latent reasoning engine.

OO cognitive loop : `dark_loop → D+ judgment → action`

### Phase 1 — Latent SFT

- [x] Config `oo_v1_mamba_130m.json`
- [x] Architecture `mamba_model.py` (OOMambaEngine + HaltingHead)
- [x] Dataset OO (`build_dataset.py`) — chat/math/code/system
- [x] Script entraînement `train_latent.py`
- [ ] Lancer Phase 1 sur `state-spaces/mamba-130m-hf`

### Phase 2 — HaltingHead

- [x] Script `train_halting_head.py` (fractional ramp labels)
- [ ] Entraîner HaltingHead sur hidden states Phase 1
- [ ] Vérifier P(halt) spread (éviter collapse)

### Phase 3 — Export bare-metal

- [x] Script `export_ssm_binary.py` (→ OOSS binary format)
- [ ] Valider layout avec `ssm_infer.c` (batteryphil fork)
- [ ] Test d'inférence sur QEMU

### Phase 4 — Tool-Use SFT

- [x] Script `build_tool_dataset.py`
- [x] Script `train_tool_sft.py`
- [ ] Générer dataset `tool_use.jsonl`
- [ ] Fine-tune Mamba sur format `[AGENT] -> <TOOL: BASH> -> <RESULT>`

### Phase 5 — Intégration OO

- [ ] Câbler export → `llm-baremetal` ssm_infer.c
- [ ] Connecter HaltingHead → système de pressure OO
- [ ] Dark loops → D+ policy cycles
- [ ] Phase 6: persistent h_t state → kv_persist

## Track C — Native Lineage (futur)

- [ ] Tokenizer propre OO
- [ ] Architecture from-scratch compacte
- [ ] Dataset OO étendu (oo-lab → oo-model pipeline)
- [ ] Première lignée native `oo-native-v1`
