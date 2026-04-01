# oo-model Roadmap

## Phase 1 — Foundation

- [x] Créer le repo `oo-model`
- [x] Définir la spec `oo-v1-15m`
- [x] Définir le schéma dataset OO
- [ ] Extraire les premières sources depuis `llm-baremetal`, `oo-host`, `oo-system`
- [ ] Générer un premier corpus `train/valid/test`

## Phase 2 — Bootstrap Model

- [ ] Choisir le modèle bootstrap open-weight
- [ ] Préparer tokenizer/vocab cible
- [ ] Lancer un premier dry-run d'entraînement
- [ ] Définir la batterie `eval_oo`

## Phase 3 — OO-Tuned Model

- [ ] Fine-tune orienté système
- [ ] Mesurer hallucination et cohérence
- [ ] Quantiser pour intégration
- [ ] Préparer export vers runtime bare-metal

## Phase 4 — Native Lineage

- [ ] Tokenizer propre OO
- [ ] Architecture from-scratch compacte
- [ ] Dataset OO étendu
- [ ] Première lignée native `oo-native-v1`
