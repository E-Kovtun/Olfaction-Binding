# scripts/

Main M2OR pipeline. Everything here runs in the **root project venv** (`uv run python scripts/...`).
Each script finds the repo root by walking up to `pyproject.toml`, so it can be invoked from anywhere.

Self-contained experiments with their own dependencies live in [`../experiments/`](../experiments/) instead
(see that folder's README for why they are kept separate, not merged here).

## Pipeline stages (rough order)

| stage | folder | what it does |
|-------|--------|--------------|
| 0. download    | `downloading/`          | fetch raw M2OR |
| 1. preprocess  | `preprocessing/`        | build the pairs table; Ballesteros–Weinstein residue numbering |
| 2. embeddings  | `embedding_generation/` | molecule (GIN) and protein (ESM2-650M: mean / ECL2 / per-residue) embeddings |
| 3. modeling    | `modeling/`             | train models, build baseline tables, run analyses |

File-name number prefixes (`00_`, `01_`, …) encode the original global order within a stage.

## modeling/ layout

Was a flat dump; now split by purpose:

```
modeling/
  train/        model-fitting entry points
    train_gnn_link.py        bipartite GraphSAGE link predictor (curated / full)
    train_gat_link.py        bipartite GAT link predictor (curated / full)
    train_graph_full_full.py GNN/GAT on full_full (LORAX folds, EC50 test, 2 regimes)
    train_mp.py              LORAX-style concat[mol||prot] -> MLP baseline
    train_attention.py       curated flat/site cross- and self-attention (incremental table)

  eval/         baselines & comparison tables (produce CSVs)
    eval_mp_table.py         MP table: protein {ESM,random} x split {stratified,molecule}
    eval_protein_variants.py XGBoost across protein-embedding variants
    eval_onehot_protein.py   control: one-hot protein blocks (no ESM)
    eval_on_lorax_splits.py  our boost on LORAX splits (reproduces their protocol)
    eval_full_full_baseline.py  no-graph boost baseline for full_full (both regimes)
    _append_concat22.py / _append_concat22pca.py   append-only helpers for
                             eval_protein_variants (must stay beside it — sibling import)

  analysis/     mechanism investigations (not just leaderboard numbers)
    pocket_binding_signal.py / _v2.py   does pocket/ECL2 divergence predict binding
                             divergence, controlling for phylogeny
    (interaction ladder + bilinear structure → notebooks/interaction_research.ipynb)
```

Notebooks in [`../notebooks/`](../notebooks/) consume the CSVs / checkpoints these write.
