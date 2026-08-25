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

```
modeling/
  train/
    train_ensemble_boost.py        THE entry point: multi-source boosting ensemble.
                                   Every paper table on M2OR/Carey/Hallem comes from here.
    run_quantile_criteria_sweep.py quantile x criterion sweep of the pipeline GNN
                                   (appendix); read by
                                   notebooks/graph/alternatives/protein_based_graph*.ipynb

  eval/         the protein-side head-to-head (Table "protein sources")
    eval_protein_variants.py       XGBoost across protein-embedding variants
    _append_concat22.py / _append_concat22pca.py   append-only helpers; they import
                                   eval_protein_variants as a sibling, so they must
                                   stay beside it
    eval_onehot_protein.py         control: one-hot protein blocks (no ESM)
    build_pocket_variants_cache.py caches the pocket/ECL2 receptor variants

  analysis/
    prot_floor_sweep.py            produces the protein-source table cited in the paper
    pocket_binding_signal_v2.py    does pocket/ECL2 divergence predict binding
                                   divergence, controlling for phylogeny
    c9_protein_repr_analysis.py    information-criteria battery (paper section 8, pending)
```

## analysis/ (repo-level dashboards)

```
analysis/
  summarize_runs.py          one row per (run, combo) over results/ensemble_logs/,
                             mean +- 95% CI, task-aware columns
  extend_runs_with_combo.py  prints the commands that add a cls+prot+mol row to
                             existing cls-only runs, reusing their checkpoints
```

## legacy/

Closed experiment lines live in [`legacy/`](legacy/), mirroring this tree one level down
(`legacy/modeling/train/`, `legacy/modeling/eval/`, `legacy/queues/`, ...). Nothing there
feeds a paper table; most of it is a negative result worth not repeating. See
[`legacy/README.md`](legacy/README.md) for what each line showed.

Notebooks in [`../notebooks/`](../notebooks/) consume the CSVs / checkpoints these write.
