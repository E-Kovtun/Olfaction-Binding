# notebooks/graph/

Evaluation of the **heterogeneous bipartite GNN/GAT link predictor** (molecule ↔ protein)
across our three dataset variants, plus a mechanism analysis.

All notebooks read checkpoints/CSVs produced by `scripts/modeling/train/` and
`scripts/modeling/eval/`. They resolve the repo root by walking up to `pyproject.toml`,
so they run correctly from this sub-folder.

## The model

A bipartite graph: molecule nodes (ChemBERTa / GIN features) and protein nodes (ESM2-650M).
Message-passing enriches the **protein** embedding from the molecules each receptor binds.
The reported head is an unentangled **XGBoost probe** on `[raw mol ‖ graph-enriched prot]`
(the MLP probe was dropped). Knobs swept: MP mode (`pos_only` / `all_edges` / `signed`),
architecture (GNN / GAT), molecule-coverage quality filter (`q87/q95/q99`), and
history depth (`[x0‖x1]` vs `[x0‖x1‖x2]`).

## Notebooks

| notebook | dataset | regimes | what it shows |
|----------|---------|---------|---------------|
| `graph_evaluation_curated.ipynb`  | curated (409 rec / 21k pairs) | transductive + inductive_molecule | MP-mode / arch / quality / history sweeps vs no-graph XGBoost (GIN‖ESM) |
| `graph_evaluation_full.ipynb`     | full (780 rec / 30k pairs)    | transductive + inductive_molecule | same sweeps on the larger noisy set |
| `graph_evaluation_full_full.ipynb`| full_full (LORAX/Hladis release) | transductive + inductive_molecule | LORAX folds; **EC50-only test**; ChemBERTa‖ESM; the main sweep grid + D (quality) + E (history) + F (best+history-depth) |
| `graph_inductive-transductive_analysis.ipynb` | full_full | both | **mechanism**: the `rawp` ablation (`[ChemBERTa ‖ raw ESM ‖ z_prot]`) that disentangles "graph compresses away raw ESM" from "graph adds transferable signal" |
| `gnn_training_diagnostics.ipynb` | full_full | inductive_molecule | **900-epoch training curves** (val vs test per epoch, loss, correlation scatter, key-epoch table); shows why 300-epoch early stop is suboptimal |

## Key findings (full_full, fold 1)

- **Quality filter is the strongest knob**: `signed + q95` lifts the graph well above plain
  `signed` (`q99` over-prunes and degrades).
- **History helps only transductive**; depth barely matters (`[x0‖x1]` ≈ `[x0‖x1‖x2]`).
- **The transductive↔inductive flip is real**: the no-graph baseline is inflated in
  transductive by **identity memorization** (test molecules 99% seen) and collapses when
  molecules go cold; the graph degrades far less.
- **`rawp` ablation verdict**: in transductive the graph is ≈ lossy compression of ESM
  (raw ESM recovers most of the gap, `z_prot` adds nothing extra); in inductive the pure
  graph **beats** baseline and raw ESM even *hurts* — the binding-profile signal genuinely
  transfers to new molecules. See the analysis notebook.

## Data sources

`scripts/modeling/train/train_gnn_link.py`, `train_gat_link.py` → `results/{checkpoints,tables}/`
and `results/full/...`; `train_graph_full_full.py` → `results/full_full/checkpoints/`;
`scripts/modeling/eval/eval_full_full_baseline.py` → `results/full_full/tables/baselines.csv`;
`eval_on_lorax_splits.py` → `results/lorax/lorax_compare.csv`.
