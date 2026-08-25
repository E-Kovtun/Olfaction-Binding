> **ARCHIVED.** This describes the graph line as it stood before the ensembler
> (`scripts/modeling/train/train_ensemble_boost.py`) replaced the standalone trainers, and
> before `orbind/gnn_extractor.py` became the surviving descendant. Paths below point at
> files that now live under `scripts/legacy/` and `notebooks/legacy/`. It is kept for the
> ledger it contains -- the "none of the alternatives beats raw boost" verdict and the
> transductive/inductive flip -- not as a guide to the current tree.

# notebooks/graph/

Evaluation of the **heterogeneous bipartite GNN/GAT link predictor** (molecule ↔ protein)
across the curated and full_full dataset variants, plus mechanism analyses and the ledger of
alternative graph formulations.

> **Protocol status — exploratory, rerun required.** Existing graph checkpoints,
> tables, plots, and notebook outputs must not be treated as final benchmark results.
> The old runs have three evaluation problems: the `full_full` cold-molecule split is
> reconstructed from the same row universe for every nominal LORAX fold; train labels
> are also used as message-passing edges; and the original sweeps used the final epoch
> rather than a validation-selected checkpoint. Future runs should create genuinely
> distinct group-molecule folds, checkpoint by validation metrics, and touch test only
> once after model selection. The notebooks remain useful as exploratory diagnostics.

All notebooks read checkpoints/CSVs produced by `scripts/modeling/train/` and
`scripts/modeling/eval/`. They resolve the repo root by walking up to `pyproject.toml`,
so they run correctly from any sub-folder depth.

## Layout

The notebooks are grouped by role. Filenames are short because the sub-folder already
carries the context (no more `graph_`/`gnn_` prefixes).

```
notebooks/graph/
  benchmarks/     the main link-predictor sweeps, one per dataset variant
  mechanism/      why/how the graph adds signal + training diagnostics
  alternatives/   molecule-side / CF / metapath graphs — the "none beats raw boost" ledger
```

## The model

A bipartite graph: molecule nodes (ChemBERTa / GIN features) and protein nodes (ESM2-650M).
Message-passing enriches the **protein** embedding from the molecules each receptor binds.
The reported head is an unentangled **XGBoost probe** on `[raw mol ‖ graph-enriched prot]`
(the MLP probe was dropped). Knobs swept: MP mode (`pos_only` / `all_edges` / `signed`),
architecture (GNN / GAT), molecule-coverage quality filter (`q87/q95/q99`), history depth
(`[x0‖x1]` vs `[x0‖x1‖x2]`), and — in v6 — representation-shaping add-ons (DGI auxiliary
loss, SimGCL contrastive noise) and an alternative BPR main loss.

Transductive runs have an additional opt-in probe, `--transductive-exp`, using
`[graph-enriched mol ‖ graph-enriched prot]`. The default transductive probe remains
`[raw mol ‖ graph-enriched prot]`; inductive-molecule always keeps the raw molecule
embedding because held-out molecules have no graph context. Checkpoint/table variants
from the new probe carry the `_transductive_exp` suffix. A transductive molecule whose
train edges were removed (for example by the quality filter) still receives the GNN/GAT
root/self transformation, but no neighbor-derived context; this is expected.

For a stricter probe protocol, `--disjoint-probe-train` partitions the original train
edges into two class-stratified subsets. GNN/GAT message passing and decoder training
use only the first; XGBoost/MLP fitting uses only the second. Thus no downstream-probe
training label was already present as the exact same MP edge. The default split is 50/50
and result variants carry the `_disjoint` suffix.

## Notebooks

### `benchmarks/`

| notebook | dataset | regimes | what it shows |
|----------|---------|---------|---------------|
| `full_full.ipynb` | full_full (LORAX/Hladis release) | transductive + inductive_molecule | **the main sweep.** LORAX folds; EC50-only test; ChemBERTa‖ESM; the v5 grid (MP-mode / arch / quality / history) + the v6 add-on studies (DGI, SimGCL, BPR) |
| `full_full_compressed.ipynb` | full_full | both | capacity variant — same folds with PCA32/PCA16 domain embeddings; two forks only (regime × supervision) |
| `curated.ipynb` | curated (409 rec / 21k pairs) | transductive + inductive_molecule | the MP-mode / arch / quality / history sweep on curated vs no-graph XGBoost (GIN‖ESM) |

### `mechanism/`

| notebook | dataset | regimes | what it shows |
|----------|---------|---------|---------------|
| `inductive_vs_transductive.ipynb` | full_full | both | **mechanism**: the `rawp` ablation (`[ChemBERTa ‖ raw ESM ‖ z_prot]`) that disentangles "graph compresses away raw ESM" from "graph adds transferable signal", plus the cold-molecule enrichment ladder and a compressed/full capacity check |
| `training_diagnostics.ipynb` | full_full | inductive_molecule | **training curves** (900-epoch study + clean 1500-epoch rerun): val vs test per epoch, loss, correlation scatter, key-epoch table; shows why an early 300-epoch stop is suboptimal |

### `alternatives/`

| notebook | what it shows |
|----------|---------------|
| `molecule_side_graphs.ipynb` | combined ledger of the molecule-side / interaction-matrix graphs — **A/B/C/D** (receptor co-response, collaborative SVD, metapath M-P-M-P), **interaction-profile CF** (borrow neighbours' binding profiles over the ChemBERTa kNN graph), and **similarity-graph MP** (one hop over the ChemBERTa kNN graph, untrained `Â·X` vs trained). Shared verdict: **none beats the raw XGBoost boost** over multiple seeds |

## Provisional observations (full_full, fold 1; require rerun)

- **Quality filter is the strongest knob**: `signed + q95` lifts the graph well above plain
  `signed` (`q99` over-prunes and degrades).
- **History helps only transductive**; depth barely matters (`[x0‖x1]` ≈ `[x0‖x1‖x2]`).
- **The transductive↔inductive flip is real**: the no-graph baseline is inflated in
  transductive by **identity memorization** (test molecules 99% seen) and collapses when
  molecules go cold; the graph degrades far less.
- **`rawp` ablation verdict**: in transductive the graph is ≈ lossy compression of ESM
  (raw ESM recovers most of the gap, `z_prot` adds nothing extra); in inductive the pure
  graph **beats** baseline and raw ESM even *hurts* — the binding-profile signal genuinely
  transfers to new molecules. See `mechanism/inductive_vs_transductive.ipynb`.
- **Alternative molecule-side graphs don't help**: co-response, SVD-CF, metapath,
  profile-CF, and similarity-MP all fail to beat raw boost over multiple seeds. See
  `alternatives/molecule_side_graphs.ipynb`.

## Data sources

`scripts/modeling/train/train_gnn_link.py`, `train_gat_link.py` → `results/curated/{checkpoints,tables}/`
and `results/full/...`; `train_graph_full_full.py` → an explicit directory under `results/graph/`
(v5/v6 sweeps live in `results/graph/full_full/...`);
`scripts/modeling/eval/eval_full_full_baseline.py` → `results/full_full/tables/baselines.csv`;
`eval_on_lorax_splits.py` → `results/full_full/article_results/lorax_compare.csv`.
