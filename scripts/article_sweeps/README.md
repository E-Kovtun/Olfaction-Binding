# scripts/article_sweeps/

Sweeps that exist **for the article's ablations** rather than for the pipeline. They
train models, so they want a GPU and they cache; `scripts/article_tables/` reads results
and this folder produces them.

In [`README3.md`](../../README3.md) these are producers **P5** (`s5_run_architecture.py`,
the architecture table) and **P6** (`s4_run_quantile_criteria.py`, Appendix C); the exact
invocations the paper uses are there. The readers with the same prefixes are in
`../article_tables/`, except the construction sweep's, which is a notebook.


Each sweep ships as a pair, the same convention the tables use:

| computes (slow, cached) | reads (fast) | ablation |
|---|---|---|
| `s4_run_quantile_criteria.py` | `notebooks/article_figures/quantile_criteria.ipynb`, over `s4_quantile_grid.py` | how the graph is BUILT: molecule-ranking criterion x coverage quantile |
| `s5_run_architecture.py` | `../article_tables/s5_architecture.py` | which message-passing OPERATOR: the graph pinned, only the operator moving |

## Why these live apart from `scripts/modeling/train/`

`run_alpha_gate_sweep.py` hardwires the message-passing variant
(`VARIANTS[args._variant]`) so that a dial run cannot change two things at once. That is
a property worth keeping, so a sweep over the variant itself is a separate script that
**imports the alpha sweep as a module** and reuses its fold preparation, coverage mask,
metric battery and boosting reference unchanged. Same folds, same head, same columns --
a row here is comparable with a row there, and nothing in the older code had to move.

## The architecture sweep

```
# the four operators (GraphSAGE, GAT, GraphConv, GIN)
.venv/bin/python scripts/article_sweeps/s5_run_architecture.py \
    --dataset m2or cc hc --regime transductive inductive \
    --prot-embeddings 'data/embeddings/proteins/esm3_{ds}.npz' \
    --max-parallel 4 --gpus 0 1 2 3

# the encoder ablations: one layer, positive edges only, unsigned edges, no message passing
.venv/bin/python scripts/article_sweeps/s5_run_architecture.py \
    --dataset m2or cc hc --regime transductive inductive \
    --conv sage:paper:1layer sage:paper:pos sage:paper:unsigned none \
    --prot-embeddings 'data/embeddings/proteins/esm3_{ds}.npz' \
    --max-parallel 4 --gpus 0 1 2 3
```

The graph is pinned per dataset (M2OR: greedy pair cover at q=0.99; insects: the complete
panel), so only the operator or the ablated component moves. Seeds 42-46 and graph
seeding are on by default. `:paper` names the paper's encoder configuration (neighbour
sampling 25/10, per-layer L2 normalisation); specs without it train the earlier,
unsampled encoder and are not in the table. Rendered by
`../article_tables/s5_architecture.py`.

## The construction sweep

```
.venv/bin/python scripts/article_sweeps/s4_run_quantile_criteria.py \
    --dataset m2or --regime inductive transductive \
    --prot-embeddings 'data/embeddings/proteins/esm3_{ds}.npz' \
    --seed-graph --seeds 42 43 44 45 46 --max-parallel 4 --gpus 0 1 2 3
```

This is the paper's run: M2OR only, both settings, five seeds. Mosquito and Fly are
complete panels, where the coverage cut removes nothing, and the paper always uses their
complete graph.

Writes one CSV per (dataset, regime) under `results/article_sweeps/quantile_criteria/`,
one row per (criterion, quantile, fold, seed, head, split), plus a `config.json` with the
interpreter and the xgboost version. Resumable at cell granularity: re-run the same
command to continue, `--force` is not needed and does not exist.

Then look at it:

```
jupyter lab notebooks/article_figures/quantile_criteria.ipynb
```

That notebook is the only reader. It draws the paper's figure (the paired difference
from XGBoost-base against q, M2OR seen and cold molecules, on validation) and, in the cell
after it, prints the numbers the appendix quotes. `s4_quantile_grid.py` is the aggregation
layer it imports -- fold means, intervals, paired deltas -- and has no command line of its
own.

### Four things that decide whether the result means anything

* **Quantiles are fractions** (`0.99`), not percents. The older study script took
  percents; a `99` reaching the extractor keeps nothing and the failure looks like a
  modelling result.
* **The cell the paper reports must be inside the grid** — `greedy_pair_cover` at
  `q=0.99` on M2OR, `coverage` at `q=0` on the insects. It is what the ablation is
  *about*; the reader marks it on every panel and flags it when it is missing.
* **`--k-mode` decides what a quantile means.** `coverage_quantile` cuts on the
  coverage distribution (M2OR's reading); `fraction` keeps the top (1-q) share
  outright. On the complete insect matrices the first keeps every molecule at every q,
  so the script switches to `fraction` there. That thinning is available but is not in
  the paper, which uses the complete insect graphs throughout.
* **Pass the protein npz the comparison run used.** The one-hot block is not what
  changes here, but the file still decides the coverage mask, and a different mask is a
  different set of rows in every fold.

### What the sweep records that is not a metric

`K` (how many molecules survived on that fold) and `status`. A cell that cannot be
computed is written as NaN with the reason rather than dropped: at a tiny K every kept
molecule can land on one side of `edge_threshold`, and signed message passing needs both
signs. The hole is a finding about that construction. The reader drops those rows from
every aggregate — a NaN inside an interval would shorten its `n` silently.

## Choosing a knob on these figures

Reading the best quantile off the curves and then reporting those same curves picks the
number and its defence from the same rows. Either run with `--no-train-scores` and read
the notebook at `SPLIT = "val"`, or make the weaker claim the figures support directly:
over the range where the peak is inside the grid's resolution, the construction does not
matter, and the cell we report is not measurably behind the best one.
