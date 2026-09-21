# scripts/article_sweeps/

Sweeps that exist **for the article's ablations** rather than for the pipeline. They
train models, so they want a GPU and they cache; `scripts/article_tables/` reads results
and this folder produces them.

Each sweep ships as a pair, the same convention the tables use:

| computes (slow, cached) | reads (fast) | ablation |
|---|---|---|
| `run_quantile_criteria.py` | `notebooks/article_figures/quantile_criteria.ipynb`, over `quantile_grid.py` | how the graph is BUILT: molecule-ranking criterion x coverage quantile |

## Why these live apart from `scripts/modeling/train/`

`run_alpha_gate_sweep.py` hardwires the message-passing variant
(`VARIANTS[args._variant]`) so that a dial run cannot change two things at once. That is
a property worth keeping, so a sweep over the variant itself is a separate script that
**imports the alpha sweep as a module** and reuses its fold preparation, coverage mask,
metric battery and boosting reference unchanged. Same folds, same head, same columns --
a row here is comparable with a row there, and nothing in the older code had to move.

## The construction sweep

```
.venv/bin/python scripts/article_sweeps/run_quantile_criteria.py \
    --dataset m2or --regime inductive transductive \
    --prot-embeddings 'data/embeddings/proteins/esm3_{ds}.npz' \
    --seeds 42 43 --max-parallel 4 --gpus 0 1 2 3
```

Writes one CSV per (dataset, regime) under `results/article_sweeps/quantile_criteria/`,
one row per (criterion, quantile, fold, seed, head, split), plus a `config.json` with the
interpreter and the xgboost version. Resumable at cell granularity: re-run the same
command to continue, `--force` is not needed and does not exist.

Then look at it:

```
jupyter lab notebooks/article_figures/quantile_criteria.ipynb
```

That notebook is the only reader. `quantile_grid.py` is the aggregation layer it
imports -- fold means, intervals, paired deltas -- and has no command line of its own:
the artifact here is a figure, and a second text rendering of the same numbers would be
one more thing to keep in agreement with it.

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
  so the whole sweep collapses to one point — the default switches for you and the run
  says so.
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
