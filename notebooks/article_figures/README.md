# notebooks/article_figures/

The paper's figures. Each notebook reads a producer's output and draws; none trains
anything. How to produce their inputs is in [`README3.md`](../../README3.md).

| notebook | paper artifact | reads | first knobs |
|---|---|---|---|
| `prediction_dial.ipynb` | `fig:dial` (Appendix B.1) | the OlfaGraph sweep, P1a | `ROOT_DIR` = that sweep's root, `SPLIT = "test"` |
| `quantile_criteria.ipynb` | `fig:construction` (Appendix C) | the construction sweep, P6 | `ROOT_DIR` = `results/article_sweeps/quantile_criteria`, `SPLIT = "val"` |
| `alpha_rank_dial.ipynb` | — | P1a | superseded by `prediction_dial` |
| `geometry_dial.ipynb` | — | P1a | the parked geometry line |
| `figkit.py` | — | — | the shared visual contract: rcParams, palette, panels, bands, legends, `savefig` |

With `SAVE_FIGS = True` each notebook writes PNG and PDF under
`results/article_figures/<notebook's folder>/`. The figures are built at their printed
width (`FIG_WIDTH_IN = 7.1`), so fonts do not shrink when the PDF is placed.

**`prediction_dial`** draws a 3×2 grid (datasets × settings) of the paired difference
between OlfaGraph and XGBoost-base along α, one curve for each form (reduced and full),
each with a least-squares line and a per-panel slope test. The slope is fitted per split,
and the five slopes are the sample of the t-test, Holm-corrected over the two forms in a
panel. The cell under the figure prints the slopes as a table. `SPLIT = "test"` is used
because nothing is chosen on this figure: α=1 is the model, not the argmax.

**`quantile_criteria`** draws M2OR's two settings, one curve per odorant-selection
criterion against the coverage quantile, as paired differences from XGBoost-base on the
validation split, with the reported configuration (greedy pair cover, q=0.99) ringed. The
cell under the figure prints, per split and per cell of the grid, the number of odorants
kept, the difference from XGBoost-base and the paired comparison with the reported cell
(Holm over the grid).

## The rule these notebooks are built around

**They draw and do not aggregate.** Every mean and interval comes from
[`scripts/analysis/alpha_grid.py`](../../scripts/analysis/alpha_grid.py) (or, for the
construction sweep, `scripts/article_sweeps/s4_quantile_grid.py`, which delegates to it),
which

1. averages the **model seeds inside each split**, then
2. takes a Student-t interval over the **splits**.

A split changes which rows are held out, which is what a claim about generalisation is
over; a seed changes only the draw the same model made on the same rows. Treating the 25
cells of a five-seed grid as 25 observations would use t(24) where the honest value is
t(4), and divide by √25 where the effective sample is 5 — roughly halving every interval.

## Two traps the notebooks refuse rather than warn about

**Two dials must not share an axis.** The paper's dial (`--dial nodes`) mixes the graph's
*input* between receptor identity (α=0) and ESM3 (α=1). An earlier parameterisation
(`--dial gate`) mixes the graph's output and runs the other way. Each notebook raises if
the loaded frame holds both.

**A root holding two runs is refused.** The producers are resumable, so an older run left
in the output directory would be concatenated with the reported one. The construction
notebook checks the provenance columns and stops if it finds more than one configuration.
