# notebooks/article_figures/

The figures the article uses, separated from the exploratory dial notebooks in
`notebooks/graph/`. Those were written to *find out* what the dial does; these are
written to *show* it, on one sweep, with one visual contract.

```text
figkit.py             the visual contract: rcParams, palette, panels, bands, legends
geometry_dial.ipynb   alignment with structure vs function, along alpha
prediction_dial.ipynb predictive metrics and the paired advantage over the base
```

Both notebooks read the ESM3 sweep (`results/graph/v13_esm3`) by default; the root is
the first knob in each.

## The rule these notebooks are built around

**They draw and do not aggregate.** Every number comes from
[`scripts/analysis/alpha_grid.py`](../../scripts/analysis/alpha_grid.py), which

1. averages the **model seeds inside each fold**, then
2. takes a Student-t interval over the **folds**.

A fold changes which rows are held out, which is what a claim about generalisation is
over; a seed changes only the draw the same model made on the same rows. Treating the
25 cells of a five-seed grid as 25 observations uses t(24) where the honest value is
t(4) and divides by √25 where the effective sample is 5 — it roughly halves every
interval, and every "significant" gap read off the figure inherits that.

So a cell here may slice, label and plot an `alpha_grid` frame. It may not compute a
mean, and it may not build an interval.

## The visual conventions

| element | what it means |
|---|---|
| solid line + band | our graph at that alpha; band = CI **over folds** |
| dashed grey | the boosting base (`prot+mol`) on the same folds |
| dotted pale | naive (constant train mean) |
| blue / orange | against ESM (structure) / against the response profile (function) |
| grey I-mark | the resolution floor — model noise left in a fold mean |

The I-mark replaces per-fold whiskers on the curve. The between-fold spread is several
times wider than the model noise, so whiskers would answer "how different are the
folds" on a figure asking "did moving the dial change anything". The mark answers the
second question: a bump shorter than it is noise whatever the band does.

The palette is taken verbatim from `notebooks/graph/alpha_gate/`, where it was checked
pairwise for colour-vision deficiency against a white ground (worst all-pairs ΔE 9.3
deutan, 17.6 normal). Reuse is deliberate — the same hue meaning two different things
across figures of one paper is harder to catch than a bad hue.

## Two traps the notebooks refuse rather than warn about

**The two dials must not share an axis.** On a v8 *gate* run alpha mixes the graph's
output against a frozen ESM branch; on a v9 *node* dial it mixes the graph's input and
runs the other way (alpha=0 is receptor identity, alpha=1 the legacy graph). Each
notebook raises if the loaded frame holds both.

**Alpha may not be chosen here.** Choosing on these curves and then reporting them
takes the number and its defence from the same rows. The honest path is
`scripts/analysis/val_rescore.py` then `alpha_choice.py --select-on val`, which is what
`notebooks/graph/node_dial/` does.
