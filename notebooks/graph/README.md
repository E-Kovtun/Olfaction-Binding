# notebooks/graph/

Live notebooks around the signed bipartite graph. All of them are **display / analysis**:
the models and tables they read are produced by scripts, not here.

The graph itself is `orbind/gnn_extractor.py` (`GnnSignedExtractor`), reached through
`scripts/modeling/train/train_ensemble_boost.py` as a `cls` source. Everything from the
earlier standalone graph line (v3-v6 trainers, benchmark sweeps, the alternative-graph
ledger) is archived under [`../../legacy/`](../../legacy/README.md).

```
notebooks/graph/
  alpha_gate/           the alpha dial: where the receptor cloud sits between
                        structure and function, and what that costs in prediction
  mechanism_holdout/    ligand-class holdout: does the refined receptor transfer a
                        MECHANISM to a chemistry it never trained on?
  alternatives/         display-only readers for the quantile x criterion sweeps
```

## `alpha_gate/alpha_gate.ipynb` -- the dial, both halves

The v8 grid in one notebook. Its subject is

    z_prot = (1 - alpha) * frozen_SVD(ESM) + alpha * signed_graph(...)

with **one-hot receptor nodes throughout**, so the protein embedding reaches the model
nowhere but through the frozen branch at weight (1 - alpha). That is what makes alpha an
honest fraction of structure rather than a mixing weight between two things that both
contain ESM, and it is why the earlier ESM-node series cannot be read as more folds of
this one -- with ESM nodes the dial has no upper end.

Two blocks, and the notebook is deliberately nothing else:

* **Geometry** -- how far the receptor cloud sits from each of the two extremes at every
  alpha: alignment to raw ESM and to the train response profile, under RSA, CCA and
  Procrustes, all oriented so larger is closer. The dial working *means* the ESM curve
  falls and the profile curve rises, so that is checked as a number (`audit`) before any
  plot is read. Then the crossover -- the alpha at which the cloud is equally far along
  both dials -- per series and per measure.
* **Performance** -- the head's own score along the same axis, against `boost`
  (XGBoost on raw ESM || molecule) and `naive` (the constant train mean); then the same
  numbers **differenced cell by cell** against boost, which is the panel that carries the
  interval that is actually about the gap. Both arms ran on the same split with the same
  draw, so an unpaired comparison would throw that away.

Knobs at the top: molecular source (`None` keeps both as separate series), node kind,
dataset, regime, model seed, CI level, geometry scale (raw or z against the permutation
null), and which metrics the battery grid shows. **Error bars are over the splits** --
five per series, Student-t, because at n=5 the 1.96 approximation is 29% too narrow.

**All computation lives in `scripts/analysis/alpha_grid.py`**, which the notebook
imports; the same module run as a script prints both blocks as text, for an ssh session
with no browser:

```sh
python scripts/analysis/alpha_grid.py --mol-source chemberta
python scripts/analysis/headline_table.py        # the scoreboard of record
python scripts/analysis/alpha_choice.py --loo    # which alpha do we report?
```

### `alpha_choice.py` -- picking the primary alpha

A different question from either of the above, and it gets its own file because it is a
DECISION and not a reading. The graph-vs-boost comparison is three tables (M2OR, Carey,
Hallem) with a row per (regime x molecule source); one cell here is one of those rows.
Every alpha, plus `boost` and `legacy`, is ranked WITHIN each cell and the ranks are
averaged across all eighteen. The rank is the headline criterion because it is the only
pooled summary that is not a unit error -- Carey's R2 and M2OR's AUROC cannot be
averaged, but their orderings can. Beside it: `cells won` against boost, the paired
advantage per table in its own units, and `dz` (the paired difference over its own
across-split spread) as the one dimensionless pooled column.

The script is built around the fact that **this is selection on the test folds**. Three
guards, all printed whether or not they are convenient:

* **the 1-SE set** -- every alpha within one standard error of the best mean rank.
  Picking the argmax out of a flat set is noise-chasing; among the tied set take the one
  with an argument behind it, and alpha=1 is that one (no protein embedding enters the
  model anywhere, which is the claim).
* **`--loo`** -- choose on two tables, report where that alpha lands on the third.
* **`--metric`** -- if the argmax moves when R2 becomes Spearman or AUROC becomes AUPRC,
  the ordering is inside the noise and only the 1-SE set means anything.

The grid itself comes from `scripts/modeling/train/run_alpha_gate_sweep.py`. The
notebook is safe to open **mid-run**: a series that has only reached its baselines keeps
its panel and says "not run yet", and `coverage` flags a ragged dial, where some alphas
rest on fewer splits than others and the wiggles are partly the run schedule.

## `mechanism_holdout/split_alternatives.ipynb` -- would a different split help?

A self-contained probe, not part of the pipeline: cluster the odorant panel by Tanimoto and by
GIN distance (HDBSCAN, plus agglomerative with k by silhouette), compare the clusters with the
SMARTS functional groups, and score both kinds of group with the same `struct_leak` /
`func_redund` controls the holdout uses. It answers whether a similarity split would be a
different experiment and a better-isolated one, before anything is implemented. Computation and
drawing live together here on purpose -- it is small.

## `mechanism_holdout/` -- one notebook, a dataset flag

Hold out every odorant of a chemical class (SMARTS), train the signed graph without it, then
ask whether the resulting receptor embedding still says something true about that class. Three
receptor representations throughout: **raw ESM** (structure only), **GNN+ESM** (both),
**GNN one-hot** (function only -- the same graph fed a one-hot receptor identity, so the
geometry comes from binding alone).

**All computing lives in `scripts/modeling/analysis/mechanism_holdout.py`.** The notebook reads
its artifacts and draws; set `DATASET` in the first code cell to `m2or`, `cc` or `hc`.

Each dataset writes its own directory and shares nothing with the others, so the three run
side by side, one per GPU -- no merge step, the artifacts land exactly where the serial run
puts them:

```sh
i=0
for d in m2or cc hc; do
  CUDA_VISIBLE_DEVICES=$i .venv/bin/python scripts/modeling/analysis/mechanism_holdout.py --dataset $d > mh_$d.log 2>&1 &
  i=$((i + 1))
done; wait
```

Which refinement graph is a flag: `--variant q99greedy` (M2OR's default -- the q99 + greedy
pair cover the rest of the M2OR paper uses) or `--variant q0cov` (the insects' default -- full
coverage, no quantile cut stacked on the class removal). A non-legacy variant writes to
`<dataset>__<variant>/`, so the two coexist and the notebook's `VARIANT` flag selects one.

`CUDA_VISIBLE_DEVICES` rather than `--device cuda:N`: it also pins whatever the boosting head
and the extractor pick up on their own. A fourth GPU has nothing to do here -- M2OR is the long
pole and stays one process. Serially, on one GPU, it is the same command with `--dataset all`.

Three readouts, deliberately different in kind:

* **Three geometric measures** -- the metrics of record, of increasing strictness: **RSA**
  (neighbour order), **CCA** (shared linear subspace), **Procrustes** (same shape). No head,
  no hyperparameters, nothing predicted. RSA feeds the paper's mechanism table; the other two
  say whether the conclusion depends on which notion of "aligned" one picks. CCA and
  Procrustes reduce both sides to a small common rank, calibrated against the permutation
  null (at rank 10 CCA's null swallows the signal entirely).
* **Predictive OOD** -- the pipeline's own boosting head fitted on pairs outside the class and
  scored on the class, against `naive` and `receptor tuning` references. Agreement between the
  two is the point: a conclusion that survives both does not live in either one's moving parts.
  `--hladis` scores a competitor on exactly the same masks. On cc/hc it runs against TWO
  targets -- the continuous response and its binarisation at the graph's own edge threshold --
  and the notebook's `OOD_METRIC` picks the series and the number together (`R2 | Pearson |
  Spearman` vs `AUROC | AUPRC | MCC | F1 | precision | recall`); M2OR has only the binary one.
  Its axis is symlog -- log in both
  directions around each class's own chance level (what the constant `naive` predictor scores
  there: the naive row for R2, 0.5 for AUROC, the prevalence for AUPRC) -- and
  the rightmost group, past a divider, is the same models aggregated over classes with the
  section-7 trust weights, plus a dimensionless table beneath it.
* **kNN** -- the original leave-one-out readout, kept as a deprecated panel. Do not quote it.

Three post-hoc passes bring an older run up to date without retraining anything, all reading
its own `embeddings.npz`: `--derive` adds the two derived representations and the per-model
nulls (and runs automatically after a fresh run), `--backfill` adds the isolation controls
and null spreads to `nulls.csv`, and `--rescore-ood` refits the boosting head for BOTH target
series and rewrites `ood.csv`:

```sh
.venv/bin/python scripts/modeling/analysis/mechanism_holdout.py --dataset all --backfill --derive --rescore-ood
```

Six representations on M2OR, five on the insects, and the notebook's `SHOW` list picks which
of them the tables and figures use: `raw ESM` (structure), `GNN one-hot` and `retained profile` (function, learned
and not learned), `GNN+ESM` and `GNN + PCA128(ESM)` (both, mixed during training and stapled
together after it), and `tested mask` (M2OR only -- the profile with the responses deleted,
so it carries assay design and no binding; it is the control for reading the profile on a
sparse matrix). The last three are derived after the fact, without training.

Section 7 collapses the per-class numbers into **one per representation**. Dimensionless
first -- each class scored against its own permutation null, in units of that null's spread,
because the three geometries sit on different floors and a small class has a wider null.
Then weighted by `trust = (1 - struct_leak)(1 - func_redund)`: how isolated the holdout
really was, structurally (a retained near-twin of the class) and functionally (the class was
just general tuning). The equal-weight mean is printed beside it, so a conclusion that
depends on the weighting is visible as one. Both leaks come from the artifacts.

`m2or` is kept for completeness but is **not** the stand for the claim -- its sparsity, receptor
cross-correlation and non-random assay design make it unreadable there. The two complete insect
matrices (CC 50x110, HC 24x110) are.

## `alternatives/`

`protein_based_graph.ipynb` (M2OR) and `protein_based_graph_carey.ipynb` (Carey/Hallem)
display the quantile x criterion sweeps behind the appendix. **Display only** -- the CSVs
come from `scripts/modeling/train/run_quantile_criteria_sweep.py`.

Note the two datasets need different readings of `q`: on M2OR the coverage quantile cuts a
long-tailed distribution, while the insect matrices are complete, so coverage is constant
and the quantile is a no-op there. The sweep's `k_mode="fraction"` is what makes the axis
mean anything on Carey/Hallem -- see `orbind/mol_selection.resolve_K`.

## `../legacy/structure_function_grid.ipynb` -- retired

The (k, phi) surface, superseded by the alpha gate: both of its axes only removed
information, so neither could pull the receptor cloud back toward ESM. Kept as
reference. Its successor is `alpha_gate/alpha_gate.ipynb`.

`refinement_geometry/{M2OR,CC,HC}` and the first curve notebook,
`alpha_gate_curves.ipynb`, were retired to [`../legacy/`](../legacy/README.md) at the
same time: the first retrained the historical model inside the notebook to photograph
its cloud, which is now one axis of the alpha grid rather than a picture per dataset;
the second read a melted CSV from a script that no longer exists.
