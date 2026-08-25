# notebooks/graph/

Live notebooks around the signed bipartite graph. All of them are **display / analysis**:
the models and tables they read are produced by scripts, not here.

The graph itself is `orbind/gnn_extractor.py` (`GnnSignedExtractor`), reached through
`scripts/modeling/train/train_ensemble_boost.py` as a `cls` source. Everything from the
earlier standalone graph line (v3-v6 trainers, benchmark sweeps, the alternative-graph
ledger) is archived under [`../../legacy/`](../../legacy/README.md).

```
notebooks/graph/
  mechanism_holdout/    ligand-class holdout: does the refined receptor transfer a
                        MECHANISM to a chemistry it never trained on?
  refinement_geometry/  what refinement does to the receptor geometry
  alternatives/         display-only readers for the quantile x criterion sweeps
```

## `mechanism_holdout/` -- `{M2OR, CC, HC}`

Hold out every molecule of a chemical class (SMARTS), train the signed graph without it,
then ask whether the resulting receptor embedding still says something true about that
class. Three receptor representations throughout: **raw ESM** (structure only),
**GNN+ESM** (both), **GNN one-hot** (function only -- the same graph fed a one-hot
receptor identity, so the geometry comes from binding alone).

Two readouts, deliberately different in kind:

* **RSA / Mantel** -- the metric of record. Spearman between the off-diagonals of
  embedding similarity and residualised held-out-class-profile similarity. No head, no
  hyperparameters, predicts nothing. Feeds the paper's mechanism table.
* **Predictive OOD** (section 5) -- the pipeline's own boosting head fitted on pairs
  outside the class and scored on the class, against `naive` and `receptor tuning`
  references. Agreement between the two is the point: a conclusion that survives both
  does not live in either one's moving parts. Section 6 optionally runs a competitor
  (Hladis) on exactly the same masks.

`M2OR` is kept for completeness but is **not** the stand for the RSA claim -- its
sparsity, receptor cross-correlation and non-random assay design make it unreadable
there. The two complete insect matrices (CC 50x110, HC 24x110) are.

The legacy LOO-kNN readout is still present as a deprecated panel; do not quote it.

## `refinement_geometry/` -- `{M2OR, CC, HC}`

The geometry companion to the same study: what moves when the receptor vector is refined.

## `alternatives/`

`protein_based_graph.ipynb` (M2OR) and `protein_based_graph_carey.ipynb` (Carey/Hallem)
display the quantile x criterion sweeps behind the appendix. **Display only** -- the CSVs
come from `scripts/modeling/train/run_quantile_criteria_sweep.py`.

Note the two datasets need different readings of `q`: on M2OR the coverage quantile cuts a
long-tailed distribution, while the insect matrices are complete, so coverage is constant
and the quantile is a no-op there. The sweep's `k_mode="fraction"` is what makes the axis
mean anything on Carey/Hallem -- see `orbind/mol_selection.resolve_K`.
