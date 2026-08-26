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
  `--hladis` scores a competitor on exactly the same masks.
* **kNN** -- the original leave-one-out readout, kept as a deprecated panel. Do not quote it.

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
