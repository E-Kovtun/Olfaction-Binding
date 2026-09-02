# Retired notebooks, kept as reference

Nothing here is wired into the current pipeline, and none of it should be extended.
They are kept because the results they made are still quoted and because the reasoning
inside them is often the record of *why* the current design looks the way it does. The
scripts they read are archived beside them in
[`../../scripts/legacy/`](../../scripts/legacy/README.md).

| notebook | what it was | what replaced it, and why |
|---|---|---|
| `refinement_geometry/{CC,HC,M2OR}.ipynb` | what refinement does to the receptor geometry: 2D PCA scatters of the receptor cloud before message passing (raw ESM) and after it, one per fold, with receptors grouped by functional and by structural clustering | the alpha gate. These notebooks **retrain the historical `_SignedSage`** inside the notebook to look at its cloud, which is both slow and a second copy of the model definition; and their question -- "how far did refinement move the receptors from ESM" -- is now one axis of a grid rather than a picture per dataset. `notebooks/graph/alpha_gate/alpha_gate.ipynb` answers it at every alpha, over five splits, with the same three measures and a permutation null under each. |
| `alpha_gate_curves.ipynb` | the first reader of the alpha sweep: geometry and prediction against alpha, one seed, ESM node features | the same notebook rebuilt on the **fully separated** grid (one-hot nodes throughout, so alpha is an honest fraction of structure), with error bars over the splits and a knob for the molecular source. It read `curves_long.csv` from `alpha_curves.py`; the current one reads the metrics CSVs directly through `scripts/analysis/alpha_grid.py`, so a knob changes a figure instead of requiring a script rerun. |
| `structure_function_grid.ipynb` | the (k, phi) surface: two ablation axes over the graph's message passing | the alpha gate. Both of those axes only ever REMOVED information, so neither could pull the receptor cloud back toward ESM -- the structural end was reachable only by not training. A frozen branch the optimizer cannot drain is what turned the pair of ablations into one dial with two known ends. |

## Reading the old results with the current tools

The pre-separation v8 sweeps -- ESM node features, both edge variants, the five-seed
series -- were moved aside as `results/graph/v8_alpha_gate_reference/` when the fully
separated grid replaced them:

```sh
python scripts/analysis/headline_table.py \
    --root results/graph/v8_alpha_gate_reference --nodes esm --all-variants --all-seeds
python scripts/analysis/alpha_grid.py \
    --root results/graph/v8_alpha_gate_reference --nodes esm
```

`alpha_grid` takes `--nodes esm` for exactly this reason. Read those numbers as a
*different experiment*, not as more folds of the current one: with ESM node features
the protein embedding still reaches the receptor vector through message passing at
alpha=1, so that dial has no upper end.
