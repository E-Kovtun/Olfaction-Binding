# Retired, kept as reference

Nothing here is wired into the current pipeline. These are superseded producers and
readers, kept because the results they made are still quoted and because the reasoning
inside them is often the record of *why* the current design looks the way it does.
Read them; do not extend them.

| file | what it was | what replaced it, and why |
|---|---|---|
| `alpha_gate_summary.py` | per-run reader of one v8 alpha sweep: prediction, geometry and a verdict for a single CSV | `scripts/analysis/headline_table.py` (the numbers, across every run at once) and `scripts/analysis/alpha_curves.py` (the geometry as a curve in alpha). One run at a time stopped being the unit of reading once the grid became one command. |
| `structure_function_grid.py` | the (k, phi) grid: two ablation axes over the graph's message passing | the alpha gate. Both of those axes only ever REMOVED information, so neither could pull the receptor cloud back toward ESM -- the structural end was reachable only by not training. A frozen branch the optimizer cannot drain is what turned the pair of ablations into one dial with two known ends. |
| `sf_grid_summary.py` | CLI reader for that grid | — |

The companion notebook is `notebooks/legacy/structure_function_grid.ipynb`.

Results produced by these live under `results/graph/` in whatever directory the run
wrote to; the pre-separation v8 sweeps (ESM node features, both edge variants, the
five-seed series) were moved aside as `results/graph/v8_alpha_gate_reference/` when the
fully separated grid replaced them. To read that archive with the current tools:

```sh
python scripts/analysis/headline_table.py \
    --root results/graph/v8_alpha_gate_reference --nodes esm --all-variants --all-seeds
```
