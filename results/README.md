# results/ layout

Organized by artifact type. Curated-dataset artifacts live at the top level;
full-dataset artifacts mirror the same structure under `full/`.

Graph `gnn_*.csv` / `gat_*.csv` tables and checkpoints are currently treated as
local generated artifacts: their original evaluation protocol needs correction and
the sweeps will likely be rerun. Baseline, attention, LORAX comparison, analysis, and
explicitly documented training-history tables remain suitable for version control.

```
results/
  checkpoints/      # trained GNN/GAT .pt files  (gnn_*.pt, gat_*.pt)
  tables/           # metric tables written by training/eval scripts
                    #   gnn_link_results_*.csv, gat_link_results_*.csv
                    #   boost_results.{csv,md}, mlp_results.{csv,md}
  analysis/         # one-off analysis CSVs
                    #   pocket_binding_signal*.csv, protein_variants_boost.csv,
                    #   onehot_protein_boost.csv
  lorax/            # external-benchmark comparison (lorax_compare.csv)
  full/
    checkpoints/    # full-dataset .pt
    tables/         # full-dataset boost_results.* etc.
```

## Who writes where
| script | output |
|---|---|
| `train_gnn_link.py --results-dir results/` | `checkpoints/`, `tables/` |
| `train_gat_link.py --results-dir results/` | `checkpoints/`, `tables/` |
| `eval_mp_table.py --out-dir results/` | `tables/boost_results.*`, `tables/mlp_results.*` |
| `eval_on_lorax_splits.py` | `lorax/lorax_compare.csv` |
| `pocket_binding_signal*.py`, `eval_protein_variants.py`, `eval_onehot_protein.py` | `analysis/` |

For the full dataset, pass `--results-dir results/full/` (or `--out-dir results/full/`).
