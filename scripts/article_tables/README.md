# scripts/article_tables/

Everything the article's tables are assembled from. All scripts **read** results already on
disk and write LaTeX + a long CSV + a plain-text summary to `results/article_tables/`.
The one exception is `02a_protein_geometry.py`, which computes geometry of frozen
embeddings (CPU, no training).

Run from the repo root with `.venv/bin/python`.

| script | table | inputs |
|---|---|---|
| `00_inventory.py` | — | reports READY / PARTIAL / MISSING for every input cell of every table |
| `01_main_tables.py` | main: all methods × {M2OR, Carey, Hallem} × {transductive, cold molecule} | sweep `results/graph/v9_seeded` (graph α=1, both heads; boost) + `results/ensemble_logs` (LORAX, ProSmith, MolOR, Hladiš) |
| `02a_protein_geometry.py` | (compute) geometry of ESM-1b / ProtT5 / ESM-2 / classical descriptors / one-hot | the sweep's own fold preparation and geometry functions |
| `02_geometry_table.py` | RSA / CCA / Procrustes vs the functional profile | sweep geometry columns at α + `02a` CSVs |
| `03_molecule_ablation.py` | graph vs boost vs Hladiš × ChemBERTa / GIN / ECFP (successor of tab:t2m2or/t2cc/t2hc) | sweep + ensemble_logs |

## Conventions shared by every table (`tablekit.py`)

- **Unit = held-out split.** Sweep rows are averaged over model seeds inside each split
  first. External baselines have one seed per split.
- **Cell = mean ± std over splits.**
- **Significance:** a paired two-sided t-test over splits against the reference row,
  Holm-corrected within the column. Wilcoxon is not used: at 5 splits its smallest
  two-sided p is 0.0625.
- **Rank:** place within each split among all rows, averaged over splits (and over the
  metrics shown, in the main table). Friedman p is in the long CSV.
- **Baseline row selection:** reuses `scripts/analysis/paper_tables.baseline_row`, with
  the molecule source matched on the file, the requested combo first, and `_timing`
  skipped. A baseline fed another molecule embedding is not usable unless
  `--allow-mol-mismatch` is passed.

## Typical order

```
.venv/bin/python scripts/article_tables/00_inventory.py
.venv/bin/python scripts/article_tables/01_main_tables.py
.venv/bin/python scripts/article_tables/03_molecule_ablation.py
.venv/bin/python scripts/article_tables/02a_protein_geometry.py --dataset cc hc
.venv/bin/python scripts/article_tables/02_geometry_table.py
```
