# scripts/article_tables/

Everything the article's tables are assembled from. All scripts **read** results already
on disk and write LaTeX + a long CSV + a plain-text summary to `results/article_tables/`.

**The prefix is the registry item** (`paper/PLAN.md`), renamed 23.09.2026: `m1_` for the
main-text battery, `s2_`/`s3_`/`s5_`/`s6_` for supplementary items A2, A3.2, A5 and A6,
and no prefix for a tool that serves every table. The old prefixes (`00`, `01`, `03`,
`05`, `05a`, `07`, `08`) were the order the scripts were written in -- they looked like a
sequence and corresponded to nothing.

Two gaps in the numbering are real and deliberate. There is no `s1_`: the supplementary
baselines table is `m1_main_tables.py` with `--baseline-combo cls --no-ours`, the same
reader answering a narrower question. There is no `s4_` here either: A4's reader is a
notebook, and its compute half lives in `../article_sweeps/`. A3.1 is a notebook
too, which is why `s3_` names only the identity control.

Retired to `scripts/legacy/` on the same day, because no current table is built from
them: `02a_protein_geometry.py` and `02_geometry_table.py` (the geometry line is parked)
and `06_alpha_choice_not_used.py` (replaced by a figure). `04_compare_runs.py` went with
them. All four still run; they are simply not in any chain.

Run from the repo root with `.venv/bin/python`.

| script | item | table | inputs |
|---|---|---|---|
| `inventory.py` | tool | — | reports READY / PARTIAL / MISSING for every input cell of every table |
| `m1_main_tables.py` | **M1** + **A1** | main: all methods × {M2OR, Carey, Hallem} × {transductive, cold molecule} | sweep `results/graph/v9_seeded` (graph α=1, both heads; boost) + `results/ensemble_logs` (LORAX, ProSmith, MolOR, Hladiš) |
| `s6_molecule_ablation.py` | **A6** | graph vs boost vs Hladiš × ChemBERTa / GIN / ECFP (successor of tab:t2m2or/t2cc/t2hc) | sweep + ensemble_logs |
| `s3_onehot_boost.py` | **A3.2** | (compute) the boosting head over [one-hot receptor ‖ molecule] — the row no sweep writes | the sweep's own fold prep, `fit_boost` and metric battery |
| `s3_alpha0_vs_boost.py` | **A3.2** | the identity control: our graph at alpha=0 vs boost over ESM and vs the one-hot boost | sweep + `s3_onehot_boost`'s CSVs |
| `s5_architecture.py` | **A5** | the architecture table (`tab:arch`): one row per message-passing operator, six columns (dataset x regime), the boosting base as the anchor | `s5_run_architecture.py`'s CSVs under results/article_sweeps/architecture |
| `s2_protein_sources.py` | **A2** | what the receptor side has to be: our graph, pLMs, the classical floor and the one-hot controls under one head; metric of record only by default | `prot_floor_sweep.py --gnn` CSVs under `results/tables/` |

Every table takes `--dataset` from `paper_tables.DATASET_ORDER`, which includes the
shrunk insect panels (`cc_shrinked`, `hc_shrinked`, `*_shrinked50`). Their graph rows
live in their own sweep, so pass it: `--sweep-root results/graph/v11_shrunk` (0.063) or
`results/graph/v12_shrunk50` (0.5). On them the `*_fun` geometry columns carry the
same assay-design contamination as M2OR's — they now have a measured-cell mask.

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
.venv/bin/python scripts/article_tables/inventory.py
.venv/bin/python scripts/article_tables/m1_main_tables.py
.venv/bin/python scripts/article_tables/s6_molecule_ablation.py
.venv/bin/python scripts/legacy/02a_protein_geometry.py --dataset cc hc
.venv/bin/python scripts/legacy/02_geometry_table.py
```
