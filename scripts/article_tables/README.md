# scripts/article_tables/

The readers that assemble the paper's tables. Every script here **reads** results already
on disk and writes LaTeX, a long CSV and a plain-text rendering under
`results/article_tables/`. None trains a graph. The one exception, `s3_onehot_boost.py`,
fits XGBoost heads and caches them, and is listed as producer P4 in
[`README3.md`](../../README3.md).

The exact command for each paper table, and which producer output it needs, is in
[`README3.md`](../../README3.md) §5. Run from the repo root with `.venv/bin/python`.

| script | paper artifact | reads |
|---|---|---|
| `m1_main_tables.py` | `tab:main_results` (main table); with `--dataset m2or --baseline-combo cls --no-ours`, `tab:esm3t1` (App. A) | `--sweep-root` (P1) + `--ensemble-root` (P2) |
| `s2_protein_sources.py` | `tab:protein_representations` | `results/tables/` (P3) |
| `s5_architecture.py` | `tab:graph_ablation` | `results/article_sweeps/architecture/` (P5) |
| `s3_onehot_boost.py` | — (fits the one-hot heads for `tab:alpha0`) | the sweep's own fold preparation; writes `results/article_tables/onehot_boost/` |
| `s3_alpha0_vs_boost.py` | `tab:alpha0` (App. B.2) | `--sweep-root` (P1) + the one-hot cache (P4) |
| `s6_molecule_ablation.py` | `tab:mol` (App. D) | `--sweep-root` (P1, all three molecular embeddings) + `--ensemble-root` (P2, Hladiš) |
| `inventory.py` | — | READY / PARTIAL / MISSING for every input cell of every table |
| `tablekit.py` | — | the conventions below, shared by every script |

The figures of Appendices B and C are drawn by notebooks in
[`notebooks/article_figures/`](../../notebooks/article_figures/), not here.

**The prefix is the table's place in the paper's argument**: `m1_` for the main table,
`s2_`–`s6_` for the ablations in the order they were planned. There is no `s1_` (App. A
is `m1_main_tables.py` answering a narrower question) and no `s4_` (the construction
ablation is a figure). `sweep-root` and `ensemble-root` have no usable default: pass them
every time, and always as a pair built on the same protein embedding.

## Conventions shared by every table (`tablekit.py`)

- **Unit = held-out split.** Sweep rows are averaged over model seeds inside each split
  first. Baseline runs have one seed per split.
- **Spread.** `m1_main_tables.py` prints mean ± standard deviation over splits;
  `s2_`, `s3_alpha0_vs_boost.py` and `s5_` print the half-width of the 95% Student-t
  interval; `s6_` prints the interval by default and the standard deviation with
  `--spread std`. At n = 5 the interval is 1.24× the standard deviation, so the two are
  not interchangeable in a caption.
- **Paired differences** (`s3_alpha0_vs_boost.py`): the difference is taken inside each
  split and then averaged, with a paired t-test Holm-corrected over the comparisons of a
  row.
- **Significance:** a paired two-sided t-test over splits against the reference row,
  Holm-corrected within the column. Wilcoxon is not used: at five splits its smallest
  two-sided p is 0.0625.
- **Rank:** place within each split among the ranked rows, averaged over splits (and over
  the metrics shown, in the main table).
- **Baseline row selection:** `scripts/analysis/paper_tables.baseline_row` finds a
  baseline run by its `config.json`, matches the molecular embedding on the file it was
  fed, takes the requested feature set and skips `_timing` rows. A baseline fed another
  molecular embedding is not used unless `--allow-mol-mismatch` is passed.

Every script also accepts the sparsified insect panels (`cc_shrinked`, `hc_shrinked`,
`*_shrinked50`) of a closed experiment; they are not in the paper.
