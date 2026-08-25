# legacy/notebooks/

Per-notebook detail for the archive. The narrative index -- what each closed line
showed -- is [`../README.md`](../README.md).

The tree mirrors the live `notebooks/` one level down. These still run: they use the
same root-walk to `pyproject.toml`, and the archived library modules they import are
kept importable as `orbind.legacy.*`.

What stayed live, and why:

* `notebooks/graph/mechanism_holdout/mechanism_holdout.ipynb` -- the ligand-class holdout, now
  ONE notebook with a dataset flag, reading artifacts from
  `scripts/modeling/analysis/mechanism_holdout.py`. The three per-dataset notebooks it replaced
  are archived here under `graph/mechanism_holdout/`: they carried the compute inline, which is
  exactly what moved to the script.
* `notebooks/graph/refinement_geometry/{M2OR,CC,HC}` -- the geometry companion.
* `notebooks/graph/alternatives/protein_based_graph{,_carey}` -- display-only readers
  for the quantile x criterion sweeps (appendix).
* `notebooks/datasets/carey_hallem_carlson_overview` -- the dataset description
  behind the paper's data section.

## What is archived here

**`protein_embeddings/`** (6) — the receptor-representation exploration that produced the
standing conclusion: ready-made protein encoders are interchangeable for this task, ESM
carries receptor *identity* plus a weak neighbourhood prior rather than ligand
specificity, and learning a receptor embedding from scratch does not beat frozen ESM
transductively. The paper's protein-side table is produced by
`scripts/modeling/analysis/prot_floor_sweep.py`, not by these.

**`molecule_embeddings/`** (2) — molecule-source screening. Superseded by the ECFP/GIN/
ChemBERTa columns that are now run through the ensembler proper.

**`baselines_research/`** (4) — attention / site-MIL and the early MLP-vs-XGBoost screen,
plus one ROC-curve notebook. Same line as `legacy/scripts/modeling/train/train_attention*`.

**`graph/benchmarks/`** (3) — `curated`, `full_full`, `full_full_compressed`: benchmark
notebooks from before results moved into `results/ensemble_logs/` and the dashboards.

**`graph/alternatives/molecule_side_graphs`** — the molecule-side aggregation null
result described in `legacy/scripts/README.md`.

**`graph/`** (3, formerly `graph/legacy/`) — `inductive_split_seed_check`,
`inductive_vs_transductive`, `training_diagnostics`. Diagnostics from the graph line;
the collapse fix they diagnosed (lr 1e-3 + grad clip + plateau scheduler) is long since
in `GnnSignedExtractor`'s defaults.

**`interaction/interaction_nature_gin_esm`** — the interaction-nature probe (bilinear /
SHAP) from the interaction branch.

## Note on file size

Roughly half of these carry saved outputs, which is most of the ~12 MB the notebooks add
to the repository. If that ever matters, stripping outputs here is safe — these are an
archive, and their numbers of record live in the memory notes and `results/`.
