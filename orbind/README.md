# orbind/

The library. Everything under `scripts/` and `notebooks/` is a caller; the logic
lives here.

Scripts put the repo root on `sys.path` themselves, so `orbind` imports without
being installed (`package = false` in `pyproject.toml` — this is an application,
not a distributable).

## Documentation

The ensembler is large enough to have its own write-up rather than a docstring:

* [`docs/ensembler.md`](docs/ensembler.md) — how the mechanism works layer by
  layer, **and which half of it we deliberately do not use** (combo stacking and
  per-head tuning are implemented, and switched off in everything reported).
* [`docs/gotchas.md`](docs/gotchas.md) — the pipeline-wide traps, each with the
  failure that produced it.

## Map

**Data and splits**

| module | what |
|---|---|
| `dataset.py` | pair tables, npz loading, the metric functions, the generic splitters |
| `filters.py` | M2OR curation rules (human only, no mutants, mono-molecular, binary, no orphan receptors) |
| `regimes.py` | the `full_full` M2OR regime over LORAX's pool: `transductive`, `inductive_molecule`, `inductive_molecule_v5` |
| `regimes_ofm.py` | Carey / Hallem: pools, the four split families, the continuous target |
| `tasks.py` | the classification/regression axis — which head, which metrics, which criterion |

`regimes.py` and `regimes_ofm.py` are deliberately not unified: different pool,
different key, different target type, and upstream ships splits in different
shapes. Each docstring says so at length.

**The engine**

| module | what |
|---|---|
| `ensemble.py` | the multi-source boosting ensemble: extractor protocol, the combo mini-language, coverage intersection, the per-combo head |
| `baselines.py` | the boosting/MLP heads themselves (`fit_boost`, `tune_boost`, `predict_scores`) |
| `mol_selection.py` | which molecules become graph nodes: 7 ranking criteria, `resolve_K`'s two readings of the quantile |

**Extractors** — one per source type in the trainer's dispatch table. Each takes
row-positions into a shared `pairs` frame and returns a vector per row; how it gets
there is its own business.

| module | source type |
|---|---|
| `gnn_extractor.py` | `gnn_signed`, `gnn_signed_dgi` — **ours** |
| `gnn_lora_extractor.py` | `gnn_lora` — two-stage LORAX-molecule + our graph |
| `lorax_extractor.py` | `lorax` |
| `prosmith_extractor.py` | `prosmith` |
| `molor_extractor.py` | `molor` (needs `.venv-molor`) |
| `hladis_extractor.py` | `hladis` |
| `attention_extractor.py` | `attn_noisy_or`, `attn_lse` (site-MIL; reachable, unused) |

Extractor modules are imported **lazily** by the trainer, so a run pulls only the
deps its sources need. That is what lets the ProSmith/LORAX controls run in a
PyG-free environment.

**`docs/`** — prose about the ensembler and the pipeline's traps, above.

**`legacy/`** — four archived modules (`hetero`, `hetero_gat`, `lorax`,
`attention`), kept importable because archived scripts and notebooks import them.
Nothing live does. `orbind.legacy.lorax` is also the reference definition that
`regimes.inductive_molecule_v5_indices` reproduces position for position.

## Conventions

* **Proteins are keyed by amino-acid sequence, molecules by InChIKey**, everywhere.
* Metrics are chosen by task, never mixed: AUROC/AUPRC/MCC/F1 for classification,
  R²/RMSE/MAE/Pearson/Spearman for regression.
* An extractor raises on an unknown key; the caller decides whether to drop
  uncovered pairs (`--on-missing drop`).
