# orbind

Research code and executable notebooks for modelling olfactory receptor–molecule interactions in M2OR. The repository contains the experiment definitions, analysis notebooks, and training/evaluation code. Datasets, embeddings, model checkpoints, logs, and generated result files are intentionally distributed separately.

## Quick start

The supported runtime is CPython 3.11. The main environment is managed by uv in the repository-level `.venv`.

```bash
uv python install 3.11
uv sync --frozen
uv run python -m ipykernel install --user --name orbind --display-name "orbind (uv)"
uv run jupyter lab
```

Open notebooks from the repository root. Their setup cells locate the root via `pyproject.toml`, so they also work when Jupyter starts inside a notebook subdirectory.

The default PyTorch build installed on this workstation is CPU-only. GPU execution is supported by the graph scripts through `--device auto`, but the CUDA PyTorch installation must be prepared and validated on the target machine separately.

## External data

`data/` is not versioned. Unpack the external data bundle into the repository so that paths begin with `data/processed/`, `data/embeddings/`, and, for the full_full/LORAX `transductive` splits, `data/splits_indexes/lorax_m2or/` (see that folder's README for why these 5 folds are kept as their own borrowed split rather than merged into our own conventions). The current code deliberately uses repository-relative paths rather than machine-specific absolute paths.

Until the data bundle receives a formal manifest, the notebooks themselves are the most precise record of the files required by each experiment.

## Notebooks

All experiment notebooks are versioned and are the primary research record.

- `notebooks/baseline_screening.ipynb` — curated baseline comparison.
- `notebooks/interaction_research.ipynb` — interaction-feature experiments.
- `notebooks/graph/graph_evaluation_curated.ipynb` — curated graph evaluation.
- `notebooks/graph/graph_evaluation_full_full.ipynb` — standard full_full GNN/GAT architecture screen and v5 analysis.
- `notebooks/graph/graph_evaluation_full_full_compressed.ipynb` — compressed-embedding graph experiments.
- `notebooks/graph/graph_inductive-transductive_analysis.ipynb` — inductive/transductive mechanisms and molecule enrichment.
- `notebooks/graph/gnn_training_diagnostics.ipynb` — training-history and stability diagnostics.
- `notebooks/molecule_embeddings/` — molecular embedding and PCA screening.
- `notebooks/protein_embeddings/` — ESM embedding, PCA, and neighbourhood analyses.

Notebook outputs retain scientific figures and concise summaries. Transient tracebacks, full run inventories, and long training logs should not be committed.

## Main commands

```bash
# Build the curated pair table from the M2OR export
uv run python scripts/preprocessing/01_build_table.py

# Generate receptor and molecule representations
uv run python scripts/embedding_generation/proteins/02_embed_receptors.py
uv run python scripts/embedding_generation/molecules/03_embed_molecules.py

# Baseline pair model
uv run python scripts/modeling/train/train_mp.py --split group_molecule

# Resumable full_full v5 graph grid (PowerShell)
./scripts/modeling/train/run_graph_full_full_v5.ps1
```

Training outputs are written below `results/`. That directory is local-only and ignored by Git; move or archive it separately when transferring experiments to a server.

## Environment policy

`pyproject.toml` and `uv.lock` are the sole dependency specification for the main project. Use `uv sync --frozen` to reproduce it; do not install packages manually into `.venv`.

`experiments/struct_interaction/` is a legacy docking exploration retained for provenance. It has no separate environment and is not part of the active notebook or modelling pipeline.

The core environment currently verified on Windows is Python 3.11 with Torch 2.2.2, DGL 2.2.1, PyG 2.8, NumPy 1.26, and XGBoost 2.1.

## Repository layout

```text
orbind/       reusable dataset, embedding, baseline, and graph code
scripts/      preprocessing, embedding generation, training, and evaluation entry points
notebooks/    versioned experiment definitions and scientific outputs
data/         external datasets and embeddings (ignored)
results/      generated metrics, logs, figures, and checkpoints (ignored)
experiments/  isolated exploratory directions
notes/        experiment notes and protocol decisions
```
