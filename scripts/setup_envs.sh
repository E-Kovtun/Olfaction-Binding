#!/usr/bin/env bash
# Create the project's isolated uv environments. Five in all:
#
#   project     -> ./.venv            OlfaGraph, XGBoost-base, Hladis, every fitter and
#                                     reader, and the molecule embeddings (GIN needs the
#                                     `gin` extra). Made by `uv sync --frozen --extra gin`,
#                                     NOT by this script.
#   env 1  controls    -> .venv-controls    ProSmith + LORAX baselines, PyG-FREE
#                         (torch + xgboost/scikit-learn/optuna + transformers/peft)
#   env 2  embeddings  -> .venv-embeddings  ProtT5 embeddings
#                         (torch + transformers + sentencepiece)
#   env 3  molor       -> .venv-molor       the MolOR baseline (dgl 2.4 + dgllife)
#   env 4  esm         -> .venv-esm         ESM3 embeddings (EvolutionaryScale `esm` SDK,
#                         whose package name clashes with fair-esm -- hence its own env)
#
# Why the split: the source dispatch in train_ensemble_boost.py is lazy, so a
# ProSmith/LORAX run imports no torch_geometric -- the controls env can drop the
# fragile torch<->PyG pin entirely. Embedding generation (deepchem/rdkit) is the
# messiest dependency set and runs once, so it gets its own throwaway env.
#
# Requires `uv` on PATH. Run from anywhere; it cd's to the repo root.
#
#   bash scripts/setup_envs.sh
#
# Override defaults if the server needs them (e.g. a different CUDA wheel index
# or python version):
#   PYVER=3.11 TORCH_INDEX=https://download.pytorch.org/whl/cu124 bash scripts/setup_envs.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PYVER="${PYVER:-3.11}"
# CUDA 12.6 driver (nvidia-smi 560.x) runs cu124 wheels. Change if your box differs.
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv not on PATH" >&2; exit 1
fi

echo "############################################################"
echo "# env 1: controls (ProSmith + LORAX) -> .venv-controls   (PyG-free)"
echo "############################################################"
uv venv .venv-controls --python "$PYVER"
# torch from the CUDA wheel index first (that index only serves torch & friends)...
uv pip install --python .venv-controls/bin/python torch --index-url "$TORCH_INDEX"
# ...then the rest from PyPI. transformers+peft are only needed by LORAX's live
# ChemBERTa; ProSmith needs neither. No torch_geometric, no deepchem here.
# xgboost is PINNED to the same major the project env uses. Two reasons, and the
# second one alone would be enough: (a) 3.x allocates its device vectors through
# CUDA virtual memory (cuMemCreate), which aborts on some of this box's cards --
# see orbind/docs/gotchas.md; (b) the paper claims ONE fixed boosting head for
# every method, and two majors are two implementations of it.
uv pip install --python .venv-controls/bin/python \
    "xgboost>=2.0,<3.0" scikit-learn optuna pandas numpy transformers peft

echo "############################################################"
echo "# env 2: embeddings (run-once) -> .venv-embeddings   (ProtT5)"
echo "############################################################"
uv venv .venv-embeddings --python "$PYVER"
uv pip install --python .venv-embeddings/bin/python torch --index-url "$TORCH_INDEX"
# sentencepiece is ProtT5's tokenizer; transformers does not pull it in by itself.
uv pip install --python .venv-embeddings/bin/python \
    transformers sentencepiece numpy pandas

echo "############################################################"
echo "# env 3: molor -> .venv-molor   (MolOR baseline)"
echo "############################################################"
# dgl ships its CUDA wheels from its own index, per torch minor version, so torch is
# pinned to the minor that index serves. dgl BEFORE dgllife: installing dgllife first
# pulls a CPU dgl from PyPI that the CUDA wheel then cannot replace cleanly.
uv venv .venv-molor --python "$PYVER"
uv pip install --python .venv-molor/bin/python "torch==2.4.*" --index-url "$TORCH_INDEX"
uv pip install --python .venv-molor/bin/python "dgl==2.4.*" \
    -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
uv pip install --python .venv-molor/bin/python \
    dgllife rdkit "xgboost>=2.0,<3.0" scikit-learn pandas numpy

echo "############################################################"
echo "# env 4: esm -> .venv-esm   (ESM3 embeddings)"
echo "############################################################"
# The weights download from HuggingFace on first use; if that answers 401,
# `export HF_TOKEN=<read token>` and rerun. httpx is imported by the SDK but not
# declared by it.
uv venv .venv-esm --python "$PYVER"
uv pip install --python .venv-esm/bin/python torch --index-url "$TORCH_INDEX"
uv pip install --python .venv-esm/bin/python esm httpx pandas scikit-learn

cat <<'EOF'

############################################################
# done.
#   env 1 controls    -> .venv-controls    (ProSmith + LORAX)
#   env 2 embeddings  -> .venv-embeddings  (ProtT5)
#   env 3 molor       -> .venv-molor       (MolOR)
#   env 4 esm         -> .venv-esm         (ESM3)
#   project           -> ./.venv, from `uv sync --frozen --extra gin` (not this script)
#
# Run a control in the controls env (note: call its python directly; the script
# adds the repo root to sys.path itself, so `orbind` imports without an install):
#
#   .venv-controls/bin/python scripts/modeling/train/train_ensemble_boost.py \
#       --regime full_full --full-full-mode inductive_molecule_v5 \
#       --run-name inductive_lorax_chemberta --source cls=lorax \
#       --combos "1" --weight-method simplex --on-missing drop \
#       --max-parallel 1 --repeats 42
############################################################
EOF
