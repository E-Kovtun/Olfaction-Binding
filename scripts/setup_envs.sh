#!/usr/bin/env bash
# Create the project's isolated uv environments.
#
#   env 1  controls    -> .venv-controls    ProSmith + LORAX baselines, PyG-FREE
#                         (torch + xgboost/scikit-learn/optuna + transformers/peft)
#   env 2  embeddings  -> .venv-embeddings  run-once embedding generation
#                         (torch + fair-esm + transformers + deepchem + rdkit)
#   env 3  project     -> your EXISTING ./.venv  (GNN + ensemble pipeline; PyG).
#                         NOT created or touched by this script.
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
echo "# env 2: embeddings (run-once) -> .venv-embeddings"
echo "#   NOTE: deepchem/rdkit resolution can be finicky against the latest"
echo "#   torch. If this step fails, pin/loosen deepchem here -- the env is"
echo "#   isolated and only used to (re)generate the cached .npz embeddings."
echo "############################################################"
uv venv .venv-embeddings --python "$PYVER"
uv pip install --python .venv-embeddings/bin/python torch --index-url "$TORCH_INDEX"
uv pip install --python .venv-embeddings/bin/python \
    fair-esm transformers deepchem rdkit numpy pandas

cat <<'EOF'

############################################################
# done.
#   env 1 controls    -> .venv-controls    (ProSmith + LORAX)
#   env 2 embeddings  -> .venv-embeddings  (run-once)
#   env 3 project     -> your existing ./.venv  (GNN + ensemble; unchanged)
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
