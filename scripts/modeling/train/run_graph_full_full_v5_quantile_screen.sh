#!/usr/bin/env bash
set -euo pipefail

# Run GNN full_full quantile screen on one GPU.
# Default is conservative serial execution. Set MAX_PARALLEL=2 to try light
# parallelization on an A100, then watch nvidia-smi for memory/utilization.
#
# Examples:
#   bash scripts/modeling/train/run_graph_full_full_v5_quantile_screen.sh
#   MAX_PARALLEL=2 bash scripts/modeling/train/run_graph_full_full_v5_quantile_screen.sh
#   DRY_RUN=1 bash scripts/modeling/train/run_graph_full_full_v5_quantile_screen.sh

RESULT_DIR="${RESULT_DIR:-results/graph/full_full/v5/quantile_screen/training}"
LOG_DIR="$RESULT_DIR/logs"
EPOCHS="${EPOCHS:-900}"
LR="${LR:-3e-3}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
DRY_RUN="${DRY_RUN:-0}"
DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"

mkdir -p "$LOG_DIR"

# Same repeat semantics as v5 architecture_screen:
# transductive: genuine LoRaX folds 1/2/3 with seeds 42/43/44
# inductive_molecule: independent cold-molecule split seeds 42/43/44; fold=1 is the source container
REPEATS=(
  "transductive 1 42 1042"
  "transductive 2 43 1043"
  "transductive 3 44 1044"
  "inductive_molecule 1 42 1042"
  "inductive_molecule 1 43 1043"
  "inductive_molecule 1 44 1044"
)

ARCHES=("gnn")
MP_MODES=("all_edges" "signed")
QUANTILES=("0" "0.87" "0.95" "0.99")

q_tag() {
  case "$1" in
    0|0.0|0.00) echo "" ;;
    0.87|.87) echo "_q87" ;;
    0.95|.95) echo "_q95" ;;
    0.99|.99) echo "_q99" ;;
    *)
      echo "Unsupported quantile '$1'; add it to q_tag()" >&2
      return 2
      ;;
  esac
}

run_one() {
  local regime="$1"
  local fold="$2"
  local gnn_seed="$3"
  local boost_seed="$4"
  local arch="$5"
  local mp="$6"
  local q="$7"

  local qt variant stem artifact log
  qt="$(q_tag "$q")"
  variant="${mp}${qt}"
  stem="${arch}_${variant}_${regime}_fold${fold}_gnn${gnn_seed}_boost${boost_seed}"
  artifact="${RESULT_DIR}/checkpoints/${arch}_${variant}_unentangled_boost_${regime}_fold${fold}_gnn${gnn_seed}_boost${boost_seed}.pt"
  log="${LOG_DIR}/${stem}.log"

  if [[ -f "$artifact" ]]; then
    echo "SKIP $stem"
    return 0
  fi

  echo "RUN  $stem -> $log"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo uv run python scripts/modeling/train/train_graph_full_full.py \
      --arch "$arch" --mp_mode "$mp" --mol_quality_q "$q" \
      --regime "$regime" --fold "$fold" \
      --epochs "$EPOCHS" --lr "$LR" --grad-clip "$GRAD_CLIP" \
      --device "$DEVICE" --seed "$gnn_seed" --boost-seed "$boost_seed" \
      --probe-checkpoint both --results-dir "$RESULT_DIR"
    return 0
  fi

  uv run python scripts/modeling/train/train_graph_full_full.py \
    --arch "$arch" --mp_mode "$mp" --mol_quality_q "$q" \
    --regime "$regime" --fold "$fold" \
    --epochs "$EPOCHS" --lr "$LR" --grad-clip "$GRAD_CLIP" \
    --device "$DEVICE" --seed "$gnn_seed" --boost-seed "$boost_seed" \
    --probe-checkpoint both --results-dir "$RESULT_DIR" 2>&1 | tee "$log"
}

wait_for_slot() {
  while [[ "$(jobs -pr | wc -l)" -ge "$MAX_PARALLEL" ]]; do
    wait -n
  done
}

total=0
for repeat in "${REPEATS[@]}"; do
  read -r regime fold gnn_seed boost_seed <<< "$repeat"
  for arch in "${ARCHES[@]}"; do
    for mp in "${MP_MODES[@]}"; do
      for q in "${QUANTILES[@]}"; do
        total=$((total + 1))
        wait_for_slot
        run_one "$regime" "$fold" "$gnn_seed" "$boost_seed" "$arch" "$mp" "$q" &
      done
    done
  done
done

wait
echo "quantile screen finished; scheduled runs: $total"
