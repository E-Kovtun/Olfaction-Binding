#!/usr/bin/env bash
# full_full v5 quantile screen: all jobs run sequentially on one GPU.
#
# Normal server launch:
#   GPU_ID=0 bash legacy/scripts/queues/quantile_screen.sh
#
# Optional partial rerun of one repeat:
#   GPU_ID=0 REPEAT_INDEX=1 bash legacy/scripts/queues/quantile_screen.sh
#
# Full queue: 36 jobs = 3 repeats x
#   {GNN all_edges, GAT signed} x {q87, q95, q99}
#   x {transductive, inductive_molecule}.

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

GPU_ID="${GPU_ID:-0}"
REPEAT_INDEX="${REPEAT_INDEX:-all}"
EPOCHS="${EPOCHS:-900}"
RESULT_ROOT="${RESULT_ROOT:-results/graph/full_full/v5/quantile_screen/training}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

CKPT_DIR="$RESULT_ROOT/checkpoints"
mkdir -p "$CKPT_DIR"

set_repeat() {
  local repeat="$1"
  case "$repeat" in
    0) TRANS_FOLD=1; GNN_SEED=42; BOOST_SEED=1042 ;;
    1) TRANS_FOLD=2; GNN_SEED=43; BOOST_SEED=1043 ;;
    2) TRANS_FOLD=3; GNN_SEED=44; BOOST_SEED=1044 ;;
    *) echo "repeat must be 0, 1, or 2" >&2; exit 2 ;;
  esac
  CURRENT_REPEAT="$repeat"
  LOG_DIR="$RESULT_ROOT/logs/repeat_${repeat}"
  mkdir -p "$LOG_DIR"
  QUEUE_LOG="$LOG_DIR/queue.log"
}

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$QUEUE_LOG"
}

run_one() {
  local arch="$1" mp="$2" regime="$3" fold="$4" q="$5" qtag="$6"
  local stem="${arch}_${mp}_q${qtag}_${regime}_fold${fold}_gnn${GNN_SEED}_boost${BOOST_SEED}"
  local artifact="$CKPT_DIR/${arch}_${mp}_q${qtag}_unentangled_boost_${regime}_fold${fold}_gnn${GNN_SEED}_boost${BOOST_SEED}.pt"
  local job_log="$LOG_DIR/${stem}.log"

  if [[ -f "$artifact" ]]; then
    log "SKIP cached: $stem"
    return 0
  fi

  log "RUN GPU=$GPU_ID: $stem"
  uv run python legacy/scripts/modeling/train/train_graph_full_full.py \
    --arch "$arch" \
    --mp_mode "$mp" \
    --regime "$regime" \
    --fold "$fold" \
    --mol_quality_q "$q" \
    --epochs "$EPOCHS" \
    --lr 3e-3 \
    --grad-clip 1.0 \
    --device cuda \
    --seed "$GNN_SEED" \
    --boost-seed "$BOOST_SEED" \
    --probe-checkpoint last \
    --results-dir "$RESULT_ROOT" \
    2>&1 | tee "$job_log"

  [[ -f "$artifact" ]] || {
    log "FAILED: process exited without expected artifact: $artifact"
    return 1
  }
  log "DONE: $stem"
}

run_config() {
  local arch="$1" mp="$2"
  for qspec in "0.87 87" "0.95 95" "0.99 99"; do
    read -r q qtag <<< "$qspec"
    run_one "$arch" "$mp" transductive "$TRANS_FOLD" "$q" "$qtag"
    run_one "$arch" "$mp" inductive_molecule 1 "$q" "$qtag"
  done
}

run_repeat() {
  set_repeat "$1"
  log "repeat start: GPU=$GPU_ID repeat=$CURRENT_REPEAT epochs=$EPOCHS"
  log "selection: GNN/all_edges and GAT/signed; q={87,95,99}"

  # Keep these calls explicit: this is the scientific selection for this screen.
  run_config gnn all_edges
  run_config gat signed

  log "repeat complete: GPU=$GPU_ID repeat=$CURRENT_REPEAT"
}

if [[ "$REPEAT_INDEX" == "all" ]]; then
  run_repeat 0
  run_repeat 1
  run_repeat 2
else
  run_repeat "$REPEAT_INDEX"
fi
