#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Run GNN full_full quantile screen on one GPU.
# Detailed logs are written per run. By default the terminal shows a compact
# dashboard; set QUIET=0 to stream raw logs to the terminal via tee.
#
# Examples:
#   bash scripts/modeling/train/run_graph_full_full_v5_quantile_screen.sh
#   MAX_PARALLEL=20 bash scripts/modeling/train/run_graph_full_full_v5_quantile_screen.sh
#   QUIET=0 MAX_PARALLEL=2 bash scripts/modeling/train/run_graph_full_full_v5_quantile_screen.sh
#   DRY_RUN=1 bash scripts/modeling/train/run_graph_full_full_v5_quantile_screen.sh

RESULT_DIR="${RESULT_DIR:-results/graph/full_full/v5/quantile_screen/training}"
LOG_DIR="$RESULT_DIR/logs"
STATUS_DIR="$RESULT_DIR/run_status"
EPOCHS="${EPOCHS:-900}"
LR="${LR:-3e-3}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
DRY_RUN="${DRY_RUN:-0}"
QUIET="${QUIET:-1}"
DASHBOARD_INTERVAL="${DASHBOARD_INTERVAL:-5}"
DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"

mkdir -p "$LOG_DIR" "$STATUS_DIR"
rm -f "$STATUS_DIR"/*.status

# Repeat semantics for the v5 quantile screen:
# transductive: genuine LoRaX folds 1..5
# inductive_molecule: independent cold-molecule split seeds 42..46; fold=1 is the source container
REPEATS=(
  "transductive 1 42 1042"
  "transductive 2 43 1043"
  "transductive 3 44 1044"
  "transductive 4 45 1045"
  "transductive 5 46 1046"
  "inductive_molecule 1 42 1042"
  "inductive_molecule 1 43 1043"
  "inductive_molecule 1 44 1044"
  "inductive_molecule 1 45 1045"
  "inductive_molecule 1 46 1046"
)

ARCHES=("gnn")
MP_MODES=("all_edges" "signed")
QUANTILES=("0" "0.87" "0.95" "0.99")
TOTAL_RUNS=$((${#REPEATS[@]} * ${#ARCHES[@]} * ${#MP_MODES[@]} * ${#QUANTILES[@]}))

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

write_status() {
  local status="$1"
  local stem="$2"
  local log="$3"
  printf '%s\t%s\t%s\t%s\n' "$status" "$stem" "$log" "$(date '+%Y-%m-%d %H:%M:%S')" > "$STATUS_DIR/$stem.status"
}

run_stage() {
  local log="$1"
  if [[ ! -f "$log" ]]; then
    echo "STARTING"
    return 0
  fi
  if grep -qE 'Traceback|RuntimeError|Error' "$log"; then
    echo "ERROR"
  elif grep -q 'UPDATED_CHECKPOINT' "$log"; then
    echo "DONE/REPAIRED"
  elif grep -q 'SKIP_HAS_BEST' "$log"; then
    echo "DONE/HAS_BEST"
  elif grep -q 'BEST_VAL_PROBE' "$log"; then
    echo "REPAIR_BEST"
  elif grep -q 'REPAIR_BEST' "$log"; then
    echo "REPAIR_SETUP"
  elif grep -q 'model snapshots ->' "$log"; then
    echo "DONE/SAVED"
  elif grep -q 'result bundle ->' "$log"; then
    echo "SAVING_MODELS"
  elif grep -q '\[last-epoch\] unentangled_boost' "$log"; then
    echo "BOOST_LAST_DONE"
  elif grep -q '\[last-epoch\] fitting XGBoost probe' "$log"; then
    echo "BOOST_LAST"
  elif grep -q 'probing best encoder' "$log"; then
    if grep -q '\[best-val.*unentangled_boost' "$log"; then
      echo "PREPARE_LAST"
    elif grep -q '\[best-val.*fitting XGBoost probe' "$log"; then
      echo "BOOST_BEST"
    else
      echo "PREPARE_BEST"
    fi
  elif grep -q 'best val AUPRC' "$log"; then
    echo "PREPARE_PROBES"
  elif grep -qE 'epoch[[:space:]]+[0-9]+' "$log"; then
    echo "TRAIN"
  else
    echo "SETUP"
  fi
}
latest_log_line() {
  local log="$1"
  if [[ ! -f "$log" ]]; then
    echo "log not created yet"
    return 0
  fi
  local line
  line=$(grep -E 'epoch[[:space:]]+[0-9]+|best val|probing best encoder|fitting XGBoost probe|REPAIR_BEST|BEST_VAL_PROBE|UPDATED_CHECKPOINT|SKIP_HAS_BEST|\[best-val|\[last-epoch\]|result bundle|model snapshots|Traceback|RuntimeError|Error' "$log" | tail -n 1 || true)
  if [[ -z "$line" ]]; then
    line=$(tail -n 1 "$log" 2>/dev/null || true)
  fi
  echo "$line"
}

dashboard() {
  local total="$1"
  while true; do
    local done=0 failed=0 skipped=0 running=0 pending=0 seen=0
    local running_rows=()
    local failed_rows=()
    local status stem log ts

    for f in "$STATUS_DIR"/*.status; do
      [[ -e "$f" ]] || continue
      IFS=$'\t' read -r status stem log ts < "$f" || true
      seen=$((seen + 1))
      case "$status" in
        DONE) done=$((done + 1)) ;;
        FAILED) failed=$((failed + 1)); failed_rows+=("$stem") ;;
        SKIPPED) skipped=$((skipped + 1)) ;;
        RUNNING) running=$((running + 1)); running_rows+=("$stem|$log") ;;
      esac
    done
    pending=$((total - seen))
    if [[ "$pending" -lt 0 ]]; then pending=0; fi

    printf '\033[H\033[2J'
    echo "full_full v5 quantile screen | $(date '+%Y-%m-%d %H:%M:%S')"
    echo "total=$total done=$done skipped=$skipped failed=$failed running=$running pending=$pending max_parallel=$MAX_PARALLEL"
    echo "logs: $LOG_DIR"
    echo
    echo "RUNNING"
    if [[ "${#running_rows[@]}" -eq 0 ]]; then
      echo "  none"
    else
      local row latest
      for row in "${running_rows[@]}"; do
        stem="${row%%|*}"
        log="${row#*|}"
        latest="$(latest_log_line "$log")"
        stage="$(run_stage "$log")"
        printf '  %-12s %-78s\n    %s\n' "$stage" "$stem" "$latest"
      done
    fi
    echo
    echo "FAILED"
    if [[ "${#failed_rows[@]}" -eq 0 ]]; then
      echo "  none"
    else
      printf '  %s\n' "${failed_rows[@]}"
    fi
    echo
    echo "Tip: tail one log with: tail -f $LOG_DIR/<run>.log"

    if [[ "$((done + skipped + failed))" -ge "$total" ]]; then
      break
    fi
    sleep "$DASHBOARD_INTERVAL"
  done
}

run_one() {
  local regime="$1"
  local fold="$2"
  local gnn_seed="$3"
  local boost_seed="$4"
  local arch="$5"
  local mp="$6"
  local q="$7"

  local qt variant stem artifact log repair_log rc
  qt="$(q_tag "$q")"
  variant="${mp}${qt}"
  stem="${arch}_${variant}_${regime}_fold${fold}_gnn${gnn_seed}_boost${boost_seed}"
  artifact="${RESULT_DIR}/checkpoints/${arch}_${variant}_unentangled_boost_${regime}_fold${fold}_gnn${gnn_seed}_boost${boost_seed}.pt"
  log="${LOG_DIR}/${stem}.log"
  repair_log="${LOG_DIR}/${stem}.best_val_repair.log"

  if [[ -f "$artifact" ]]; then
    write_status "RUNNING" "$stem" "$repair_log"
    if [[ "$DRY_RUN" == "1" ]]; then
      {
        echo uv run python scripts/modeling/eval/eval_graph_full_full_quantile_best.py \
          --checkpoint "$artifact" --device "$DEVICE"
      } > "$repair_log"
      write_status "DONE" "$stem" "$repair_log"
      return 0
    fi

    set +e
    if [[ "$QUIET" == "1" ]]; then
      uv run python scripts/modeling/eval/eval_graph_full_full_quantile_best.py \
        --checkpoint "$artifact" --device "$DEVICE" > "$repair_log" 2>&1
      rc=$?
    else
      uv run python scripts/modeling/eval/eval_graph_full_full_quantile_best.py \
        --checkpoint "$artifact" --device "$DEVICE" 2>&1 | tee "$repair_log"
      rc=${PIPESTATUS[0]}
    fi
    set -e

    if [[ "$rc" -eq 0 ]]; then
      write_status "DONE" "$stem" "$repair_log"
    else
      write_status "FAILED" "$stem" "$repair_log"
    fi
    return "$rc"
  fi

  write_status "RUNNING" "$stem" "$log"
  if [[ "$DRY_RUN" == "1" ]]; then
    {
      echo uv run python scripts/modeling/train/train_graph_full_full.py \
        --arch "$arch" --mp_mode "$mp" --mol_quality_q "$q" \
        --regime "$regime" --fold "$fold" \
        --epochs "$EPOCHS" --lr "$LR" --grad-clip "$GRAD_CLIP" \
        --device "$DEVICE" --seed "$gnn_seed" --boost-seed "$boost_seed" \
        --probe-checkpoint both --results-dir "$RESULT_DIR"
    } > "$log"
    write_status "DONE" "$stem" "$log"
    return 0
  fi

  set +e
  if [[ "$QUIET" == "1" ]]; then
    uv run python scripts/modeling/train/train_graph_full_full.py \
      --arch "$arch" --mp_mode "$mp" --mol_quality_q "$q" \
      --regime "$regime" --fold "$fold" \
      --epochs "$EPOCHS" --lr "$LR" --grad-clip "$GRAD_CLIP" \
      --device "$DEVICE" --seed "$gnn_seed" --boost-seed "$boost_seed" \
      --probe-checkpoint both --results-dir "$RESULT_DIR" > "$log" 2>&1
    rc=$?
  else
    uv run python scripts/modeling/train/train_graph_full_full.py \
      --arch "$arch" --mp_mode "$mp" --mol_quality_q "$q" \
      --regime "$regime" --fold "$fold" \
      --epochs "$EPOCHS" --lr "$LR" --grad-clip "$GRAD_CLIP" \
      --device "$DEVICE" --seed "$gnn_seed" --boost-seed "$boost_seed" \
      --probe-checkpoint both --results-dir "$RESULT_DIR" 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
  fi
  set -e

  if [[ "$rc" -eq 0 ]]; then
    write_status "DONE" "$stem" "$log"
  else
    write_status "FAILED" "$stem" "$log"
  fi
  return "$rc"
}

RUNNING_PIDS=()
prune_pids() {
  local alive=()
  local pid
  for pid in "${RUNNING_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      alive+=("$pid")
    fi
  done
  RUNNING_PIDS=("${alive[@]}")
}

wait_for_slot() {
  while true; do
    prune_pids
    if [[ "${#RUNNING_PIDS[@]}" -lt "$MAX_PARALLEL" ]]; then
      break
    fi
    sleep 2
  done
}

MONITOR_PID=""
if [[ "$QUIET" == "1" ]]; then
  dashboard "$TOTAL_RUNS" &
  MONITOR_PID="$!"
  trap '[[ -n "${MONITOR_PID:-}" ]] && kill "$MONITOR_PID" 2>/dev/null || true' EXIT
fi

for repeat in "${REPEATS[@]}"; do
  read -r regime fold gnn_seed boost_seed <<< "$repeat"
  for arch in "${ARCHES[@]}"; do
    for mp in "${MP_MODES[@]}"; do
      for q in "${QUANTILES[@]}"; do
        wait_for_slot
        run_one "$regime" "$fold" "$gnn_seed" "$boost_seed" "$arch" "$mp" "$q" &
        RUNNING_PIDS+=("$!")
      done
    done
  done
done

fail=0
for pid in "${RUNNING_PIDS[@]}"; do
  if ! wait "$pid"; then
    fail=1
  fi
done

if [[ -n "$MONITOR_PID" ]]; then
  wait "$MONITOR_PID" || true
fi

if [[ "$fail" -ne 0 ]]; then
  echo "quantile screen finished with failures; inspect logs in $LOG_DIR" >&2
  exit 1
fi

echo "quantile screen finished successfully; scheduled runs: $TOTAL_RUNS"