#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# full_full v6: SimGCL contrastive add-on screen (Yu et al., SIGIR'22).
#
# GNN setup: arch=gnn, mp_mode=signed, 900 epochs, lr 3e-3, grad-clip 1.0,
# last-epoch probe. The only change vs the matching v5 point is the training
# objective: main link loss + cl_weight * InfoNCE over two SimGCL-noised views.
# The cl_weight=0 control is the existing v5 signed run at the SAME quantile
# (NOT re-run here) -- pick that quantile in the notebook cell.
#
# Sweep: QUANTILES (default "0 0.95 0.99") x SimGCL arms (cl_eps x cl_weight).
# MAIN_LOSS (bce default | bpr) sets the main link loss; variant names carry
# both the quantile and the loss so bce/bpr and different q coexist in one dir.
#   default: 4 arms x 10 repeats x 3 quantiles = 120 runs.
#
# Examples:
#   bash legacy/scripts/modeling/train/run_graph_full_full_v6_simgcl.sh
#   MAX_PARALLEL=20 bash legacy/scripts/modeling/train/run_graph_full_full_v6_simgcl.sh
#   QUIET=0 MAX_PARALLEL=2 bash legacy/scripts/modeling/train/run_graph_full_full_v6_simgcl.sh
#   DRY_RUN=1 bash legacy/scripts/modeling/train/run_graph_full_full_v6_simgcl.sh

RESULT_DIR="${RESULT_DIR:-results/graph/full_full/v6/simgcl_screen/training}"
LOG_DIR="$RESULT_DIR/logs"
STATUS_DIR="$RESULT_DIR/run_status"
EPOCHS="${EPOCHS:-900}"
LR="${LR:-3e-3}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
MP_MODE="${MP_MODE:-signed}"
ARCH="${ARCH:-gnn}"
QUANTILES="${QUANTILES:-0 0.95 0.99}"   # space-separated molecule-coverage quantiles to sweep
MAIN_LOSS="${MAIN_LOSS:-bce}"      # main link loss: bce (default) or bpr
CL_TEMP="${CL_TEMP:-0.2}"          # InfoNCE temperature (fixed, paper default)
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

# transductive: LoRaX folds 1..5 ; inductive_molecule: cold seeds 42..46 (fold=1 container)
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

# "cl_eps cl_weight" arms. cl_weight=0 control comes from v5, not re-run here.
# Edit this list to change the grid.
SIMGCL_ARMS=(
  "0.1 0.2"
  "0.1 0.5"
  "0.2 0.2"
  "0.2 0.5"
)
NQ=$(wc -w <<< "$QUANTILES")
TOTAL_RUNS=$((${#REPEATS[@]} * ${#SIMGCL_ARMS[@]} * NQ))

write_status() {
  local status="$1" stem="$2" log="$3"
  printf '%s\t%s\t%s\t%s\n' "$status" "$stem" "$log" "$(date '+%Y-%m-%d %H:%M:%S')" > "$STATUS_DIR/$stem.status"
}

run_stage() {
  local log="$1"
  if [[ ! -f "$log" ]]; then echo "STARTING"; return 0; fi
  if grep -qE 'Traceback|RuntimeError|Error' "$log"; then echo "ERROR"
  elif grep -q 'model snapshots ->' "$log"; then echo "DONE/SAVED"
  elif grep -q 'result bundle ->' "$log"; then echo "SAVING_MODELS"
  elif grep -q '\[last-epoch\] unentangled_boost' "$log"; then echo "BOOST_LAST_DONE"
  elif grep -q '\[last-epoch\] fitting XGBoost probe' "$log"; then echo "BOOST_LAST"
  elif grep -q 'best val AUPRC' "$log"; then echo "PREPARE_PROBES"
  elif grep -qE 'epoch[[:space:]]+[0-9]+' "$log"; then echo "TRAIN"
  else echo "SETUP"; fi
}

latest_log_line() {
  local log="$1" line
  if [[ ! -f "$log" ]]; then echo "log not created yet"; return 0; fi
  line=$(grep -E 'epoch[[:space:]]+[0-9]+|best val|fitting XGBoost probe|\[last-epoch\]|result bundle|model snapshots|Traceback|RuntimeError|Error' "$log" | tail -n 1 || true)
  [[ -z "$line" ]] && line=$(tail -n 1 "$log" 2>/dev/null || true)
  echo "$line"
}

dashboard() {
  local total="$1"
  while true; do
    local done=0 failed=0 running=0 pending=0 seen=0
    local running_rows=() failed_rows=()
    local status stem log ts
    for f in "$STATUS_DIR"/*.status; do
      [[ -e "$f" ]] || continue
      IFS=$'\t' read -r status stem log ts < "$f" || true
      seen=$((seen + 1))
      case "$status" in
        DONE) done=$((done + 1)) ;;
        FAILED) failed=$((failed + 1)); failed_rows+=("$stem") ;;
        RUNNING) running=$((running + 1)); running_rows+=("$stem|$log") ;;
      esac
    done
    pending=$((total - seen)); [[ "$pending" -lt 0 ]] && pending=0
    printf '\033[H\033[2J'
    echo "full_full v6 SimGCL screen | $(date '+%Y-%m-%d %H:%M:%S')"
    echo "total=$total done=$done failed=$failed running=$running pending=$pending max_parallel=$MAX_PARALLEL"
    echo "logs: $LOG_DIR"
    echo; echo "RUNNING"
    if [[ "${#running_rows[@]}" -eq 0 ]]; then echo "  none"; else
      local row latest stage
      for row in "${running_rows[@]}"; do
        stem="${row%%|*}"; log="${row#*|}"
        latest="$(latest_log_line "$log")"; stage="$(run_stage "$log")"
        printf '  %-14s %-70s\n    %s\n' "$stage" "$stem" "$latest"
      done
    fi
    echo; echo "FAILED"
    if [[ "${#failed_rows[@]}" -eq 0 ]]; then echo "  none"; else printf '  %s\n' "${failed_rows[@]}"; fi
    echo; echo "Tip: tail one log with: tail -f $LOG_DIR/<run>.log"
    if [[ "$((done + failed))" -ge "$total" ]]; then break; fi
    sleep "$DASHBOARD_INTERVAL"
  done
}

run_one() {
  local regime="$1" fold="$2" gnn_seed="$3" boost_seed="$4" eps="$5" weight="$6" q="$7"
  local ep wp qt variant stem artifact log rc
  ep="$(awk "BEGIN{printf \"%d\", $eps*100}")"
  wp="$(awk "BEGIN{printf \"%d\", $weight*100}")"
  qt="$(awk "BEGIN{printf \"%d\", $q*100}")"
  local q_seg=""; [[ "$qt" != "0" ]] && q_seg="_q${qt}"
  local loss_seg=""; [[ "$MAIN_LOSS" == "bpr" ]] && loss_seg="_bpr"
  variant="${MP_MODE}${q_seg}${loss_seg}_simgcl_e${ep}_w${wp}"
  stem="${ARCH}_${variant}_${regime}_fold${fold}_gnn${gnn_seed}_boost${boost_seed}"
  artifact="${RESULT_DIR}/checkpoints/${ARCH}_${variant}_unentangled_boost_${regime}_fold${fold}_gnn${gnn_seed}_boost${boost_seed}.pt"
  log="${LOG_DIR}/${stem}.log"

  if [[ -f "$artifact" ]]; then
    write_status "DONE" "$stem" "$log"; echo "SKIP (exists): $stem"; return 0
  fi

  write_status "RUNNING" "$stem" "$log"
  if [[ "$DRY_RUN" == "1" ]]; then
    {
      echo uv run python legacy/scripts/modeling/train/train_graph_full_full.py \
        --arch "$ARCH" --mp_mode "$MP_MODE" --mol_quality_q "$q" --main-loss "$MAIN_LOSS" \
        --cl-eps "$eps" --cl-weight "$weight" --cl-temp "$CL_TEMP" \
        --regime "$regime" --fold "$fold" \
        --epochs "$EPOCHS" --lr "$LR" --grad-clip "$GRAD_CLIP" \
        --device "$DEVICE" --seed "$gnn_seed" --boost-seed "$boost_seed" \
        --probe-checkpoint last --results-dir "$RESULT_DIR"
    } > "$log"
    write_status "DONE" "$stem" "$log"; return 0
  fi

  set +e
  if [[ "$QUIET" == "1" ]]; then
    uv run python legacy/scripts/modeling/train/train_graph_full_full.py \
      --arch "$ARCH" --mp_mode "$MP_MODE" --mol_quality_q "$q" --main-loss "$MAIN_LOSS" \
      --cl-eps "$eps" --cl-weight "$weight" --cl-temp "$CL_TEMP" \
      --regime "$regime" --fold "$fold" \
      --epochs "$EPOCHS" --lr "$LR" --grad-clip "$GRAD_CLIP" \
      --device "$DEVICE" --seed "$gnn_seed" --boost-seed "$boost_seed" \
      --probe-checkpoint last --results-dir "$RESULT_DIR" > "$log" 2>&1
    rc=$?
  else
    uv run python legacy/scripts/modeling/train/train_graph_full_full.py \
      --arch "$ARCH" --mp_mode "$MP_MODE" --mol_quality_q "$q" --main-loss "$MAIN_LOSS" \
      --cl-eps "$eps" --cl-weight "$weight" --cl-temp "$CL_TEMP" \
      --regime "$regime" --fold "$fold" \
      --epochs "$EPOCHS" --lr "$LR" --grad-clip "$GRAD_CLIP" \
      --device "$DEVICE" --seed "$gnn_seed" --boost-seed "$boost_seed" \
      --probe-checkpoint last --results-dir "$RESULT_DIR" 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
  fi
  set -e

  if [[ "$rc" -eq 0 ]]; then write_status "DONE" "$stem" "$log"; else write_status "FAILED" "$stem" "$log"; fi
  return "$rc"
}

RUNNING_PIDS=()
prune_pids() {
  local alive=() pid
  for pid in "${RUNNING_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then alive+=("$pid"); fi
  done
  RUNNING_PIDS=("${alive[@]}")
}
wait_for_slot() {
  while true; do
    prune_pids
    if [[ "${#RUNNING_PIDS[@]}" -lt "$MAX_PARALLEL" ]]; then break; fi
    sleep 2
  done
}

MONITOR_PID=""
if [[ "$QUIET" == "1" ]]; then
  dashboard "$TOTAL_RUNS" &
  MONITOR_PID="$!"
  trap '[[ -n "${MONITOR_PID:-}" ]] && kill "$MONITOR_PID" 2>/dev/null || true' EXIT
fi

echo "full_full v6 SimGCL: $TOTAL_RUNS runs (${#SIMGCL_ARMS[@]} arms x ${#REPEATS[@]} repeats x $NQ quantiles); arch=$ARCH mp=$MP_MODE q={$QUANTILES} main_loss=$MAIN_LOSS tau=$CL_TEMP"
for repeat in "${REPEATS[@]}"; do
  read -r regime fold gnn_seed boost_seed <<< "$repeat"
  for q in $QUANTILES; do
    for arm in "${SIMGCL_ARMS[@]}"; do
      read -r eps weight <<< "$arm"
      wait_for_slot
      run_one "$regime" "$fold" "$gnn_seed" "$boost_seed" "$eps" "$weight" "$q" &
      RUNNING_PIDS+=("$!")
    done
  done
done

fail=0
for pid in "${RUNNING_PIDS[@]}"; do
  if ! wait "$pid"; then fail=1; fi
done
[[ -n "$MONITOR_PID" ]] && wait "$MONITOR_PID" 2>/dev/null || true

if [[ "$fail" -ne 0 ]]; then
  echo "v6 SimGCL screen finished with failures; inspect logs in $LOG_DIR" >&2
  exit 1
fi
echo "v6 SimGCL screen finished successfully; scheduled runs: $TOTAL_RUNS"
