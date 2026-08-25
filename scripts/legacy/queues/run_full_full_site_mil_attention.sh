#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Full_full LoRaX site-level MIL/attention screen with a live dashboard.
#
# Examples:
#   bash scripts/queues/run_full_full_site_mil_attention.sh
#   MAX_PARALLEL=2 MODEL=attention_excess_max bash scripts/queues/run_full_full_site_mil_attention.sh
#   REGIME=transductive REPEATS="1 2 3 4 5" MODEL=all bash scripts/queues/run_full_full_site_mil_attention.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

RESULT_DIR="${RESULT_DIR:-results/attention/full_full/site_mil}"
RUNS_DIR="$RESULT_DIR/runs"
LOG_DIR="$RESULT_DIR/logs"
STATUS_DIR="$RESULT_DIR/run_status"
DEVICE="${DEVICE:-cuda:0}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
QUIET="${QUIET:-1}"
DASHBOARD_INTERVAL="${DASHBOARD_INTERVAL:-5}"

REGIME="${REGIME:-all}"
MODEL="${MODEL:-all}"
TRANS_REPEATS=(${TRANS_REPEATS:-1 2 3 4 5})
INDUCTIVE_REPEATS=(${INDUCTIVE_REPEATS:-42 43 44 45 46})
MODELS=(${MODELS:-boost_gin_mean mil_max mil_noisy_or mil_lse attention_mil attention_excess_max})

EPOCHS="${EPOCHS:-80}"
PATIENCE="${PATIENCE:-12}"
BATCH_SIZE="${BATCH_SIZE:-256}"
DIM="${DIM:-64}"
DROPOUT="${DROPOUT:-0.1}"
TEMPERATURE="${TEMPERATURE:-1.0}"
LR="${LR:-3e-4}"
POS_FRACTION="${POS_FRACTION:-0.5}"
NEG_THRESHOLD="${NEG_THRESHOLD:-0.2}"
POS_THRESHOLD="${POS_THRESHOLD:-0.4}"
MARGIN_WEIGHT="${MARGIN_WEIGHT:-1.0}"
DECISION_THRESHOLD="${DECISION_THRESHOLD:-}"
USE_MARGIN_THRESHOLD="${USE_MARGIN_THRESHOLD:-0}"
NUM_WORKERS="${NUM_WORKERS:-0}"
TRANS_SEED_OFFSET="${TRANS_SEED_OFFSET:-41}"
BOOST_SEED_OFFSET="${BOOST_SEED_OFFSET:-1000}"
FORCE="${FORCE:-0}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

mkdir -p "$RUNS_DIR" "$LOG_DIR" "$STATUS_DIR"
rm -f "$STATUS_DIR"/*.status

selected_models=()
if [[ "$MODEL" == "all" ]]; then
  selected_models=("${MODELS[@]}")
else
  selected_models=("$MODEL")
fi

units=()
if [[ "$REGIME" == "all" || "$REGIME" == "transductive" ]]; then
  for rep in "${TRANS_REPEATS[@]}"; do
    for m in "${selected_models[@]}"; do units+=("transductive|$rep|$m"); done
  done
fi
if [[ "$REGIME" == "all" || "$REGIME" == "inductive_molecule" ]]; then
  for rep in "${INDUCTIVE_REPEATS[@]}"; do
    for m in "${selected_models[@]}"; do units+=("inductive_molecule|$rep|$m"); done
  done
fi
TOTAL_RUNS=${#units[@]}

write_status() {
  printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$(date '+%Y-%m-%d %H:%M:%S')" > "$STATUS_DIR/$2.status"
}

run_stage() {
  local log="$1"
  [[ -f "$log" ]] || { echo "STARTING"; return 0; }
  if grep -qE 'Traceback|RuntimeError|Error|ValueError|XGBoostError' "$log"; then echo "ERROR"
  elif grep -q 'UNIT_DONE' "$log"; then echo "DONE"
  elif grep -q 'epoch' "$log"; then echo "TRAIN"
  elif grep -q 'RUN ' "$log"; then echo "INIT"
  else echo "STARTING"; fi
}

latest_log_line() {
  local log="$1"
  [[ -f "$log" ]] || { echo "log not created yet"; return 0; }
  local line
  line=$(grep -E 'RUN |epoch|SKIP cached|UNIT_DONE|AUROC=|Traceback|RuntimeError|ValueError|Error' "$log" | tail -n 1 || true)
  [[ -z "$line" ]] && line=$(tail -n 1 "$log" 2>/dev/null || true)
  echo "$line"
}

dashboard() {
  local total="$1"
  while true; do
    local done=0 failed=0 running=0 seen=0 pending
    local running_rows=() failed_rows=()
    local f status stem log ts row
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
    pending=$((total - seen)); [[ $pending -lt 0 ]] && pending=0
    printf '\033[H\033[2J'
    echo "full_full LoRaX site MIL/attention | $(date '+%H:%M:%S') | device=$DEVICE"
    echo "total=$total done=$done failed=$failed running=$running pending=$pending max_parallel=$MAX_PARALLEL"
    echo "epochs=$EPOCHS patience=$PATIENCE batch=$BATCH_SIZE dim=$DIM model=$MODEL regime=$REGIME"
    echo "pos_fraction=$POS_FRACTION neg_t=$NEG_THRESHOLD pos_t=$POS_THRESHOLD logs=$LOG_DIR"
    echo
    echo "RUNNING"
    if [[ ${#running_rows[@]} -eq 0 ]]; then
      echo "  none"
    else
      for row in "${running_rows[@]}"; do
        stem="${row%%|*}"; log="${row#*|}"
        printf '  %-8s %-60s\n    %s\n' "$(run_stage "$log")" "$stem" "$(latest_log_line "$log")"
      done
    fi
    echo
    echo "FAILED"
    if [[ ${#failed_rows[@]} -eq 0 ]]; then echo "  none"; else printf '  %s\n' "${failed_rows[@]}"; fi
    [[ $((done + failed)) -ge $total ]] && break
    sleep "$DASHBOARD_INTERVAL"
  done
}

run_one_unit() {
  local regime="$1"
  local rep="$2"
  local model="$3"
  local seed
  if [[ "$regime" == "transductive" ]]; then seed=$((rep + TRANS_SEED_OFFSET)); else seed="$rep"; fi
  local boost_seed=$((seed + BOOST_SEED_OFFSET))
  local stem="${regime}_rep${rep}_${model}_seed${seed}_boost${boost_seed}"
  local log="$LOG_DIR/$stem.log"
  local metrics="$RUNS_DIR/$stem/metrics.json"
  if [[ -s "$metrics" && "$FORCE" != "1" ]]; then
    write_status "DONE" "$stem" "$log"
    return 0
  fi
  write_status "RUNNING" "$stem" "$log"
  local extra_args=()
  if [[ -n "$DECISION_THRESHOLD" ]]; then extra_args+=(--decision-threshold "$DECISION_THRESHOLD"); fi
  if [[ "$USE_MARGIN_THRESHOLD" == "1" ]]; then extra_args+=(--use-margin-threshold); fi
  if [[ "$FORCE" == "1" ]]; then extra_args+=(--force); fi

  set +e
  uv run python scripts/modeling/train/train_full_full_site_mil_attention.py \
    --regime "$regime" \
    --repeat "$rep" \
    --model "$model" \
    --seed "$seed" \
    --boost-seed "$boost_seed" \
    --out-dir "$RESULT_DIR" \
    --device "$DEVICE" \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --batch-size "$BATCH_SIZE" \
    --dim "$DIM" \
    --dropout "$DROPOUT" \
    --temperature "$TEMPERATURE" \
    --lr "$LR" \
    --pos-fraction "$POS_FRACTION" \
    --neg-threshold "$NEG_THRESHOLD" \
    --pos-threshold "$POS_THRESHOLD" \
    --margin-weight "$MARGIN_WEIGHT" \
    --num-workers "$NUM_WORKERS" \
    "${extra_args[@]}" > "$log" 2>&1
  local rc=$?
  set -e
  if [[ $rc -eq 0 ]]; then write_status "DONE" "$stem" "$log"; else write_status "FAILED" "$stem" "$log"; fi
  return $rc
}

RUNNING_PIDS=()
prune_pids() {
  local alive=() pid
  for pid in "${RUNNING_PIDS[@]}"; do kill -0 "$pid" 2>/dev/null && alive+=("$pid"); done
  RUNNING_PIDS=("${alive[@]}")
}
wait_for_slot() {
  while true; do
    prune_pids
    [[ ${#RUNNING_PIDS[@]} -lt $MAX_PARALLEL ]] && break
    sleep 2
  done
}

MONITOR_PID=""
if [[ "$QUIET" == "1" ]]; then
  dashboard "$TOTAL_RUNS" &
  MONITOR_PID="$!"
  trap '[[ -n "${MONITOR_PID:-}" ]] && kill "$MONITOR_PID" 2>/dev/null || true' EXIT
fi

for unit in "${units[@]}"; do
  IFS='|' read -r regime rep model <<< "$unit"
  wait_for_slot
  if [[ "$QUIET" == "1" ]]; then
    run_one_unit "$regime" "$rep" "$model" &
  else
    run_one_unit "$regime" "$rep" "$model" 2>&1 | tee -a "$LOG_DIR/stream.log" &
  fi
  RUNNING_PIDS+=("$!")
done

fail=0
for pid in "${RUNNING_PIDS[@]}"; do wait "$pid" || fail=1; done
[[ -n "$MONITOR_PID" ]] && wait "$MONITOR_PID" 2>/dev/null || true

uv run python - "$RUNS_DIR" "$RESULT_DIR" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

runs_dir = Path(sys.argv[1])
result_dir = Path(sys.argv[2])
rows = []
for path in sorted(runs_dir.glob("*/metrics.json")):
    with open(path, "r", encoding="utf-8") as f:
        rec = json.load(f)
    rec.pop("config", None)
    rows.append(rec)
if rows:
    df = pd.DataFrame(rows)
    cols = [c for c in ["regime", "repeat", "model", "seed", "boost_seed"] if c in df.columns]
    df = df.sort_values(cols).reset_index(drop=True)
    out = result_dir / "metrics.csv"
    df.to_csv(out, index=False)
    print(f"merged {len(rows)} metric files -> {out}")
else:
    print("no metric files found")
PY

if [[ $fail -ne 0 ]]; then
  echo "finished with failures; inspect logs in $LOG_DIR" >&2
  exit 1
fi
echo "full_full site MIL/attention finished -> $RESULT_DIR/metrics.csv"
