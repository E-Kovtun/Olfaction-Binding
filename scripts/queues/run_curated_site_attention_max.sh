#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Curated attention-as-score screen with a live dashboard.
# One queue unit is one seed; each unit runs both regimes and writes into an
# isolated seed directory. The launcher merges metrics/predictions into the root
# result directory read by the notebook.
#
# Examples:
#   bash scripts/queues/run_curated_site_attention_max.sh
#   MAX_PARALLEL=2 bash scripts/queues/run_curated_site_attention_max.sh
#   POS_FRACTION=0.5 NEG_THRESHOLD=0.20 POS_THRESHOLD=0.40 bash scripts/queues/run_curated_site_attention_max.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

RESULT_DIR="${RESULT_DIR:-results/attention/curated/site_max}"
RUNS_DIR="$RESULT_DIR/seed_runs"
LOG_DIR="$RESULT_DIR/logs"
STATUS_DIR="$RESULT_DIR/run_status"
SEEDS=(${SEEDS:-42 43 44 45 46})
DEVICE="${DEVICE:-cuda:0}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
QUIET="${QUIET:-1}"
DASHBOARD_INTERVAL="${DASHBOARD_INTERVAL:-4}"
EPOCHS="${EPOCHS:-80}"
PATIENCE="${PATIENCE:-12}"
BATCH_SIZE="${BATCH_SIZE:-128}"
DIM="${DIM:-64}"
TEMPERATURE="${TEMPERATURE:-1.0}"
DROPOUT="${DROPOUT:-0.1}"
NEG_THRESHOLD="${NEG_THRESHOLD:-0.2}"
POS_THRESHOLD="${POS_THRESHOLD:-0.4}"
MARGIN_WEIGHT="${MARGIN_WEIGHT:-1.0}"
POS_FRACTION="${POS_FRACTION:-0.5}"
DECISION_THRESHOLD="${DECISION_THRESHOLD:-}"
REGIME="${REGIME:-all}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

mkdir -p "$RUNS_DIR" "$LOG_DIR" "$STATUS_DIR"
rm -f "$STATUS_DIR"/*.status
TOTAL_RUNS=${#SEEDS[@]}

write_status() {
  printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$(date '+%Y-%m-%d %H:%M:%S')" > "$STATUS_DIR/$2.status"
}

run_stage() {
  local log="$1"
  [[ -f "$log" ]] || { echo "STARTING"; return 0; }
  if grep -qE 'Traceback|RuntimeError|Error|ValueError' "$log"; then echo "ERROR"
  elif grep -q 'SEED_DONE' "$log"; then echo "DONE"
  elif grep -q 'inductive_molecule' "$log"; then echo "INDUCTIVE"
  elif grep -q 'transductive' "$log"; then echo "TRANSDUCTIVE"
  else echo "INIT"; fi
}

latest_log_line() {
  local log="$1"
  [[ -f "$log" ]] || { echo "log not created yet"; return 0; }
  local line
  line=$(grep -E 'RUN |epoch|SKIP cached|SEED_DONE|Traceback|RuntimeError|ValueError|Error' "$log" | tail -n 1 || true)
  [[ -z "$line" ]] && line=$(tail -n 1 "$log" 2>/dev/null || true)
  echo "$line"
}

dashboard() {
  local total="$1"
  while true; do
    local done=0 failed=0 running=0 seen=0
    local running_rows=() failed_rows=()
    local f status stem log ts row pending
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
    echo "curated attention max-score | $(date '+%H:%M:%S') | device=$DEVICE"
    echo "total=$total done=$done failed=$failed running=$running pending=$pending max_parallel=$MAX_PARALLEL"
    echo "epochs=$EPOCHS patience=$PATIENCE batch=$BATCH_SIZE dim=$DIM neg_t=$NEG_THRESHOLD pos_t=$POS_THRESHOLD margin=$MARGIN_WEIGHT pos_fraction=$POS_FRACTION"
    echo "logs: $LOG_DIR"
    echo
    echo "RUNNING"
    if [[ ${#running_rows[@]} -eq 0 ]]; then
      echo "  none"
    else
      for row in "${running_rows[@]}"; do
        stem="${row%%|*}"; log="${row#*|}"
        printf '  %-12s %-14s\n    %s\n' "$(run_stage "$log")" "$stem" "$(latest_log_line "$log")"
      done
    fi
    echo
    echo "FAILED"
    if [[ ${#failed_rows[@]} -eq 0 ]]; then echo "  none"; else printf '  %s\n' "${failed_rows[@]}"; fi
    [[ $((done + failed)) -ge $total ]] && break
    sleep "$DASHBOARD_INTERVAL"
  done
}

run_one_seed() {
  local seed="$1"
  local stem="seed_${seed}"
  local seed_dir="$RUNS_DIR/$stem"
  local log="$LOG_DIR/$stem.log"
  mkdir -p "$seed_dir"
  if [[ -s "$seed_dir/metrics.csv" ]] && grep -q 'SEED_DONE' "$log" 2>/dev/null; then
    write_status "DONE" "$stem" "$log"
    return 0
  fi
  write_status "RUNNING" "$stem" "$log"
  local extra_args=()
  if [[ -n "$DECISION_THRESHOLD" ]]; then
    extra_args+=(--decision-threshold "$DECISION_THRESHOLD")
  fi
  set +e
  uv run python scripts/modeling/train/train_curated_site_attention_max.py \
    --out-dir "$seed_dir" \
    --device "$DEVICE" \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --batch-size "$BATCH_SIZE" \
    --dim "$DIM" \
    --temperature "$TEMPERATURE" \
    --dropout "$DROPOUT" \
    --neg-threshold "$NEG_THRESHOLD" \
    --pos-threshold "$POS_THRESHOLD" \
    --margin-weight "$MARGIN_WEIGHT" \
    --pos-fraction "$POS_FRACTION" \
    "${extra_args[@]}" \
    --seed "$seed" \
    --regime "$REGIME" > "$log" 2>&1
  local rc=$?
  if [[ $rc -eq 0 ]]; then echo "SEED_DONE seed=$seed" >> "$log"; fi
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

for seed in "${SEEDS[@]}"; do
  wait_for_slot
  if [[ "$QUIET" == "1" ]]; then
    run_one_seed "$seed" &
  else
    run_one_seed "$seed" 2>&1 | tee -a "$LOG_DIR/stream.log" &
  fi
  RUNNING_PIDS+=("$!")
done

fail=0
for pid in "${RUNNING_PIDS[@]}"; do wait "$pid" || fail=1; done
[[ -n "$MONITOR_PID" ]] && wait "$MONITOR_PID" 2>/dev/null || true

uv run python - "$RUNS_DIR" "$RESULT_DIR" <<'PY'
import glob
import shutil
import sys
from pathlib import Path

import pandas as pd

runs_dir = Path(sys.argv[1])
result_dir = Path(sys.argv[2])
metric_files = sorted(runs_dir.glob('seed_*/metrics.csv'))
if metric_files:
    df = pd.concat([pd.read_csv(path) for path in metric_files], ignore_index=True)
    if 'pos_fraction' not in df.columns:
        df['pos_fraction'] = pd.NA
    subset = ['regime', 'model', 'seed', 'neg_threshold', 'pos_threshold', 'pos_fraction']
    df = df.drop_duplicates(subset=subset, keep='last')
    df = df.sort_values(['regime', 'model', 'seed']).reset_index(drop=True)
    out = result_dir / 'metrics.csv'
    df.to_csv(out, index=False)
    print(f'merged {len(metric_files)} seed metric files -> {out} ({len(df)} rows)')
else:
    print('no seed metrics found')

for pred in sorted(runs_dir.glob('seed_*/pred_*.npz')):
    shutil.copy2(pred, result_dir / pred.name)
print(f'copied {len(glob.glob(str(runs_dir / "seed_*" / "pred_*.npz")))} prediction files -> {result_dir}')
PY

if [[ $fail -ne 0 ]]; then
  echo "finished with failures; inspect logs in $LOG_DIR" >&2
  exit 1
fi
echo "curated attention max-score finished; seeds: $TOTAL_RUNS -> $RESULT_DIR/metrics.csv"
