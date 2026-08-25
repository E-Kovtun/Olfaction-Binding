#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Interaction-profile collaborative-filtering experiment, with a live dashboard.
# Schedules 10 units = {transductive, inductive_molecule} x 5 repeats. Each unit computes
# raw / profile_A / profile_B (+ profile_A_own / profile_B_own in transductive), stratified
# matched-prevalence inductive split, and writes a per-unit CSV under tables/runs/. On completion
# the per-unit CSVs are concatenated into one CSV that
# notebooks/graph/molecule_profile_cf.ipynb reads and plots.
#
# Parameter-free (numpy + XGBoost, no neural training) so this runs fine on CPU too.
#
# Examples:
#   bash legacy/scripts/modeling/eval/run_molecule_profile_cf.sh
#   MAX_PARALLEL=4 bash legacy/scripts/modeling/eval/run_molecule_profile_cf.sh
#   K=15 KP=10 QUIET=0 bash legacy/scripts/modeling/eval/run_molecule_profile_cf.sh

RESULT_DIR="${RESULT_DIR:-results/graph/mol_profile_cf}"
RUNS_DIR="$RESULT_DIR/tables/runs"
OUT="${OUT:-$RESULT_DIR/tables/molecule_profile_cf_runs.csv}"
LOG_DIR="$RESULT_DIR/logs"
STATUS_DIR="$RESULT_DIR/run_status"
K="${K:-10}"
KP="${KP:-10}"
BOOST_SEED="${BOOST_SEED:-42}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
QUIET="${QUIET:-1}"
DASHBOARD_INTERVAL="${DASHBOARD_INTERVAL:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

mkdir -p "$RUNS_DIR" "$LOG_DIR" "$STATUS_DIR"
rm -f "$STATUS_DIR"/*.status

UNITS=(
  "transductive 1" "transductive 2" "transductive 3" "transductive 4" "transductive 5"
  "inductive_molecule 42" "inductive_molecule 43" "inductive_molecule 44"
  "inductive_molecule 45" "inductive_molecule 46"
)
TOTAL_RUNS=${#UNITS[@]}

write_status() {
  printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$(date '+%Y-%m-%d %H:%M:%S')" > "$STATUS_DIR/$2.status"
}

run_stage() {
  local log="$1"
  [[ -f "$log" ]] || { echo "STARTING"; return 0; }
  if grep -qE 'Traceback|RuntimeError|Error' "$log"; then echo "ERROR"
  elif grep -q 'UNIT_DONE' "$log"; then echo "DONE"
  elif grep -q 'probe profile_B_own' "$log"; then echo "PROBE_B_OWN"
  elif grep -q 'probe profile_A_own' "$log"; then echo "PROBE_A_OWN"
  elif grep -q 'probe profile_B' "$log"; then echo "PROBE_B"
  elif grep -q 'probe profile_A' "$log"; then echo "PROBE_A"
  elif grep -q 'probe raw' "$log"; then echo "PROBE_RAW"
  elif grep -q 'building profiles' "$log"; then echo "SETUP"
  else echo "INIT"; fi
}

latest_log_line() {
  local log="$1"
  [[ -f "$log" ]] || { echo "log not created yet"; return 0; }
  local line
  line=$(grep -E 'prev=|probe |building profiles|UNIT_DONE|Traceback|RuntimeError|Error' "$log" | tail -n 1 || true)
  [[ -z "$line" ]] && line=$(tail -n 1 "$log" 2>/dev/null || true)
  echo "$line"
}

dashboard() {
  local total="$1"
  while true; do
    local done=0 failed=0 running=0 seen=0
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
    local pending=$((total - seen)); [[ $pending -lt 0 ]] && pending=0
    printf '\033[H\033[2J'
    echo "molecule profile-CF | $(date '+%H:%M:%S') | k=$K kp=$KP boost_seed=$BOOST_SEED"
    echo "total=$total done=$done failed=$failed running=$running pending=$pending max_parallel=$MAX_PARALLEL"
    echo "logs: $LOG_DIR"
    echo; echo "RUNNING"
    if [[ ${#running_rows[@]} -eq 0 ]]; then echo "  none"; else
      local row
      for row in "${running_rows[@]}"; do
        stem="${row%%|*}"; log="${row#*|}"
        printf '  %-15s %-26s\n    %s\n' "$(run_stage "$log")" "$stem" "$(latest_log_line "$log")"
      done
    fi
    echo; echo "FAILED"
    if [[ ${#failed_rows[@]} -eq 0 ]]; then echo "  none"; else printf '  %s\n' "${failed_rows[@]}"; fi
    [[ $((done + failed)) -ge $total ]] && break
    sleep "$DASHBOARD_INTERVAL"
  done
}

run_one() {
  local regime="$1" rep="$2"
  local stem="${regime}_rep${rep}"
  local unit_out="$RUNS_DIR/${stem}.csv"
  local log="$LOG_DIR/${stem}.log"
  if [[ -s "$unit_out" ]] && grep -q 'UNIT_DONE' "$log" 2>/dev/null; then
    write_status "DONE" "$stem" "$log"; return 0
  fi
  write_status "RUNNING" "$stem" "$log"
  set +e
  uv run python legacy/scripts/modeling/eval/run_molecule_profile_cf.py \
    --regime "$regime" --repeat "$rep" --k "$K" --kp "$KP" --boost-seed "$BOOST_SEED" \
    --out "$unit_out" > "$log" 2>&1
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

for unit in "${UNITS[@]}"; do
  read -r regime rep <<< "$unit"
  wait_for_slot
  if [[ "$QUIET" == "1" ]]; then
    run_one "$regime" "$rep" &
  else
    run_one "$regime" "$rep" 2>&1 | tee -a "$LOG_DIR/stream.log" &
  fi
  RUNNING_PIDS+=("$!")
done

fail=0
for pid in "${RUNNING_PIDS[@]}"; do wait "$pid" || fail=1; done
[[ -n "$MONITOR_PID" ]] && wait "$MONITOR_PID" 2>/dev/null || true

# concatenate per-unit CSVs into the combined table the notebook reads
uv run python - "$RUNS_DIR" "$OUT" <<'PY'
import sys, glob, pandas as pd
runs_dir, out = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(f"{runs_dir}/*.csv"))
if files:
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.to_csv(out, index=False)
    print(f"combined {len(files)} unit CSVs -> {out}  ({len(df)} rows)")
else:
    print("no unit CSVs found")
PY

if [[ $fail -ne 0 ]]; then
  echo "finished with failures; inspect logs in $LOG_DIR" >&2
  exit 1
fi
echo "molecule profile-CF finished; units: $TOTAL_RUNS -> $OUT"
