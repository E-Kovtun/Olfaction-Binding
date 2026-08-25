#!/usr/bin/env bash
set -euo pipefail

# Run the no-graph reference XGBoost baseline for the same 5-repeat protocol
# used by the v5 full_full quantile screen:
#   transductive        -> LoRaX folds 1..5
#   inductive_molecule -> cold-molecule seeds 42..46, fold=1 as source container
#
# Outputs are written by eval_full_full_baseline.py to:
#   results/full_full/tables/baseline_runs.csv
#   results/full_full/tables/baselines_ci95.csv
#   results/full_full/tables/baselines.csv
#
# Examples:
#   bash legacy/scripts/modeling/eval/run_full_full_baseline_5repeat.sh
#   BOOST_SEED=42 bash legacy/scripts/modeling/eval/run_full_full_baseline_5repeat.sh
#   DRY_RUN=1 bash legacy/scripts/modeling/eval/run_full_full_baseline_5repeat.sh

BOOST_SEED="${BOOST_SEED:-42}"
DRY_RUN="${DRY_RUN:-0}"
LOG_DIR="${LOG_DIR:-results/full_full/logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/baseline_5repeat_boost${BOOST_SEED}.log}"

mkdir -p "$LOG_DIR"

cmd=(
  uv run python legacy/scripts/modeling/eval/eval_full_full_baseline.py
  --folds 1 2 3 4 5
  --cold-seeds 42 43 44 45 46
  --boost-seed "$BOOST_SEED"
)

if [[ "$DRY_RUN" == "1" ]]; then
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

printf 'full_full no-graph baseline | boost_seed=%s\n' "$BOOST_SEED" | tee "$LOG_FILE"
printf 'command:' | tee -a "$LOG_FILE"
printf ' %q' "${cmd[@]}" | tee -a "$LOG_FILE"
printf '\n\n' | tee -a "$LOG_FILE"

"${cmd[@]}" 2>&1 | tee -a "$LOG_FILE"
