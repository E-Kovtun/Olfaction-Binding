#!/usr/bin/env bash
set -euo pipefail

# Run downstream XGBoost feature probes on already trained full_full v5
# GNN signed q99 checkpoints. No GNN training is performed.
#
# Default protocol matches the quantile screen:
#   transductive        -> LoRaX folds 1..5
#   inductive_molecule -> cold seeds 42..46
#
# Probe variants are implemented in eval_graph_full_full_signed_q99_feature_probes.py:
#   z2_rawmol
#   z2_rawprot_rawmol
#   z2_x0_rawmol
#   z2_rawprot_x0_x1_rawmol
#
# Examples:
#   bash legacy/scripts/modeling/eval/run_graph_full_full_signed_q99_feature_probes.sh
#   CHECKPOINT=best_val bash legacy/scripts/modeling/eval/run_graph_full_full_signed_q99_feature_probes.sh
#   REGIME=inductive_molecule bash legacy/scripts/modeling/eval/run_graph_full_full_signed_q99_feature_probes.sh
#   DEVICE=cpu DRY_RUN=1 bash legacy/scripts/modeling/eval/run_graph_full_full_signed_q99_feature_probes.sh

RESULT_DIR="${RESULT_DIR:-results/graph/full_full/v5/quantile_screen/training}"
OUT_DIR="${OUT_DIR:-results/graph/full_full/v5/signed_q99_feature_probes}"
CHECKPOINT="${CHECKPOINT:-last_epoch}"
REGIME="${REGIME:-both}"
DEVICE="${DEVICE:-cuda}"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/signed_q99_feature_probes_${CHECKPOINT}_${REGIME}.log}"

mkdir -p "$LOG_DIR"

cmd=(
  uv run python legacy/scripts/modeling/eval/eval_graph_full_full_signed_q99_feature_probes.py
  --results-dir "$RESULT_DIR"
  --out-dir "$OUT_DIR"
  --checkpoint "$CHECKPOINT"
  --regime "$REGIME"
  --device "$DEVICE"
)
if [[ "$FORCE" == "1" ]]; then
  cmd+=(--force)
fi

if [[ "$DRY_RUN" == "1" ]]; then
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

printf 'signed q99 feature probes | checkpoint=%s regime=%s device=%s\n' "$CHECKPOINT" "$REGIME" "$DEVICE" | tee "$LOG_FILE"
printf 'command:' | tee -a "$LOG_FILE"
printf ' %q' "${cmd[@]}" | tee -a "$LOG_FILE"
printf '\n\n' | tee -a "$LOG_FILE"

"${cmd[@]}" 2>&1 | tee -a "$LOG_FILE"
