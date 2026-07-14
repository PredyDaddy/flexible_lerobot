#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../" && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"

export DATASET_NAME="${DATASET_NAME:-jz_robot_pin_timed_curated_42eps_20260713}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export NUM_WORKERS="${NUM_WORKERS:-0}"
export STEPS_OVERRIDE="${STEPS_OVERRIDE:-2}"
export SAVE_FREQ_OVERRIDE="${SAVE_FREQ_OVERRIDE:-${STEPS_OVERRIDE}}"
export LOG_FREQ_OVERRIDE="${LOG_FREQ_OVERRIDE:-1}"
export RUN_NAME="${RUN_NAME:-smoke_act_${DATASET_NAME}_${STAMP}}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/tests/outputs/${RUN_NAME}}"

if [[ "${RUN_QUICK_AUDIT:-1}" == "1" ]]; then
  FULL_VIDEO_DECODE=0 \
  REPORT_DIR="${OUTPUT_DIR}_audit" \
  bash "${SCRIPT_DIR}/audit_dataset.sh"
fi

bash "${SCRIPT_DIR}/train_act.sh"
echo "status=PASS smoke_output=${OUTPUT_DIR}"
