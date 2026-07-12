#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
require_lerobot_flex_host
require_local_env
enable_offline_mode
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "[ERROR] Invalid RUN_ID: ${RUN_ID}" >&2
  exit 1
fi

RUN_DIR="${GR00T17_ROOT}/outputs/smoke/${RUN_ID}"
TRAIN_DIR="${RUN_DIR}/train"
LOG_DIR="${RUN_DIR}/logs"
RUN_REPORT_DIR="${RUN_DIR}/reports"
mkdir -p "${LOG_DIR}" "${RUN_REPORT_DIR}"
require_path_within_root "${RUN_DIR}"

STAGE_ONE_LOG="${LOG_DIR}/stage1.log"
STAGE_TWO_LOG="${LOG_DIR}/stage2_resume.log"
GPU_LOG="${LOG_DIR}/gpu_usage.log"

"${GR00T17_ENV}/bin/python" "${GR00T17_ROOT}/scripts/preflight.py" \
  --root "${GR00T17_ROOT}" \
  --output-dir "${TRAIN_DIR}" \
  --report "${RUN_REPORT_DIR}/preflight_stage1.json" \
  --min-free-gpu-mib 40000 \
  --shard-size 64 \
  --episode-sampling-rate 0.1

start_gpu_monitor "${GPU_LOG}" 2
trap stop_gpu_monitor EXIT INT TERM

echo "[SMOKE] Stage 1: one optimizer step and full resumable checkpoint"
run_n17_finetune "${TRAIN_DIR}" 1 1 1 1 0 64 3 2>&1 | tee "${STAGE_ONE_LOG}"

"${GR00T17_ENV}/bin/python" "${GR00T17_ROOT}/scripts/preflight.py" \
  --root "${GR00T17_ROOT}" \
  --output-dir "${TRAIN_DIR}" \
  --report "${RUN_REPORT_DIR}/preflight_stage2.json" \
  --min-free-gpu-mib 40000 \
  --shard-size 64 \
  --episode-sampling-rate 0.1 \
  --allow-resume

echo "[SMOKE] Stage 2: resume checkpoint-1 and advance to step 2"
run_n17_finetune "${TRAIN_DIR}" 2 1 1 1 0 64 3 2>&1 | tee "${STAGE_TWO_LOG}"
stop_gpu_monitor

"${GR00T17_ENV}/bin/python" "${GR00T17_ROOT}/scripts/preflight.py" \
  --root "${GR00T17_ROOT}" \
  --output-dir "${TRAIN_DIR}" \
  --report "${RUN_REPORT_DIR}/post_smoke_integrity.json" \
  --min-free-gpu-mib 0 \
  --shard-size 64 \
  --episode-sampling-rate 0.1 \
  --allow-resume

"${GR00T17_ENV}/bin/python" "${GR00T17_ROOT}/scripts/validate_smoke_train.py" \
  --root "${GR00T17_ROOT}" \
  --base-model "${GR00T17_MODEL_DIR}" \
  --train-dir "${TRAIN_DIR}" \
  --stage-one-log "${STAGE_ONE_LOG}" \
  --stage-two-log "${STAGE_TWO_LOG}" \
  --gpu-log "${GPU_LOG}" \
  --report "${GR00T17_ROOT}/reports/smoke_train.json"

echo "[OK] Smoke training passed: ${TRAIN_DIR}"
echo "[OK] Smoke report: ${GR00T17_ROOT}/reports/smoke_train.json"
