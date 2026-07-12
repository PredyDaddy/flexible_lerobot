#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
require_lerobot_flex_host
require_local_env
enable_offline_mode
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RESUME="${RESUME:-0}"
MAX_STEPS="${MAX_STEPS:-2000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
SHARD_SIZE="${SHARD_SIZE:-1024}"
SAVE_STEPS="${SAVE_STEPS:-500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"

if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "[ERROR] Invalid RUN_ID: ${RUN_ID}" >&2
  exit 1
fi

RUN_DIR="${GR00T17_ROOT}/outputs/formal/${RUN_ID}"
TRAIN_DIR="${RUN_DIR}/train"
LOG_DIR="${RUN_DIR}/logs"
RUN_REPORT_DIR="${RUN_DIR}/reports"
mkdir -p "${LOG_DIR}" "${RUN_REPORT_DIR}"
require_path_within_root "${RUN_DIR}"

TERMINAL_LOG="${LOG_DIR}/train_terminal.log"
GPU_LOG="${LOG_DIR}/gpu_usage.log"
PREFLIGHT_REPORT="${RUN_REPORT_DIR}/preflight_$(date +%Y%m%d_%H%M%S).json"

if [[ "${RESUME}" != "0" && "${RESUME}" != "1" ]]; then
  echo "[ERROR] RESUME must be 0 or 1, got: ${RESUME}" >&2
  exit 1
fi

PREFLIGHT_RESUME_ARGS=()
if [[ "${RESUME}" == "1" ]]; then
  PREFLIGHT_RESUME_ARGS+=(--allow-resume)
fi

"${GR00T17_ENV}/bin/python" "${GR00T17_ROOT}/scripts/preflight.py" \
  --root "${GR00T17_ROOT}" \
  --output-dir "${TRAIN_DIR}" \
  --report "${PREFLIGHT_REPORT}" \
  --min-free-gpu-mib 40000 \
  --shard-size "${SHARD_SIZE}" \
  --episode-sampling-rate 0.1 \
  --require-smoke \
  "${PREFLIGHT_RESUME_ARGS[@]}"

{
  echo "RUN_ID=${RUN_ID}"
  echo "RESUME=${RESUME}"
  echo "TRAIN_DIR=${TRAIN_DIR}"
  echo "MAX_STEPS=${MAX_STEPS}"
  echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}"
  echo "GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}"
  echo "DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS}"
  echo "SHARD_SIZE=${SHARD_SIZE}"
  echo "SAVE_STEPS=${SAVE_STEPS}"
  echo "SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT}"
} >>"${RUN_REPORT_DIR}/launch_parameters.env"

echo "[TRAIN] Output: ${TRAIN_DIR}"
echo "[TRAIN] Terminal log: ${TERMINAL_LOG}"
echo "[TRAIN] GPU log: ${GPU_LOG}"
start_gpu_monitor "${GPU_LOG}"
trap stop_gpu_monitor EXIT INT TERM

run_n17_finetune \
  "${TRAIN_DIR}" \
  "${MAX_STEPS}" \
  "${SAVE_STEPS}" \
  "${GLOBAL_BATCH_SIZE}" \
  "${GRADIENT_ACCUMULATION_STEPS}" \
  "${DATALOADER_NUM_WORKERS}" \
  "${SHARD_SIZE}" \
  "${SAVE_TOTAL_LIMIT}" 2>&1 | tee -a "${TERMINAL_LOG}"

echo "[OK] Formal training completed: ${TRAIN_DIR}"
