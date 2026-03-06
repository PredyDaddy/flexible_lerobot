#!/usr/bin/env bash

set -Eeuo pipefail

# -----------------------------------------------------------------------------
# Full training template (long run, configurable).
# Purpose: close to old long-training style, but keeps local/offline flexibility.
# -----------------------------------------------------------------------------

# Runtime env
CONDA_ENV_NAME="${CONDA_ENV_NAME:-lerobot_flex}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/hf_datasets}"

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/outputs/train}"

# Dataset defaults
DATASET_REPO_ID="${DATASET_REPO_ID:-local__web_collection_demo}"
DATASET_ROOT="${DATASET_ROOT:-${ROOT_DIR}/datasets/lerobot_datasets/local__web_collection_demo}"

# Full defaults (long run)
JOB_NAME="${JOB_NAME:-act_web_collection_full}"
POLICY_TYPE="${POLICY_TYPE:-act}"
POLICY_DEVICE="${POLICY_DEVICE:-auto}" # auto|cuda|cpu
BATCH_SIZE="${BATCH_SIZE:-32}"
STEPS="${STEPS:-100000}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
EVAL_FREQ="${EVAL_FREQ:--1}" # set to 10000 when env/eval path is ready
LOG_FREQ="${LOG_FREQ:-100}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-1000}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"

# Optional overrides
DATASET_EPISODES="${DATASET_EPISODES:-}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}" "${HF_DATASETS_CACHE}"

if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "ERROR: dataset root not found: ${DATASET_ROOT}" >&2
  exit 2
fi

if [[ -z "${CUDA_VISIBLE_DEVICES}" ]]; then
  unset CUDA_VISIBLE_DEVICES
else
  export CUDA_VISIBLE_DEVICES
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found in PATH." >&2
  exit 2
fi

RUNNER=()
if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV_NAME}" ]]; then
  RUNNER=(conda run --no-capture-output -n "${CONDA_ENV_NAME}")
fi

if [[ "${POLICY_DEVICE}" == "auto" ]]; then
  detected_device="$("${RUNNER[@]}" python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')" || true)"
  if [[ "${detected_device}" == "cuda" ]]; then
    POLICY_DEVICE="cuda"
  else
    POLICY_DEVICE="cpu"
  fi
fi

timestamp="$(date +'%Y%m%d_%H%M%S')"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${timestamp}_${JOB_NAME}}"
LOG_FILE="${LOG_DIR}/train_${JOB_NAME}_${timestamp}.log"

TRAIN_CMD=(
  "${RUNNER[@]}"
  lerobot-train
  "--dataset.repo_id=${DATASET_REPO_ID}"
  "--dataset.root=${DATASET_ROOT}"
  "--policy.type=${POLICY_TYPE}"
  "--output_dir=${OUTPUT_DIR}"
  "--job_name=${JOB_NAME}"
  "--policy.device=${POLICY_DEVICE}"
  "--policy.push_to_hub=${PUSH_TO_HUB}"
  "--batch_size=${BATCH_SIZE}"
  "--steps=${STEPS}"
  "--save_freq=${SAVE_FREQ}"
  "--eval_freq=${EVAL_FREQ}"
  "--log_freq=${LOG_FREQ}"
  "--num_workers=${NUM_WORKERS}"
  "--seed=${SEED}"
  "--wandb.enable=${WANDB_ENABLE}"
)

if [[ -n "${DATASET_EPISODES}" ]]; then
  TRAIN_CMD+=("--dataset.episodes=${DATASET_EPISODES}")
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[DRY_RUN] Would execute:"
  printf '%q ' "${TRAIN_CMD[@]}" "$@"
  echo
  echo "[DRY_RUN] Log file: ${LOG_FILE}"
  exit 0
fi

echo "[start] mode=full conda_env=${CONDA_ENV_NAME}"
echo "[start] dataset_root=${DATASET_ROOT}"
echo "[start] output_dir=${OUTPUT_DIR}"
echo "[start] policy_device=${POLICY_DEVICE}"
echo "[start] steps=${STEPS} batch_size=${BATCH_SIZE} num_workers=${NUM_WORKERS}"
echo "[start] eval_freq=${EVAL_FREQ} log_file=${LOG_FILE}"

"${TRAIN_CMD[@]}" "$@" 2>&1 | tee "${LOG_FILE}"
