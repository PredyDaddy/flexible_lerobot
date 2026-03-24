#!/usr/bin/env bash

set -Eeuo pipefail

# -----------------------------------------------------------------------------
# Agilex ACT 单臂训练脚本（右臂）
# 说明：
# - 默认训练 datasets/lerobot_datasets/first_test_right
# - 默认按 15 epochs 自动换算总 steps，而不是手填固定 steps
# - 默认 batch size 取 16，先偏稳妥，后续可由环境变量自行上调
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
DATASET_REPO_ID="${DATASET_REPO_ID:-first_test_right}"
DATASET_ROOT="${DATASET_ROOT:-${ROOT_DIR}/datasets/lerobot_datasets/first_test_right}"
DATASET_VIDEO_BACKEND="${DATASET_VIDEO_BACKEND:-pyav}"

# Training defaults
EPOCHS="${EPOCHS:-15}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS:-5}"
JOB_NAME="${JOB_NAME:-act_agilex_first_test_right_e${EPOCHS}}"
POLICY_TYPE="${POLICY_TYPE:-act}"
POLICY_DEVICE="${POLICY_DEVICE:-auto}"
EVAL_FREQ="${EVAL_FREQ:--1}"
LOG_FREQ="${LOG_FREQ:-100}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-1000}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"

# Optional overrides
DATASET_EPISODES="${DATASET_EPISODES:-}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
DRY_RUN="${DRY_RUN:-0}"
STEPS="${STEPS:-}"
SAVE_FREQ="${SAVE_FREQ:-}"

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

if [[ ! "${EPOCHS}" =~ ^[0-9]+$ ]] || [[ "${EPOCHS}" -le 0 ]]; then
  echo "ERROR: EPOCHS must be a positive integer, got: ${EPOCHS}" >&2
  exit 2
fi

if [[ ! "${BATCH_SIZE}" =~ ^[0-9]+$ ]] || [[ "${BATCH_SIZE}" -le 0 ]]; then
  echo "ERROR: BATCH_SIZE must be a positive integer, got: ${BATCH_SIZE}" >&2
  exit 2
fi

if [[ ! "${SAVE_EVERY_EPOCHS}" =~ ^[0-9]+$ ]] || [[ "${SAVE_EVERY_EPOCHS}" -le 0 ]]; then
  echo "ERROR: SAVE_EVERY_EPOCHS must be a positive integer, got: ${SAVE_EVERY_EPOCHS}" >&2
  exit 2
fi

if [[ -n "${STEPS}" ]] && { [[ ! "${STEPS}" =~ ^[0-9]+$ ]] || [[ "${STEPS}" -le 0 ]]; }; then
  echo "ERROR: STEPS must be a positive integer when provided, got: ${STEPS}" >&2
  exit 2
fi

if [[ -n "${SAVE_FREQ}" ]] && { [[ ! "${SAVE_FREQ}" =~ ^[0-9]+$ ]] || [[ "${SAVE_FREQ}" -le 0 ]]; }; then
  echo "ERROR: SAVE_FREQ must be a positive integer when provided, got: ${SAVE_FREQ}" >&2
  exit 2
fi

if [[ "${POLICY_DEVICE}" == "auto" ]]; then
  detected_device="$("${RUNNER[@]}" python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')" || true)"
  if [[ "${detected_device}" == "cuda" ]]; then
    POLICY_DEVICE="cuda"
  else
    POLICY_DEVICE="cpu"
  fi
fi

INFO_JSON="${DATASET_ROOT}/meta/info.json"
if [[ ! -f "${INFO_JSON}" ]]; then
  echo "ERROR: dataset meta/info.json not found: ${INFO_JSON}" >&2
  exit 2
fi

TOTAL_FRAMES="$("${RUNNER[@]}" python - <<'PY' "${INFO_JSON}"
import json
import sys
from pathlib import Path

info_path = Path(sys.argv[1])
info = json.loads(info_path.read_text())
total_frames = info.get("total_frames")
if not isinstance(total_frames, int) or total_frames <= 0:
    raise SystemExit(f"Invalid total_frames in {info_path}: {total_frames}")
print(total_frames)
PY
)"

STEPS_PER_EPOCH="$(( (TOTAL_FRAMES + BATCH_SIZE - 1) / BATCH_SIZE ))"
CALCULATED_STEPS="$(( STEPS_PER_EPOCH * EPOCHS ))"
CALCULATED_SAVE_FREQ="$(( STEPS_PER_EPOCH * SAVE_EVERY_EPOCHS ))"

if [[ -n "${STEPS}" ]]; then
  TOTAL_STEPS="${STEPS}"
else
  TOTAL_STEPS="${CALCULATED_STEPS}"
fi

if [[ -n "${SAVE_FREQ}" ]]; then
  TOTAL_SAVE_FREQ="${SAVE_FREQ}"
else
  TOTAL_SAVE_FREQ="${CALCULATED_SAVE_FREQ}"
fi

timestamp="$(date +'%Y%m%d_%H%M%S')"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${timestamp}_${JOB_NAME}}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/train_${JOB_NAME}_${timestamp}.log}"

TRAIN_CMD=(
  "${RUNNER[@]}"
  lerobot-train
  "--dataset.repo_id=${DATASET_REPO_ID}"
  "--dataset.root=${DATASET_ROOT}"
  "--dataset.video_backend=${DATASET_VIDEO_BACKEND}"
  "--policy.type=${POLICY_TYPE}"
  "--output_dir=${OUTPUT_DIR}"
  "--job_name=${JOB_NAME}"
  "--policy.device=${POLICY_DEVICE}"
  "--policy.push_to_hub=${PUSH_TO_HUB}"
  "--batch_size=${BATCH_SIZE}"
  "--steps=${TOTAL_STEPS}"
  "--save_freq=${TOTAL_SAVE_FREQ}"
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

echo "[calc] dataset_repo_id=${DATASET_REPO_ID}"
echo "[calc] dataset_root=${DATASET_ROOT}"
echo "[calc] total_frames=${TOTAL_FRAMES}"
echo "[calc] batch_size=${BATCH_SIZE}"
echo "[calc] epochs=${EPOCHS}"
echo "[calc] steps_per_epoch=${STEPS_PER_EPOCH}"
echo "[calc] calculated_steps=${CALCULATED_STEPS}"
echo "[calc] calculated_save_freq=${CALCULATED_SAVE_FREQ} (save_every_epochs=${SAVE_EVERY_EPOCHS})"
if [[ -n "${STEPS}" ]]; then
  echo "[calc] manual_steps_override=${STEPS}"
fi
if [[ -n "${SAVE_FREQ}" ]]; then
  echo "[calc] manual_save_freq_override=${SAVE_FREQ}"
fi
echo "[calc] total_steps=${TOTAL_STEPS}"
echo "[calc] save_freq=${TOTAL_SAVE_FREQ}"
echo "[calc] dataset_video_backend=${DATASET_VIDEO_BACKEND}"
echo "[start] script=train_first_test_right.sh conda_env=${CONDA_ENV_NAME}"
echo "[start] output_dir=${OUTPUT_DIR}"
echo "[start] policy_device=${POLICY_DEVICE}"
echo "[start] num_workers=${NUM_WORKERS} eval_freq=${EVAL_FREQ} log_freq=${LOG_FREQ}"
echo "[start] log_file=${LOG_FILE}"

"${TRAIN_CMD[@]}" "$@" 2>&1 | tee "${LOG_FILE}"
