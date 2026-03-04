#!/usr/bin/env bash
set -euo pipefail

# Reproduce GROOT training used in the previous successful run.
#
# Fresh run example:
#   DRY_RUN=true bash my_devs/train/groot/run_groot_train_repro.sh
#   bash my_devs/train/groot/run_groot_train_repro.sh
#
# Resume example:
#   RESUME=true \
#   CONFIG_PATH=/data/cqy_workspace/flexible_lerobot/outputs/train/groot_grasp_block_in_bin1_repro_20260302_223413/bs32_20260302_223447/checkpoints/010000/pretrained_model/train_config.json \
#   bash my_devs/train/groot/run_groot_train_repro.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR_DEFAULT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${REPO_DIR_DEFAULT}" ]]; then
  REPO_DIR_DEFAULT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi
REPO_DIR="${REPO_DIR:-${REPO_DIR_DEFAULT}}"
cd "${REPO_DIR}"

CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-lerobot_flex}"

# Runtime env
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Resume control
RESUME="${RESUME:-false}"
CONFIG_PATH="${CONFIG_PATH:-}"

# Core training config (same as prior repro run, with safer defaults)
POLICY_TYPE="${POLICY_TYPE:-groot}"
POLICY_REPO_ID="${POLICY_REPO_ID:-robotech/groot}"
POLICY_PUSH_TO_HUB="${POLICY_PUSH_TO_HUB:-false}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
POLICY_USE_BF16="${POLICY_USE_BF16:-true}"
POLICY_BASE_MODEL_PATH="${POLICY_BASE_MODEL_PATH:-/data/cqy_workspace/flexible_lerobot/assets/modelscope/GR00T-N1.5-3B}"

DATASET_REPO_ID="${DATASET_REPO_ID:-admin123/grasp_block_in_bin1}"
DATASET_ROOT="${DATASET_ROOT:-/home/cqy/.cache/huggingface/lerobot/admin123/grasp_block_in_bin1}"
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"

BATCH_SIZE="${BATCH_SIZE:-32}"
STEPS="${STEPS:-15000}"
SAVE_FREQ="${SAVE_FREQ:-2000}"
EVAL_FREQ="${EVAL_FREQ:-20000}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"
JOB_NAME="${JOB_NAME:-groot_grasp_block_in_bin1_repro}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/cqy_workspace/flexible_lerobot/outputs/train/groot_grasp_block_in_bin1_repro_${RUN_ID}}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/bs${BATCH_SIZE}_${RUN_ID}}"

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "[ERROR] conda init script not found: ${CONDA_SH}" >&2
  exit 1
fi

if [[ "${RESUME}" != "true" ]]; then
  if [[ ! -d "${POLICY_BASE_MODEL_PATH}" ]]; then
    echo "[ERROR] base model path does not exist: ${POLICY_BASE_MODEL_PATH}" >&2
    exit 1
  fi
  if [[ ! -d "${DATASET_ROOT}" ]]; then
    echo "[ERROR] dataset root does not exist: ${DATASET_ROOT}" >&2
    exit 1
  fi
  if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "[ERROR] output dir already exists: ${OUTPUT_DIR}" >&2
    echo "        Set a new RUN_ID/OUTPUT_DIR, or use RESUME=true with CONFIG_PATH." >&2
    exit 1
  fi
fi

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

TRAIN_ENTRY=""
if command -v lerobot-train >/dev/null 2>&1; then
  TRAIN_ENTRY="lerobot-train"
else
  TRAIN_ENTRY="python src/lerobot/scripts/lerobot_train.py"
fi

export PYTHONUNBUFFERED=1
export HF_ENDPOINT
export TOKENIZERS_PARALLELISM

if [[ "${RESUME}" == "true" ]]; then
  if [[ -z "${CONFIG_PATH}" ]]; then
    echo "[ERROR] RESUME=true but CONFIG_PATH is empty." >&2
    exit 1
  fi
  if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "[ERROR] resume config file not found: ${CONFIG_PATH}" >&2
    exit 1
  fi

  CMD=(
    ${TRAIN_ENTRY}
    "--config_path=${CONFIG_PATH}"
    "--resume=true"
  )
else
  CMD=(
    ${TRAIN_ENTRY}
    "--policy.type=${POLICY_TYPE}"
    "--policy.repo_id=${POLICY_REPO_ID}"
    "--policy.push_to_hub=${POLICY_PUSH_TO_HUB}"
    "--dataset.repo_id=${DATASET_REPO_ID}"
    "--dataset.root=${DATASET_ROOT}"
    "--dataset.video_backend=${VIDEO_BACKEND}"
    "--batch_size=${BATCH_SIZE}"
    "--steps=${STEPS}"
    "--output_dir=${OUTPUT_DIR}"
    "--job_name=${JOB_NAME}"
    "--policy.device=${POLICY_DEVICE}"
    "--wandb.enable=${WANDB_ENABLE}"
    "--policy.base_model_path=${POLICY_BASE_MODEL_PATH}"
    "--save_freq=${SAVE_FREQ}"
    "--eval_freq=${EVAL_FREQ}"
    "--policy.use_bf16=${POLICY_USE_BF16}"
  )
fi

echo "[INFO] Repo: ${REPO_DIR}"
echo "[INFO] Conda env: ${CONDA_ENV}"
echo "[INFO] HF_ENDPOINT: ${HF_ENDPOINT}"
echo "[INFO] TRAIN_ENTRY: ${TRAIN_ENTRY}"
echo "[INFO] RESUME: ${RESUME}"
if [[ "${RESUME}" == "true" ]]; then
  echo "[INFO] CONFIG_PATH: ${CONFIG_PATH}"
else
  echo "[INFO] DATASET_REPO_ID: ${DATASET_REPO_ID}"
  echo "[INFO] DATASET_ROOT: ${DATASET_ROOT}"
  echo "[INFO] BASE_MODEL_PATH: ${POLICY_BASE_MODEL_PATH}"
  echo "[INFO] OUTPUT_DIR: ${OUTPUT_DIR}"
  echo "[INFO] STEPS: ${STEPS}, BATCH_SIZE: ${BATCH_SIZE}"
fi

echo "[INFO] Command:"
printf '  %q' "${CMD[@]}"
echo

if [[ "${DRY_RUN:-false}" == "true" ]]; then
  echo "[INFO] DRY_RUN=true, exit without execution."
  exit 0
fi

exec "${CMD[@]}"

