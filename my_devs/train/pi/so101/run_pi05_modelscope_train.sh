#!/usr/bin/env bash
set -euo pipefail

find_repo_root() {
  local dir="$1"
  while [[ "${dir}" != "/" ]]; do
    if [[ -f "${dir}/pyproject.toml" && -d "${dir}/src/lerobot" ]]; then
      printf '%s\n' "${dir}"
      return 0
    fi
    dir="$(dirname "${dir}")"
  done
  return 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT_DEFAULT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${PROJECT_ROOT_DEFAULT}" ]]; then
  PROJECT_ROOT_DEFAULT="$(find_repo_root "${SCRIPT_DIR}" || true)"
fi
if [[ -z "${PROJECT_ROOT_DEFAULT}" ]]; then
  echo "[ERROR] failed to locate repo root from script dir: ${SCRIPT_DIR}" >&2
  exit 1
fi

PROJECT_ROOT=${PROJECT_ROOT:-${PROJECT_ROOT_DEFAULT}}
cd "${PROJECT_ROOT}"

CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-lerobot_flex}"

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "[ERROR] conda init script not found: ${CONDA_SH}" >&2
  exit 1
fi

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

DATASET_REPO_ID=${DATASET_REPO_ID:-admin123/grasp_block_in_bin1}
DATASET_ROOT=${DATASET_ROOT:-/home/cqy/.cache/huggingface/lerobot/admin123/grasp_block_in_bin1}
MODEL_DIR=${MODEL_DIR:-${PROJECT_ROOT}/assets/modelscope/lerobot/pi05_base}
RUN_MODE=${RUN_MODE:-full}  # smoke | full

if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "[ERROR] DATASET_ROOT not found: ${DATASET_ROOT}" >&2
  exit 1
fi

if [[ ! -e "${MODEL_DIR}/config.json" ]]; then
  echo "[ERROR] MODEL_DIR not ready: ${MODEL_DIR}" >&2
  exit 1
fi

RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
OUTPUT_ROOT=${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/train/pi05_grasp_block_in_bin1_repro_${RUN_ID}}

case "${RUN_MODE}" in
  smoke)
    STEPS=${STEPS:-200}
    BATCH_SIZE=${BATCH_SIZE:-8}
    JOB_NAME=${JOB_NAME:-pi05_grasp_block_in_bin1_smoke}
    ;;
  full)
    STEPS=${STEPS:-15000}
    BATCH_SIZE=${BATCH_SIZE:-32}
    JOB_NAME=${JOB_NAME:-pi05_grasp_block_in_bin1_repro}
    ;;
  *)
    echo "[ERROR] RUN_MODE must be smoke or full, got: ${RUN_MODE}" >&2
    exit 1
    ;;
esac

SAVE_FREQ=${SAVE_FREQ:-1000}
LOG_FREQ=${LOG_FREQ:-50}
NUM_WORKERS=${NUM_WORKERS:-4}
POLICY_COMPILE_MODEL=${POLICY_COMPILE_MODEL:-false}
POLICY_GRADIENT_CHECKPOINTING=${POLICY_GRADIENT_CHECKPOINTING:-true}

OUT_DIR=${OUTPUT_ROOT}/bs${BATCH_SIZE}_$(date +%Y%m%d_%H%M%S)
LOG_FILE=${OUT_DIR}.log
mkdir -p "${OUTPUT_ROOT}"

echo "[INFO] RUN_MODE=${RUN_MODE}"
echo "[INFO] MODEL_DIR=${MODEL_DIR}"
echo "[INFO] DATASET_REPO_ID=${DATASET_REPO_ID}"
echo "[INFO] DATASET_ROOT=${DATASET_ROOT}"
echo "[INFO] OUT_DIR=${OUT_DIR}"
echo "[INFO] LOG_FILE=${LOG_FILE}"
echo "[INFO] POLICY_COMPILE_MODEL=${POLICY_COMPILE_MODEL}"
echo "[INFO] POLICY_GRADIENT_CHECKPOINTING=${POLICY_GRADIENT_CHECKPOINTING}"

set -x
python ${PROJECT_ROOT}/src/lerobot/scripts/lerobot_train.py \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.video_backend=pyav \
  --policy.type=pi05 \
  --policy.pretrained_path="${MODEL_DIR}" \
  --policy.push_to_hub=false \
  --policy.compile_model="${POLICY_COMPILE_MODEL}" \
  --policy.gradient_checkpointing="${POLICY_GRADIENT_CHECKPOINTING}" \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
  --output_dir="${OUT_DIR}" \
  --job_name="${JOB_NAME}" \
  --batch_size="${BATCH_SIZE}" \
  --num_workers="${NUM_WORKERS}" \
  --steps="${STEPS}" \
  --save_freq="${SAVE_FREQ}" \
  --log_freq="${LOG_FREQ}" \
  |& tee "${LOG_FILE}"
