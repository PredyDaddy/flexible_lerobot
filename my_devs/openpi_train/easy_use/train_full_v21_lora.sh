#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/cqy_workspace/flexible_lerobot/my_devs/openpi_train"
DATASET_ROOT="${ROOT}/easy_use/data/lerobot_v21/desk_cleanup_v1/eraser_cup_multi_task"
BASE_PARAMS="${ROOT}/assets/openpi_cache/openpi-assets/checkpoints/pi05_base/params"
ASSETS_BASE_DIR="${ROOT}/assets/openpi_assets"
CHECKPOINT_BASE_DIR="${ROOT}/outputs/checkpoints"
ASSET_ID="desk_cleanup_v1/eraser_cup_multi_task_v21_full"
RUN_ID="${RUN_ID:-full_v21_$(date +%Y%m%d_%H%M%S)}"
EXP_NAME="${EXP_NAME:-so101_lora_${RUN_ID}}"
LOG_DIR="${ROOT}/logs/${EXP_NAME}"

mkdir -p "${LOG_DIR}"

source "${ROOT}/scripts/env.sh"

if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "Missing converted v2.1 dataset: ${DATASET_ROOT}" >&2
  echo "Run ${ROOT}/easy_use/convert_full_v21.sh first." >&2
  exit 1
fi

if [[ ! -f "${ASSETS_BASE_DIR}/pi05_so101_eraser_cup_lora/${ASSET_ID}/norm_stats.json" ]]; then
  echo "Missing norm stats for ${ASSET_ID}." >&2
  echo "Run ${ROOT}/easy_use/compute_full_v21_norm_stats.sh first." >&2
  exit 1
fi

python -m openpi_so101.train \
  --exp-name "${EXP_NAME}" \
  --dataset-format v21 \
  --converted-root "${DATASET_ROOT}" \
  --asset-id "${ASSET_ID}" \
  --base-params "${BASE_PARAMS}" \
  --num-train-steps "${OPENPI_SO101_STEPS:-33300}" \
  --batch-size "${OPENPI_SO101_BATCH_SIZE:-1}" \
  --max-frames "${OPENPI_SO101_MAX_FRAMES:-53235}" \
  --action-horizon "${OPENPI_SO101_ACTION_HORIZON:-50}" \
  --learning-rate "${OPENPI_SO101_LR:-5e-5}" \
  --save-interval "${OPENPI_SO101_SAVE_INTERVAL:-1000}" \
  --log-interval "${OPENPI_SO101_LOG_INTERVAL:-10}" \
  --overwrite \
  "$@" 2>&1 | tee "${LOG_DIR}/train.log"

echo "EXP_NAME=${EXP_NAME}"
echo "LOG=${LOG_DIR}/train.log"
echo "CHECKPOINT_ROOT=${CHECKPOINT_BASE_DIR}/pi05_so101_eraser_cup_lora/${EXP_NAME}"
