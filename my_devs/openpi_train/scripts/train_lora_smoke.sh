#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/cqy_workspace/flexible_lerobot/my_devs/openpi_train"
source "${ROOT}/scripts/env.sh"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
EXP_NAME="${EXP_NAME:-so101_lora_smoke_${RUN_ID}}"
LOG_DIR="${ROOT}/logs/${EXP_NAME}"
mkdir -p "${LOG_DIR}"

python -m openpi_so101.train \
  --exp-name "${EXP_NAME}" \
  --num-train-steps "${OPENPI_SO101_STEPS:-3}" \
  --batch-size "${OPENPI_SO101_BATCH_SIZE:-1}" \
  --max-frames "${OPENPI_SO101_MAX_FRAMES:-128}" \
  --dataset-format "${OPENPI_SO101_DATASET_FORMAT:-v3}" \
  --save-interval "${OPENPI_SO101_SAVE_INTERVAL:-3}" \
  --log-interval "${OPENPI_SO101_LOG_INTERVAL:-1}" \
  --overwrite \
  "$@" 2>&1 | tee "${LOG_DIR}/train.log"

echo "EXP_NAME=${EXP_NAME}"
echo "LOG=${LOG_DIR}/train.log"
echo "CHECKPOINT_ROOT=${OPENPI_SO101_CHECKPOINT_BASE_DIR}/pi05_so101_eraser_cup_lora/${EXP_NAME}"
