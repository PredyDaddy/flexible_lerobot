#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/cqy_workspace/flexible_lerobot/my_devs/openpi_train"
DATASET_ROOT="${ROOT}/easy_use/data/lerobot_v21/desk_cleanup_v1/eraser_cup_multi_task"
CHECKPOINT_DIR="${ROOT}/outputs/checkpoints/pi05_so101_eraser_cup_lora/so101_lora_full_v21_10epoch_bs48_20260702_171345/11090"
ASSET_ID="desk_cleanup_v1/eraser_cup_multi_task_v21_full"
DEFAULT_PROMPT="${DEFAULT_PROMPT:-Put the eraser into the small box}"
PORT="${PORT:-8000}"
PYTORCH_DEVICE="${PYTORCH_DEVICE:-cuda}"

source "${ROOT}/scripts/env.sh"

if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
  echo "Missing checkpoint: ${CHECKPOINT_DIR}" >&2
  exit 1
fi

if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "Missing converted v2.1 dataset: ${DATASET_ROOT}" >&2
  exit 1
fi

if [[ ! -f "${ROOT}/assets/openpi_assets/pi05_so101_eraser_cup_lora/${ASSET_ID}/norm_stats.json" ]]; then
  echo "Missing norm stats for ${ASSET_ID}" >&2
  exit 1
fi

echo "Serving SO101 OpenPI policy"
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
echo "ASSET_ID=${ASSET_ID}"
echo "DATASET_ROOT=${DATASET_ROOT}"
echo "DEFAULT_PROMPT=${DEFAULT_PROMPT}"
echo "PORT=${PORT}"
echo "PYTORCH_DEVICE=${PYTORCH_DEVICE}"

OPENPI_SO101_V21_ROOT="${DATASET_ROOT}" \
python -m openpi_so101.serve_policy \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --dataset-format v21 \
  --asset-id "${ASSET_ID}" \
  --default-prompt "${DEFAULT_PROMPT}" \
  --port "${PORT}" \
  --pytorch-device "${PYTORCH_DEVICE}" \
  "$@"
