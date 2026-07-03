#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/cqy_workspace/flexible_lerobot/my_devs/openpi_train"
DATASET_ROOT="${ROOT}/easy_use/data/lerobot_v21/desk_cleanup_v1/eraser_cup_multi_task"
ASSET_ID="desk_cleanup_v1/eraser_cup_multi_task_v21_full"
HOST="${HOST:-localhost}"
PORT="${PORT:-8000}"
PROMPT="${PROMPT:-Put the eraser into the small box}"
SOURCE="${SOURCE:-dataset}"
DATASET_INDEX="${DATASET_INDEX:-0}"
NUM_REQUESTS="${NUM_REQUESTS:-1}"

source "${ROOT}/scripts/env.sh"

OPENPI_SO101_V21_ROOT="${DATASET_ROOT}" \
python -m openpi_so101.remote_infer_client \
  --host "${HOST}" \
  --port "${PORT}" \
  --prompt "${PROMPT}" \
  --source "${SOURCE}" \
  --dataset-format v21 \
  --dataset-index "${DATASET_INDEX}" \
  --asset-id "${ASSET_ID}" \
  --num-requests "${NUM_REQUESTS}" \
  "$@"
