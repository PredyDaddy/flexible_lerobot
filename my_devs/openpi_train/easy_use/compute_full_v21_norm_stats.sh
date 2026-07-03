#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/cqy_workspace/flexible_lerobot/my_devs/openpi_train"
DATASET_ROOT="${ROOT}/easy_use/data/lerobot_v21/desk_cleanup_v1/eraser_cup_multi_task"
ASSET_ID="desk_cleanup_v1/eraser_cup_multi_task_v21_full"

source "${ROOT}/scripts/env.sh"

if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "Missing converted v2.1 dataset: ${DATASET_ROOT}" >&2
  echo "Run ${ROOT}/easy_use/convert_full_v21.sh first." >&2
  exit 1
fi

OPENPI_SO101_V21_ROOT="${DATASET_ROOT}" \
python -m openpi_so101.compute_norm_stats \
  --dataset-format v21 \
  --asset-id "${ASSET_ID}" \
  --max-frames "${OPENPI_SO101_NORM_MAX_FRAMES:-53235}" \
  --batch-size "${OPENPI_SO101_BATCH_SIZE:-64}" \
  --action-horizon "${OPENPI_SO101_ACTION_HORIZON:-50}" \
  --no-decode-images \
  "$@"
