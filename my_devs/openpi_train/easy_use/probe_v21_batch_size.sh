#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/cqy_workspace/flexible_lerobot/my_devs/openpi_train"
DATASET_ROOT="${ROOT}/easy_use/data/lerobot_v21/desk_cleanup_v1/eraser_cup_multi_task"
BASE_PARAMS="${ROOT}/assets/openpi_cache/openpi-assets/checkpoints/pi05_base/params"
ASSET_ID="desk_cleanup_v1/eraser_cup_multi_task_v21_full"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <batch_size> [extra probe args]" >&2
  exit 2
fi

BATCH_SIZE="$1"
shift

source "${ROOT}/scripts/env.sh"

python "${ROOT}/easy_use/probe_v21_batch_size.py" \
  --batch-size "${BATCH_SIZE}" \
  --dataset-root "${DATASET_ROOT}" \
  --base-params "${BASE_PARAMS}" \
  --asset-id "${ASSET_ID}" \
  "$@"
