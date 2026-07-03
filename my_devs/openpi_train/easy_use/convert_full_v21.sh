#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/cqy_workspace/flexible_lerobot/my_devs/openpi_train"
SOURCE_ROOT="/data/cqy_workspace/flexible_lerobot/datasets/desk_cleanup_v1/eraser_cup_multi_task"
OUTPUT_ROOT="${ROOT}/easy_use/data/lerobot_v21/desk_cleanup_v1/eraser_cup_multi_task"
REPO_ID="desk_cleanup_v1/eraser_cup_multi_task"
EPISODES="${EPISODES:-157}"

source "${ROOT}/scripts/env.sh"

python -m openpi_so101.convert_v3_to_v21_pilot \
  --source-root "${SOURCE_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --repo-id "${REPO_ID}" \
  --episodes "${EPISODES}" \
  --allow-more-episodes \
  --overwrite

echo "Converted LeRobot v2.1 dataset:"
echo "${OUTPUT_ROOT}"
