#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/cqy_workspace/flexible_lerobot/my_devs/openpi_train"

source "${ROOT}/scripts/env.sh"
python "${ROOT}/easy_use/check_lerobot_v21_dataset.py" "$@"
