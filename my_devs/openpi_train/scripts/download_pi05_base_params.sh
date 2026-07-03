#!/usr/bin/env bash
set -euo pipefail

source /data/cqy_workspace/flexible_lerobot/my_devs/openpi_train/scripts/env.sh
python /data/cqy_workspace/flexible_lerobot/my_devs/openpi_train/scripts/download_pi05_base_params.py "$@"
