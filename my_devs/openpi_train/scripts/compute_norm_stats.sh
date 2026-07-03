#!/usr/bin/env bash
set -euo pipefail

source /data/cqy_workspace/flexible_lerobot/my_devs/openpi_train/scripts/env.sh
python -m openpi_so101.compute_norm_stats "$@"
