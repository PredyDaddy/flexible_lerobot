#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/cqy_workspace/flexible_lerobot/my_devs/openpi_train"
REPO_ROOT="/data/cqy_workspace/flexible_lerobot"
CONDA_ROOT="${CONDA_ROOT:-/home/cqy/miniconda3}"
ROBOT_CONDA_ENV="${ROBOT_CONDA_ENV:-lerobot_flex}"

HOST="${HOST:-localhost}"
PORT="${PORT:-8000}"
TASK="${TASK:-Put the eraser into the small box}"
EXECUTE_ACTIONS="${EXECUTE_ACTIONS:-false}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${ROBOT_CONDA_ENV}"

export PYTHONPATH="${ROOT}:${ROOT}/openpi-main/packages/openpi-client/src:${REPO_ROOT}/src:${PYTHONPATH:-}"

python -m openpi_so101.robot_remote_client \
  --host "${HOST}" \
  --port "${PORT}" \
  --task "${TASK}" \
  --execute-actions "${EXECUTE_ACTIONS}" \
  "$@"
