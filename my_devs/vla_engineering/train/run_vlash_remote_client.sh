#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/data/cqy_workspace/flexible_lerobot"
TRAIN_ROOT="${REPO_ROOT}/my_devs/vla_engineering/train"
CONDA_ROOT="${CONDA_ROOT:-/home/cqy/miniconda3}"
ROBOT_ENV="${ROBOT_ENV:-lerobot_flex}"

HOST="${HOST:-localhost}"
PORT="${PORT:-8005}"
TASK="${TASK:-Put the eraser into the small box}"
EXECUTE_ACTIONS="${EXECUTE_ACTIONS:-false}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${ROBOT_ENV}"

cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

python "${TRAIN_ROOT}/run_vlash_remote_client.py" \
  --host "${HOST}" \
  --port "${PORT}" \
  --task "${TASK}" \
  --execute-actions "${EXECUTE_ACTIONS}" \
  "$@"
