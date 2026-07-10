#!/usr/bin/env bash
set -euo pipefail

BACKEND_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENPI_TRAIN_ROOT="$(cd "${BACKEND_ROOT}/.." && pwd)"
REPO_ROOT="$(cd "${OPENPI_TRAIN_ROOT}/../.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-/home/cqy/miniconda3}"
ROBOT_CONDA_ENV="${ROBOT_CONDA_ENV:-lerobot_flex}"

CONFIG="${CONFIG:-${BACKEND_ROOT}/client/configs/so101_http.yaml}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${ROBOT_CONDA_ENV}"

export PYTHONPATH="${BACKEND_ROOT}/client:${OPENPI_TRAIN_ROOT}:${REPO_ROOT}/src:${PYTHONPATH:-}"

echo "[run_so101_client] CONFIG=${CONFIG}"
echo "[run_so101_client] CONDA_ENV=${ROBOT_CONDA_ENV}"

exec python "${BACKEND_ROOT}/client/local_client.py" \
  --config "${CONFIG}" \
  "$@"

