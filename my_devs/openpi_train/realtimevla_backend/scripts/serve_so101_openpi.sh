#!/usr/bin/env bash
set -euo pipefail

BACKEND_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENPI_TRAIN_ROOT="$(cd "${BACKEND_ROOT}/.." && pwd)"

CONFIG="${CONFIG:-${BACKEND_ROOT}/server/configs/so101_openpi.yaml}"
PORT="${PORT:-18080}"
PYTHON="${PYTHON:-${BACKEND_ROOT}/.venv/bin/python}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing server Python: ${PYTHON}" >&2
  echo "Run: ${BACKEND_ROOT}/scripts/create_server_venv.sh" >&2
  exit 1
fi

export PYTHONPATH="${BACKEND_ROOT}/server:${OPENPI_TRAIN_ROOT}:${OPENPI_TRAIN_ROOT}/openpi-main:${OPENPI_TRAIN_ROOT}/openpi-main/src:${OPENPI_TRAIN_ROOT}/openpi-main/packages/openpi-client/src:${PYTHONPATH:-}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${OPENPI_TRAIN_ROOT}/assets/openpi_cache}"
export OPENPI_SO101_DATASET_ROOT="${OPENPI_SO101_DATASET_ROOT:-/data/cqy_workspace/flexible_lerobot/datasets/desk_cleanup_v1/eraser_cup_multi_task}"
export OPENPI_SO101_REPO_ID="${OPENPI_SO101_REPO_ID:-desk_cleanup_v1/eraser_cup_multi_task}"
export OPENPI_SO101_V21_ROOT="${OPENPI_SO101_V21_ROOT:-${OPENPI_TRAIN_ROOT}/easy_use/data/lerobot_v21/desk_cleanup_v1/eraser_cup_multi_task}"
export OPENPI_SO101_ASSETS_BASE_DIR="${OPENPI_SO101_ASSETS_BASE_DIR:-${OPENPI_TRAIN_ROOT}/assets/openpi_assets}"
export OPENPI_SO101_CHECKPOINT_BASE_DIR="${OPENPI_SO101_CHECKPOINT_BASE_DIR:-${OPENPI_TRAIN_ROOT}/outputs/checkpoints}"
export OPENPI_PI05_BASE_PARAMS="${OPENPI_PI05_BASE_PARAMS:-${OPENPI_DATA_HOME}/openpi-assets/checkpoints/pi05_base/params}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

echo "[serve_so101_openpi] CONFIG=${CONFIG}"
echo "[serve_so101_openpi] PORT=${PORT}"
echo "[serve_so101_openpi] PYTHON=${PYTHON}"

exec "${PYTHON}" "${BACKEND_ROOT}/server/infer_server.py" \
  --config "${CONFIG}" \
  --port "${PORT}" \
  "$@"
