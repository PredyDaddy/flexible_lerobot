#!/usr/bin/env bash
set -euo pipefail

OPENPI_TRAIN_ROOT="${OPENPI_TRAIN_ROOT:-/data/cqy_workspace/flexible_lerobot/my_devs/openpi_train}"
OPENPI_ROOT="${OPENPI_ROOT:-${OPENPI_TRAIN_ROOT}/openpi-main}"
CONDA_ROOT="${CONDA_ROOT:-/home/cqy/miniconda3}"
OPENPI_CONDA_ENV="${OPENPI_CONDA_ENV:-openpi_train}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${OPENPI_CONDA_ENV}"
python - <<'PY'
import os
import sys

if sys.version_info < (3, 11):
    env = os.environ.get("CONDA_DEFAULT_ENV", "<unknown>")
    raise SystemExit(
        f"OpenPI scripts require Python >= 3.11, but {env} is running "
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}."
    )
PY

export PYTHONPATH="${OPENPI_TRAIN_ROOT}:${OPENPI_ROOT}:${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src:${PYTHONPATH:-}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${OPENPI_TRAIN_ROOT}/assets/openpi_cache}"
export OPENPI_SO101_DATASET_ROOT="${OPENPI_SO101_DATASET_ROOT:-/data/cqy_workspace/flexible_lerobot/datasets/desk_cleanup_v1/eraser_cup_multi_task}"
export OPENPI_SO101_REPO_ID="${OPENPI_SO101_REPO_ID:-desk_cleanup_v1/eraser_cup_multi_task}"
export OPENPI_SO101_V21_ROOT="${OPENPI_SO101_V21_ROOT:-${OPENPI_TRAIN_ROOT}/data/lerobot_v21_pilot/desk_cleanup_v1/eraser_cup_multi_task}"
export OPENPI_SO101_ASSETS_BASE_DIR="${OPENPI_SO101_ASSETS_BASE_DIR:-${OPENPI_TRAIN_ROOT}/assets/openpi_assets}"
export OPENPI_SO101_CHECKPOINT_BASE_DIR="${OPENPI_SO101_CHECKPOINT_BASE_DIR:-${OPENPI_TRAIN_ROOT}/outputs/checkpoints}"
export OPENPI_PI05_BASE_PARAMS="${OPENPI_PI05_BASE_PARAMS:-${OPENPI_DATA_HOME}/openpi-assets/checkpoints/pi05_base/params}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
