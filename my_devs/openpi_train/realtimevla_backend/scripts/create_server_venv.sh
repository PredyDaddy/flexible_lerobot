#!/usr/bin/env bash
set -euo pipefail

BACKEND_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-/home/cqy/miniconda3}"
OPENPI_CONDA_ENV="${OPENPI_CONDA_ENV:-openpi_train}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${OPENPI_CONDA_ENV}"

python - <<'PY'
import sys

if sys.version_info < (3, 11):
    raise SystemExit("RealtimeVLA SO101 server requires Python >= 3.11.")
PY

python -m venv --system-site-packages "${BACKEND_ROOT}/.venv"
"${BACKEND_ROOT}/.venv/bin/python" -m pip install -r "${BACKEND_ROOT}/requirements/server.txt"

echo "[create_server_venv] ready: ${BACKEND_ROOT}/.venv"

