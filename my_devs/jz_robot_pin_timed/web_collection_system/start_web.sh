#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../" && pwd)"
CONDA_ENV="${CONDA_ENV:-lerobot_flex}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8010}"

cd "${REPO_ROOT}"

if [[ "${CONDA_DEFAULT_ENV:-}" == "${CONDA_ENV}" && -x "${CONDA_PREFIX:-}/bin/python" ]]; then
  PYTHON_CMD=("${CONDA_PREFIX}/bin/python")
else
  PYTHON_CMD=(conda run --no-capture-output -n "${CONDA_ENV}" python)
fi

echo "[jz_timed/web] url=http://${HOST}:${PORT}"
echo "[jz_timed/web] armed_actions=${JZ_WEB_ARMED_ACTIONS:-0} mock=${JZ_WEB_MOCK_COMMANDS:-0}"

exec "${PYTHON_CMD[@]}" \
  -m my_devs.jz_robot_pin_timed.web_collection_system.server \
  --host "${HOST}" \
  --port "${PORT}"
