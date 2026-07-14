#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMED_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${TIMED_ROOT}/lib/common.sh"

CONDA_ENV="${CONDA_ENV:-${LEROBOT_CONDA_ENV:-lerobot_flex}}"
ORIN_IP="${ORIN_IP:-192.168.1.81}"
DURATION_S="${DURATION_S:-10}"
MIN_FPS="${MIN_FPS:-25}"
REPORT_JSON="${REPORT_JSON:-${REPO_ROOT}/tests/outputs/jz_direct_zmq_camera_probe.json}"

timed_make_python_cmd "${CONDA_ENV}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

echo "[timed/zmq_probe] read-only; no robot state, teleop, action, or command sockets are opened"
exec "${PYTHON_CMD[@]}" "${SCRIPT_DIR}/probe_direct_zmq_cameras.py" \
  --host "${ORIN_IP}" \
  --duration-s "${DURATION_S}" \
  --min-fps "${MIN_FPS}" \
  --report-json "${REPORT_JSON}"
