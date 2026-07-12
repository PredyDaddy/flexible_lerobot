#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BASE_SCRIPT="${REPO_ROOT}/my_devs/jz_robot_pin/edge/stop_pin_replay.sh"
CAMERA_STOP_SCRIPT="${REPO_ROOT}/my_devs/jz_robot_pin_timed/edge/stop_direct_realsense_zmq.sh"

if [[ ! -f "${BASE_SCRIPT}" ]]; then
  echo "[timed/edge/stop] missing shared Pin edge script: ${BASE_SCRIPT}" >&2
  exit 1
fi
if [[ ! -f "${CAMERA_STOP_SCRIPT}" ]]; then
  echo "[timed/edge/stop] missing direct camera stop script: ${CAMERA_STOP_SCRIPT}" >&2
  exit 1
fi

status=0
echo "[timed/edge/stop] stopping the shared Pin/timed state bridge and executor"
bash "${BASE_SCRIPT}" "$@" || status=$?
echo "[timed/edge/stop] stopping direct RealSense ZMQ cameras"
bash "${CAMERA_STOP_SCRIPT}" || status=$?
exit "${status}"
