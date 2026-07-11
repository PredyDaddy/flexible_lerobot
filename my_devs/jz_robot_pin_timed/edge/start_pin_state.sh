#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BASE_SCRIPT="${REPO_ROOT}/my_devs/jz_robot_pin/edge/start_pin_state.sh"
STATE_HZ="${STATE_HZ:-30}"

if [[ ! -f "${BASE_SCRIPT}" ]]; then
  echo "[timed/edge/state] missing shared Pin edge script: ${BASE_SCRIPT}" >&2
  exit 1
fi

echo "[timed/edge/state] reusing the shared read-only ROS state UDP bridge"
echo "[timed/edge/state] STATE_HZ=${STATE_HZ} (timed default matches 30 FPS observation)"
exec env STATE_HZ="${STATE_HZ}" bash "${BASE_SCRIPT}" "$@"
