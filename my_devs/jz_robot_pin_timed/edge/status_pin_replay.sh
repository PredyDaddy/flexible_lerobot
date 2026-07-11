#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BASE_SCRIPT="${REPO_ROOT}/my_devs/jz_robot_pin/edge/status_pin_replay.sh"

if [[ ! -f "${BASE_SCRIPT}" ]]; then
  echo "[timed/edge/status] missing shared Pin edge script: ${BASE_SCRIPT}" >&2
  exit 1
fi

echo "[timed/edge/status] shared Pin/timed transport status"
exec bash "${BASE_SCRIPT}" "$@"

