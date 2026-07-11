#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BASE_SCRIPT="${REPO_ROOT}/my_devs/jz_robot_pin/edge/stop_pin_replay.sh"

if [[ ! -f "${BASE_SCRIPT}" ]]; then
  echo "[timed/edge/stop] missing shared Pin edge script: ${BASE_SCRIPT}" >&2
  exit 1
fi

echo "[timed/edge/stop] stopping the shared Pin/timed state bridge and executor"
exec bash "${BASE_SCRIPT}" "$@"

