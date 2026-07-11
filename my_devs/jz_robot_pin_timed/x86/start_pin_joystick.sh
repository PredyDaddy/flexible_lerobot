#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"

BASE_SCRIPT="${REPO_ROOT}/my_devs/jz_robot_pin/x86/start_pin_joystick.sh"
if [[ ! -f "${BASE_SCRIPT}" ]]; then
  echo "[timed/x86/joystick] missing shared Pin joystick script: ${BASE_SCRIPT}" >&2
  exit 1
fi

echo "[timed/x86/joystick] reusing the shared VR, IK, target-action, and Meshcat publisher"
echo "[timed/x86/joystick] target action remains udp://${TARGET_ACTION_IP:-127.0.0.1}:${TARGET_ACTION_PORT:-39030}"

# Keep this timed wrapper as the parent process so the timed stop script can
# identify the publisher without matching another robot's visualizer globally.
bash "${BASE_SCRIPT}" "$@"

