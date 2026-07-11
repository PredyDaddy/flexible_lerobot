#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BASE_SCRIPT="${REPO_ROOT}/my_devs/jz_robot_pin/edge/start_pin_state.sh"
STATE_HZ="${STATE_HZ:-30}"
JZ_TIMED_NON_30_STATE_HZ_CONFIRM="${JZ_TIMED_NON_30_STATE_HZ_CONFIRM:-}"

if [[ ! "${STATE_HZ}" =~ ^[0-9]+([.][0-9]+)?$ ]] || ! awk -v hz="${STATE_HZ}" 'BEGIN { exit !(hz > 0) }'; then
  echo "[timed/edge/state] invalid STATE_HZ=${STATE_HZ}; expected a positive number" >&2
  exit 2
fi

NON_30_OVERRIDE_CONFIRMED=false
if ! awk -v hz="${STATE_HZ}" 'BEGIN { exit !(hz == 30) }'; then
  if [[ "${JZ_TIMED_NON_30_STATE_HZ_CONFIRM}" != "${STATE_HZ}" ]]; then
    echo "[timed/edge/state] refusing non-30 timed state rate STATE_HZ=${STATE_HZ}" >&2
    echo "[timed/edge/state] set JZ_TIMED_NON_30_STATE_HZ_CONFIRM=${STATE_HZ} to confirm this exact override" >&2
    exit 2
  fi
  NON_30_OVERRIDE_CONFIRMED=true
fi

if [[ ! -f "${BASE_SCRIPT}" ]]; then
  echo "[timed/edge/state] missing shared Pin edge script: ${BASE_SCRIPT}" >&2
  exit 1
fi

echo "[timed/edge/state] reusing the shared read-only ROS state UDP bridge"
echo "[timed/edge/state] profile=timed requested_hz=${STATE_HZ} expected_hz=${STATE_HZ} non_30_override_confirmed=${NON_30_OVERRIDE_CONFIRMED}"
exec env \
  STATE_HZ="${STATE_HZ}" \
  JZ_STATE_HZ_PROFILE=timed \
  JZ_EXPECTED_STATE_HZ="${STATE_HZ}" \
  JZ_TIMED_NON_30_STATE_HZ_CONFIRM="${JZ_TIMED_NON_30_STATE_HZ_CONFIRM}" \
  bash "${BASE_SCRIPT}" "$@"
