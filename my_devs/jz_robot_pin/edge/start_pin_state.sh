#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"

ORIN_IP="${ORIN_IP:-192.168.1.81}"
X86_IP="${X86_IP:-192.168.1.106}"
STATE_PORT="${STATE_PORT:-39010}"
STATE_HZ="${STATE_HZ:-20}"
JZ_STATE_HZ_PROFILE="${JZ_STATE_HZ_PROFILE:-legacy}"
JZ_EXPECTED_STATE_HZ="${JZ_EXPECTED_STATE_HZ:-${STATE_HZ}}"
JZ_TIMED_NON_30_STATE_HZ_CONFIRM="${JZ_TIMED_NON_30_STATE_HZ_CONFIRM:-}"
PYTHON_CMD="${PYTHON_CMD:-conda run --no-capture-output -n lerobot python}"
AUTO_TAIL="${AUTO_TAIL:-0}"

if [[ ! "${STATE_HZ}" =~ ^[0-9]+([.][0-9]+)?$ ]] || ! awk -v hz="${STATE_HZ}" 'BEGIN { exit !(hz > 0) }'; then
  echo "[edge/start_pin_state] invalid STATE_HZ=${STATE_HZ}" >&2
  exit 2
fi
if [[ ! "${JZ_EXPECTED_STATE_HZ}" =~ ^[0-9]+([.][0-9]+)?$ ]] || \
   ! awk -v actual="${STATE_HZ}" -v expected="${JZ_EXPECTED_STATE_HZ}" 'BEGIN { exit !(actual == expected) }'; then
  echo "[edge/start_pin_state] STATE_HZ=${STATE_HZ} does not match JZ_EXPECTED_STATE_HZ=${JZ_EXPECTED_STATE_HZ}" >&2
  exit 2
fi
if [[ "${JZ_STATE_HZ_PROFILE}" == "timed" ]] && \
   ! awk -v hz="${STATE_HZ}" 'BEGIN { exit !(hz == 30) }' && \
   [[ "${JZ_TIMED_NON_30_STATE_HZ_CONFIRM}" != "${STATE_HZ}" ]]; then
  echo "[edge/start_pin_state] timed profile refuses unconfirmed STATE_HZ=${STATE_HZ}" >&2
  exit 2
fi
if [[ "${JZ_STATE_HZ_PROFILE}" != "legacy" && "${JZ_STATE_HZ_PROFILE}" != "timed" ]]; then
  echo "[edge/start_pin_state] unsupported JZ_STATE_HZ_PROFILE=${JZ_STATE_HZ_PROFILE}" >&2
  exit 2
fi

cd "${REPO_ROOT}"
echo "[edge/start_pin_state] ORIN_IP=${ORIN_IP} X86_IP=${X86_IP} STATE_PORT=${STATE_PORT} profile=${JZ_STATE_HZ_PROFILE} requested_hz=${STATE_HZ} expected_hz=${JZ_EXPECTED_STATE_HZ}"
ORIN_IP="${ORIN_IP}" \
X86_IP="${X86_IP}" \
STATE_PORT="${STATE_PORT}" \
STATE_HZ="${STATE_HZ}" \
JZ_STATE_HZ_PROFILE="${JZ_STATE_HZ_PROFILE}" \
JZ_EXPECTED_STATE_HZ="${JZ_EXPECTED_STATE_HZ}" \
JZ_TIMED_NON_30_STATE_HZ_CONFIRM="${JZ_TIMED_NON_30_STATE_HZ_CONFIRM}" \
PYTHON_CMD="${PYTHON_CMD}" \
AUTO_TAIL="${AUTO_TAIL}" \
bash udp_test/server_bash/orin_arm/start.sh
