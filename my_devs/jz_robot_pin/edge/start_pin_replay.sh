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
COMMAND_PORT="${COMMAND_PORT:-39020}"
EXECUTION="${EXECUTION:-armed}"
PYTHON_CMD="${PYTHON_CMD:-conda run --no-capture-output -n lerobot python}"
CONFIG="${CONFIG:-${REPO_ROOT}/udp_test/test_scripts/arm_side/orin_phase3_executor_config_armed_hold.yaml}"
LEGACY_READY_TIMEOUT_S="${READY_TIMEOUT_S:-}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-15}"
BRIDGE_START_TIMEOUT_S="${BRIDGE_START_TIMEOUT_S:-${LEGACY_READY_TIMEOUT_S:-30}}"
STATE_WAIT_TIMEOUT_S="${STATE_WAIT_TIMEOUT_S:-${LEGACY_READY_TIMEOUT_S:-15}}"
STATE_READY_TIMEOUT_S="${STATE_READY_TIMEOUT_S:-${LEGACY_READY_TIMEOUT_S:-20}}"
EXECUTOR_READY_TIMEOUT_S="${EXECUTOR_READY_TIMEOUT_S:-${LEGACY_READY_TIMEOUT_S:-15}}"

if [[ ! "${STATE_HZ}" =~ ^[0-9]+([.][0-9]+)?$ ]] || ! awk -v hz="${STATE_HZ}" 'BEGIN { exit !(hz > 0) }'; then
  echo "[edge/start_pin_replay] invalid STATE_HZ=${STATE_HZ}" >&2
  exit 2
fi
if [[ ! "${JZ_EXPECTED_STATE_HZ}" =~ ^[0-9]+([.][0-9]+)?$ ]] || \
   ! awk -v actual="${STATE_HZ}" -v expected="${JZ_EXPECTED_STATE_HZ}" 'BEGIN { exit !(actual == expected) }'; then
  echo "[edge/start_pin_replay] STATE_HZ=${STATE_HZ} does not match JZ_EXPECTED_STATE_HZ=${JZ_EXPECTED_STATE_HZ}" >&2
  exit 2
fi
case "${JZ_STATE_HZ_PROFILE}" in
  legacy) ;;
  timed)
    if ! awk -v hz="${STATE_HZ}" 'BEGIN { exit !(hz == 30) }' && \
       [[ "${JZ_TIMED_NON_30_STATE_HZ_CONFIRM}" != "${STATE_HZ}" ]]; then
      echo "[edge/start_pin_replay] timed profile refuses unconfirmed STATE_HZ=${STATE_HZ}" >&2
      exit 2
    fi
    ;;
  *)
    echo "[edge/start_pin_replay] unsupported JZ_STATE_HZ_PROFILE=${JZ_STATE_HZ_PROFILE}" >&2
    exit 2
    ;;
esac

if [[ "${EXECUTION}" == "armed" && "${JZ_UDP_EXECUTOR_ARMED:-}" != "1" ]]; then
  echo "[edge/start_pin_replay] refusing armed replay: set JZ_UDP_EXECUTOR_ARMED=1" >&2
  exit 2
fi

cd "${REPO_ROOT}"
echo "[edge/start_pin_replay] ORIN_IP=${ORIN_IP} X86_IP=${X86_IP}"
echo "[edge/start_pin_replay] STATE_PORT=${STATE_PORT} COMMAND_PORT=${COMMAND_PORT} EXECUTION=${EXECUTION}"
echo "[edge/start_pin_replay] profile=${JZ_STATE_HZ_PROFILE} requested_hz=${STATE_HZ} expected_hz=${JZ_EXPECTED_STATE_HZ}"
echo "[edge/start_pin_replay] bridge timeouts: start=${BRIDGE_START_TIMEOUT_S}s state_wait=${STATE_WAIT_TIMEOUT_S}s"
echo "[edge/start_pin_replay] readiness timeouts: state=${STATE_READY_TIMEOUT_S}s executor=${EXECUTOR_READY_TIMEOUT_S}s"
ORIN_IP="${ORIN_IP}" \
X86_IP="${X86_IP}" \
STATE_PORT="${STATE_PORT}" \
STATE_HZ="${STATE_HZ}" \
JZ_STATE_HZ_PROFILE="${JZ_STATE_HZ_PROFILE}" \
JZ_EXPECTED_STATE_HZ="${JZ_EXPECTED_STATE_HZ}" \
JZ_TIMED_NON_30_STATE_HZ_CONFIRM="${JZ_TIMED_NON_30_STATE_HZ_CONFIRM}" \
COMMAND_PORT="${COMMAND_PORT}" \
EXECUTION="${EXECUTION}" \
PYTHON_CMD="${PYTHON_CMD}" \
CONFIG="${CONFIG}" \
READY_TIMEOUT_S="${READY_TIMEOUT_S}" \
BRIDGE_START_TIMEOUT_S="${BRIDGE_START_TIMEOUT_S}" \
STATE_WAIT_TIMEOUT_S="${STATE_WAIT_TIMEOUT_S}" \
STATE_READY_TIMEOUT_S="${STATE_READY_TIMEOUT_S}" \
EXECUTOR_READY_TIMEOUT_S="${EXECUTOR_READY_TIMEOUT_S}" \
bash udp_test/all/start_replay.sh
