#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"

ORIN_IP="${ORIN_IP:-192.168.1.81}"
X86_IP="${X86_IP:-192.168.1.106}"
STATE_PORT="${STATE_PORT:-39010}"
STATE_HZ="${STATE_HZ:-20}"
COMMAND_PORT="${COMMAND_PORT:-39020}"
EXECUTION="${EXECUTION:-armed}"
PYTHON_CMD="${PYTHON_CMD:-conda run --no-capture-output -n lerobot python}"
CONFIG="${CONFIG:-${REPO_ROOT}/udp_test/test_scripts/arm_side/orin_phase3_executor_config_armed_hold.yaml}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-15}"

if [[ "${EXECUTION}" == "armed" && "${JZ_UDP_EXECUTOR_ARMED:-}" != "1" ]]; then
  echo "[edge/start_pin_replay] refusing armed replay: set JZ_UDP_EXECUTOR_ARMED=1" >&2
  exit 2
fi

cd "${REPO_ROOT}"
echo "[edge/start_pin_replay] ORIN_IP=${ORIN_IP} X86_IP=${X86_IP}"
echo "[edge/start_pin_replay] STATE_PORT=${STATE_PORT} COMMAND_PORT=${COMMAND_PORT} EXECUTION=${EXECUTION}"
ORIN_IP="${ORIN_IP}" \
X86_IP="${X86_IP}" \
STATE_PORT="${STATE_PORT}" \
STATE_HZ="${STATE_HZ}" \
COMMAND_PORT="${COMMAND_PORT}" \
EXECUTION="${EXECUTION}" \
PYTHON_CMD="${PYTHON_CMD}" \
CONFIG="${CONFIG}" \
READY_TIMEOUT_S="${READY_TIMEOUT_S}" \
bash udp_test/all/start_replay.sh
