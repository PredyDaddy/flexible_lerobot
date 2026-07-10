#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"

ORIN_IP="${ORIN_IP:-192.168.1.81}"
X86_IP="${X86_IP:-192.168.1.106}"
STATE_PORT="${STATE_PORT:-39010}"
STATE_HZ="${STATE_HZ:-20}"
PYTHON_CMD="${PYTHON_CMD:-conda run --no-capture-output -n lerobot python}"
AUTO_TAIL="${AUTO_TAIL:-0}"

cd "${REPO_ROOT}"
echo "[edge/start_pin_state] ORIN_IP=${ORIN_IP} X86_IP=${X86_IP} STATE_PORT=${STATE_PORT} STATE_HZ=${STATE_HZ}"
ORIN_IP="${ORIN_IP}" \
X86_IP="${X86_IP}" \
STATE_PORT="${STATE_PORT}" \
STATE_HZ="${STATE_HZ}" \
PYTHON_CMD="${PYTHON_CMD}" \
AUTO_TAIL="${AUTO_TAIL}" \
bash udp_test/server_bash/orin_arm/start.sh
