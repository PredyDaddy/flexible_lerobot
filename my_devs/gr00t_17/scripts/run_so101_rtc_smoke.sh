#!/usr/bin/env bash
set -euo pipefail

LEROBOT_FLEX_PYTHON="${CONDA_PREFIX:-}/bin/python"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
require_lerobot_flex_host
require_local_env
enable_offline_mode

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${GR00T17_ROOT}/outputs/inference/rtc_smoke/${RUN_ID}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${GR00T17_ROOT}/outputs/formal/so101_n17_b2_e10_20260711/train/checkpoint-63600}"
SERVER_HOST="127.0.0.1"
SERVER_PORT="${SERVER_PORT:-5556}"
mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/reports"

SERVER_PID=""
cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

CHECKPOINT_PATH="${CHECKPOINT_PATH}" SERVER_HOST="${SERVER_HOST}" SERVER_PORT="${SERVER_PORT}" \
  bash "${GR00T17_ROOT}/scripts/run_so101_rtc_policy_server.sh" \
  >"${RUN_DIR}/logs/policy_server.log" 2>&1 &
SERVER_PID=$!

PING=("${LEROBOT_FLEX_PYTHON}" "${GR00T17_ROOT}/scripts/so101_rtc_robot_client.py"
  --mode ping --server-host "${SERVER_HOST}" --server-port "${SERVER_PORT}"
  --checkpoint-path "${CHECKPOINT_PATH}" --request-timeout-s 1)
for _ in $(seq 1 120); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    tail -n 100 "${RUN_DIR}/logs/policy_server.log" >&2
    exit 1
  fi
  if "${PING[@]}" >/dev/null 2>&1; then break; fi
  sleep 1
done
"${PING[@]}"

"${LEROBOT_FLEX_PYTHON}" "${GR00T17_ROOT}/scripts/so101_rtc_robot_client.py" \
  --mode dry-run \
  --server-host "${SERVER_HOST}" \
  --server-port "${SERVER_PORT}" \
  --expected-backend tensorrt \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --run-time-s 3 \
  --execution-horizon 8 \
  --control-hz 30 \
  --request-timeout-s 2 \
  --max-inference-s 2 \
  --report "${RUN_DIR}/reports/summary.json" \
  "$@" 2>&1 | tee "${RUN_DIR}/logs/robot_client.log"

echo "[OK] RTC no-actuation smoke completed: ${RUN_DIR}"
