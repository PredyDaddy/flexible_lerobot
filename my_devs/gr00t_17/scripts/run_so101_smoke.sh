#!/usr/bin/env bash
set -euo pipefail

LEROBOT_FLEX_PYTHON="${CONDA_PREFIX:-}/bin/python"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
require_lerobot_flex_host
require_local_env
enable_offline_mode

if [[ ! -x "${LEROBOT_FLEX_PYTHON}" ]]; then
  echo "[ERROR] lerobot_flex Python is missing: ${LEROBOT_FLEX_PYTHON}" >&2
  exit 1
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${GR00T17_ROOT}/outputs/inference/smoke/${RUN_ID}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${GR00T17_ROOT}/outputs/formal/so101_n17_b2_e10_20260711/train/checkpoint-63600}"
SERVER_HOST="127.0.0.1"
SERVER_PORT="${SERVER_PORT:-5555}"
TASK="${TASK:-Put the eraser into the small box}"
ROBOT_PORT="${ROBOT_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00}"
TOP_CAM="${TOP_CAM:-/dev/video4}"
WRIST_CAM="${WRIST_CAM:-/dev/video6}"
ENABLE_ACTUATION="${ENABLE_ACTUATION:-0}"
ACTUATION_CONFIRM="${ACTUATION_CONFIRM:-}"

if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "[ERROR] Invalid RUN_ID: ${RUN_ID}" >&2
  exit 1
fi
if [[ "${ENABLE_ACTUATION}" != "0" && "${ENABLE_ACTUATION}" != "1" ]]; then
  echo "[ERROR] ENABLE_ACTUATION must be 0 or 1." >&2
  exit 1
fi
if [[ "${ENABLE_ACTUATION}" == "1" && "${ACTUATION_CONFIRM}" != "SO101_GR00T_N17" ]]; then
  echo "[ERROR] Actuated smoke requires ACTUATION_CONFIRM=SO101_GR00T_N17." >&2
  exit 1
fi

require_path_within_root "${RUN_DIR}"
if [[ -e "${RUN_DIR}" ]]; then
  echo "[ERROR] Refusing to reuse smoke run directory: ${RUN_DIR}" >&2
  exit 1
fi
mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/reports"

CLIENT=(
  "${LEROBOT_FLEX_PYTHON}"
  "${GR00T17_ROOT}/scripts/so101_robot_client.py"
  --server-host "${SERVER_HOST}"
  --server-port "${SERVER_PORT}"
  --checkpoint-path "${CHECKPOINT_PATH}"
  --robot-port "${ROBOT_PORT}"
  --top-cam "${TOP_CAM}"
  --wrist-cam "${WRIST_CAM}"
  --task "${TASK}"
)

echo "[SMOKE] Read-only motor bus preflight; torque is disabled on exit"
"${LEROBOT_FLEX_PYTHON}" "${GR00T17_ROOT}/scripts/check_so101_bus.py" \
  --robot-port "${ROBOT_PORT}" \
  --report "${RUN_DIR}/reports/bus_preflight.json"

SERVER_PID=""
cleanup() {
  stop_gpu_monitor || true
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

start_gpu_monitor "${RUN_DIR}/logs/gpu_usage.log" 2
CHECKPOINT_PATH="${CHECKPOINT_PATH}" SERVER_HOST="${SERVER_HOST}" SERVER_PORT="${SERVER_PORT}" \
  bash "${GR00T17_ROOT}/scripts/run_so101_policy_server.sh" \
  >"${RUN_DIR}/logs/policy_server.log" 2>&1 &
SERVER_PID=$!

echo "[SMOKE] Waiting for policy server PID ${SERVER_PID}"
for _ in $(seq 1 120); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[ERROR] Policy server exited during startup." >&2
    tail -n 80 "${RUN_DIR}/logs/policy_server.log" >&2
    exit 1
  fi
  if "${CLIENT[@]}" --mode ping --request-timeout-s 1 >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
"${CLIENT[@]}" --mode ping --request-timeout-s 2

echo "[SMOKE] Live cameras + robot state + two policy predictions; no motor commands"
"${CLIENT[@]}" \
  --mode predict \
  --prediction-count 2 \
  --max-command-delta 1 \
  --max-relative-target 1 \
  --report "${RUN_DIR}/reports/predict.json" \
  2>&1 | tee "${RUN_DIR}/logs/predict.log"

if [[ "${ENABLE_ACTUATION}" == "1" ]]; then
  echo "[SMOKE] One-second guarded actuation, 5 Hz, horizon=1, max delta=0.25"
  "${CLIENT[@]}" \
    --mode actuate \
    --run-time-s 1 \
    --execution-horizon 1 \
    --control-hz 5 \
    --max-command-delta 0.25 \
    --max-relative-target 0.25 \
    --enable-actuation \
    --confirm-actuation "${ACTUATION_CONFIRM}" \
    --report "${RUN_DIR}/reports/actuate.json" \
    2>&1 | tee "${RUN_DIR}/logs/actuate.log"
else
  echo "[SMOKE] Actuation intentionally skipped. Set ENABLE_ACTUATION=1 with confirmation after clearing the workspace."
fi

"${LEROBOT_FLEX_PYTHON}" "${GR00T17_ROOT}/scripts/validate_robot_smoke.py" \
  --root "${GR00T17_ROOT}" \
  --run-dir "${RUN_DIR}" \
  --expect-actuation "${ENABLE_ACTUATION}" \
  --report "${RUN_DIR}/reports/summary.json"

echo "[OK] SO101 smoke completed: ${RUN_DIR}"
