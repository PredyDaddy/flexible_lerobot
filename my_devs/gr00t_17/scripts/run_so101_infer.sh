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
RUN_DIR="${GR00T17_ROOT}/outputs/inference/formal/${RUN_ID}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${GR00T17_ROOT}/outputs/formal/so101_n17_b2_e10_20260711/train/checkpoint-63600}"
SERVER_HOST="127.0.0.1"
SERVER_PORT="${SERVER_PORT:-5555}"
ACTUATED_SMOKE_REPORT="${ACTUATED_SMOKE_REPORT:-}"

if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "[ERROR] Invalid RUN_ID: ${RUN_ID}" >&2
  exit 1
fi
ENABLE_ACTUATION_ARG=0
ACTUATION_CONFIRM_ARG=""
CLIENT_ARGS=("$@")
for ((ARG_INDEX = 0; ARG_INDEX < ${#CLIENT_ARGS[@]}; ARG_INDEX++)); do
  ARG="${CLIENT_ARGS[ARG_INDEX]}"
  case "${ARG}" in
    --enable-actuation)
      ENABLE_ACTUATION_ARG=1
      ;;
    --confirm-actuation)
      if ((ARG_INDEX + 1 >= ${#CLIENT_ARGS[@]})); then
        echo "[ERROR] --confirm-actuation requires a value." >&2
        exit 1
      fi
      ACTUATION_CONFIRM_ARG="${CLIENT_ARGS[ARG_INDEX + 1]}"
      ((ARG_INDEX += 1))
      ;;
    --confirm-actuation=*)
      ACTUATION_CONFIRM_ARG="${ARG#*=}"
      ;;
    --mode|--mode=*|--report|--report=*|--server-host|--server-host=*|--server-port|--server-port=*|--checkpoint-path|--checkpoint-path=*)
      echo "[ERROR] ${ARG} is managed by run_so101_infer.sh and cannot be overridden." >&2
      exit 1
      ;;
  esac
done
if [[ "${ENABLE_ACTUATION_ARG}" != "1" || "${ACTUATION_CONFIRM_ARG}" != "SO101_GR00T_N17" ]]; then
  echo "[ERROR] Formal inference requires --enable-actuation and --confirm-actuation SO101_GR00T_N17." >&2
  exit 1
fi

require_path_within_root "${RUN_DIR}"
if [[ -e "${RUN_DIR}" ]]; then
  echo "[ERROR] Refusing to reuse inference run directory: ${RUN_DIR}" >&2
  exit 1
fi

ROBOT_PORT_ARG="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00"
TASK_ARG="Put the eraser into the small box"
for ((ARG_INDEX = 0; ARG_INDEX < ${#CLIENT_ARGS[@]}; ARG_INDEX++)); do
  ARG="${CLIENT_ARGS[ARG_INDEX]}"
  case "${ARG}" in
    --robot-port)
      if ((ARG_INDEX + 1 >= ${#CLIENT_ARGS[@]})); then
        echo "[ERROR] --robot-port requires a value." >&2
        exit 1
      fi
      ROBOT_PORT_ARG="${CLIENT_ARGS[ARG_INDEX + 1]}"
      ((ARG_INDEX += 1))
      ;;
    --robot-port=*)
      ROBOT_PORT_ARG="${ARG#*=}"
      ;;
    --task)
      if ((ARG_INDEX + 1 >= ${#CLIENT_ARGS[@]})); then
        echo "[ERROR] --task requires a value." >&2
        exit 1
      fi
      TASK_ARG="${CLIENT_ARGS[ARG_INDEX + 1]}"
      ((ARG_INDEX += 1))
      ;;
    --task=*)
      TASK_ARG="${ARG#*=}"
      ;;
  esac
done

if [[ -z "${ACTUATED_SMOKE_REPORT}" ]]; then
  echo "[ERROR] Set ACTUATED_SMOKE_REPORT to a passed guarded-actuation summary." >&2
  exit 1
fi
require_path_within_root "${ACTUATED_SMOKE_REPORT}"
if [[ ! -f "${ACTUATED_SMOKE_REPORT}" ]]; then
  echo "[ERROR] Guarded-actuation smoke report does not exist: ${ACTUATED_SMOKE_REPORT}" >&2
  echo "[ERROR] Run scripts/run_so101_smoke.sh with ENABLE_ACTUATION=1 first." >&2
  exit 1
fi
"${LEROBOT_FLEX_PYTHON}" "${GR00T17_ROOT}/scripts/validate_actuation_gate.py" \
  --root "${GR00T17_ROOT}" \
  --smoke-report "${ACTUATED_SMOKE_REPORT}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --task "${TASK_ARG}" \
  --report "${RUN_DIR}/reports/actuation_gate.json"

mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/reports"

echo "[INFER] Read-only motor bus preflight; torque is disabled on exit"
"${LEROBOT_FLEX_PYTHON}" "${GR00T17_ROOT}/scripts/check_so101_bus.py" \
  --robot-port "${ROBOT_PORT_ARG}" \
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

PING=(
  "${LEROBOT_FLEX_PYTHON}"
  "${GR00T17_ROOT}/scripts/so101_robot_client.py"
  --mode ping
  --server-host "${SERVER_HOST}"
  --server-port "${SERVER_PORT}"
  --checkpoint-path "${CHECKPOINT_PATH}"
  --request-timeout-s 1
)
for _ in $(seq 1 120); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[ERROR] Policy server exited during startup." >&2
    tail -n 80 "${RUN_DIR}/logs/policy_server.log" >&2
    exit 1
  fi
  if "${PING[@]}" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
"${PING[@]}"

"${LEROBOT_FLEX_PYTHON}" "${GR00T17_ROOT}/scripts/so101_robot_client.py" \
  --mode actuate \
  --server-host "${SERVER_HOST}" \
  --server-port "${SERVER_PORT}" \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --report "${RUN_DIR}/reports/inference.json" \
  "$@" 2>&1 | tee "${RUN_DIR}/logs/robot_client.log"

echo "[OK] SO101 formal inference completed: ${RUN_DIR}"
