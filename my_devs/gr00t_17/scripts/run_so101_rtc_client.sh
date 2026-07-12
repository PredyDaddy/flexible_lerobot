#!/usr/bin/env bash
set -euo pipefail

LEROBOT_FLEX_PYTHON="${CONDA_PREFIX:-}/bin/python"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
require_lerobot_flex_host
require_local_env
enable_offline_mode

RUN_ID="${RUN_ID:-so101_n17_rtc_client_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${GR00T17_ROOT}/outputs/inference/rtc_client/${RUN_ID}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${GR00T17_ROOT}/outputs/formal/so101_n17_b2_e10_20260711/train/checkpoint-63600}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-5556}"

if [[ "${SERVER_HOST}" != "127.0.0.1" && "${SERVER_HOST}" != "localhost" ]]; then
  echo "[ERROR] Local RTC client only connects to 127.0.0.1/localhost." >&2
  exit 1
fi
if [[ ! "${SERVER_PORT}" =~ ^[0-9]+$ ]] || ((SERVER_PORT < 1 || SERVER_PORT > 65535)); then
  echo "[ERROR] Invalid SERVER_PORT: ${SERVER_PORT}" >&2
  exit 1
fi
if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "[ERROR] Invalid RUN_ID: ${RUN_ID}" >&2
  exit 1
fi
require_path_within_root "${CHECKPOINT_PATH}"
require_path_within_root "${RUN_DIR}"
if [[ ! -f "${CHECKPOINT_PATH}/model.safetensors.index.json" ]]; then
  echo "[ERROR] RTC checkpoint is incomplete: ${CHECKPOINT_PATH}" >&2
  exit 1
fi
if [[ -e "${RUN_DIR}" ]]; then
  echo "[ERROR] Refusing to reuse RTC client run directory: ${RUN_DIR}" >&2
  exit 1
fi
mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/reports"

PING=(
  "${LEROBOT_FLEX_PYTHON}"
  "${GR00T17_ROOT}/scripts/so101_rtc_robot_client.py"
  --mode ping
  --server-host "${SERVER_HOST}"
  --server-port "${SERVER_PORT}"
  --expected-backend tensorrt
  --checkpoint-path "${CHECKPOINT_PATH}"
  --request-timeout-s 1
)
if ! "${PING[@]}"; then
  echo "[ERROR] RTC policy server is not reachable at ${SERVER_HOST}:${SERVER_PORT}." >&2
  echo "[ERROR] Start run_so101_rtc_policy_server.sh in the server terminal first." >&2
  exit 1
fi

"${LEROBOT_FLEX_PYTHON}" "${GR00T17_ROOT}/scripts/so101_rtc_robot_client.py" \
  --mode actuate \
  --server-host "${SERVER_HOST}" \
  --server-port "${SERVER_PORT}" \
  --expected-backend tensorrt \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --run-time-s 120 \
  --execution-horizon 8 \
  --control-hz 30 \
  --camera-warmup-s 2 \
  --bounds-mode physical \
  --max-command-delta 200 \
  --max-relative-target 200 \
  --request-timeout-s 2 \
  --max-inference-s 2 \
  --enable-actuation \
  --confirm-actuation SO101_GR00T_N17 \
  --report "${RUN_DIR}/reports/inference.json" \
  "$@" 2>&1 | tee "${RUN_DIR}/logs/robot_client.log"

echo "[OK] SO101 RTC client inference completed: ${RUN_DIR}"
