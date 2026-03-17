#!/usr/bin/env bash

set -Eeuo pipefail

CONDA_ENV_NAME="${CONDA_ENV_NAME:-lerobot_flex}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR_DEFAULT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${ROOT_DIR_DEFAULT}" ]]; then
  ROOT_DIR_DEFAULT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
fi
ROOT_DIR="${ROOT_DIR:-${ROOT_DIR_DEFAULT}}"
cd "${ROOT_DIR}"

LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs}"
JOB_NAME="${JOB_NAME:-groot_agilex_remote_policy_server}"

HOST="${REMOTE_GROOT_SERVER_HOST:-0.0.0.0}"
PORT="${REMOTE_GROOT_SERVER_PORT:-5560}"
POLICY_PATH="${POLICY_PATH:-}"
DATASET_TASK="${DATASET_TASK:-Execute the trained AgileX GR00T task}"
ROBOT_TYPE="${ROBOT_TYPE:-agilex}"
INFER_BACKEND="${INFER_BACKEND:-pytorch}"
TRT_ENGINE_PATH="${TRT_ENGINE_PATH:-}"
TRT_VIT_DTYPE="${TRT_VIT_DTYPE:-fp16}"
TRT_LLM_DTYPE="${TRT_LLM_DTYPE:-fp16}"
TRT_DIT_DTYPE="${TRT_DIT_DTYPE:-fp16}"
TRT_ACTION_HEAD_ONLY="${TRT_ACTION_HEAD_ONLY:-false}"
POLICY_DEVICE_OVERRIDE="${POLICY_DEVICE_OVERRIDE:-${POLICY_DEVICE:-}}"
CONTROL_ARM="${CONTROL_ARM:-auto}"
REMOTE_GROOT_SOCKET_TIMEOUT_S="${REMOTE_GROOT_SOCKET_TIMEOUT_S:-30}"
REMOTE_GROOT_MAX_MESSAGE_BYTES="${REMOTE_GROOT_MAX_MESSAGE_BYTES:-67108864}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${LOG_DIR}"

if [[ -n "${POLICY_PATH}" && ! -d "${POLICY_PATH}" ]]; then
  echo "ERROR: policy path does not exist: ${POLICY_PATH}" >&2
  exit 2
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found in PATH." >&2
  exit 2
fi

RUNNER=()
if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV_NAME}" ]]; then
  RUNNER=(conda run --no-capture-output -n "${CONDA_ENV_NAME}")
fi

timestamp="$(date +'%Y%m%d_%H%M%S')"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${JOB_NAME}_${timestamp}.log}"

CMD=(
  "${RUNNER[@]}"
  python
  my_devs/train/groot/agilex/run_groot_remote_policy_server.py
  "--host=${HOST}"
  "--port=${PORT}"
  "--task=${DATASET_TASK}"
  "--robot-type=${ROBOT_TYPE}"
  "--backend=${INFER_BACKEND}"
  "--vit-dtype=${TRT_VIT_DTYPE}"
  "--llm-dtype=${TRT_LLM_DTYPE}"
  "--dit-dtype=${TRT_DIT_DTYPE}"
  "--trt-action-head-only=${TRT_ACTION_HEAD_ONLY}"
  "--control-arm=${CONTROL_ARM}"
  "--socket-timeout-s=${REMOTE_GROOT_SOCKET_TIMEOUT_S}"
  "--max-message-bytes=${REMOTE_GROOT_MAX_MESSAGE_BYTES}"
  "--log-level=${LOG_LEVEL}"
)

if [[ -n "${POLICY_PATH}" ]]; then
  CMD+=("--policy-path=${POLICY_PATH}")
fi

if [[ -n "${TRT_ENGINE_PATH}" ]]; then
  CMD+=("--trt-engine-path=${TRT_ENGINE_PATH}")
fi

if [[ -n "${POLICY_DEVICE_OVERRIDE}" ]]; then
  CMD+=("--policy-device=${POLICY_DEVICE_OVERRIDE}")
fi

if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" ]]; then
  CMD+=("--dry-run=true")
  echo "[DRY_RUN] Would execute:"
  printf '%q ' "${CMD[@]}" "$@"
  echo
  echo "[DRY_RUN] Log file: ${LOG_FILE}"
  exit 0
fi

echo "[start] script=run_groot_remote_policy_server.sh conda_env=${CONDA_ENV_NAME}"
echo "[start] host=${HOST} port=${PORT}"
echo "[start] policy_path=${POLICY_PATH:-<client handshake or empty>}"
echo "[start] task=${DATASET_TASK}"
echo "[start] backend=${INFER_BACKEND}"
echo "[start] control_arm=${CONTROL_ARM}"
echo "[start] policy_device_override=${POLICY_DEVICE_OVERRIDE:-<checkpoint default>}"
echo "[start] log_file=${LOG_FILE}"

"${CMD[@]}" "$@" 2>&1 | tee "${LOG_FILE}"
