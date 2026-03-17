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
JOB_NAME="${JOB_NAME:-groot_agilex_remote_robot_client}"

ROBOT_ID="${ROBOT_ID:-my_agilex}"
CONTROL_MODE="${CONTROL_MODE:-passive_follow}"
STATE_LEFT_TOPIC="${STATE_LEFT_TOPIC:-/puppet/joint_left}"
STATE_RIGHT_TOPIC="${STATE_RIGHT_TOPIC:-/puppet/joint_right}"
COMMAND_LEFT_TOPIC="${COMMAND_LEFT_TOPIC:-/master/joint_left}"
COMMAND_RIGHT_TOPIC="${COMMAND_RIGHT_TOPIC:-/master/joint_right}"
FRONT_CAMERA_TOPIC="${FRONT_CAMERA_TOPIC:-/camera_f/color/image_raw}"
LEFT_CAMERA_TOPIC="${LEFT_CAMERA_TOPIC:-/camera_l/color/image_raw}"
RIGHT_CAMERA_TOPIC="${RIGHT_CAMERA_TOPIC:-/camera_r/color/image_raw}"
OBSERVATION_TIMEOUT_S="${OBSERVATION_TIMEOUT_S:-2.0}"
QUEUE_SIZE="${QUEUE_SIZE:-1}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-480}"
IMAGE_WIDTH="${IMAGE_WIDTH:-640}"
FPS="${FPS:-30}"

DATASET_TASK="${DATASET_TASK:-Execute the trained AgileX GR00T task}"
ROBOT_TYPE="${ROBOT_TYPE:-agilex}"
REMOTE_POLICY_PATH="${REMOTE_POLICY_PATH:-${POLICY_PATH:-}}"
INFER_BACKEND="${INFER_BACKEND:-pytorch}"
TRT_ENGINE_PATH="${TRT_ENGINE_PATH:-}"
TRT_VIT_DTYPE="${TRT_VIT_DTYPE:-fp16}"
TRT_LLM_DTYPE="${TRT_LLM_DTYPE:-fp16}"
TRT_DIT_DTYPE="${TRT_DIT_DTYPE:-fp16}"
TRT_ACTION_HEAD_ONLY="${TRT_ACTION_HEAD_ONLY:-false}"
POLICY_DEVICE_OVERRIDE="${POLICY_DEVICE_OVERRIDE:-${POLICY_DEVICE:-}}"
CONTROL_ARM="${CONTROL_ARM:-auto}"

REMOTE_GROOT_SERVER_HOST="${REMOTE_GROOT_SERVER_HOST:-10.1.26.37}"
REMOTE_GROOT_SERVER_PORT="${REMOTE_GROOT_SERVER_PORT:-5560}"
REMOTE_GROOT_SOCKET_TIMEOUT_S="${REMOTE_GROOT_SOCKET_TIMEOUT_S:-30}"
REMOTE_GROOT_RECONNECT_RETRIES="${REMOTE_GROOT_RECONNECT_RETRIES:-5}"
REMOTE_GROOT_RECONNECT_RETRY_DELAY_S="${REMOTE_GROOT_RECONNECT_RETRY_DELAY_S:-1.0}"

RUN_TIME_S="${RUN_TIME_S:-0}"
LOG_INTERVAL="${LOG_INTERVAL:-30}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${LOG_DIR}"

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
  my_devs/train/groot/agilex/run_groot_remote_robot_client.py
  "--robot-id=${ROBOT_ID}"
  "--control-mode=${CONTROL_MODE}"
  "--state-left-topic=${STATE_LEFT_TOPIC}"
  "--state-right-topic=${STATE_RIGHT_TOPIC}"
  "--command-left-topic=${COMMAND_LEFT_TOPIC}"
  "--command-right-topic=${COMMAND_RIGHT_TOPIC}"
  "--front-camera-topic=${FRONT_CAMERA_TOPIC}"
  "--left-camera-topic=${LEFT_CAMERA_TOPIC}"
  "--right-camera-topic=${RIGHT_CAMERA_TOPIC}"
  "--observation-timeout-s=${OBSERVATION_TIMEOUT_S}"
  "--queue-size=${QUEUE_SIZE}"
  "--image-height=${IMAGE_HEIGHT}"
  "--image-width=${IMAGE_WIDTH}"
  "--fps=${FPS}"
  "--task=${DATASET_TASK}"
  "--robot-type=${ROBOT_TYPE}"
  "--backend=${INFER_BACKEND}"
  "--vit-dtype=${TRT_VIT_DTYPE}"
  "--llm-dtype=${TRT_LLM_DTYPE}"
  "--dit-dtype=${TRT_DIT_DTYPE}"
  "--trt-action-head-only=${TRT_ACTION_HEAD_ONLY}"
  "--control-arm=${CONTROL_ARM}"
  "--server-host=${REMOTE_GROOT_SERVER_HOST}"
  "--server-port=${REMOTE_GROOT_SERVER_PORT}"
  "--socket-timeout-s=${REMOTE_GROOT_SOCKET_TIMEOUT_S}"
  "--reconnect-retries=${REMOTE_GROOT_RECONNECT_RETRIES}"
  "--reconnect-retry-delay-s=${REMOTE_GROOT_RECONNECT_RETRY_DELAY_S}"
  "--run-time-s=${RUN_TIME_S}"
  "--log-interval=${LOG_INTERVAL}"
  "--log-level=${LOG_LEVEL}"
)

if [[ -n "${REMOTE_POLICY_PATH}" ]]; then
  CMD+=("--policy-path=${REMOTE_POLICY_PATH}")
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

echo "[start] script=run_groot_remote_robot_client.sh conda_env=${CONDA_ENV_NAME}"
echo "[start] robot_id=${ROBOT_ID} control_mode=${CONTROL_MODE}"
echo "[start] server=${REMOTE_GROOT_SERVER_HOST}:${REMOTE_GROOT_SERVER_PORT}"
echo "[start] remote_policy_path=${REMOTE_POLICY_PATH:-<server default>}"
echo "[start] task=${DATASET_TASK}"
echo "[start] control_arm=${CONTROL_ARM}"
echo "[start] fps=${FPS} run_time_s=${RUN_TIME_S}"
echo "[start] state_topics=${STATE_LEFT_TOPIC},${STATE_RIGHT_TOPIC}"
echo "[start] command_topics=${COMMAND_LEFT_TOPIC},${COMMAND_RIGHT_TOPIC}"
echo "[start] camera_topics=${FRONT_CAMERA_TOPIC},${LEFT_CAMERA_TOPIC},${RIGHT_CAMERA_TOPIC}"
echo "[start] log_file=${LOG_FILE}"

"${CMD[@]}" "$@" 2>&1 | tee "${LOG_FILE}"
