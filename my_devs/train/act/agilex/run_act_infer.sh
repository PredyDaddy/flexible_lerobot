#!/usr/bin/env bash

set -Eeuo pipefail

# -----------------------------------------------------------------------------
# Agilex ACT 上机推理脚本
# 说明：
# - 对应 Python 主脚本：my_devs/train/act/agilex/run_act_infer.py
# - 支持 dry-run、只读观测检查、影子推理、真实闭环
# - 默认使用当前已训练完成的双臂 Agilex ACT checkpoint
# -----------------------------------------------------------------------------

CONDA_ENV_NAME="${CONDA_ENV_NAME:-lerobot_flex}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR_DEFAULT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${ROOT_DIR_DEFAULT}" ]]; then
  ROOT_DIR_DEFAULT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
fi
ROOT_DIR="${ROOT_DIR:-${ROOT_DIR_DEFAULT}}"
cd "${ROOT_DIR}"

LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs}"
JOB_NAME="${JOB_NAME:-act_agilex_first_test_infer}"
DEFAULT_POLICY_PATH="${ROOT_DIR}/outputs/train/20260313_194500_act_agilex_first_test_full/checkpoints/100000/pretrained_model"
POLICY_PATH="${POLICY_PATH:-${DEFAULT_POLICY_PATH}}"

ROBOT_ID="${ROBOT_ID:-my_agilex}"
EXECUTION_MODE="${EXECUTION_MODE:-policy_inference}"
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
FPS="${FPS:-30}"

POLICY_DEVICE_OVERRIDE="${POLICY_DEVICE_OVERRIDE:-${POLICY_DEVICE:-}}"
POLICY_N_ACTION_STEPS="${POLICY_N_ACTION_STEPS:-}"
POLICY_TEMPORAL_ENSEMBLE_COEFF="${POLICY_TEMPORAL_ENSEMBLE_COEFF:-}"
DATASET_TASK="${DATASET_TASK:-Execute the trained Agilex ACT task}"
RUN_TIME_S="${RUN_TIME_S:-0}"
LOG_INTERVAL="${LOG_INTERVAL:-30}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${LOG_DIR}"

if [[ ! -d "${POLICY_PATH}" ]]; then
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
LOG_FILE="${LOG_FILE:-${LOG_DIR}/infer_${JOB_NAME}_${timestamp}.log}"

CMD=(
  "${RUNNER[@]}"
  python
  my_devs/train/act/agilex/run_act_infer.py
  "--robot-id=${ROBOT_ID}"
  "--execution-mode=${EXECUTION_MODE}"
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
  "--fps=${FPS}"
  "--policy-path=${POLICY_PATH}"
  "--task=${DATASET_TASK}"
  "--run-time-s=${RUN_TIME_S}"
  "--log-interval=${LOG_INTERVAL}"
)

if [[ -n "${POLICY_DEVICE_OVERRIDE}" ]]; then
  CMD+=("--policy-device=${POLICY_DEVICE_OVERRIDE}")
fi

if [[ -n "${POLICY_N_ACTION_STEPS}" ]]; then
  CMD+=("--policy-n-action-steps=${POLICY_N_ACTION_STEPS}")
fi

if [[ -n "${POLICY_TEMPORAL_ENSEMBLE_COEFF}" ]]; then
  CMD+=("--policy-temporal-ensemble-coeff=${POLICY_TEMPORAL_ENSEMBLE_COEFF}")
fi

if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" ]]; then
  CMD+=("--dry-run=true")
  echo "[DRY_RUN] Would execute:"
  printf '%q ' "${CMD[@]}" "$@"
  echo
  echo "[DRY_RUN] Log file: ${LOG_FILE}"
  exit 0
fi

echo "[start] script=run_act_infer.sh conda_env=${CONDA_ENV_NAME}"
echo "[start] execution_mode=${EXECUTION_MODE} control_mode=${CONTROL_MODE}"
echo "[start] robot_id=${ROBOT_ID}"
echo "[start] policy_path=${POLICY_PATH}"
echo "[start] policy_device_override=${POLICY_DEVICE_OVERRIDE:-<checkpoint default>}"
echo "[start] policy_n_action_steps=${POLICY_N_ACTION_STEPS:-<checkpoint default>}"
echo "[start] policy_temporal_ensemble_coeff=${POLICY_TEMPORAL_ENSEMBLE_COEFF:-<checkpoint default>}"
echo "[start] fps=${FPS} run_time_s=${RUN_TIME_S} log_interval=${LOG_INTERVAL}"
echo "[start] state_topics=${STATE_LEFT_TOPIC},${STATE_RIGHT_TOPIC}"
echo "[start] command_topics=${COMMAND_LEFT_TOPIC},${COMMAND_RIGHT_TOPIC}"
echo "[start] camera_topics=${FRONT_CAMERA_TOPIC},${LEFT_CAMERA_TOPIC},${RIGHT_CAMERA_TOPIC}"
echo "[start] log_file=${LOG_FILE}"

"${CMD[@]}" "$@" 2>&1 | tee "${LOG_FILE}"
