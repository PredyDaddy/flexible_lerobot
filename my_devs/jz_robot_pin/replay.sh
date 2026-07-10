#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"

CONDA_ENV="${CONDA_ENV:-${LEROBOT_CONDA_ENV:-lerobot_flex}}"
ORIN_IP="${ORIN_IP:-192.168.1.81}"
STATE_BIND_IP="${STATE_BIND_IP:-0.0.0.0}"
STATE_PORT="${STATE_PORT:-39010}"
STATE_TIMEOUT_S="${STATE_TIMEOUT_S:-1.0}"
CONNECT_TIMEOUT_S="${CONNECT_TIMEOUT_S:-300.0}"
COMMAND_PORT="${COMMAND_PORT:-39020}"
EXECUTION="${EXECUTION:-dry_run}"
SEND_ACTION_TRANSPORT="${SEND_ACTION_TRANSPORT:-udp}"

DATASET_NAME="${DATASET_NAME:-jz_robot_pin_vr_001}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/tests/outputs/${DATASET_NAME}}"
DATASET_REPO_ID="${DATASET_REPO_ID:-local/${DATASET_NAME}}"
EPISODE="${EPISODE:-0}"
REPLAY_FPS="${REPLAY_FPS:-30}"
PLAY_SOUNDS="${PLAY_SOUNDS:-true}"
MAX_INITIAL_JOINT_DELTA_RAD="${MAX_INITIAL_JOINT_DELTA_RAD:-0.02}"
MAX_JOINT_STEP_RAD="${MAX_JOINT_STEP_RAD:-0.02}"

case "${EXECUTION}" in
  dry_run|armed) ;;
  *)
    echo "[replay.sh] unsupported EXECUTION=${EXECUTION}; use dry_run or armed" >&2
    exit 2
    ;;
esac

pin_require_armed_confirmation "${EXECUTION}" "replay"
pin_make_python_cmd "${CONDA_ENV}"

export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

echo "[replay.sh] dataset=${DATASET_ROOT} repo_id=${DATASET_REPO_ID} episode=${EPISODE}"
echo "[replay.sh] robot state: udp://${STATE_BIND_IP}:${STATE_PORT} expected=${ORIN_IP}"
echo "[replay.sh] robot command: udp://${ORIN_IP}:${COMMAND_PORT} execution=${EXECUTION}"
echo "[replay.sh] conda_env=${CONDA_ENV} replay_fps=${REPLAY_FPS}"

cd "${REPO_ROOT}"
exec "${PYTHON_CMD[@]}" -m lerobot.scripts.lerobot_replay \
  --robot.type=jz_robot_pin \
  --robot.id=jz_robot_pin_replay \
  --robot.bind_ip="${STATE_BIND_IP}" \
  --robot.state_port="${STATE_PORT}" \
  --robot.allowed_state_sender_ip="${ORIN_IP}" \
  --robot.connect_timeout_s="${CONNECT_TIMEOUT_S}" \
  --robot.state_timeout_s="${STATE_TIMEOUT_S}" \
  --robot.command_target_ip="${ORIN_IP}" \
  --robot.command_target_port="${COMMAND_PORT}" \
  --robot.send_action_transport="${SEND_ACTION_TRANSPORT}" \
  --robot.send_action_execution="${EXECUTION}" \
  --robot.max_initial_joint_delta_rad="${MAX_INITIAL_JOINT_DELTA_RAD}" \
  --robot.max_joint_step_rad="${MAX_JOINT_STEP_RAD}" \
  --robot.rtsp_cameras='{}' \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.episode="${EPISODE}" \
  --dataset.fps="${REPLAY_FPS}" \
  --play_sounds="${PLAY_SOUNDS}"
