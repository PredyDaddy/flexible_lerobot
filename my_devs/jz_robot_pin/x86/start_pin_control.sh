#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"

CONDA_ENV="${CONDA_ENV:-${LEROBOT_CONDA_ENV:-lerobot_flex}}"
ORIN_IP="${ORIN_IP:-192.168.1.81}"
STATE_BIND_IP="${STATE_BIND_IP:-0.0.0.0}"
STATE_PORT="${STATE_PORT:-39010}"
STATE_TIMEOUT_S="${STATE_TIMEOUT_S:-1.0}"
CONNECT_TIMEOUT_S="${CONNECT_TIMEOUT_S:-300.0}"
COMMAND_PORT="${COMMAND_PORT:-39020}"
TARGET_ACTION_BIND_IP="${TARGET_ACTION_BIND_IP:-0.0.0.0}"
TARGET_ACTION_PORT="${TARGET_ACTION_PORT:-39030}"
TARGET_ACTION_ALLOWED_SENDER_IP="${TARGET_ACTION_ALLOWED_SENDER_IP:-127.0.0.1}"
TARGET_ACTION_TIMEOUT_S="${TARGET_ACTION_TIMEOUT_S:-0.5}"
EXECUTION="${EXECUTION:-armed}"
SEND_ACTION_TRANSPORT="${SEND_ACTION_TRANSPORT:-udp}"
FPS="${FPS:-80}"
DISPLAY_DATA="${DISPLAY_DATA:-false}"
DISPLAY_COMPRESSED_IMAGES="${DISPLAY_COMPRESSED_IMAGES:-false}"
MAX_INITIAL_JOINT_DELTA_RAD="${MAX_INITIAL_JOINT_DELTA_RAD:-0.02}"
MAX_JOINT_STEP_RAD="${MAX_JOINT_STEP_RAD:-0.02}"
GRIPPER_WIDTH_MIN="${GRIPPER_WIDTH_MIN:-0.0}"
GRIPPER_WIDTH_MAX="${GRIPPER_WIDTH_MAX:-100.0}"
GRIPPER_FORCE_MIN="${GRIPPER_FORCE_MIN:-0.0}"
GRIPPER_FORCE_MAX="${GRIPPER_FORCE_MAX:-100.0}"

case "${EXECUTION}" in
  dry_run|armed) ;;
  *)
    echo "[start_pin_control] unsupported EXECUTION=${EXECUTION}; use dry_run or armed" >&2
    exit 2
    ;;
esac

pin_require_armed_confirmation "${EXECUTION}" "control"
pin_make_python_cmd "${CONDA_ENV}"

export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

echo "[start_pin_control] robot state: udp://${STATE_BIND_IP}:${STATE_PORT} expected=${ORIN_IP}"
echo "[start_pin_control] robot command: udp://${ORIN_IP}:${COMMAND_PORT} execution=${EXECUTION}"
echo "[start_pin_control] target_action: udp://${TARGET_ACTION_BIND_IP}:${TARGET_ACTION_PORT} expected=${TARGET_ACTION_ALLOWED_SENDER_IP}"
echo "[start_pin_control] conda_env=${CONDA_ENV} fps=${FPS}"
if [[ "${EXECUTION}" == "armed" ]]; then
  echo "JZ ROBOT PIN ARMED"
  echo "WILL send UDP commands to Orin executor"
  echo "emergency stop is physical fallback, not a replacement for software limits"
fi

cd "${REPO_ROOT}"
exec "${PYTHON_CMD[@]}" -m lerobot.scripts.lerobot_teleoperate \
  --robot.type=jz_robot_pin \
  --robot.id=jz_robot_pin_default \
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
  --robot.gripper_width_min="${GRIPPER_WIDTH_MIN}" \
  --robot.gripper_width_max="${GRIPPER_WIDTH_MAX}" \
  --robot.gripper_force_min="${GRIPPER_FORCE_MIN}" \
  --robot.gripper_force_max="${GRIPPER_FORCE_MAX}" \
  --robot.rtsp_cameras='{}' \
  --teleop.type=jz_robot_pin_target_action \
  --teleop.id=jz_robot_pin_target_action \
  --teleop.bind_ip="${TARGET_ACTION_BIND_IP}" \
  --teleop.target_action_port="${TARGET_ACTION_PORT}" \
  --teleop.allowed_sender_ip="${TARGET_ACTION_ALLOWED_SENDER_IP}" \
  --teleop.connect_timeout_s=0.0 \
  --teleop.target_action_timeout_s="${TARGET_ACTION_TIMEOUT_S}" \
  --teleop.stale_policy=hold_current \
  --fps="${FPS}" \
  --display_data="${DISPLAY_DATA}" \
  --display_compressed_images="${DISPLAY_COMPRESSED_IMAGES}"
