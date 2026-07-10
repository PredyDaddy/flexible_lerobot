#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"

BRIDGE_ROOT="${REPO_ROOT}/my_devs/my_var_tp/live_vr_replay_bridge"
VISUAL_CONDA_ENV="${VISUAL_CONDA_ENV:-light_tp}"
VR_HOST="${VR_HOST:-10.1.42.3}"
VR_PORT="${VR_PORT:-8080}"
TARGET_ACTION_IP="${TARGET_ACTION_IP:-127.0.0.1}"
TARGET_ACTION_PORT="${TARGET_ACTION_PORT:-39030}"
LEFT_EE_FRAME="${LEFT_EE_FRAME:-left_arm_link7}"
RIGHT_EE_FRAME="${RIGHT_EE_FRAME:-right_arm_link7}"
TCP_CONTROL_OFFSET="${TCP_CONTROL_OFFSET:-0.0}"
VR_DEBUG_TARGET_FORWARD_OFFSET="${VR_DEBUG_TARGET_FORWARD_OFFSET:-0.0}"
VISUAL_FREQUENCY="${VISUAL_FREQUENCY:-80}"
TARGET_MAX_SPEED="${TARGET_MAX_SPEED:-0.4}"
PUBLISH_EVERY="${PUBLISH_EVERY:-1}"
RESET_PUBLISH_DURATION_S="${RESET_PUBLISH_DURATION_S:-0.5}"
JOINT_MOTION_COST="${JOINT_MOTION_COST:-0.0}"
JOINT_MOTION_COST_PROFILE="${JOINT_MOTION_COST_PROFILE:-uniform}"
GRIPPER_INPUT="${GRIPPER_INPUT:-trigger}"
GRIPPER_OPEN_WIDTH="${GRIPPER_OPEN_WIDTH:-0}"
GRIPPER_CLOSED_WIDTH="${GRIPPER_CLOSED_WIDTH:-100}"
GRIPPER_FORCE="${GRIPPER_FORCE:-80}"
GRIPPER_PUBLISH_MODE="${GRIPPER_PUBLISH_MODE:-on-change}"

if [[ ! -x "${BRIDGE_ROOT}/scripts/run_visual_publisher.sh" ]]; then
  echo "[start_pin_joystick] missing visual publisher script: ${BRIDGE_ROOT}/scripts/run_visual_publisher.sh" >&2
  exit 1
fi

echo "[start_pin_joystick] VR input: udp://${VR_HOST}:${VR_PORT}"
echo "[start_pin_joystick] target_action: udp://${TARGET_ACTION_IP}:${TARGET_ACTION_PORT}"
echo "[start_pin_joystick] visual_conda_env=${VISUAL_CONDA_ENV}"
echo "[start_pin_joystick] Meshcat: http://127.0.0.1:7000/static/"

cd "${BRIDGE_ROOT}"
exec env -u PYTHON \
  TARGET_ACTION_IP="${TARGET_ACTION_IP}" \
  TARGET_ACTION_PORT="${TARGET_ACTION_PORT}" \
  VISUALIZE_WHOLE_ROBOT=true \
  CONDA_ENV="${VISUAL_CONDA_ENV}" \
  ./scripts/run_visual_publisher.sh "${VR_HOST}" "${VR_PORT}" \
    --left-ee-frame "${LEFT_EE_FRAME}" \
    --right-ee-frame "${RIGHT_EE_FRAME}" \
    --tcp-control-offset "${TCP_CONTROL_OFFSET}" \
    --vr-debug-target-forward-offset "${VR_DEBUG_TARGET_FORWARD_OFFSET}" \
    --frequency "${VISUAL_FREQUENCY}" \
    --target-max-speed "${TARGET_MAX_SPEED}" \
    --publish-every "${PUBLISH_EVERY}" \
    --reset-publish-duration-s "${RESET_PUBLISH_DURATION_S}" \
    --joint-motion-cost "${JOINT_MOTION_COST}" \
    --joint-motion-cost-profile "${JOINT_MOTION_COST_PROFILE}" \
    --no-arm-meshes-only \
    --gripper-input "${GRIPPER_INPUT}" \
    --gripper-open-width "${GRIPPER_OPEN_WIDTH}" \
    --gripper-closed-width "${GRIPPER_CLOSED_WIDTH}" \
    --gripper-force "${GRIPPER_FORCE}" \
    --gripper-publish-mode "${GRIPPER_PUBLISH_MODE}"
