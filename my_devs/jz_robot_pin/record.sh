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
TARGET_ACTION_BIND_IP="${TARGET_ACTION_BIND_IP:-0.0.0.0}"
TARGET_ACTION_PORT="${TARGET_ACTION_PORT:-39030}"
TARGET_ACTION_ALLOWED_SENDER_IP="${TARGET_ACTION_ALLOWED_SENDER_IP:-127.0.0.1}"
TARGET_ACTION_CONNECT_TIMEOUT_S="${TARGET_ACTION_CONNECT_TIMEOUT_S:-5.0}"
TARGET_ACTION_TIMEOUT_S="${TARGET_ACTION_TIMEOUT_S:-0.5}"
TARGET_ACTION_STALE_POLICY="${TARGET_ACTION_STALE_POLICY:-raise}"
EXECUTION="${EXECUTION:-armed}"
SEND_ACTION_TRANSPORT="${SEND_ACTION_TRANSPORT:-udp}"

DATASET_NAME="${DATASET_NAME:-jz_robot_pin_vr_001}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/tests/outputs/${DATASET_NAME}}"
DATASET_REPO_ID="${DATASET_REPO_ID:-local/${DATASET_NAME}}"
SINGLE_TASK="${SINGLE_TASK:-jz robot pin vr teleoperation}"
NUM_EPISODES="${NUM_EPISODES:-30}"
EPISODE_TIME_S="${EPISODE_TIME_S:-10}"
RESET_TIME_S="${RESET_TIME_S:-5}"
RECORD_FPS="${RECORD_FPS:-30}"
DISPLAY_DATA="${DISPLAY_DATA:-false}"
DISPLAY_COMPRESSED_IMAGES="${DISPLAY_COMPRESSED_IMAGES:-false}"
PLAY_SOUNDS="${PLAY_SOUNDS:-true}"
RESUME="${RESUME:-true}"
VIDEO="${VIDEO:-true}"
VCODEC="${VCODEC:-h264}"
VIDEO_ENCODING_BATCH_SIZE="${VIDEO_ENCODING_BATCH_SIZE:-${NUM_EPISODES}}"
RTSP_PRESET="${RTSP_PRESET:-jz_three_rtsp}"
RTSP_WARMUP_FRAMES="${RTSP_WARMUP_FRAMES:-1}"
RTSP_STALE_FRAME_TIMEOUT_MS="${RTSP_STALE_FRAME_TIMEOUT_MS:-1000}"
RTSP_FFMPEG_CAPTURE_OPTIONS="${RTSP_FFMPEG_CAPTURE_OPTIONS:-}"
MAX_INITIAL_JOINT_DELTA_RAD="${MAX_INITIAL_JOINT_DELTA_RAD:-0.02}"
MAX_JOINT_STEP_RAD="${MAX_JOINT_STEP_RAD:-0.02}"
GRIPPER_WIDTH_MIN="${GRIPPER_WIDTH_MIN:-0.0}"
GRIPPER_WIDTH_MAX="${GRIPPER_WIDTH_MAX:-100.0}"
GRIPPER_FORCE_MIN="${GRIPPER_FORCE_MIN:-0.0}"
GRIPPER_FORCE_MAX="${GRIPPER_FORCE_MAX:-100.0}"

case "${EXECUTION}" in
  dry_run|armed) ;;
  *)
    echo "[record.sh] unsupported EXECUTION=${EXECUTION}; use dry_run or armed" >&2
    exit 2
    ;;
esac

pin_require_armed_confirmation "${EXECUTION}" "record"
pin_make_python_cmd "${CONDA_ENV}"

if [[ "${RTSP_PRESET}" == "none" ]]; then
  RTSP_CAMERAS="{}"
elif [[ "${RTSP_PRESET}" == "jz_three_rtsp" ]]; then
  RTSP_CAMERAS='{
    "camera_head": {
      "url": "rtsp://'"${ORIN_IP}"':8554/robot_camera/camera_head",
      "fps": 30,
      "width": 1280,
      "height": 720,
      "timeout_ms": 5000,
      "warmup_frames": '"${RTSP_WARMUP_FRAMES}"',
      "color_mode": "rgb",
      "transport": "tcp",
      "threaded_reader": true,
      "stale_frame_timeout_ms": '"${RTSP_STALE_FRAME_TIMEOUT_MS}"',
      "ffmpeg_capture_options": "'"${RTSP_FFMPEG_CAPTURE_OPTIONS}"'"
    },
    "camera_left": {
      "url": "rtsp://'"${ORIN_IP}"':8554/robot_camera/camera_left",
      "fps": 30,
      "width": 640,
      "height": 480,
      "timeout_ms": 5000,
      "warmup_frames": '"${RTSP_WARMUP_FRAMES}"',
      "color_mode": "rgb",
      "transport": "tcp",
      "threaded_reader": true,
      "stale_frame_timeout_ms": '"${RTSP_STALE_FRAME_TIMEOUT_MS}"',
      "ffmpeg_capture_options": "'"${RTSP_FFMPEG_CAPTURE_OPTIONS}"'"
    },
    "camera_right": {
      "url": "rtsp://'"${ORIN_IP}"':8554/robot_camera/camera_right",
      "fps": 30,
      "width": 640,
      "height": 480,
      "timeout_ms": 5000,
      "warmup_frames": '"${RTSP_WARMUP_FRAMES}"',
      "color_mode": "rgb",
      "transport": "tcp",
      "threaded_reader": true,
      "stale_frame_timeout_ms": '"${RTSP_STALE_FRAME_TIMEOUT_MS}"',
      "ffmpeg_capture_options": "'"${RTSP_FFMPEG_CAPTURE_OPTIONS}"'"
    }
  }'
else
  echo "[record.sh] unsupported RTSP_PRESET=${RTSP_PRESET}; use jz_three_rtsp or none" >&2
  exit 2
fi

export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

echo "[record.sh] dataset=${DATASET_ROOT}"
echo "[record.sh] repo_id=${DATASET_REPO_ID} episodes=${NUM_EPISODES} fps=${RECORD_FPS}"
echo "[record.sh] robot state: udp://${STATE_BIND_IP}:${STATE_PORT} expected=${ORIN_IP}"
echo "[record.sh] robot command: udp://${ORIN_IP}:${COMMAND_PORT} execution=${EXECUTION}"
echo "[record.sh] target_action: udp://${TARGET_ACTION_BIND_IP}:${TARGET_ACTION_PORT} expected=${TARGET_ACTION_ALLOWED_SENDER_IP}"
echo "[record.sh] target_action connect_timeout_s=${TARGET_ACTION_CONNECT_TIMEOUT_S} timeout_s=${TARGET_ACTION_TIMEOUT_S} stale_policy=${TARGET_ACTION_STALE_POLICY}"
echo "[record.sh] conda_env=${CONDA_ENV} rtsp_preset=${RTSP_PRESET}"

cd "${REPO_ROOT}"
exec "${PYTHON_CMD[@]}" -m lerobot.scripts.lerobot_record \
  --robot.type=jz_robot_pin \
  --robot.id=jz_robot_pin_record \
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
  --robot.rtsp_cameras="${RTSP_CAMERAS}" \
  --teleop.type=jz_robot_pin_target_action \
  --teleop.id=jz_robot_pin_target_action_record \
  --teleop.bind_ip="${TARGET_ACTION_BIND_IP}" \
  --teleop.target_action_port="${TARGET_ACTION_PORT}" \
  --teleop.allowed_sender_ip="${TARGET_ACTION_ALLOWED_SENDER_IP}" \
  --teleop.connect_timeout_s="${TARGET_ACTION_CONNECT_TIMEOUT_S}" \
  --teleop.target_action_timeout_s="${TARGET_ACTION_TIMEOUT_S}" \
  --teleop.stale_policy="${TARGET_ACTION_STALE_POLICY}" \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.num_episodes="${NUM_EPISODES}" \
  --dataset.episode_time_s="${EPISODE_TIME_S}" \
  --dataset.reset_time_s="${RESET_TIME_S}" \
  --dataset.fps="${RECORD_FPS}" \
  --dataset.single_task="${SINGLE_TASK}" \
  --dataset.push_to_hub=false \
  --dataset.video="${VIDEO}" \
  --dataset.vcodec="${VCODEC}" \
  --dataset.video_encoding_batch_size="${VIDEO_ENCODING_BATCH_SIZE}" \
  --display_data="${DISPLAY_DATA}" \
  --display_compressed_images="${DISPLAY_COMPRESSED_IMAGES}" \
  --play_sounds="${PLAY_SOUNDS}" \
  --resume="${RESUME}"
