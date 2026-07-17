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

DATASET_NAME="${DATASET_NAME:-jz_robot_pin_timed_vr_001}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/tests/outputs/${DATASET_NAME}}"
DATASET_REPO_ID="${DATASET_REPO_ID:-local/${DATASET_NAME}}"
SINGLE_TASK="${SINGLE_TASK:-jz robot pin timed vr teleoperation}"
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
VIDEO_CRF="${VIDEO_CRF:-18}"
KEEP_IMAGE_FILES="${KEEP_IMAGE_FILES:-false}"
VIDEO_ENCODING_BATCH_SIZE="${VIDEO_ENCODING_BATCH_SIZE:-${NUM_EPISODES}}"
ZMQ_PRESET="${ZMQ_PRESET:-jz_three_zmq}"
RTSP_PRESET="${RTSP_PRESET:-jz_three_rtsp}"
RTSP_WARMUP_FRAMES="${RTSP_WARMUP_FRAMES:-1}"
RTSP_STALE_FRAME_TIMEOUT_MS="${RTSP_STALE_FRAME_TIMEOUT_MS:-1000}"
RTSP_FFMPEG_CAPTURE_OPTIONS="${RTSP_FFMPEG_CAPTURE_OPTIONS:-}"
CAMERA_BUFFER_SIZE="${CAMERA_BUFFER_SIZE:-8}"
CAMERA_RECONNECT_DELAY_MS="${CAMERA_RECONNECT_DELAY_MS:-250}"
MAX_CAMERA_STATE_RECEIVE_SKEW_MS="${MAX_CAMERA_STATE_RECEIVE_SKEW_MS:-200.0}"
ENFORCE_CAMERA_STATE_RECEIVE_SKEW="${ENFORCE_CAMERA_STATE_RECEIVE_SKEW:-true}"
REJECT_REUSED_CAMERA_FRAMES="${REJECT_REUSED_CAMERA_FRAMES:-false}"
TIMING_LOG_EVERY_N="${TIMING_LOG_EVERY_N:-30}"
TIMING_SIDECAR="${TIMING_SIDECAR:-true}"
REQUIRE_STATE_SOURCE_TIMING="${REQUIRE_STATE_SOURCE_TIMING:-false}"
REQUIRE_STATE_ADVANCE_PER_OBSERVATION="${REQUIRE_STATE_ADVANCE_PER_OBSERVATION:-true}"
STATE_ADVANCE_TIMEOUT_S="${STATE_ADVANCE_TIMEOUT_S:-0.1}"
LEFT_GRIPPER_OBSERVATION_SOURCE="${LEFT_GRIPPER_OBSERVATION_SOURCE:-unavailable}"
RIGHT_GRIPPER_OBSERVATION_SOURCE="${RIGHT_GRIPPER_OBSERVATION_SOURCE:-unavailable}"
LEFT_GRIPPER_OBSERVATION_RAW_CLOSED="${LEFT_GRIPPER_OBSERVATION_RAW_CLOSED:-0.0}"
LEFT_GRIPPER_OBSERVATION_RAW_OPEN="${LEFT_GRIPPER_OBSERVATION_RAW_OPEN:-100.0}"
RIGHT_GRIPPER_OBSERVATION_RAW_CLOSED="${RIGHT_GRIPPER_OBSERVATION_RAW_CLOSED:-100.0}"
RIGHT_GRIPPER_OBSERVATION_RAW_OPEN="${RIGHT_GRIPPER_OBSERVATION_RAW_OPEN:-0.0}"
LEFT_GRIPPER_ACTION_RAW_CLOSED="${LEFT_GRIPPER_ACTION_RAW_CLOSED:-100.0}"
LEFT_GRIPPER_ACTION_RAW_OPEN="${LEFT_GRIPPER_ACTION_RAW_OPEN:-0.0}"
RIGHT_GRIPPER_ACTION_RAW_CLOSED="${RIGHT_GRIPPER_ACTION_RAW_CLOSED:-100.0}"
RIGHT_GRIPPER_ACTION_RAW_OPEN="${RIGHT_GRIPPER_ACTION_RAW_OPEN:-0.0}"
LEFT_GRIPPER_TRAINING_COMMAND_FORCE="${LEFT_GRIPPER_TRAINING_COMMAND_FORCE:-80.0}"
RIGHT_GRIPPER_TRAINING_COMMAND_FORCE="${RIGHT_GRIPPER_TRAINING_COMMAND_FORCE:-80.0}"
MAX_INITIAL_JOINT_DELTA_RAD="${MAX_INITIAL_JOINT_DELTA_RAD:-0.02}"
MAX_JOINT_STEP_RAD="${MAX_JOINT_STEP_RAD:-0.02}"
GRIPPER_WIDTH_MIN="${GRIPPER_WIDTH_MIN:-0.0}"
GRIPPER_WIDTH_MAX="${GRIPPER_WIDTH_MAX:-100.0}"
GRIPPER_FORCE_MIN="${GRIPPER_FORCE_MIN:-0.0}"
GRIPPER_FORCE_MAX="${GRIPPER_FORCE_MAX:-100.0}"

case "${EXECUTION}" in
  dry_run|armed) ;;
  *)
    echo "[timed/record] unsupported EXECUTION=${EXECUTION}; use dry_run or armed" >&2
    exit 2
    ;;
esac

timed_require_armed_confirmation "${EXECUTION}" "record"
automatic_episode_reset_value="${AUTOMATIC_EPISODE_RESET:-false}"
case "${automatic_episode_reset_value,,}" in
  0|false|no|off) ;;
  1|true|yes|on)
    echo "[timed/record] AUTOMATIC_EPISODE_RESET has been retired;" \
      "X86 must not call jz_pin_reset_control/39040 or trigger choreography" >&2
    exit 2
    ;;
  *)
    echo "[timed/record] unsupported retired AUTOMATIC_EPISODE_RESET=" \
      "${automatic_episode_reset_value}" >&2
    exit 2
    ;;
esac
timed_make_python_cmd "${CONDA_ENV}"

if [[ "${ZMQ_PRESET}" == "jz_three_zmq" ]]; then
  ZMQ_CAMERAS='{
    "camera_head": {
      "server_address": "'"${ORIN_IP}"'", "port": 5555, "camera_name": "camera_head",
      "fps": 30, "width": 1280, "height": 720, "color_mode": "rgb", "timeout_ms": 5000
    },
    "camera_left": {
      "server_address": "'"${ORIN_IP}"'", "port": 5556, "camera_name": "camera_left",
      "fps": 30, "width": 640, "height": 480, "color_mode": "rgb", "timeout_ms": 5000
    },
    "camera_right": {
      "server_address": "'"${ORIN_IP}"'", "port": 5557, "camera_name": "camera_right",
      "fps": 30, "width": 640, "height": 480, "color_mode": "rgb", "timeout_ms": 5000
    }
  }'
  RTSP_CAMERAS="{}"
elif [[ "${ZMQ_PRESET}" != "none" ]]; then
  echo "[timed/record] unsupported ZMQ_PRESET=${ZMQ_PRESET}; use jz_three_zmq or none" >&2
  exit 2
elif [[ "${RTSP_PRESET}" == "none" ]]; then
  ZMQ_CAMERAS="{}"
  RTSP_CAMERAS="{}"
elif [[ "${RTSP_PRESET}" == "jz_three_rtsp" ]]; then
  ZMQ_CAMERAS="{}"
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
  echo "[timed/record] unsupported RTSP_PRESET=${RTSP_PRESET}; use jz_three_rtsp or none" >&2
  exit 2
fi

export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

echo "[timed/record] dataset=${DATASET_ROOT}"
echo "[timed/record] repo_id=${DATASET_REPO_ID} episodes=${NUM_EPISODES} fps=${RECORD_FPS}"
echo "[timed/record] video codec=${VCODEC} crf=${VIDEO_CRF} pix_fmt=yuv420p gop=2"
echo "[timed/record] keep_image_files=${KEEP_IMAGE_FILES} image_format=png"
echo "[timed/record] robot state: udp://${STATE_BIND_IP}:${STATE_PORT} expected=${ORIN_IP}"
echo "[timed/record] robot command: udp://${ORIN_IP}:${COMMAND_PORT} execution=${EXECUTION}"
echo "[timed/record] target_action: udp://${TARGET_ACTION_BIND_IP}:${TARGET_ACTION_PORT}" \
  "expected=${TARGET_ACTION_ALLOWED_SENDER_IP} stale_policy=${TARGET_ACTION_STALE_POLICY}"
echo "[timed/record] camera_receiver zmq_preset=${ZMQ_PRESET} rtsp_fallback=${RTSP_PRESET}" \
  "buffer=${CAMERA_BUFFER_SIZE} max_receive_skew_ms=${MAX_CAMERA_STATE_RECEIVE_SKEW_MS}"
echo "[timed/record] timing_sidecar=${TIMING_SIDECAR}" \
  "require_state_source_timing=${REQUIRE_STATE_SOURCE_TIMING}" \
  "require_state_advance_per_observation=${REQUIRE_STATE_ADVANCE_PER_OBSERVATION}" \
  "state_advance_timeout_s=${STATE_ADVANCE_TIMEOUT_S}" \
  "dataset_path=meta/timing/episode-*.jsonl"
echo "[timed/record] training_schema_sidecar=required" \
  "left_source=${LEFT_GRIPPER_OBSERVATION_SOURCE}" \
  "right_source=${RIGHT_GRIPPER_OBSERVATION_SOURCE}" \
  "canonical_direction=0_closed_100_open"
echo "[timed/record] conda_env=${CONDA_ENV} zmq_preset=${ZMQ_PRESET} rtsp_fallback=${RTSP_PRESET}"

cd "${REPO_ROOT}"
exec "${PYTHON_CMD[@]}" -m lerobot.scripts.lerobot_record \
  --robot.type=jz_robot_pin_timed \
  --robot.id=jz_robot_pin_timed_record \
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
  --robot.camera_buffer_size="${CAMERA_BUFFER_SIZE}" \
  --robot.camera_reconnect_delay_ms="${CAMERA_RECONNECT_DELAY_MS}" \
  --robot.max_camera_state_receive_skew_ms="${MAX_CAMERA_STATE_RECEIVE_SKEW_MS}" \
  --robot.enforce_camera_state_receive_skew="${ENFORCE_CAMERA_STATE_RECEIVE_SKEW}" \
  --robot.reject_reused_camera_frames="${REJECT_REUSED_CAMERA_FRAMES}" \
  --robot.timing_log_every_n="${TIMING_LOG_EVERY_N}" \
  --robot.timing_sidecar="${TIMING_SIDECAR}" \
  --robot.require_state_source_timing="${REQUIRE_STATE_SOURCE_TIMING}" \
  --robot.require_state_advance_per_observation="${REQUIRE_STATE_ADVANCE_PER_OBSERVATION}" \
  --robot.state_advance_timeout_s="${STATE_ADVANCE_TIMEOUT_S}" \
  --robot.left_gripper_observation_source="${LEFT_GRIPPER_OBSERVATION_SOURCE}" \
  --robot.right_gripper_observation_source="${RIGHT_GRIPPER_OBSERVATION_SOURCE}" \
  --robot.left_gripper_observation_raw_closed="${LEFT_GRIPPER_OBSERVATION_RAW_CLOSED}" \
  --robot.left_gripper_observation_raw_open="${LEFT_GRIPPER_OBSERVATION_RAW_OPEN}" \
  --robot.right_gripper_observation_raw_closed="${RIGHT_GRIPPER_OBSERVATION_RAW_CLOSED}" \
  --robot.right_gripper_observation_raw_open="${RIGHT_GRIPPER_OBSERVATION_RAW_OPEN}" \
  --robot.left_gripper_action_raw_closed="${LEFT_GRIPPER_ACTION_RAW_CLOSED}" \
  --robot.left_gripper_action_raw_open="${LEFT_GRIPPER_ACTION_RAW_OPEN}" \
  --robot.right_gripper_action_raw_closed="${RIGHT_GRIPPER_ACTION_RAW_CLOSED}" \
  --robot.right_gripper_action_raw_open="${RIGHT_GRIPPER_ACTION_RAW_OPEN}" \
  --robot.left_gripper_training_command_force="${LEFT_GRIPPER_TRAINING_COMMAND_FORCE}" \
  --robot.right_gripper_training_command_force="${RIGHT_GRIPPER_TRAINING_COMMAND_FORCE}" \
  --robot.zmq_cameras="${ZMQ_CAMERAS}" \
  --robot.rtsp_cameras="${RTSP_CAMERAS}" \
  --teleop.type=jz_robot_pin_target_action \
  --teleop.id=jz_robot_pin_timed_target_action_record \
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
  --dataset.video_crf="${VIDEO_CRF}" \
  --dataset.keep_image_files="${KEEP_IMAGE_FILES}" \
  --dataset.video_encoding_batch_size="${VIDEO_ENCODING_BATCH_SIZE}" \
  --display_data="${DISPLAY_DATA}" \
  --display_compressed_images="${DISPLAY_COMPRESSED_IMAGES}" \
  --play_sounds="${PLAY_SOUNDS}" \
  --resume="${RESUME}"
