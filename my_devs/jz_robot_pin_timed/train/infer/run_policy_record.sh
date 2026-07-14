#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

infer_require_policy
infer_require_reference_dataset

EXECUTION="${EXECUTION:-dry_run}"
SEND_ACTION_TRANSPORT="${SEND_ACTION_TRANSPORT:-local}"
case "${EXECUTION}" in
  dry_run)
    if [[ "${SEND_ACTION_TRANSPORT}" != "local" ]]; then
      echo "[jz_pin_timed/infer] dry_run must use SEND_ACTION_TRANSPORT=local" >&2
      exit 2
    fi
    ;;
  armed)
    if [[ "${SEND_ACTION_TRANSPORT}" != "udp" ]]; then
      echo "[jz_pin_timed/infer] armed inference must use SEND_ACTION_TRANSPORT=udp" >&2
      exit 2
    fi
    infer_require_armed_confirmation
    ;;
  *)
    echo "[jz_pin_timed/infer] EXECUTION must be dry_run or armed" >&2
    exit 2
    ;;
esac

ORIN_IP="${ORIN_IP:-192.168.1.81}"
STATE_BIND_IP="${STATE_BIND_IP:-0.0.0.0}"
STATE_PORT="${STATE_PORT:-39010}"
STATE_TIMEOUT_S="${STATE_TIMEOUT_S:-1.0}"
CONNECT_TIMEOUT_S="${CONNECT_TIMEOUT_S:-300.0}"
COMMAND_PORT="${COMMAND_PORT:-39020}"

POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
POLICY_N_ACTION_STEPS="${POLICY_N_ACTION_STEPS:-}"
STAMP="$(date +%Y%m%d_%H%M%S)"
POLICY_RUN_NAME="$(basename "$(dirname "$(dirname "$(dirname "${POLICY_PATH}")")")")"
DATASET_NAME="${DATASET_NAME:-eval_${POLICY_RUN_NAME}_${EXECUTION}_${STAMP}}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/tests/outputs/${DATASET_NAME}}"
DATASET_REPO_ID="${DATASET_REPO_ID:-local/${DATASET_NAME}}"
SINGLE_TASK="${SINGLE_TASK:-jz robot pin timed policy evaluation}"
NUM_EPISODES="${NUM_EPISODES:-1}"
EPISODE_TIME_S="${EPISODE_TIME_S:-10}"
RESET_TIME_S="${RESET_TIME_S:-0}"
RECORD_FPS="${RECORD_FPS:-20}"
DISPLAY_DATA="${DISPLAY_DATA:-false}"
DISPLAY_COMPRESSED_IMAGES="${DISPLAY_COMPRESSED_IMAGES:-false}"
PLAY_SOUNDS="${PLAY_SOUNDS:-true}"
VIDEO="${VIDEO:-true}"
VCODEC="${VCODEC:-h264}"
VIDEO_CRF="${VIDEO_CRF:-18}"
KEEP_IMAGE_FILES="${KEEP_IMAGE_FILES:-false}"
VIDEO_ENCODING_BATCH_SIZE="${VIDEO_ENCODING_BATCH_SIZE:-${NUM_EPISODES}}"

CAMERA_BUFFER_SIZE="${CAMERA_BUFFER_SIZE:-8}"
CAMERA_RECONNECT_DELAY_MS="${CAMERA_RECONNECT_DELAY_MS:-250}"
CAMERA_STALE_FRAME_TIMEOUT_MS="${CAMERA_STALE_FRAME_TIMEOUT_MS:-1000}"
MAX_CAMERA_STATE_RECEIVE_SKEW_MS="${MAX_CAMERA_STATE_RECEIVE_SKEW_MS:-100.0}"
ENFORCE_CAMERA_STATE_RECEIVE_SKEW="${ENFORCE_CAMERA_STATE_RECEIVE_SKEW:-true}"
REJECT_REUSED_CAMERA_FRAMES="${REJECT_REUSED_CAMERA_FRAMES:-false}"
TIMING_LOG_EVERY_N="${TIMING_LOG_EVERY_N:-20}"
TIMING_SIDECAR="${TIMING_SIDECAR:-true}"
REQUIRE_STATE_SOURCE_TIMING="${REQUIRE_STATE_SOURCE_TIMING:-true}"
REQUIRE_STATE_ADVANCE_PER_OBSERVATION="${REQUIRE_STATE_ADVANCE_PER_OBSERVATION:-true}"
STATE_ADVANCE_TIMEOUT_S="${STATE_ADVANCE_TIMEOUT_S:-0.1}"

LEFT_GRIPPER_OBSERVATION_SOURCE="${LEFT_GRIPPER_OBSERVATION_SOURCE:-measured_opening}"
RIGHT_GRIPPER_OBSERVATION_SOURCE="${RIGHT_GRIPPER_OBSERVATION_SOURCE:-commanded_opening}"
MAX_INITIAL_JOINT_DELTA_RAD="${MAX_INITIAL_JOINT_DELTA_RAD:-0.02}"
MAX_JOINT_STEP_RAD="${MAX_JOINT_STEP_RAD:-0.02}"
GRIPPER_WIDTH_MIN="${GRIPPER_WIDTH_MIN:-0.0}"
GRIPPER_WIDTH_MAX="${GRIPPER_WIDTH_MAX:-100.0}"
GRIPPER_FORCE_MIN="${GRIPPER_FORCE_MIN:-0.0}"
GRIPPER_FORCE_MAX="${GRIPPER_FORCE_MAX:-100.0}"

if [[ -e "${DATASET_ROOT}" ]]; then
  echo "[jz_pin_timed/infer] evaluation dataset already exists: ${DATASET_ROOT}" >&2
  exit 2
fi

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

CMD=(
  "${CONDA_PYTHON}"
  -m lerobot.scripts.lerobot_record
  --robot.type=jz_robot_pin_timed
  --robot.id=jz_robot_pin_timed_policy_infer
  "--robot.bind_ip=${STATE_BIND_IP}"
  "--robot.state_port=${STATE_PORT}"
  "--robot.allowed_state_sender_ip=${ORIN_IP}"
  "--robot.connect_timeout_s=${CONNECT_TIMEOUT_S}"
  "--robot.state_timeout_s=${STATE_TIMEOUT_S}"
  "--robot.command_target_ip=${ORIN_IP}"
  "--robot.command_target_port=${COMMAND_PORT}"
  "--robot.send_action_transport=${SEND_ACTION_TRANSPORT}"
  "--robot.send_action_execution=${EXECUTION}"
  "--robot.max_initial_joint_delta_rad=${MAX_INITIAL_JOINT_DELTA_RAD}"
  "--robot.max_joint_step_rad=${MAX_JOINT_STEP_RAD}"
  "--robot.gripper_width_min=${GRIPPER_WIDTH_MIN}"
  "--robot.gripper_width_max=${GRIPPER_WIDTH_MAX}"
  "--robot.gripper_force_min=${GRIPPER_FORCE_MIN}"
  "--robot.gripper_force_max=${GRIPPER_FORCE_MAX}"
  "--robot.camera_buffer_size=${CAMERA_BUFFER_SIZE}"
  "--robot.camera_reconnect_delay_ms=${CAMERA_RECONNECT_DELAY_MS}"
  "--robot.camera_stale_frame_timeout_ms=${CAMERA_STALE_FRAME_TIMEOUT_MS}"
  "--robot.max_camera_state_receive_skew_ms=${MAX_CAMERA_STATE_RECEIVE_SKEW_MS}"
  "--robot.enforce_camera_state_receive_skew=${ENFORCE_CAMERA_STATE_RECEIVE_SKEW}"
  "--robot.reject_reused_camera_frames=${REJECT_REUSED_CAMERA_FRAMES}"
  "--robot.timing_log_every_n=${TIMING_LOG_EVERY_N}"
  "--robot.timing_sidecar=${TIMING_SIDECAR}"
  "--robot.require_state_source_timing=${REQUIRE_STATE_SOURCE_TIMING}"
  "--robot.require_state_advance_per_observation=${REQUIRE_STATE_ADVANCE_PER_OBSERVATION}"
  "--robot.state_advance_timeout_s=${STATE_ADVANCE_TIMEOUT_S}"
  "--robot.left_gripper_observation_source=${LEFT_GRIPPER_OBSERVATION_SOURCE}"
  "--robot.right_gripper_observation_source=${RIGHT_GRIPPER_OBSERVATION_SOURCE}"
  --robot.left_gripper_observation_raw_closed=0.0
  --robot.left_gripper_observation_raw_open=100.0
  --robot.right_gripper_observation_raw_closed=100.0
  --robot.right_gripper_observation_raw_open=0.0
  --robot.left_gripper_action_raw_closed=100.0
  --robot.left_gripper_action_raw_open=0.0
  --robot.right_gripper_action_raw_closed=100.0
  --robot.right_gripper_action_raw_open=0.0
  --robot.left_gripper_training_command_force=80.0
  --robot.right_gripper_training_command_force=80.0
  "--robot.zmq_cameras=${ZMQ_CAMERAS}"
  --robot.rtsp_cameras={}
  "--policy.path=${POLICY_PATH}"
  "--policy.device=${POLICY_DEVICE}"
  "--dataset.repo_id=${DATASET_REPO_ID}"
  "--dataset.root=${DATASET_ROOT}"
  "--dataset.num_episodes=${NUM_EPISODES}"
  "--dataset.episode_time_s=${EPISODE_TIME_S}"
  "--dataset.reset_time_s=${RESET_TIME_S}"
  "--dataset.fps=${RECORD_FPS}"
  "--dataset.single_task=${SINGLE_TASK}"
  --dataset.push_to_hub=false
  "--dataset.video=${VIDEO}"
  "--dataset.vcodec=${VCODEC}"
  "--dataset.video_crf=${VIDEO_CRF}"
  "--dataset.keep_image_files=${KEEP_IMAGE_FILES}"
  "--dataset.video_encoding_batch_size=${VIDEO_ENCODING_BATCH_SIZE}"
  "--display_data=${DISPLAY_DATA}"
  "--display_compressed_images=${DISPLAY_COMPRESSED_IMAGES}"
  "--play_sounds=${PLAY_SOUNDS}"
  --resume=false
)
if [[ -n "${POLICY_N_ACTION_STEPS}" ]]; then
  CMD+=("--policy.n_action_steps=${POLICY_N_ACTION_STEPS}")
fi

echo "[jz_pin_timed/infer] mode=${EXECUTION} transport=${SEND_ACTION_TRANSPORT}"
echo "[jz_pin_timed/infer] policy=${POLICY_PATH} device=${POLICY_DEVICE}"
echo "[jz_pin_timed/infer] output=${DATASET_ROOT} episodes=${NUM_EPISODES} time_s=${EPISODE_TIME_S}"
echo "[jz_pin_timed/infer] state=udp://${STATE_BIND_IP}:${STATE_PORT} source=${ORIN_IP}"
echo "[jz_pin_timed/infer] cameras=zmq://${ORIN_IP}:5555,5556,5557"
echo "[jz_pin_timed/infer] command:"
infer_print_command "${CMD[@]}"

if [[ "${PRINT_COMMAND_ONLY:-0}" == "1" ]]; then
  echo "[jz_pin_timed/infer] PRINT_COMMAND_ONLY=1; no process was started"
  exit 0
fi

if [[ "${RUN_OFFLINE_PREFLIGHT:-1}" == "1" ]]; then
  POLICY_PATH="${POLICY_PATH}" \
  REFERENCE_DATASET_ROOT="${REFERENCE_DATASET_ROOT}" \
  REFERENCE_DATASET_REPO_ID="${REFERENCE_DATASET_REPO_ID}" \
  SAMPLE_INDICES=first \
  DEVICE="${POLICY_DEVICE}" \
    bash "${SCRIPT_DIR}/offline_infer.sh"
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
cd "${REPO_ROOT}"
exec "${CMD[@]}"
