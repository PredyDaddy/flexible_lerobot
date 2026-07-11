#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMED_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${TIMED_ROOT}/lib/common.sh"

CONDA_ENV="${CONDA_ENV:-${LEROBOT_CONDA_ENV:-lerobot_flex}}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
DATASET_NAME="${DATASET_NAME:-jz_robot_pin_timed_data_check_${RUN_STAMP}}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/tests/outputs/${DATASET_NAME}}"
DATASET_REPO_ID="${DATASET_REPO_ID:-local/${DATASET_NAME}}"
EPISODE_TIME_S="${EPISODE_TIME_S:-10}"
RESET_TIME_S="${RESET_TIME_S:-5}"
RECORD_FPS="${RECORD_FPS:-30}"
EXECUTION="${EXECUTION:-armed}"
SEND_ACTION_TRANSPORT="${SEND_ACTION_TRANSPORT:-udp}"
VCODEC="${VCODEC:-h264}"
VIDEO_CRF="${VIDEO_CRF:-18}"
MAX_INITIAL_JOINT_DELTA_RAD="${MAX_INITIAL_JOINT_DELTA_RAD:-10.0}"
MAX_JOINT_STEP_RAD="${MAX_JOINT_STEP_RAD:-10.0}"
MAX_TIMING_CAMERA_AGE_MS="${MAX_TIMING_CAMERA_AGE_MS:-${RTSP_STALE_FRAME_TIMEOUT_MS:-1000.0}}"
DEFAULT_TIMING_CAMERA_STATE_SKEW_MS="${MAX_CAMERA_STATE_RECEIVE_SKEW_MS:-100.0}"
MAX_TIMING_CAMERA_STATE_SKEW_MS="${MAX_TIMING_CAMERA_STATE_SKEW_MS:-${DEFAULT_TIMING_CAMERA_STATE_SKEW_MS}}"
MAX_TIMING_SOURCE_AGE_MS="${MAX_TIMING_SOURCE_AGE_MS:-50.0}"
MAX_TIMING_SOURCE_SKEW_MS="${MAX_TIMING_SOURCE_SKEW_MS:-20.0}"

if [[ -e "${DATASET_ROOT}" ]]; then
  echo "[timed/record_and_check_3] refusing to reuse existing dataset root: ${DATASET_ROOT}" >&2
  echo "[timed/record_and_check_3] choose a new DATASET_NAME or DATASET_ROOT" >&2
  exit 2
fi

echo "[timed/record_and_check_3] dataset_root=${DATASET_ROOT}"
echo "[timed/record_and_check_3] recording exactly 3 episodes"
echo "[timed/record_and_check_3] episode_time_s=${EPISODE_TIME_S}" \
  "reset_time_s=${RESET_TIME_S} fps=${RECORD_FPS} video=${VCODEC}/crf${VIDEO_CRF}"
echo "[timed/record_and_check_3] joint delta guards: initial=${MAX_INITIAL_JOINT_DELTA_RAD}rad" \
  "step=${MAX_JOINT_STEP_RAD}rad"
echo "[timed/record_and_check_3] source timing: required on every recorded frame" \
  "max_age=${MAX_TIMING_SOURCE_AGE_MS}ms max_skew=${MAX_TIMING_SOURCE_SKEW_MS}ms"
echo "[timed/record_and_check_3] start the pin joystick publisher before this command" \
  "and keep it publishing"

NUM_EPISODES=3 \
RESUME=false \
VIDEO=true \
VCODEC="${VCODEC}" \
VIDEO_CRF="${VIDEO_CRF}" \
VIDEO_ENCODING_BATCH_SIZE=3 \
RTSP_PRESET=jz_three_rtsp \
TIMING_SIDECAR=true \
REQUIRE_STATE_SOURCE_TIMING=true \
MAX_INITIAL_JOINT_DELTA_RAD="${MAX_INITIAL_JOINT_DELTA_RAD}" \
MAX_JOINT_STEP_RAD="${MAX_JOINT_STEP_RAD}" \
TARGET_ACTION_CONNECT_TIMEOUT_S="${TARGET_ACTION_CONNECT_TIMEOUT_S:-5.0}" \
TARGET_ACTION_STALE_POLICY="${TARGET_ACTION_STALE_POLICY:-raise}" \
CONDA_ENV="${CONDA_ENV}" \
DATASET_NAME="${DATASET_NAME}" \
DATASET_ROOT="${DATASET_ROOT}" \
DATASET_REPO_ID="${DATASET_REPO_ID}" \
EPISODE_TIME_S="${EPISODE_TIME_S}" \
RESET_TIME_S="${RESET_TIME_S}" \
RECORD_FPS="${RECORD_FPS}" \
EXECUTION="${EXECUTION}" \
SEND_ACTION_TRANSPORT="${SEND_ACTION_TRANSPORT}" \
  bash "${TIMED_ROOT}/record.sh"

timed_make_python_cmd "${CONDA_ENV}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

DATA_REPORT_JSON="${DATA_REPORT_JSON:-${DATASET_ROOT}/data_check_report.json}"
TIMING_REPORT_JSON="${TIMING_REPORT_JSON:-${DATASET_ROOT}/timing_check_report.json}"

echo "[timed/record_and_check_3] recording completed; validating data and timing sidecars"
"${PYTHON_CMD[@]}" "${SCRIPT_DIR}/check_3_episodes.py" \
  --dataset-root "${DATASET_ROOT}" \
  --expected-episode-time-s "${EPISODE_TIME_S}" \
  --expected-fps "${RECORD_FPS}" \
  --max-initial-joint-delta-rad "${MAX_INITIAL_JOINT_DELTA_RAD}" \
  --max-action-joint-step-rad "${MAX_JOINT_STEP_RAD}" \
  --report-json "${DATA_REPORT_JSON}"

TIMING_CHECK_ARGS=(
  --dataset-root "${DATASET_ROOT}"
  --expected-codec "${VCODEC}"
  --expected-crf "${VIDEO_CRF}"
  --expected-camera-fps "${RECORD_FPS}"
  --expected-command-mode "${EXECUTION}"
  --expected-command-transport "${SEND_ACTION_TRANSPORT}"
  --expected-action-key-count 18
  --require-source-timing
  --max-source-age-ms "${MAX_TIMING_SOURCE_AGE_MS}"
  --max-source-skew-ms "${MAX_TIMING_SOURCE_SKEW_MS}"
  --max-camera-age-ms "${MAX_TIMING_CAMERA_AGE_MS}"
  --max-camera-state-skew-ms "${MAX_TIMING_CAMERA_STATE_SKEW_MS}"
  --report-json "${TIMING_REPORT_JSON}"
)
if [[ -n "${MAX_TIMING_REUSE_FRACTION:-}" ]]; then
  TIMING_CHECK_ARGS+=(--max-reuse-fraction "${MAX_TIMING_REUSE_FRACTION}")
fi
if [[ -n "${MAX_TIMING_STATE_REUSE_FRACTION:-}" ]]; then
  TIMING_CHECK_ARGS+=(--max-state-reuse-fraction "${MAX_TIMING_STATE_REUSE_FRACTION}")
fi
"${PYTHON_CMD[@]}" "${SCRIPT_DIR}/check_timing.py" "${TIMING_CHECK_ARGS[@]}"

echo "[timed/record_and_check_3] PASS data_report=${DATA_REPORT_JSON}"
echo "[timed/record_and_check_3] PASS timing_report=${TIMING_REPORT_JSON}"
