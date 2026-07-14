#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMED_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${TIMED_ROOT}/lib/common.sh"

CONDA_ENV="${CONDA_ENV:-${LEROBOT_CONDA_ENV:-lerobot_flex}}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
DATASET_NAME="${DATASET_NAME:-jz_robot_pin_timed_color_diag_${RUN_STAMP}}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/tests/outputs/${DATASET_NAME}}"
DATASET_REPO_ID="${DATASET_REPO_ID:-local/${DATASET_NAME}}"
EPISODE_TIME_S="${EPISODE_TIME_S:-10}"
RECORD_FPS="${RECORD_FPS:-20}"
VCODEC="${VCODEC:-h264}"
VIDEO_CRF="${VIDEO_CRF:-18}"
DISPLAY_DATA="${DISPLAY_DATA:-true}"
MAX_INITIAL_JOINT_DELTA_RAD="${MAX_INITIAL_JOINT_DELTA_RAD:-10.0}"
MAX_JOINT_STEP_RAD="${MAX_JOINT_STEP_RAD:-10.0}"
LEFT_GRIPPER_OBSERVATION_SOURCE="${LEFT_GRIPPER_OBSERVATION_SOURCE:-measured_opening}"
RIGHT_GRIPPER_OBSERVATION_SOURCE="${RIGHT_GRIPPER_OBSERVATION_SOURCE:-commanded_opening}"

if [[ -e "${DATASET_ROOT}" ]]; then
  echo "[timed/color_diag] refusing to reuse existing dataset root: ${DATASET_ROOT}" >&2
  exit 2
fi

echo "[timed/color_diag] recording one ${EPISODE_TIME_S}s episode at ${RECORD_FPS}fps"
echo "[timed/color_diag] retained pre-encode PNG root: ${DATASET_ROOT}/images"
echo "[timed/color_diag] encoded MP4 root: ${DATASET_ROOT}/videos"
echo "[timed/color_diag] PNG frames are after Orin JPEG/X86 decode and before dataset H.264 encoding"
echo "[timed/color_diag] robot motion enabled with explicit diagnostic joint guards:" \
  "initial=${MAX_INITIAL_JOINT_DELTA_RAD}rad step=${MAX_JOINT_STEP_RAD}rad"
echo "[timed/color_diag] display_data=${DISPLAY_DATA} (requires a valid DISPLAY for on-screen cameras)"

NUM_EPISODES=1 \
RESUME=false \
VIDEO=true \
KEEP_IMAGE_FILES=true \
VIDEO_ENCODING_BATCH_SIZE=1 \
ZMQ_PRESET=jz_three_zmq \
RTSP_PRESET=none \
TIMING_SIDECAR=true \
REQUIRE_STATE_SOURCE_TIMING=true \
REQUIRE_STATE_ADVANCE_PER_OBSERVATION=true \
DISPLAY_DATA="${DISPLAY_DATA}" \
MAX_INITIAL_JOINT_DELTA_RAD="${MAX_INITIAL_JOINT_DELTA_RAD}" \
MAX_JOINT_STEP_RAD="${MAX_JOINT_STEP_RAD}" \
LEFT_GRIPPER_OBSERVATION_SOURCE="${LEFT_GRIPPER_OBSERVATION_SOURCE}" \
RIGHT_GRIPPER_OBSERVATION_SOURCE="${RIGHT_GRIPPER_OBSERVATION_SOURCE}" \
CONDA_ENV="${CONDA_ENV}" \
DATASET_NAME="${DATASET_NAME}" \
DATASET_ROOT="${DATASET_ROOT}" \
DATASET_REPO_ID="${DATASET_REPO_ID}" \
EPISODE_TIME_S="${EPISODE_TIME_S}" \
RESET_TIME_S=0 \
RECORD_FPS="${RECORD_FPS}" \
VCODEC="${VCODEC}" \
VIDEO_CRF="${VIDEO_CRF}" \
EXECUTION="${EXECUTION:-armed}" \
SEND_ACTION_TRANSPORT="${SEND_ACTION_TRANSPORT:-udp}" \
  bash "${TIMED_ROOT}/record.sh"

timed_make_python_cmd "${CONDA_ENV}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

"${PYTHON_CMD[@]}" "${SCRIPT_DIR}/compare_preencode_images.py" \
  --dataset-root "${DATASET_ROOT}" \
  --sample-count "${COLOR_DIAG_SAMPLE_COUNT:-5}" \
  --report-json "${DATASET_ROOT}/color_encoding_comparison.json"

echo "[timed/color_diag] completed; retained PNG files were not deleted"
