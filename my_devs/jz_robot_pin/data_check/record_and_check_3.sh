#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIN_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${PIN_ROOT}/lib/common.sh"

CONDA_ENV="${CONDA_ENV:-${LEROBOT_CONDA_ENV:-lerobot_flex}}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
DATASET_NAME="${DATASET_NAME:-jz_robot_pin_data_check_${RUN_STAMP}}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/tests/outputs/${DATASET_NAME}}"
DATASET_REPO_ID="${DATASET_REPO_ID:-local/${DATASET_NAME}}"
EPISODE_TIME_S="${EPISODE_TIME_S:-10}"
RESET_TIME_S="${RESET_TIME_S:-5}"
RECORD_FPS="${RECORD_FPS:-30}"
EXECUTION="${EXECUTION:-armed}"

if [[ -e "${DATASET_ROOT}" ]]; then
  echo "[record_and_check_3] refusing to reuse existing dataset root: ${DATASET_ROOT}" >&2
  echo "[record_and_check_3] choose a new DATASET_NAME or DATASET_ROOT" >&2
  exit 2
fi

echo "[record_and_check_3] dataset_root=${DATASET_ROOT}"
echo "[record_and_check_3] recording exactly 3 episodes"
echo "[record_and_check_3] episode_time_s=${EPISODE_TIME_S} reset_time_s=${RESET_TIME_S} fps=${RECORD_FPS}"
echo "[record_and_check_3] start the pin joystick publisher before this command and keep it publishing"

NUM_EPISODES=3 \
RESUME=false \
VIDEO=true \
VIDEO_ENCODING_BATCH_SIZE=3 \
RTSP_PRESET=jz_three_rtsp \
MAX_INITIAL_JOINT_DELTA_RAD="${MAX_INITIAL_JOINT_DELTA_RAD:-0.02}" \
MAX_JOINT_STEP_RAD="${MAX_JOINT_STEP_RAD:-0.02}" \
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
  bash "${PIN_ROOT}/record.sh"

pin_make_python_cmd "${CONDA_ENV}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

REPORT_JSON="${REPORT_JSON:-${DATASET_ROOT}/data_check_report.json}"
echo "[record_and_check_3] recording completed; validating all 3 episodes"
"${PYTHON_CMD[@]}" "${SCRIPT_DIR}/check_3_episodes.py" \
  --dataset-root "${DATASET_ROOT}" \
  --expected-episode-time-s "${EPISODE_TIME_S}" \
  --expected-fps "${RECORD_FPS}" \
  --report-json "${REPORT_JSON}"

echo "[record_and_check_3] PASS report=${REPORT_JSON}"
