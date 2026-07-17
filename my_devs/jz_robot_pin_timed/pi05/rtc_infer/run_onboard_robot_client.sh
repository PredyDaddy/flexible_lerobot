#!/usr/bin/env bash
set -euo pipefail

# Real-machine armed client launcher. The default is one short, low-rate
# single-step run against a final checkpoint. The known 010470 checkpoint and
# RTC each require their own explicit confirmation.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CONDA_ROOT="${CONDA_ROOT:-/home/luzhuang/miniconda3}"
export CONDA_ENV="${CONDA_ENV:-lerobot_flex}"
export CONDA_PYTHON="${CONDA_PYTHON:-${CONDA_ROOT}/envs/${CONDA_ENV}/bin/python}"

source "${SCRIPT_DIR}/common.sh"

[[ "$#" -eq 0 ]] \
  || rtc_die "run_onboard_robot_client.sh accepts no CLI arguments; configure it with environment variables"

ONBOARD_MODE="${ONBOARD_MODE:-single_step}"
rtc_require_choice ONBOARD_MODE "${ONBOARD_MODE}" single_step rtc
ONBOARD_CHECKPOINT="${ONBOARD_CHECKPOINT:-015705}"
rtc_require_choice ONBOARD_CHECKPOINT "${ONBOARD_CHECKPOINT}" 015705 010470
rtc_require_armed_confirmation

DISABLE_JOINT_DELTA_CHECKS="${JZ_PI05_DISABLE_JOINT_DELTA_CHECKS:-0}"
rtc_require_choice JZ_PI05_DISABLE_JOINT_DELTA_CHECKS "${DISABLE_JOINT_DELTA_CHECKS}" 0 1
if [[ "${DISABLE_JOINT_DELTA_CHECKS}" == "1" ]]; then
  [[ "${I_UNDERSTAND_JOINT_DELTA_CHECKS_ARE_DISABLED:-}" == "1" ]] \
    || rtc_die \
      "JZ_PI05_DISABLE_JOINT_DELTA_CHECKS=1 requires I_UNDERSTAND_JOINT_DELTA_CHECKS_ARE_DISABLED=1"
  JOINT_DELTA_CHECK_LABEL=disabled
else
  [[ "${I_UNDERSTAND_JOINT_DELTA_CHECKS_ARE_DISABLED:-0}" != "1" ]] \
    || rtc_die \
      "I_UNDERSTAND_JOINT_DELTA_CHECKS_ARE_DISABLED=1 requires JZ_PI05_DISABLE_JOINT_DELTA_CHECKS=1"
  JOINT_DELTA_CHECK_LABEL=enabled
fi

if [[ "${ONBOARD_CHECKPOINT}" == "010470" ]]; then
  [[ "${JZ_PI05_INTERMEDIATE_010470_CONFIRMED:-}" == "1" ]] \
    || rtc_die \
      "checkpoint 010470 requires JZ_PI05_INTERMEDIATE_010470_CONFIRMED=1"
  INTERMEDIATE_010470_CONFIRMED=1
  CHECKPOINT_LABEL=intermediate_010470
else
  [[ "${JZ_PI05_INTERMEDIATE_010470_CONFIRMED:-0}" != "1" ]] \
    || rtc_die \
      "JZ_PI05_INTERMEDIATE_010470_CONFIRMED=1 requires ONBOARD_CHECKPOINT=010470"
  INTERMEDIATE_010470_CONFIRMED=0
  CHECKPOINT_LABEL=final_015705
fi

SERVER_URL="${SERVER_URL:-http://127.0.0.1:8088}"
ORIN_IP="${ORIN_IP:-192.168.1.81}"
STATE_BIND_IP="${STATE_BIND_IP:-0.0.0.0}"
STATE_PORT="${STATE_PORT:-39010}"
COMMAND_PORT="${COMMAND_PORT:-39020}"
TASK="jz robot pin timed vr teleoperation"
EMPTY_QUEUE_STRATEGY=stop

require_onboard_fps() {
  local name="$1"
  local value="$2"
  [[ "${value}" =~ ^[0-9]+$ ]] \
    || rtc_die "${name} must be an integer in 1..20, got ${value}"
  (( value >= 1 && value <= 20 )) \
    || rtc_die "${name} must be an integer in 1..20, got ${value}"
}

require_onboard_run_time() {
  local value="$1"
  [[ "${value}" =~ ^[0-9]+$ ]] \
    || rtc_die "ONBOARD_RUN_TIME_S must be an integer in 1..300, got ${value}"
  (( value >= 1 && value <= 300 )) \
    || rtc_die "ONBOARD_RUN_TIME_S must be an integer in 1..300, got ${value}"
}

if [[ "${ONBOARD_MODE}" == "single_step" ]]; then
  SENSOR_FPS="${ONBOARD_SENSOR_FPS:-5}"
  CONTROL_FPS="${ONBOARD_CONTROL_FPS:-5}"
  RUN_TIME_S="${ONBOARD_RUN_TIME_S:-1}"
  QUEUE_LOW_WATERMARK=30
  MAX_QUEUE_SIZE=50
  RTC_EXECUTION_HORIZON=10
else
  [[ "${JZ_PI05_SINGLE_STEP_ARMED_PASSED:-}" == "1" ]] \
    || rtc_die "RTC onboard inference requires JZ_PI05_SINGLE_STEP_ARMED_PASSED=1 after the armed single-step check"
  SENSOR_FPS="${ONBOARD_SENSOR_FPS:-20}"
  CONTROL_FPS="${ONBOARD_CONTROL_FPS:-20}"
  RUN_TIME_S="${ONBOARD_RUN_TIME_S:-10}"
  QUEUE_LOW_WATERMARK=30
  MAX_QUEUE_SIZE=50
  RTC_EXECUTION_HORIZON=10
fi

require_onboard_fps ONBOARD_SENSOR_FPS "${SENSOR_FPS}"
require_onboard_fps ONBOARD_CONTROL_FPS "${CONTROL_FPS}"
require_onboard_run_time "${RUN_TIME_S}"

echo "[jz/pi05/onboard/client] REAL ROBOT ARMED mode=${ONBOARD_MODE}"
echo "[jz/pi05/onboard/client] checkpoint=${CHECKPOINT_LABEL}"
echo "[jz/pi05/onboard/client] joint_delta_checks=${JOINT_DELTA_CHECK_LABEL}"
echo "[jz/pi05/onboard/client] task=${TASK}"
echo "[jz/pi05/onboard/client] server=${SERVER_URL} orin=${ORIN_IP}"
echo "[jz/pi05/onboard/client] sensor_fps=${SENSOR_FPS} control_fps=${CONTROL_FPS} run_time_s=${RUN_TIME_S}"

CLIENT_ENV=(
  "MODE=${ONBOARD_MODE}"
  "EXECUTION=armed"
  "CONFIG_ONLY=false"
  "HEALTH_ONLY=false"
  "CONNECT_SMOKE=false"
  "INFERENCE_SMOKE=false"
  "JZ_PI05_INTERMEDIATE_010470_CONFIRMED=${INTERMEDIATE_010470_CONFIRMED}"
  "JZ_PI05_DISABLE_JOINT_DELTA_CHECKS=${DISABLE_JOINT_DELTA_CHECKS}"
  "I_UNDERSTAND_JOINT_DELTA_CHECKS_ARE_DISABLED=${I_UNDERSTAND_JOINT_DELTA_CHECKS_ARE_DISABLED:-0}"
  "SERVER_URL=${SERVER_URL}"
  "ORIN_IP=${ORIN_IP}"
  "STATE_BIND_IP=${STATE_BIND_IP}"
  "STATE_PORT=${STATE_PORT}"
  "COMMAND_PORT=${COMMAND_PORT}"
  "TASK=${TASK}"
  "SENSOR_FPS=${SENSOR_FPS}"
  "CONTROL_FPS=${CONTROL_FPS}"
  "RUN_TIME_S=${RUN_TIME_S}"
  "QUEUE_LOW_WATERMARK=${QUEUE_LOW_WATERMARK}"
  "MAX_QUEUE_SIZE=${MAX_QUEUE_SIZE}"
  "RTC_EXECUTION_HORIZON=${RTC_EXECUTION_HORIZON}"
  "EMPTY_QUEUE_STRATEGY=${EMPTY_QUEUE_STRATEGY}"
)

exec env "${CLIENT_ENV[@]}" bash "${SCRIPT_DIR}/run_client.sh"
