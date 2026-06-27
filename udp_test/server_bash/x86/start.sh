#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LOG_DIR="$ROOT_DIR/udp_test/server_bash/x86/logs"
PID_DIR="$ROOT_DIR/udp_test/server_bash/x86/pids"
mkdir -p "$LOG_DIR" "$PID_DIR"

OBS_COUNT="${OBS_COUNT:-0}"
OBS_HZ="${OBS_HZ:-5}"
ROBOT_CONFIG="${ROBOT_CONFIG:-$ROOT_DIR/src/lerobot/configs/robot/jz_robot_udp_three_rtsp.yaml}"
SKIP_CAMERAS="${SKIP_CAMERAS:-0}"
PYTHON_CMD="${PYTHON_CMD:-conda run --no-capture-output -n lerobot_flex python}"
read -r -a PYTHON_ARGS <<< "$PYTHON_CMD"

cd "$ROOT_DIR"

PID_FILE="$PID_DIR/jz_robot_udp_observation_check.pid"
bash "$ROOT_DIR/udp_test/server_bash/x86/stop.sh"

if [[ "$OBS_COUNT" == "0" ]]; then
  OBS_COUNT_ARG="1000000000"
else
  OBS_COUNT_ARG="$OBS_COUNT"
fi

echo "[x86/start] READONLY ONLY"
echo "[x86/start] starting JZRobotUDP observation monitor"
EXTRA_ARGS=()
if [[ "$SKIP_CAMERAS" == "1" ]]; then
  EXTRA_ARGS+=(--skip-cameras)
fi
nohup "${PYTHON_ARGS[@]}" udp_test/test_scripts/x86_side/x86_jz_robot_udp_observation_check.py \
  --robot-config "$ROBOT_CONFIG" \
  --count "$OBS_COUNT_ARG" \
  --hz "$OBS_HZ" \
  --print-every 1 \
  --continue-on-stale \
  "${EXTRA_ARGS[@]}" \
  > "$LOG_DIR/jz_robot_udp_observation_check.log" 2>&1 &
echo "$!" > "$PID_FILE"

echo "[x86/start] started pid=$(cat "$PID_FILE")"
echo "[x86/start] log: $LOG_DIR/jz_robot_udp_observation_check.log"
