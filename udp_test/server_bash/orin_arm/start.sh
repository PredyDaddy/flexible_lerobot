#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LOG_DIR="$ROOT_DIR/udp_test/server_bash/orin_arm/logs"
PID_DIR="$ROOT_DIR/udp_test/server_bash/orin_arm/pids"
mkdir -p "$LOG_DIR" "$PID_DIR"

ORIN_IP="${ORIN_IP:-192.168.1.81}"
X86_IP="${X86_IP:-192.168.1.106}"
STATE_PORT="${STATE_PORT:-39010}"
STATE_HZ="${STATE_HZ:-20}"
PYTHON_CMD="${PYTHON_CMD:-python}"
RUN_READINESS="${RUN_READINESS:-0}"
read -r -a PYTHON_ARGS <<< "$PYTHON_CMD"

cd "$ROOT_DIR"

echo "[orin_arm/start] READONLY ONLY"
if [[ "$RUN_READINESS" == "1" ]]; then
  echo "[orin_arm/start] local readiness check..."
  "${PYTHON_ARGS[@]}" udp_test/local_test/local_robot_readiness_check.py --skip-cameras | tee "$LOG_DIR/local_readiness.log"
else
  echo "[orin_arm/start] skip local readiness check. Set RUN_READINESS=1 to run it."
fi

echo "[orin_arm/start] starting ROS state UDP bridge: $ORIN_IP -> $X86_IP:$STATE_PORT"
nohup "${PYTHON_ARGS[@]}" udp_test/test_scripts/arm_side/orin_ros_state_udp_bridge.py \
  --bind-ip "$ORIN_IP" \
  --target-ip "$X86_IP" \
  --target-port "$STATE_PORT" \
  --hz "$STATE_HZ" \
  --print-every "$STATE_HZ" \
  > "$LOG_DIR/ros_state_udp_bridge.log" 2>&1 &
echo "$!" > "$PID_DIR/ros_state_udp_bridge.pid"

echo "[orin_arm/start] started pid=$(cat "$PID_DIR/ros_state_udp_bridge.pid")"
echo "[orin_arm/start] log: $LOG_DIR/ros_state_udp_bridge.log"
