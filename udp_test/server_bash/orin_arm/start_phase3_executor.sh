#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LOG_DIR="$ROOT_DIR/udp_test/server_bash/orin_arm/logs"
PID_DIR="$ROOT_DIR/udp_test/server_bash/orin_arm/pids"
mkdir -p "$LOG_DIR" "$PID_DIR"

ORIN_IP="${ORIN_IP:-192.168.1.81}"
X86_IP="${X86_IP:-192.168.1.106}"
COMMAND_PORT="${COMMAND_PORT:-39020}"
COMMAND_COUNT="${COMMAND_COUNT:-0}"
PRINT_EVERY="${PRINT_EVERY:-1}"
EXECUTION="${EXECUTION:-dry_run}"
RESET_CONTROL_PORT="${RESET_CONTROL_PORT:-39040}"
RESET_CONTROL_ENABLED="${RESET_CONTROL_ENABLED:-1}"
RESET_ACTIONS_PATH="${RESET_ACTIONS_PATH:-/home/data/test/workspace/teleop_ws/install/multi_robot_choreographer/share/multi_robot_choreographer/config/actions/actions.yaml}"
CONFIG="${CONFIG:-$ROOT_DIR/udp_test/test_scripts/arm_side/orin_phase3_executor_config.yaml}"
PYTHON_CMD="${PYTHON_CMD:-conda run --no-capture-output -n lerobot python}"
AUTO_TAIL="${AUTO_TAIL:-1}"
read -r -a PYTHON_ARGS <<< "$PYTHON_CMD"

cd "$ROOT_DIR"

PID_FILE="$PID_DIR/orin_phase3_command_executor.pid"
bash "$ROOT_DIR/udp_test/server_bash/orin_arm/stop_phase3_executor.sh"

COMMON_ARGS=(
  udp_test/test_scripts/arm_side/orin_phase3_command_executor.py
  --config "$CONFIG"
  --bind-ip "$ORIN_IP"
  --port "$COMMAND_PORT"
  --allowed-sender-ip "$X86_IP"
  --execution "$EXECUTION"
  --count "$COMMAND_COUNT"
  --print-every "$PRINT_EVERY"
)

if [[ "$EXECUTION" == "armed" ]]; then
  if [[ "${JZ_UDP_EXECUTOR_ARMED:-}" != "1" ]]; then
    echo "PHASE3 COMMAND EXECUTOR ARMED"
    echo "[orin_arm/start_phase3_executor] refusing armed start: JZ_UDP_EXECUTOR_ARMED=1 is required"
    exit 2
  fi
  echo "PHASE3 COMMAND EXECUTOR ARMED"
  echo "WILL publish ROS command topics"
  echo "emergency stop is physical fallback, not a replacement for software limits"
  COMMON_ARGS+=(--i-understand-this-publishes-robot-commands)
  if [[ "$RESET_CONTROL_ENABLED" != "1" ]]; then
    echo "[orin_arm/start_phase3_executor] refusing armed start: RESET_CONTROL_ENABLED=1 is required"
    exit 2
  fi
  COMMON_ARGS+=(
    --reset-control-enabled
    --reset-control-port "$RESET_CONTROL_PORT"
    --reset-control-allowed-sender-ip "$X86_IP"
    --reset-actions-path "$RESET_ACTIONS_PATH"
  )
  echo "[orin_arm/start_phase3_executor] reset_control=$ORIN_IP:$RESET_CONTROL_PORT allowed=$X86_IP"
  echo "[orin_arm/start_phase3_executor] reset allowlist=VR_inital_no_waist"
elif [[ "$EXECUTION" == "dry_run" ]]; then
  echo "PHASE3 COMMAND EXECUTOR DRY-RUN"
  echo "NOT publishing ROS command topics"
  echo "robot should not move"
else
  echo "[orin_arm/start_phase3_executor] unsupported EXECUTION=$EXECUTION"
  exit 2
fi

echo "[orin_arm/start_phase3_executor] starting phase3 executor: $ORIN_IP:$COMMAND_PORT allowed=$X86_IP execution=$EXECUTION"
nohup "${PYTHON_ARGS[@]}" "${COMMON_ARGS[@]}" \
  > "$LOG_DIR/orin_phase3_command_executor.log" 2>&1 &
echo "$!" > "$PID_FILE"
sleep 0.5
if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "[orin_arm/start_phase3_executor] executor exited during startup"
  tail -n 80 "$LOG_DIR/orin_phase3_command_executor.log" || true
  rm -f "$PID_FILE"
  exit 1
fi

echo "[orin_arm/start_phase3_executor] started pid=$(cat "$PID_FILE")"
echo "[orin_arm/start_phase3_executor] log: $LOG_DIR/orin_phase3_command_executor.log"
if [[ "$AUTO_TAIL" == "1" ]]; then
  echo "[orin_arm/start_phase3_executor] following log now. Ctrl-C only exits tail; use stop_phase3_executor.sh to stop service."
  tail -f "$LOG_DIR/orin_phase3_command_executor.log"
fi
