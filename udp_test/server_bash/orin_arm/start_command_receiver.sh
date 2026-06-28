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
PYTHON_CMD="${PYTHON_CMD:-conda run --no-capture-output -n lerobot python}"
AUTO_TAIL="${AUTO_TAIL:-1}"
read -r -a PYTHON_ARGS <<< "$PYTHON_CMD"

cd "$ROOT_DIR"

PID_FILE="$PID_DIR/orin_udp_command_receiver.pid"
bash "$ROOT_DIR/udp_test/server_bash/orin_arm/stop_command_receiver.sh"

echo "PHASE2 COMMAND DRY-RUN ONLY"
echo "NOT publishing ROS command topics"
echo "[orin_arm/start_command_receiver] starting command dry-run receiver: $ORIN_IP:$COMMAND_PORT allowed=$X86_IP"
nohup "${PYTHON_ARGS[@]}" udp_test/test_scripts/arm_side/orin_udp_command_receiver.py \
  --bind-ip "$ORIN_IP" \
  --port "$COMMAND_PORT" \
  --allowed-sender-ip "$X86_IP" \
  --count "$COMMAND_COUNT" \
  --print-every "$PRINT_EVERY" \
  > "$LOG_DIR/orin_udp_command_receiver.log" 2>&1 &
echo "$!" > "$PID_FILE"

echo "[orin_arm/start_command_receiver] started pid=$(cat "$PID_FILE")"
echo "[orin_arm/start_command_receiver] log: $LOG_DIR/orin_udp_command_receiver.log"
if [[ "$AUTO_TAIL" == "1" ]]; then
  echo "[orin_arm/start_command_receiver] following log now. Ctrl-C only exits tail; use stop_command_receiver.sh to stop service."
  tail -f "$LOG_DIR/orin_udp_command_receiver.log"
fi
