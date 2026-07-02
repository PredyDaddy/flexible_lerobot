#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ORIN_ARM_DIR="$ROOT_DIR/udp_test/server_bash/orin_arm"
LOG_DIR="$ORIN_ARM_DIR/logs"
PID_DIR="$ORIN_ARM_DIR/pids"
mkdir -p "$LOG_DIR" "$PID_DIR"

ORIN_IP="${ORIN_IP:-192.168.1.81}"
X86_IP="${X86_IP:-192.168.1.106}"
STATE_PORT="${STATE_PORT:-39010}"
STATE_HZ="${STATE_HZ:-20}"
TARGET_ACTION_PORT="${TARGET_ACTION_PORT:-39030}"
TARGET_ACTION_HZ="${TARGET_ACTION_HZ:-30}"
TARGET_ACTION_PRINT_EVERY="${TARGET_ACTION_PRINT_EVERY:-30}"
PYTHON_CMD="${PYTHON_CMD:-conda run --no-capture-output -n lerobot python}"
TAIL="${TAIL:-0}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-15}"
read -r -a PYTHON_ARGS <<< "$PYTHON_CMD"

TARGET_ACTION_PID_FILE="$PID_DIR/ros_target_action_udp_bridge.pid"
TARGET_ACTION_LOG="$LOG_DIR/ros_target_action_udp_bridge.log"

cd "$ROOT_DIR"

log() {
  echo "[all/start_record] $*"
}

cleanup_on_error() {
  local exit_code=$?
  log "startup failed with exit code $exit_code; stopping record services"
  bash "$ROOT_DIR/udp_test/all/stop_record.sh" || true
  exit "$exit_code"
}

trap cleanup_on_error ERR

show_log_tail() {
  local log_file="$1"
  if [[ -f "$log_file" ]]; then
    log "last lines from $log_file:"
    tail -n 20 "$log_file" || true
  else
    log "log file not created yet: $log_file"
  fi
}

ensure_alive() {
  local name="$1"
  local pid_file="$2"
  local log_file="$3"
  local pid

  if [[ ! -f "$pid_file" ]]; then
    log "$name did not create pid file: $pid_file"
    show_log_tail "$log_file"
    return 1
  fi

  pid="$(cat "$pid_file")"
  if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
    log "$name failed to stay running. pid=${pid:-missing}"
    show_log_tail "$log_file"
    rm -f "$pid_file"
    return 1
  fi

  log "$name OK pid=$pid"
  log "$name log: $log_file"
  show_log_tail "$log_file"
}

wait_for_log_pattern() {
  local name="$1"
  local pid_file="$2"
  local log_file="$3"
  local pattern="$4"
  local timeout_s="$5"
  local pid
  local deadline

  pid="$(cat "$pid_file")"
  deadline=$((SECONDS + timeout_s))
  while (( SECONDS < deadline )); do
    if ! kill -0 "$pid" 2>/dev/null; then
      log "$name exited before readiness pattern appeared: $pattern"
      show_log_tail "$log_file"
      rm -f "$pid_file"
      return 1
    fi
    if [[ -f "$log_file" ]] && grep -q "$pattern" "$log_file"; then
      log "$name confirmed ready from log pattern: $pattern"
      return
    fi
    sleep 0.5
  done

  log "$name is running but readiness pattern did not appear within ${timeout_s}s: $pattern"
  show_log_tail "$log_file"
  return 1
}

log "mode=record"
log "ORIN_IP=$ORIN_IP X86_IP=$X86_IP STATE_PORT=$STATE_PORT TARGET_ACTION_PORT=$TARGET_ACTION_PORT"
log "PYTHON_CMD=$PYTHON_CMD"
log "READY_TIMEOUT_S=$READY_TIMEOUT_S"
log "logs: $LOG_DIR"
log "pids: $PID_DIR"

log "starting ROS state UDP bridge..."
ORIN_IP="$ORIN_IP" \
X86_IP="$X86_IP" \
STATE_PORT="$STATE_PORT" \
STATE_HZ="$STATE_HZ" \
PYTHON_CMD="$PYTHON_CMD" \
AUTO_TAIL=0 \
  bash "$ORIN_ARM_DIR/start.sh"
sleep 0.5
ensure_alive "ros_state_udp_bridge" "$PID_DIR/ros_state_udp_bridge.pid" "$LOG_DIR/ros_state_udp_bridge.log"
wait_for_log_pattern \
  "ros_state_udp_bridge" \
  "$PID_DIR/ros_state_udp_bridge.pid" \
  "$LOG_DIR/ros_state_udp_bridge.log" \
  "sent seq=" \
  "$READY_TIMEOUT_S"

log "stopping any existing target-action UDP bridge..."
bash "$ROOT_DIR/udp_test/all/stop_record.sh" --target-action-only

log "starting ROS target-action UDP bridge..."
nohup "${PYTHON_ARGS[@]}" udp_test/test_scripts/arm_side/orin_ros_target_action_udp_bridge.py \
  --target-ip "$X86_IP" \
  --target-port "$TARGET_ACTION_PORT" \
  --bind-ip "$ORIN_IP" \
  --hz "$TARGET_ACTION_HZ" \
  --print-every "$TARGET_ACTION_PRINT_EVERY" \
  > "$TARGET_ACTION_LOG" 2>&1 &
echo "$!" > "$TARGET_ACTION_PID_FILE"
sleep 0.5
ensure_alive "ros_target_action_udp_bridge" "$TARGET_ACTION_PID_FILE" "$TARGET_ACTION_LOG"
wait_for_log_pattern \
  "ros_target_action_udp_bridge" \
  "$TARGET_ACTION_PID_FILE" \
  "$TARGET_ACTION_LOG" \
  "sent seq=" \
  "$READY_TIMEOUT_S"

log "record services ready."
log "stop with: bash udp_test/all/stop_record.sh"
trap - ERR

if [[ "$TAIL" == "1" ]]; then
  log "following logs now. Ctrl-C exits tail only; use stop_record.sh to stop services."
  tail -f "$LOG_DIR/ros_state_udp_bridge.log" "$TARGET_ACTION_LOG"
fi
