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
COMMAND_PORT="${COMMAND_PORT:-39020}"
CONFIG="${CONFIG:-$ROOT_DIR/udp_test/test_scripts/arm_side/orin_phase3_executor_config_armed_hold.yaml}"
EXECUTION="${EXECUTION:-armed}"
PYTHON_CMD="${PYTHON_CMD:-conda run --no-capture-output -n lerobot python}"
TAIL="${TAIL:-0}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-15}"

cd "$ROOT_DIR"

log() {
  echo "[all/start_replay] $*"
}

cleanup_on_error() {
  local exit_code=$?
  log "startup failed with exit code $exit_code; stopping replay services"
  bash "$ROOT_DIR/udp_test/all/stop_replay.sh" || true
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

if [[ "$EXECUTION" == "armed" ]]; then
  export JZ_UDP_EXECUTOR_ARMED="${JZ_UDP_EXECUTOR_ARMED:-1}"
else
  log "unsupported EXECUTION=$EXECUTION. start_replay.sh is for armed replay only."
  exit 2
fi

log "mode=replay"
log "ORIN_IP=$ORIN_IP X86_IP=$X86_IP STATE_PORT=$STATE_PORT COMMAND_PORT=$COMMAND_PORT"
log "CONFIG=$CONFIG"
log "EXECUTION=$EXECUTION JZ_UDP_EXECUTOR_ARMED=${JZ_UDP_EXECUTOR_ARMED:-}"
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

log "starting Phase 3 command executor..."
CONFIG="$CONFIG" \
EXECUTION=armed \
JZ_UDP_EXECUTOR_ARMED=1 \
AUTO_TAIL=0 \
ORIN_IP="$ORIN_IP" \
X86_IP="$X86_IP" \
COMMAND_PORT="$COMMAND_PORT" \
PYTHON_CMD="$PYTHON_CMD" \
  bash "$ORIN_ARM_DIR/start_phase3_executor.sh"
sleep 0.5
ensure_alive "orin_phase3_command_executor" \
  "$PID_DIR/orin_phase3_command_executor.pid" \
  "$LOG_DIR/orin_phase3_command_executor.log"
wait_for_log_pattern \
  "orin_phase3_command_executor" \
  "$PID_DIR/orin_phase3_command_executor.pid" \
  "$LOG_DIR/orin_phase3_command_executor.log" \
  "PHASE3 COMMAND EXECUTOR ARMED" \
  "$READY_TIMEOUT_S"
wait_for_log_pattern \
  "orin_phase3_command_executor" \
  "$PID_DIR/orin_phase3_command_executor.pid" \
  "$LOG_DIR/orin_phase3_command_executor.log" \
  "port=$COMMAND_PORT" \
  "$READY_TIMEOUT_S"

log "replay services ready."
log "stop with: bash udp_test/all/stop_replay.sh"
trap - ERR

if [[ "$TAIL" == "1" ]]; then
  log "following logs now. Ctrl-C exits tail only; use stop_replay.sh to stop services."
  tail -f "$LOG_DIR/ros_state_udp_bridge.log" "$LOG_DIR/orin_phase3_command_executor.log"
fi
