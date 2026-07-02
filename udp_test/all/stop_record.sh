#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ORIN_ARM_DIR="$ROOT_DIR/udp_test/server_bash/orin_arm"
PID_DIR="$ORIN_ARM_DIR/pids"
PATTERN="udp_test/test_scripts/arm_side/orin_ros_target_action_udp_bridge.py"
TARGET_ACTION_ONLY="${1:-}"

log() {
  echo "[all/stop_record] $*"
}

stop_pid() {
  local pid="$1"
  if [[ "$pid" == "$$" || "$pid" == "$PPID" ]]; then
    return
  fi
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    log "stopping target-action pid=$pid"
    kill "$pid" 2>/dev/null || true
  fi
}

stop_pid_file() {
  local pid_file="$1"
  local pid
  if [[ ! -f "$pid_file" ]]; then
    return
  fi
  pid="$(cat "$pid_file")"
  stop_pid "$pid"
  rm -f "$pid_file"
}

cd "$ROOT_DIR"

log "stopping ROS target-action UDP bridge"
stop_pid_file "$PID_DIR/ros_target_action_udp_bridge.pid"
while read -r pid _cmd; do
  stop_pid "$pid"
done < <(pgrep -af "$PATTERN" || true)
sleep 0.5
while read -r pid _cmd; do
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    log "force stopping target-action pid=$pid"
    kill -9 "$pid" 2>/dev/null || true
  fi
done < <(pgrep -af "$PATTERN" || true)
log "target-action bridge stopped"

if [[ "$TARGET_ACTION_ONLY" == "--target-action-only" ]]; then
  exit 0
fi

log "stopping ROS state UDP bridge"
bash "$ORIN_ARM_DIR/stop.sh"

log "status after stop:"
bash "$ORIN_ARM_DIR/status.sh"
