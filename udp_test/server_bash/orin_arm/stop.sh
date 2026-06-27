#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PID_DIR="$ROOT_DIR/udp_test/server_bash/orin_arm/pids"

stop_pid_file() {
  local pid_file="$1"
  if [[ ! -f "$pid_file" ]]; then
    return
  fi
  local pid
  pid="$(cat "$pid_file")"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "[orin_arm/stop] stopping pid=$pid"
    kill "$pid"
  fi
  rm -f "$pid_file"
}

stop_pid_file "$PID_DIR/ros_state_udp_bridge.pid"
echo "[orin_arm/stop] stopped"
