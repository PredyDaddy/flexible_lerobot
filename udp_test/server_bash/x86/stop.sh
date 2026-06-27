#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PID_DIR="$ROOT_DIR/udp_test/server_bash/x86/pids"

stop_pid_file() {
  local pid_file="$1"
  if [[ ! -f "$pid_file" ]]; then
    return
  fi
  local pid
  pid="$(cat "$pid_file")"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "[x86/stop] stopping pid=$pid"
    kill "$pid"
  fi
  rm -f "$pid_file"
}

stop_pid_file "$PID_DIR/jz_robot_udp_observation_check.pid"
echo "[x86/stop] stopped"
