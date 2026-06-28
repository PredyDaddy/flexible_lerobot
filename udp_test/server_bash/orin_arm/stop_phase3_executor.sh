#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PID_DIR="$ROOT_DIR/udp_test/server_bash/orin_arm/pids"
PATTERN="udp_test/test_scripts/arm_side/orin_phase3_command_executor.py"

stop_pid() {
  local pid="$1"
  if [[ "$pid" == "$$" || "$pid" == "$PPID" ]]; then
    return
  fi
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "[orin_arm/stop_phase3_executor] stopping pid=$pid"
    kill "$pid" 2>/dev/null || true
  fi
}

stop_pid_file() {
  local pid_file="$1"
  if [[ ! -f "$pid_file" ]]; then
    return
  fi
  local pid
  pid="$(cat "$pid_file")"
  stop_pid "$pid"
  rm -f "$pid_file"
}

stop_pid_file "$PID_DIR/orin_phase3_command_executor.pid"
while read -r pid _cmd; do
  stop_pid "$pid"
done < <(pgrep -af "$PATTERN" || true)
sleep 0.5
while read -r pid _cmd; do
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "[orin_arm/stop_phase3_executor] force stopping pid=$pid"
    kill -9 "$pid" 2>/dev/null || true
  fi
done < <(pgrep -af "$PATTERN" || true)
echo "[orin_arm/stop_phase3_executor] stopped"
