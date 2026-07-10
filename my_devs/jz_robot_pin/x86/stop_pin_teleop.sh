#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"

PID_FILE="${PIN_RUN_DIR}/pin_teleop.pids"
GRACE_SECONDS="${GRACE_SECONDS:-3}"
STOP_PIN_TELEOP_EXCLUDE_PID="${STOP_PIN_TELEOP_EXCLUDE_PID:-}"
PIN_PIDS=()

add_pid() {
  local pid="${1:-}"
  [[ "${pid}" =~ ^[0-9]+$ ]] || return 0
  [[ "${pid}" != "$$" ]] || return 0
  [[ -z "${STOP_PIN_TELEOP_EXCLUDE_PID}" || "${pid}" != "${STOP_PIN_TELEOP_EXCLUDE_PID}" ]] || return 0
  kill -0 "${pid}" 2>/dev/null || return 0
  local existing
  for existing in "${PIN_PIDS[@]}"; do
    [[ "${existing}" != "${pid}" ]] || return 0
  done
  PIN_PIDS+=("${pid}")
  pin_add_descendants "${pid}"
}

load_pid_file() {
  [[ -f "${PID_FILE}" ]] || return 0
  local key value
  while IFS="=" read -r key value; do
    case "${key}" in
      start_pid|control_pid|joystick_pid)
        add_pid "${value}"
        ;;
    esac
  done <"${PID_FILE}"
}

load_fallback_processes() {
  local pattern pid
  local patterns=(
    "lerobot.scripts.lerobot_teleoperate"
    "jz_robot_pin_target_action"
    "start_pin_control.sh"
    "start_pin_joystick.sh"
    "start_pin_teleop.sh"
    "live_vr_replay_bridge.vr_visual_publisher"
  )
  for pattern in "${patterns[@]}"; do
    while IFS= read -r pid; do
      add_pid "${pid}"
    done < <(pgrep -f "${pattern}" 2>/dev/null || true)
  done
}

any_alive() {
  local pid
  for pid in "${PIN_PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      return 0
    fi
  done
  return 1
}

send_signal() {
  local signal="$1"
  local pid
  for pid in "${PIN_PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      echo "[stop_pin_teleop] ${signal} pid=${pid} cmd=$(pin_pid_command "${pid}")"
      kill "-${signal}" "${pid}" 2>/dev/null || true
    fi
  done
}

load_pid_file
load_fallback_processes

if [[ "${#PIN_PIDS[@]}" -eq 0 ]]; then
  echo "[stop_pin_teleop] no pin teleop processes found"
  rm -f "${PID_FILE}"
  exit 0
fi

send_signal TERM
deadline=$((SECONDS + GRACE_SECONDS))
while any_alive && ((SECONDS < deadline)); do
  sleep 0.2
done
if any_alive; then
  send_signal KILL
fi

rm -f "${PID_FILE}"
echo "[stop_pin_teleop] stopped"
