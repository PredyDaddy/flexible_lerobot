#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"

PID_FILE="${TIMED_RUN_DIR}/pin_timed_teleop.pids"
GRACE_SECONDS="${GRACE_SECONDS:-3}"
STOP_TIMED_TELEOP_EXCLUDE_PID="${STOP_TIMED_TELEOP_EXCLUDE_PID:-}"
TIMED_PIDS=()

is_excluded() {
  local pid="$1"
  [[ "${pid}" == "$$" ]] || \
    [[ -n "${STOP_TIMED_TELEOP_EXCLUDE_PID}" && "${pid}" == "${STOP_TIMED_TELEOP_EXCLUDE_PID}" ]]
}

contains_pid() {
  local candidate="$1"
  local existing
  for existing in "${TIMED_PIDS[@]}"; do
    [[ "${existing}" != "${candidate}" ]] || return 0
  done
  return 1
}

add_pid() {
  local pid="${1:-}"
  [[ "${pid}" =~ ^[0-9]+$ ]] || return 0
  is_excluded "${pid}" && return 0
  kill -0 "${pid}" 2>/dev/null || return 0
  contains_pid "${pid}" && return 0
  TIMED_PIDS+=("${pid}")
}

add_descendants() {
  local parent_pid="$1"
  local child_pid
  while IFS= read -r child_pid; do
    add_pid "${child_pid}"
    add_descendants "${child_pid}"
  done < <(pgrep -P "${parent_pid}" 2>/dev/null || true)
}

matches_role() {
  local role="$1"
  local pid="$2"
  local command
  command="$(timed_pid_command "${pid}")"
  case "${role}" in
    start_pid) [[ "${command}" == *"${SCRIPT_DIR}/start_pin_teleop.sh"* ]] ;;
    control_pid)
      [[ "${command}" == *"lerobot.scripts.lerobot_teleoperate"* ]] && \
        [[ "${command}" == *"--robot.id=jz_robot_pin_timed_control"* ]]
      ;;
    joystick_pid) [[ "${command}" == *"${SCRIPT_DIR}/start_pin_joystick.sh"* ]] ;;
    *) return 1 ;;
  esac
}

load_pid_file() {
  [[ -f "${PID_FILE}" ]] || return 0
  local role pid
  while IFS="=" read -r role pid; do
    case "${role}" in
      start_pid|control_pid|joystick_pid)
        if is_excluded "${pid}"; then
          continue
        elif matches_role "${role}" "${pid}"; then
          add_pid "${pid}"
          add_descendants "${pid}"
        else
          echo "[timed/x86/stop] ignoring stale ${role}=${pid}; command fingerprint changed" >&2
        fi
        ;;
    esac
  done <"${PID_FILE}"
}

load_precise_fallbacks() {
  local pattern pid command
  local patterns=(
    "${SCRIPT_DIR}/start_pin_teleop.sh"
    "${SCRIPT_DIR}/start_pin_control.sh"
    "${SCRIPT_DIR}/start_pin_joystick.sh"
    "--robot\\.id=jz_robot_pin_timed_control"
  )
  for pattern in "${patterns[@]}"; do
    while IFS= read -r pid; do
      is_excluded "${pid}" && continue
      command="$(timed_pid_command "${pid}")"
      if [[ "${command}" == *"${SCRIPT_DIR}/start_pin_"* ]] || \
        { [[ "${command}" == *"lerobot.scripts.lerobot_teleoperate"* ]] && \
          [[ "${command}" == *"--robot.id=jz_robot_pin_timed_control"* ]]; }; then
        add_pid "${pid}"
        add_descendants "${pid}"
      fi
    done < <(pgrep -f -- "${pattern}" 2>/dev/null || true)
  done
}

any_alive() {
  local pid
  for pid in "${TIMED_PIDS[@]}"; do
    kill -0 "${pid}" 2>/dev/null && return 0
  done
  return 1
}

send_signal() {
  local signal="$1"
  local index pid
  for ((index=${#TIMED_PIDS[@]} - 1; index >= 0; index--)); do
    pid="${TIMED_PIDS[index]}"
    if kill -0 "${pid}" 2>/dev/null; then
      echo "[timed/x86/stop] ${signal} pid=${pid} cmd=$(timed_pid_command "${pid}")"
      kill "-${signal}" "${pid}" 2>/dev/null || true
    fi
  done
}

load_pid_file
load_precise_fallbacks

if [[ "${#TIMED_PIDS[@]}" -eq 0 ]]; then
  echo "[timed/x86/stop] no jz_robot_pin_timed teleop processes found"
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
echo "[timed/x86/stop] stopped ${#TIMED_PIDS[@]} timed process(es)"
