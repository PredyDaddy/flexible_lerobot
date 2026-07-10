#!/usr/bin/env bash
set -euo pipefail
trap 'status=$?; echo "[start_pin_teleop] error at line ${LINENO}, status=${status}" >&2' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"

PID_FILE="${PIN_RUN_DIR}/pin_teleop.pids"
CONTROL_LOG="${PIN_LOG_DIR}/pin_control.stdout.log"
JOYSTICK_LOG="${PIN_LOG_DIR}/pin_joystick.stdout.log"
PRESTOP_LOG="${PIN_LOG_DIR}/pin_teleop_prestop.log"
CONTROL_STARTUP_WAIT_S="${CONTROL_STARTUP_WAIT_S:-5}"
JOYSTICK_STARTUP_WAIT_S="${JOYSTICK_STARTUP_WAIT_S:-5}"

EXECUTION="${EXECUTION:-armed}"
pin_require_armed_confirmation "${EXECUTION}" "teleop"

echo "[start_pin_teleop] cleaning old pin teleop processes..."
STOP_PIN_TELEOP_EXCLUDE_PID="$$" "${SCRIPT_DIR}/stop_pin_teleop.sh" >"${PRESTOP_LOG}" 2>&1 || true

: >"${CONTROL_LOG}"
: >"${JOYSTICK_LOG}"

control_pid=""
joystick_pid=""

write_pid_file() {
  {
    echo "start_pid=$$"
    [[ -n "${control_pid}" ]] && echo "control_pid=${control_pid}"
    [[ -n "${joystick_pid}" ]] && echo "joystick_pid=${joystick_pid}"
  } >"${PID_FILE}"
}

cleanup() {
  trap - EXIT INT TERM
  "${SCRIPT_DIR}/stop_pin_teleop.sh" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

echo "[start_pin_teleop] starting LeRobot pin control..."
EXECUTION="${EXECUTION}" "${SCRIPT_DIR}/start_pin_control.sh" >"${CONTROL_LOG}" 2>&1 &
control_pid="$!"
write_pid_file
sleep "${CONTROL_STARTUP_WAIT_S}"
if ! kill -0 "${control_pid}" 2>/dev/null; then
  echo "[start_pin_teleop] pin control exited during startup" >&2
  tail -n 120 "${CONTROL_LOG}" >&2 || true
  wait "${control_pid}"
fi

echo "[start_pin_teleop] starting VR/joystick visual publisher..."
"${SCRIPT_DIR}/start_pin_joystick.sh" >"${JOYSTICK_LOG}" 2>&1 &
joystick_pid="$!"
write_pid_file
sleep "${JOYSTICK_STARTUP_WAIT_S}"
if ! kill -0 "${joystick_pid}" 2>/dev/null; then
  echo "[start_pin_teleop] pin joystick exited during startup" >&2
  tail -n 120 "${JOYSTICK_LOG}" >&2 || true
  wait "${joystick_pid}"
fi

echo "[start_pin_teleop] running"
echo "[start_pin_teleop] control log:  ${CONTROL_LOG}"
echo "[start_pin_teleop] joystick log: ${JOYSTICK_LOG}"
echo "[start_pin_teleop] stop with:    ${SCRIPT_DIR}/stop_pin_teleop.sh"

wait "${joystick_pid}"
