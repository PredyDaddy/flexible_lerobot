#!/usr/bin/env bash
set -euo pipefail
trap 'status=$?; echo "[timed/x86/teleop] error at line ${LINENO}, status=${status}" >&2' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"

PID_FILE="${TIMED_RUN_DIR}/pin_timed_teleop.pids"
CONTROL_LOG="${TIMED_LOG_DIR}/pin_timed_control.stdout.log"
JOYSTICK_LOG="${TIMED_LOG_DIR}/pin_timed_joystick.stdout.log"
PRESTOP_LOG="${TIMED_LOG_DIR}/pin_timed_teleop_prestop.log"
CONTROL_STARTUP_WAIT_S="${CONTROL_STARTUP_WAIT_S:-5}"
JOYSTICK_STARTUP_WAIT_S="${JOYSTICK_STARTUP_WAIT_S:-5}"
EXECUTION="${EXECUTION:-armed}"

timed_require_armed_confirmation "${EXECUTION}" "teleop"

echo "[timed/x86/teleop] cleaning only an earlier timed teleop session"
STOP_TIMED_TELEOP_EXCLUDE_PID="$$" \
  bash "${SCRIPT_DIR}/stop_pin_teleop.sh" >"${PRESTOP_LOG}" 2>&1 || true

: >"${CONTROL_LOG}"
: >"${JOYSTICK_LOG}"

control_pid=""
joystick_pid=""

write_pid_file() {
  local temp_file="${PID_FILE}.tmp.$$"
  {
    echo "start_pid=$$"
    [[ -n "${control_pid}" ]] && echo "control_pid=${control_pid}"
    [[ -n "${joystick_pid}" ]] && echo "joystick_pid=${joystick_pid}"
  } >"${temp_file}"
  mv "${temp_file}" "${PID_FILE}"
}

cleanup() {
  trap - EXIT INT TERM
  STOP_TIMED_TELEOP_EXCLUDE_PID="$$" bash "${SCRIPT_DIR}/stop_pin_teleop.sh" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

echo "[timed/x86/teleop] starting timed LeRobot control"
EXECUTION="${EXECUTION}" bash "${SCRIPT_DIR}/start_pin_control.sh" >"${CONTROL_LOG}" 2>&1 &
control_pid="$!"
write_pid_file
sleep "${CONTROL_STARTUP_WAIT_S}"
if ! kill -0 "${control_pid}" 2>/dev/null; then
  echo "[timed/x86/teleop] timed control exited during startup" >&2
  tail -n 120 "${CONTROL_LOG}" >&2 || true
  wait "${control_pid}"
fi

echo "[timed/x86/teleop] starting shared VR/joystick visual publisher"
bash "${SCRIPT_DIR}/start_pin_joystick.sh" >"${JOYSTICK_LOG}" 2>&1 &
joystick_pid="$!"
write_pid_file
sleep "${JOYSTICK_STARTUP_WAIT_S}"
if ! kill -0 "${joystick_pid}" 2>/dev/null; then
  echo "[timed/x86/teleop] joystick publisher exited during startup" >&2
  tail -n 120 "${JOYSTICK_LOG}" >&2 || true
  wait "${joystick_pid}"
fi

echo "[timed/x86/teleop] running"
echo "[timed/x86/teleop] pid file:     ${PID_FILE}"
echo "[timed/x86/teleop] control log:  ${CONTROL_LOG}"
echo "[timed/x86/teleop] joystick log: ${JOYSTICK_LOG}"
echo "[timed/x86/teleop] stop with:    bash ${SCRIPT_DIR}/stop_pin_teleop.sh"

wait "${joystick_pid}"

