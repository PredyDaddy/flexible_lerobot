#!/usr/bin/env bash

TIMED_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${TIMED_ROOT}/../.." && pwd)"
TIMED_LOG_DIR="${TIMED_ROOT}/logs"
TIMED_RUN_DIR="${TIMED_ROOT}/run"
mkdir -p "${TIMED_LOG_DIR}" "${TIMED_RUN_DIR}"

timed_make_python_cmd() {
  local env_name="$1"
  if [[ "${CONDA_DEFAULT_ENV:-}" == "${env_name}" && -x "${CONDA_PREFIX:-}/bin/python" ]]; then
    PYTHON_CMD=("${CONDA_PREFIX}/bin/python")
  else
    PYTHON_CMD=(conda run --no-capture-output -n "${env_name}" python)
  fi
}

timed_require_armed_confirmation() {
  local execution="${1:-dry_run}"
  local action_name="${2:-control}"
  if [[ "${execution}" != "armed" ]]; then
    return 0
  fi
  if [[ "${JZ_ROBOT_PIN_ARMED:-}" != "1" ]]; then
    echo "[jz_robot_pin_timed] refusing armed ${action_name}: set JZ_ROBOT_PIN_ARMED=1" >&2
    exit 2
  fi
  if [[ "${I_UNDERSTAND_JZ_ROBOT_PIN_MOVES_ROBOT:-}" != "1" ]]; then
    echo "[jz_robot_pin_timed] refusing armed ${action_name}:" \
      "set I_UNDERSTAND_JZ_ROBOT_PIN_MOVES_ROBOT=1" >&2
    exit 2
  fi
}

timed_pid_command() {
  ps -p "$1" -o args= 2>/dev/null || true
}

timed_add_descendants() {
  local parent_pid="$1"
  local child_pid
  while IFS= read -r child_pid; do
    TIMED_PIDS+=("${child_pid}")
    timed_add_descendants "${child_pid}"
  done < <(pgrep -P "${parent_pid}" 2>/dev/null || true)
}

