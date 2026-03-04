#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <train_log_file> [target_step] [milestone_step] [poll_seconds]" >&2
  exit 1
fi

LOG_FILE=$1
TARGET_STEP=${2:-15000}
MILESTONE_STEP=${3:-10000}
POLL_SECONDS=${4:-60}

if [[ ! -f "${LOG_FILE}" ]]; then
  echo "[ERROR] log file not found: ${LOG_FILE}" >&2
  exit 1
fi

RUN_DIR="${LOG_FILE%.log}"
CKPT_DIR="${RUN_DIR}/checkpoints"
STATUS_FILE="${RUN_DIR}/monitor_status.txt"
mkdir -p "${RUN_DIR}"

parse_k_step_to_int() {
  local s="$1"
  s=${s//,/}
  if [[ "$s" == *K ]]; then
    local base=${s%K}
    awk -v v="$base" 'BEGIN { printf "%d", int(v * 1000) }'
  else
    printf "%d" "$s"
  fi
}

max_step_from_log() {
  if [[ ! -f "${LOG_FILE}" ]]; then
    echo 0
    return
  fi
  local last
  last=$(rg -o "step:[0-9]+(\.[0-9]+)?K?" "${LOG_FILE}" | tail -n 1 | sed 's/step://') || true
  if [[ -z "${last:-}" ]]; then
    echo 0
    return
  fi
  parse_k_step_to_int "$last"
}

max_step_from_ckpt() {
  if [[ ! -d "${CKPT_DIR}" ]]; then
    echo 0
    return
  fi
  local ck
  ck=$(find "${CKPT_DIR}" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | rg '^[0-9]{6}$' | sort -n | tail -n 1) || true
  if [[ -z "${ck:-}" ]]; then
    echo 0
    return
  fi
  printf "%d" "${ck#0}"
}

best_step() {
  local a b
  a=$(max_step_from_log)
  b=$(max_step_from_ckpt)
  if (( a > b )); then
    echo "$a"
  else
    echo "$b"
  fi
}

milestone_reported=0

while true; do
  now=$(date '+%Y-%m-%d %H:%M:%S')
  current_step=$(best_step)
  last_line=$(tail -n 1 "${LOG_FILE}" 2>/dev/null || true)

  {
    echo "timestamp=${now}"
    echo "log_file=${LOG_FILE}"
    echo "run_dir=${RUN_DIR}"
    echo "target_step=${TARGET_STEP}"
    echo "milestone_step=${MILESTONE_STEP}"
    echo "current_step=${current_step}"
    echo "last_log_line=${last_line}"
  } > "${STATUS_FILE}"

  if (( current_step >= MILESTONE_STEP && milestone_reported == 0 )); then
    echo "[MONITOR] ${now} reached milestone step ${MILESTONE_STEP} (current=${current_step})" | tee -a "${STATUS_FILE}"
    milestone_reported=1
  fi

  if (( current_step >= TARGET_STEP )); then
    echo "[MONITOR] ${now} reached target step ${TARGET_STEP}." | tee -a "${STATUS_FILE}"
    exit 0
  fi

  sleep "${POLL_SECONDS}"
done
