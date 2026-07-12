#!/usr/bin/env bash
set -euo pipefail

RUNTIME_DIR="${JZ_DIRECT_CAMERA_RUNTIME_DIR:-/tmp/jz_pin_timed_direct_cameras}"
PID_FILE="$RUNTIME_DIR/orin_realsense_zmq.pid"
WORKER_PATTERN='^python([0-9.]*)? -m lerobot\.robots\.jz_robot_pin_timed\.orin_realsense_zmq([[:space:]]|$)'

if [[ -f "$PID_FILE" ]]; then
  pid="$(<"$PID_FILE")"
  if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
    echo "[direct cameras] stopping pid=$pid"
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    for _ in {1..40}; do
      kill -0 "$pid" 2>/dev/null || break
      sleep 0.25
    done
    if kill -0 "$pid" 2>/dev/null; then
      echo "[direct cameras] force stopping pid=$pid"
      kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
    fi
  fi
  rm -f "$PID_FILE"
fi

mapfile -t worker_pids < <(pgrep -f "$WORKER_PATTERN" || true)
if (( ${#worker_pids[@]} > 0 )); then
  echo "[direct cameras] waiting for worker pids=${worker_pids[*]}"
  for pid in "${worker_pids[@]}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  for _ in {1..40}; do
    mapfile -t worker_pids < <(pgrep -f "$WORKER_PATTERN" || true)
    (( ${#worker_pids[@]} == 0 )) && break
    sleep 0.25
  done
  if (( ${#worker_pids[@]} > 0 )); then
    echo "[direct cameras] force stopping worker pids=${worker_pids[*]}"
    kill -KILL "${worker_pids[@]}" 2>/dev/null || true
  fi
fi

echo "[direct cameras] stopped"
