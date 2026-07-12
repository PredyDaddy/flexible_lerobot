#!/usr/bin/env bash
set -euo pipefail

RUNTIME_DIR="${JZ_DIRECT_CAMERA_RUNTIME_DIR:-/tmp/jz_pin_timed_direct_cameras}"
PID_FILE="$RUNTIME_DIR/orin_realsense_zmq.pid"
WORKER_PATTERN='^python([0-9.]*)? -m lerobot\.robots\.jz_robot_pin_timed\.orin_realsense_zmq([[:space:]]|$)'

if [[ -f "$PID_FILE" ]]; then
  pid="$(<"$PID_FILE")"
  if kill -0 "$pid" 2>/dev/null; then
    echo "[direct cameras] running pid=$pid"
  else
    echo "[direct cameras] stale pid=$pid"
  fi
else
  echo "[direct cameras] not running"
fi

pgrep -af "$WORKER_PATTERN" || true
