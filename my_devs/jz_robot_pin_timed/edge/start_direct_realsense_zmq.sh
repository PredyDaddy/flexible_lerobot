#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUNTIME_DIR="${JZ_DIRECT_CAMERA_RUNTIME_DIR:-/tmp/jz_pin_timed_direct_cameras}"
LOG_FILE="$RUNTIME_DIR/orin_realsense_zmq.log"
PID_FILE="$RUNTIME_DIR/orin_realsense_zmq.pid"
JPEG_QUALITY="${JZ_DIRECT_CAMERA_JPEG_QUALITY:-95}"
TRACE_EVERY_FRAME="${JZ_DIRECT_CAMERA_TRACE_EVERY_FRAME:-0}"
HEAD_WHITE_BALANCE="${JZ_DIRECT_CAMERA_HEAD_WHITE_BALANCE:-}"
LEFT_WHITE_BALANCE="${JZ_DIRECT_CAMERA_LEFT_WHITE_BALANCE:-}"
RIGHT_WHITE_BALANCE="${JZ_DIRECT_CAMERA_RIGHT_WHITE_BALANCE:-}"
DIAGNOSTIC_DIR="${JZ_DIRECT_CAMERA_DIAGNOSTIC_DIR:-}"

case "${TRACE_EVERY_FRAME,,}" in
  1|true|yes|on) TRACE_ARGS=(--trace-every-frame) ;;
  0|false|no|off) TRACE_ARGS=() ;;
  *)
    echo "[direct cameras] invalid JZ_DIRECT_CAMERA_TRACE_EVERY_FRAME=$TRACE_EVERY_FRAME" >&2
    exit 2
    ;;
esac

WHITE_BALANCE_ARGS=()
for camera in HEAD LEFT RIGHT; do
  variable_name="${camera}_WHITE_BALANCE"
  value="${!variable_name}"
  if [[ -z "$value" ]]; then
    echo "[direct cameras] ${camera,,} white_balance=device_current (AWB will still be forced off)"
    continue
  fi
  if [[ ! "$value" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "[direct cameras] invalid JZ_DIRECT_CAMERA_${camera}_WHITE_BALANCE=$value" >&2
    exit 2
  fi
  WHITE_BALANCE_ARGS+=("--camera-${camera,,}-white-balance" "$value")
  echo "[direct cameras] ${camera,,} white_balance_requested=$value"
done

DIAGNOSTIC_ARGS=()
if [[ -n "$DIAGNOSTIC_DIR" ]]; then
  DIAGNOSTIC_ARGS=(--diagnostic-dir "$DIAGNOSTIC_DIR")
  echo "[direct cameras] diagnostic_dir=$DIAGNOSTIC_DIR sample_ordinals=1,150,300"
fi

mkdir -p "$RUNTIME_DIR"

DEBUG_CAMERA_OWNER_PATTERN='(^|/)(realsense-viewer|rs-capture|rs-color|rs-depth)([[:space:]]|$)'
mapfile -t debug_camera_owners < <(pgrep -af "$DEBUG_CAMERA_OWNER_PATTERN" || true)
if (( ${#debug_camera_owners[@]} > 0 )); then
  echo "[direct cameras] refusing to start: RealSense debug tool owns camera devices" >&2
  printf '[direct cameras] owner: %s\n' "${debug_camera_owners[@]}" >&2
  echo "[direct cameras] close the debug tool and retry; it will not be terminated automatically" >&2
  exit 1
fi
if pgrep -f '/robot_camera_node([[:space:]]|$)' >/dev/null; then
  echo "[direct cameras] refusing to start: robot_camera_node still owns the RealSense devices" >&2
  exit 1
fi
if pgrep -f '/camera_bridge_node([[:space:]]|$)' >/dev/null; then
  echo "[direct cameras] refusing to start: camera_bridge_node is still running" >&2
  exit 1
fi
if systemctl is-active --quiet robot_bringup.service; then
  echo "[direct cameras] robot_bringup.service active without legacy camera owners; continuing"
fi

bash "$ROOT_DIR/my_devs/jz_robot_pin_timed/edge/stop_direct_realsense_zmq.sh"
mkdir -p "$RUNTIME_DIR"

setsid env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR" \
  conda run --no-capture-output -n lerobot python \
  -m lerobot.robots.jz_robot_pin_timed.orin_realsense_zmq \
  --jpeg-quality "$JPEG_QUALITY" "${TRACE_ARGS[@]}" "${WHITE_BALANCE_ARGS[@]}" \
  "${DIAGNOSTIC_ARGS[@]}" \
  >"$LOG_FILE" 2>&1 &
pid=$!
echo "$pid" >"$PID_FILE"
echo "[direct cameras] starting pid=$pid log=$LOG_FILE"

deadline=$((SECONDS + 30))
while (( SECONDS < deadline )); do
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "[direct cameras] server exited before ready" >&2
    tail -100 "$LOG_FILE" >&2 || true
    bash "$ROOT_DIR/my_devs/jz_robot_pin_timed/edge/stop_direct_realsense_zmq.sh" || true
    exit 1
  fi
  if grep -Fq '[direct realsense zmq] all cameras ready' "$LOG_FILE"; then
    echo "[direct cameras] all cameras ready"
    echo "[direct cameras] camera_head=tcp://192.168.1.81:5555"
    echo "[direct cameras] camera_left=tcp://192.168.1.81:5556"
    echo "[direct cameras] camera_right=tcp://192.168.1.81:5557"
    exit 0
  fi
  sleep 0.25
done

echo "[direct cameras] timeout waiting for cameras" >&2
tail -100 "$LOG_FILE" >&2 || true
bash "$ROOT_DIR/my_devs/jz_robot_pin_timed/edge/stop_direct_realsense_zmq.sh"
exit 1
