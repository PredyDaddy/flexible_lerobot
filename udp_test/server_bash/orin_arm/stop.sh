#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PID_DIR="$ROOT_DIR/udp_test/server_bash/orin_arm/pids"
PATTERN="udp_test/test_scripts/arm_side/orin_ros_state_udp_bridge.py"
STARTUP_FILE="$PID_DIR/ros_state_udp_bridge.startup"
PID_FILE="$PID_DIR/ros_state_udp_bridge.pid"
BRIDGE_EXECUTOR_SHUTDOWN_S=5
STOP_GRACE_S="${STOP_GRACE_S:-6}"

if [[ ! "$STOP_GRACE_S" =~ ^[1-9][0-9]*$ ]] || ((STOP_GRACE_S < BRIDGE_EXECUTOR_SHUTDOWN_S)); then
  echo "[orin_arm/stop] invalid STOP_GRACE_S=$STOP_GRACE_S; expected integer >= $BRIDGE_EXECUTOR_SHUTDOWN_S" >&2
  exit 2
fi

pid_matches_bridge() {
  local pid="$1"
  local -a argv=()
  local index
  [[ "$pid" =~ ^[1-9][0-9]*$ && "$pid" != "$$" && "$pid" != "$PPID" ]] || return 1
  mapfile -d '' -t argv 2>/dev/null < "/proc/$pid/cmdline" || return 1
  for ((index = 1; index < ${#argv[@]}; index++)); do
    if [[ "${argv[index]}" == "$PATTERN" || "${argv[index]}" == */"$PATTERN" ]]; then
      [[ "${argv[index - 1]##*/}" == python* ]]
      return
    fi
  done
  return 1
}

declare -A TARGET_PIDS=()

consider_pid() {
  local source="$1"
  local pid="$2"
  [[ -n "$pid" ]] || return
  if pid_matches_bridge "$pid"; then
    TARGET_PIDS["$pid"]="$source"
  elif kill -0 "$pid" 2>/dev/null; then
    echo "[orin_arm/stop] ignoring $source pid=$pid because cmdline is not the state bridge" >&2
  fi
}

if [[ -f "$PID_FILE" ]]; then
  consider_pid "pid_file" "$(cat "$PID_FILE")"
fi
if [[ -f "$STARTUP_FILE" ]]; then
  while IFS='=' read -r key value; do
    case "$key" in
      bridge_pid|launcher_pid) consider_pid "startup_$key" "$value" ;;
    esac
  done < "$STARTUP_FILE"
fi
while IFS= read -r pid; do
  consider_pid "process_scan" "$pid"
done < <(pgrep -f "$PATTERN" 2>/dev/null || true)

for pid in "${!TARGET_PIDS[@]}"; do
  if kill -0 "$pid" 2>/dev/null; then
    echo "[orin_arm/stop] stopping pid=$pid source=${TARGET_PIDS[$pid]}"
    kill -TERM "$pid" 2>/dev/null || true
  fi
done

deadline=$((SECONDS + STOP_GRACE_S))
while ((SECONDS < deadline)); do
  any_alive=false
  for pid in "${!TARGET_PIDS[@]}"; do
    if pid_matches_bridge "$pid" && kill -0 "$pid" 2>/dev/null; then
      any_alive=true
      break
    fi
  done
  [[ "$any_alive" == "true" ]] || break
  sleep 0.1
done

for pid in "${!TARGET_PIDS[@]}"; do
  if pid_matches_bridge "$pid" && kill -0 "$pid" 2>/dev/null; then
    echo "[orin_arm/stop] force stopping pid=$pid after ${STOP_GRACE_S}s grace"
    kill -KILL "$pid" 2>/dev/null || true
  fi
done

rm -f "$PID_FILE" "$STARTUP_FILE"
echo "[orin_arm/stop] stopped"
