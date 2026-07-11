#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LOG_DIR="$ROOT_DIR/udp_test/server_bash/orin_arm/logs"
PID_DIR="$ROOT_DIR/udp_test/server_bash/orin_arm/pids"
mkdir -p "$LOG_DIR" "$PID_DIR"

ORIN_IP="${ORIN_IP:-192.168.1.81}"
X86_IP="${X86_IP:-192.168.1.106}"
STATE_PORT="${STATE_PORT:-39010}"
STATE_HZ="${STATE_HZ:-20}"
JZ_STATE_HZ_PROFILE="${JZ_STATE_HZ_PROFILE:-legacy}"
JZ_EXPECTED_STATE_HZ="${JZ_EXPECTED_STATE_HZ:-${STATE_HZ}}"
JZ_TIMED_NON_30_STATE_HZ_CONFIRM="${JZ_TIMED_NON_30_STATE_HZ_CONFIRM:-}"
STATE_WAIT_TIMEOUT_S="${STATE_WAIT_TIMEOUT_S:-15}"
STATE_PROCESS_START_TIMEOUT_S="${STATE_PROCESS_START_TIMEOUT_S:-30}"
STATE_EXECUTOR_THREADS="${STATE_EXECUTOR_THREADS:-4}"
STATE_PRINT_EVERY="${STATE_PRINT_EVERY:-30}"
MAX_SOURCE_AGE_MS="${MAX_SOURCE_AGE_MS:-50}"
MAX_SOURCE_SKEW_MS="${MAX_SOURCE_SKEW_MS:-20}"
REQUIRE_ALL_SOURCES_ADVANCED="${REQUIRE_ALL_SOURCES_ADVANCED:-true}"
PYTHON_CMD="${PYTHON_CMD:-python}"
RUN_READINESS="${RUN_READINESS:-0}"
AUTO_TAIL="${AUTO_TAIL:-1}"
read -r -a PYTHON_ARGS <<< "$PYTHON_CMD"

PID_FILE="$PID_DIR/ros_state_udp_bridge.pid"
STARTUP_FILE="$PID_DIR/ros_state_udp_bridge.startup"
PID_FILE_TMP="$PID_FILE.tmp.$$"
STARTUP_FILE_TMP="$STARTUP_FILE.tmp.$$"
LOG_FILE="$LOG_DIR/ros_state_udp_bridge.log"
BRIDGE_SCRIPT="udp_test/test_scripts/arm_side/orin_ros_state_udp_bridge.py"

validate_hz_contract() {
  if [[ ! "$STATE_HZ" =~ ^[0-9]+([.][0-9]+)?$ ]] || ! awk -v hz="$STATE_HZ" 'BEGIN { exit !(hz > 0) }'; then
    echo "[orin_arm/start] invalid STATE_HZ=$STATE_HZ" >&2
    return 2
  fi
  if [[ ! "$JZ_EXPECTED_STATE_HZ" =~ ^[0-9]+([.][0-9]+)?$ ]] || \
     ! awk -v actual="$STATE_HZ" -v expected="$JZ_EXPECTED_STATE_HZ" 'BEGIN { exit !(actual == expected) }'; then
    echo "[orin_arm/start] requested STATE_HZ=$STATE_HZ does not match expected=$JZ_EXPECTED_STATE_HZ" >&2
    return 2
  fi
  case "$JZ_STATE_HZ_PROFILE" in
    legacy) ;;
    timed)
      if ! awk -v hz="$STATE_HZ" 'BEGIN { exit !(hz == 30) }' && \
         [[ "$JZ_TIMED_NON_30_STATE_HZ_CONFIRM" != "$STATE_HZ" ]]; then
        echo "[orin_arm/start] timed profile refuses unconfirmed STATE_HZ=$STATE_HZ" >&2
        return 2
      fi
      ;;
    *)
      echo "[orin_arm/start] unsupported JZ_STATE_HZ_PROFILE=$JZ_STATE_HZ_PROFILE" >&2
      return 2
      ;;
  esac
}

hz_equal() {
  awk -v left="$1" -v right="$2" 'BEGIN { exit !(left == right) }'
}

normalize_hz() {
  awk -v hz="$1" 'BEGIN { printf "%.12g\n", hz }'
}

read_bridge_hz() {
  local pid="$1"
  local -a argv=()
  local index
  mapfile -d '' -t argv < "/proc/$pid/cmdline" 2>/dev/null || return 1
  [[ "${#argv[@]}" -gt 1 && "$(basename "${argv[0]}")" == python* ]] || return 1
  [[ "${argv[1]}" == "$BRIDGE_SCRIPT" || "${argv[1]}" == */"$BRIDGE_SCRIPT" ]] || return 1
  for ((index = 2; index < ${#argv[@]}; index++)); do
    if [[ "${argv[index]}" == "--hz" && $((index + 1)) -lt ${#argv[@]} ]]; then
      printf '%s\n' "${argv[index + 1]}"
      return 0
    fi
    if [[ "${argv[index]}" == --hz=* ]]; then
      printf '%s\n' "${argv[index]#--hz=}"
      return 0
    fi
  done
  return 1
}

descendant_pids() {
  local parent_pid="$1"
  local child_pid
  while IFS= read -r child_pid; do
    [[ -n "$child_pid" ]] || continue
    printf '%s\n' "$child_pid"
    descendant_pids "$child_pid"
  done < <(pgrep -P "$parent_pid" 2>/dev/null || true)
}

find_bridge_process() {
  local launcher_pid="$1"
  local candidate
  local configured_hz
  while IFS= read -r candidate; do
    [[ -r "/proc/$candidate/cmdline" ]] || continue
    if configured_hz="$(read_bridge_hz "$candidate")"; then
      printf '%s %s\n' "$candidate" "$configured_hz"
      return 0
    fi
  done < <(descendant_pids "$launcher_pid"; printf '%s\n' "$launcher_pid")
  return 1
}

stop_failed_launch() {
  local launcher_pid="$1"
  local -a descendants=()
  local index
  if [[ -z "$launcher_pid" ]]; then
    rm -f "$PID_FILE" "$STARTUP_FILE" "$PID_FILE_TMP" "$STARTUP_FILE_TMP"
    return
  fi
  mapfile -t descendants < <(descendant_pids "$launcher_pid")
  for ((index = ${#descendants[@]} - 1; index >= 0; index--)); do
    kill "${descendants[index]}" 2>/dev/null || true
  done
  kill "$launcher_pid" 2>/dev/null || true
  rm -f "$PID_FILE" "$STARTUP_FILE" "$PID_FILE_TMP" "$STARTUP_FILE_TMP"
}

validate_positive_integer() {
  local name="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "[orin_arm/start] invalid $name=$value; expected a positive integer" >&2
    return 2
  fi
}

validate_nonnegative_integer() {
  local name="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "[orin_arm/start] invalid $name=$value; expected a nonnegative integer" >&2
    return 2
  fi
}

LAUNCHER_PID=""
LAUNCH_COMMITTED=false

cleanup_uncommitted_launch() {
  local exit_code=$?
  trap - EXIT ERR INT TERM
  if [[ "$LAUNCH_COMMITTED" != "true" ]]; then
    stop_failed_launch "$LAUNCHER_PID" || true
  fi
  exit "$exit_code"
}

validate_hz_contract
validate_positive_integer STATE_PROCESS_START_TIMEOUT_S "$STATE_PROCESS_START_TIMEOUT_S"
validate_positive_integer STATE_EXECUTOR_THREADS "$STATE_EXECUTOR_THREADS"
validate_nonnegative_integer STATE_PRINT_EVERY "$STATE_PRINT_EVERY"
REQUESTED_HZ_NORMALIZED="$(normalize_hz "$STATE_HZ")"
EXPECTED_HZ_NORMALIZED="$(normalize_hz "$JZ_EXPECTED_STATE_HZ")"

case "${REQUIRE_ALL_SOURCES_ADVANCED,,}" in
  1|true|yes|on) ADVANCED_ARG="--require-all-sources-advanced" ;;
  0|false|no|off) ADVANCED_ARG="--no-require-all-sources-advanced" ;;
  *)
    echo "[orin_arm/start] invalid REQUIRE_ALL_SOURCES_ADVANCED=$REQUIRE_ALL_SOURCES_ADVANCED" >&2
    exit 2
    ;;
esac

cd "$ROOT_DIR"
bash "$ROOT_DIR/udp_test/server_bash/orin_arm/stop.sh"
rm -f "$STARTUP_FILE"
: > "$LOG_FILE"

echo "[orin_arm/start] READONLY ONLY"
echo "[orin_arm/start] profile=$JZ_STATE_HZ_PROFILE requested_hz=$STATE_HZ expected_hz=$JZ_EXPECTED_STATE_HZ"
echo "[orin_arm/start] executor=per_source_process_single_threaded workers=$STATE_EXECUTOR_THREADS"
echo "[orin_arm/start] max_source_age_ms=$MAX_SOURCE_AGE_MS max_source_skew_ms=$MAX_SOURCE_SKEW_MS require_all_sources_advanced=${REQUIRE_ALL_SOURCES_ADVANCED,,}"
if [[ "$RUN_READINESS" == "1" ]]; then
  echo "[orin_arm/start] local readiness check..."
  "${PYTHON_ARGS[@]}" udp_test/local_test/local_robot_readiness_check.py --skip-cameras | tee "$LOG_DIR/local_readiness.log"
else
  echo "[orin_arm/start] skip local readiness check. Set RUN_READINESS=1 to run it."
fi

echo "[orin_arm/start] starting ROS state UDP bridge: $ORIN_IP -> $X86_IP:$STATE_PORT"
trap cleanup_uncommitted_launch EXIT ERR
trap 'exit 130' INT
trap 'exit 143' TERM
nohup "${PYTHON_ARGS[@]}" "$BRIDGE_SCRIPT" \
  --bind-ip "$ORIN_IP" \
  --target-ip "$X86_IP" \
  --target-port "$STATE_PORT" \
  --hz "$STATE_HZ" \
  --executor-threads "$STATE_EXECUTOR_THREADS" \
  --wait-timeout-s "$STATE_WAIT_TIMEOUT_S" \
  --max-source-age-ms "$MAX_SOURCE_AGE_MS" \
  --max-source-skew-ms "$MAX_SOURCE_SKEW_MS" \
  "$ADVANCED_ARG" \
  --print-every "$STATE_PRINT_EVERY" \
  > "$LOG_FILE" 2>&1 &
LAUNCHER_PID=$!

deadline=$((SECONDS + STATE_PROCESS_START_TIMEOUT_S))
BRIDGE_PID=""
CONFIGURED_HZ_RAW=""
while ((SECONDS < deadline)); do
  if ! kill -0 "$LAUNCHER_PID" 2>/dev/null; then
    echo "[orin_arm/start] bridge launcher exited before process verification" >&2
    tail -n 20 "$LOG_FILE" >&2 || true
    stop_failed_launch "$LAUNCHER_PID"
    exit 1
  fi
  if read -r BRIDGE_PID CONFIGURED_HZ_RAW < <(find_bridge_process "$LAUNCHER_PID"); then
    break
  fi
  sleep 0.1
done

if [[ -z "$BRIDGE_PID" || -z "$CONFIGURED_HZ_RAW" ]]; then
  echo "[orin_arm/start] timed out locating the Python bridge process" >&2
  stop_failed_launch "$LAUNCHER_PID"
  exit 1
fi
CONFIGURED_HZ="$(normalize_hz "$CONFIGURED_HZ_RAW")"
if ! hz_equal "$CONFIGURED_HZ" "$EXPECTED_HZ_NORMALIZED"; then
  echo "[orin_arm/start] bridge argv mismatch requested_hz=$REQUESTED_HZ_NORMALIZED configured_hz=$CONFIGURED_HZ expected_hz=$EXPECTED_HZ_NORMALIZED" >&2
  stop_failed_launch "$LAUNCHER_PID"
  exit 1
fi

deadline=$((SECONDS + STATE_PROCESS_START_TIMEOUT_S))
STARTUP_PATTERN="pid=$BRIDGE_PID local="
HZ_PATTERN="configured_hz=$CONFIGURED_HZ"
while ((SECONDS < deadline)); do
  if ! kill -0 "$BRIDGE_PID" 2>/dev/null; then
    echo "[orin_arm/start] bridge exited before logging verified hz" >&2
    tail -n 20 "$LOG_FILE" >&2 || true
    stop_failed_launch "$LAUNCHER_PID"
    exit 1
  fi
  if grep -Fq "$STARTUP_PATTERN" "$LOG_FILE" && grep -Fq "$HZ_PATTERN" "$LOG_FILE"; then
    break
  fi
  sleep 0.1
done
if ! grep -Fq "$STARTUP_PATTERN" "$LOG_FILE" || ! grep -Fq "$HZ_PATTERN" "$LOG_FILE"; then
  echo "[orin_arm/start] bridge log hz verification failed requested_hz=$REQUESTED_HZ_NORMALIZED configured_hz=$CONFIGURED_HZ" >&2
  tail -n 20 "$LOG_FILE" >&2 || true
  stop_failed_launch "$LAUNCHER_PID"
  exit 1
fi

echo "$BRIDGE_PID" > "$PID_FILE_TMP"
{
  printf 'profile=%s\n' "$JZ_STATE_HZ_PROFILE"
  printf 'requested_hz=%s\n' "$REQUESTED_HZ_NORMALIZED"
  printf 'expected_hz=%s\n' "$EXPECTED_HZ_NORMALIZED"
  printf 'configured_hz=%s\n' "$CONFIGURED_HZ"
  printf 'bridge_pid=%s\n' "$BRIDGE_PID"
  printf 'launcher_pid=%s\n' "$LAUNCHER_PID"
} > "$STARTUP_FILE_TMP"
mv "$STARTUP_FILE_TMP" "$STARTUP_FILE"
mv "$PID_FILE_TMP" "$PID_FILE"
LAUNCH_COMMITTED=true
trap - EXIT ERR INT TERM

echo "[orin_arm/start] verified bridge pid=$BRIDGE_PID launcher_pid=$LAUNCHER_PID requested_hz=$REQUESTED_HZ_NORMALIZED configured_hz=$CONFIGURED_HZ"
echo "[orin_arm/start] log: $LOG_FILE"
if [[ "$AUTO_TAIL" == "1" ]]; then
  echo "[orin_arm/start] following log now. Ctrl-C only exits tail; use orin_arm/stop.sh to stop service."
  tail -f "$LOG_FILE"
fi
