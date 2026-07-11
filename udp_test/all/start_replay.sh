#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ORIN_ARM_DIR="$ROOT_DIR/udp_test/server_bash/orin_arm"
LOG_DIR="$ORIN_ARM_DIR/logs"
PID_DIR="$ORIN_ARM_DIR/pids"
mkdir -p "$LOG_DIR" "$PID_DIR"

ORIN_IP="${ORIN_IP:-192.168.1.81}"
X86_IP="${X86_IP:-192.168.1.106}"
STATE_PORT="${STATE_PORT:-39010}"
STATE_HZ="${STATE_HZ:-20}"
JZ_STATE_HZ_PROFILE="${JZ_STATE_HZ_PROFILE:-legacy}"
JZ_EXPECTED_STATE_HZ="${JZ_EXPECTED_STATE_HZ:-${STATE_HZ}}"
JZ_TIMED_NON_30_STATE_HZ_CONFIRM="${JZ_TIMED_NON_30_STATE_HZ_CONFIRM:-}"
MAX_SOURCE_AGE_MS="${MAX_SOURCE_AGE_MS:-50}"
MAX_SOURCE_SKEW_MS="${MAX_SOURCE_SKEW_MS:-20}"
REQUIRE_ALL_SOURCES_ADVANCED="${REQUIRE_ALL_SOURCES_ADVANCED:-true}"
COMMAND_PORT="${COMMAND_PORT:-39020}"
CONFIG="${CONFIG:-$ROOT_DIR/udp_test/test_scripts/arm_side/orin_phase3_executor_config_armed_hold.yaml}"
EXECUTION="${EXECUTION:-armed}"
PYTHON_CMD="${PYTHON_CMD:-conda run --no-capture-output -n lerobot python}"
TAIL="${TAIL:-0}"
LEGACY_READY_TIMEOUT_S="${READY_TIMEOUT_S:-}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-15}"
BRIDGE_START_TIMEOUT_S="${BRIDGE_START_TIMEOUT_S:-${LEGACY_READY_TIMEOUT_S:-30}}"
STATE_WAIT_TIMEOUT_S="${STATE_WAIT_TIMEOUT_S:-${LEGACY_READY_TIMEOUT_S:-15}}"
STATE_READY_TIMEOUT_S="${STATE_READY_TIMEOUT_S:-${LEGACY_READY_TIMEOUT_S:-20}}"
EXECUTOR_READY_TIMEOUT_S="${EXECUTOR_READY_TIMEOUT_S:-${LEGACY_READY_TIMEOUT_S:-15}}"

cd "$ROOT_DIR"

log() {
  echo "[all/start_replay] $*"
}

cleanup_on_error() {
  local exit_code=$?
  log "startup failed with exit code $exit_code; stopping replay services"
  bash "$ROOT_DIR/udp_test/all/stop_replay.sh" || true
  exit "$exit_code"
}

show_log_tail() {
  local log_file="$1"
  if [[ -f "$log_file" ]]; then
    log "last lines from $log_file:"
    tail -n 20 "$log_file" || true
  else
    log "log file not created yet: $log_file"
  fi
}

ensure_alive() {
  local name="$1"
  local pid_file="$2"
  local log_file="$3"
  local pid

  if [[ ! -f "$pid_file" ]]; then
    log "$name did not create pid file: $pid_file"
    show_log_tail "$log_file"
    return 1
  fi

  pid="$(cat "$pid_file")"
  if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
    log "$name failed to stay running. pid=${pid:-missing}"
    show_log_tail "$log_file"
    rm -f "$pid_file"
    return 1
  fi

  log "$name OK pid=$pid"
  log "$name log: $log_file"
  show_log_tail "$log_file"
}

wait_for_log_pattern() {
  local name="$1"
  local pid_file="$2"
  local log_file="$3"
  local pattern="$4"
  local timeout_s="$5"
  local pid
  local deadline

  pid="$(cat "$pid_file")"
  deadline=$((SECONDS + timeout_s))
  while (( SECONDS < deadline )); do
    if ! kill -0 "$pid" 2>/dev/null; then
      log "$name exited before readiness pattern appeared: $pattern"
      show_log_tail "$log_file"
      rm -f "$pid_file"
      return 1
    fi
    if [[ -f "$log_file" ]] && grep -Fq "$pattern" "$log_file"; then
      log "$name confirmed ready from log pattern: $pattern"
      return
    fi
    sleep 0.5
  done

  log "$name is running but readiness pattern did not appear within ${timeout_s}s: $pattern"
  show_log_tail "$log_file"
  return 1
}

normalize_hz() {
  awk -v hz="$1" 'BEGIN { printf "%.12g\n", hz }'
}

validate_hz_contract() {
  if [[ ! "$STATE_HZ" =~ ^[0-9]+([.][0-9]+)?$ ]] || ! awk -v hz="$STATE_HZ" 'BEGIN { exit !(hz > 0) }'; then
    log "invalid STATE_HZ=$STATE_HZ"
    return 2
  fi
  if [[ ! "$JZ_EXPECTED_STATE_HZ" =~ ^[0-9]+([.][0-9]+)?$ ]] || \
     ! awk -v actual="$STATE_HZ" -v expected="$JZ_EXPECTED_STATE_HZ" 'BEGIN { exit !(actual == expected) }'; then
    log "STATE_HZ=$STATE_HZ does not match JZ_EXPECTED_STATE_HZ=$JZ_EXPECTED_STATE_HZ"
    return 2
  fi
  case "$JZ_STATE_HZ_PROFILE" in
    legacy) ;;
    timed)
      if ! awk -v hz="$STATE_HZ" 'BEGIN { exit !(hz == 30) }' && \
         [[ "$JZ_TIMED_NON_30_STATE_HZ_CONFIRM" != "$STATE_HZ" ]]; then
        log "timed profile refuses unconfirmed STATE_HZ=$STATE_HZ"
        return 2
      fi
      ;;
    *)
      log "unsupported JZ_STATE_HZ_PROFILE=$JZ_STATE_HZ_PROFILE"
      return 2
      ;;
  esac
}

bridge_process_hz() {
  local pid="$1"
  local -a argv=()
  local index
  local found_bridge=false
  mapfile -d '' -t argv < "/proc/$pid/cmdline" 2>/dev/null || return 1
  for ((index = 0; index < ${#argv[@]}; index++)); do
    if [[ "${argv[index]}" == "udp_test/test_scripts/arm_side/orin_ros_state_udp_bridge.py" || \
          "${argv[index]}" == */udp_test/test_scripts/arm_side/orin_ros_state_udp_bridge.py ]]; then
      found_bridge=true
    fi
  done
  [[ "$found_bridge" == "true" ]] || return 1
  for ((index = 0; index < ${#argv[@]}; index++)); do
    if [[ "${argv[index]}" == "--hz" && $((index + 1)) -lt ${#argv[@]} ]]; then
      normalize_hz "${argv[index + 1]}"
      return 0
    fi
    if [[ "${argv[index]}" == --hz=* ]]; then
      normalize_hz "${argv[index]#--hz=}"
      return 0
    fi
  done
  return 1
}

verify_bridge_startup() {
  local pid_file="$PID_DIR/ros_state_udp_bridge.pid"
  local startup_file="$PID_DIR/ros_state_udp_bridge.startup"
  local log_file="$LOG_DIR/ros_state_udp_bridge.log"
  local pid
  local argv_hz
  local expected_normalized
  local requested_normalized
  local key value
  local profile="" requested_hz="" expected_hz="" actual_hz="" bridge_pid="" launcher_pid=""

  [[ -f "$startup_file" ]] || { log "missing bridge startup metadata: $startup_file"; return 1; }
  while IFS='=' read -r key value; do
    case "$key" in
      profile) profile="$value" ;;
      requested_hz) requested_hz="$value" ;;
      expected_hz) expected_hz="$value" ;;
      actual_hz) actual_hz="$value" ;;
      bridge_pid) bridge_pid="$value" ;;
      launcher_pid) launcher_pid="$value" ;;
    esac
  done < "$startup_file"
  pid="$(cat "$pid_file")"
  expected_normalized="$(normalize_hz "$JZ_EXPECTED_STATE_HZ")"
  requested_normalized="$(normalize_hz "$STATE_HZ")"
  [[ "$bridge_pid" == "$pid" ]] || { log "bridge pid metadata mismatch pid_file=$pid metadata=$bridge_pid"; return 1; }
  [[ "$profile" == "$JZ_STATE_HZ_PROFILE" ]] || { log "bridge profile mismatch requested=$JZ_STATE_HZ_PROFILE actual=$profile"; return 1; }
  [[ "$requested_hz" == "$requested_normalized" ]] || { log "bridge requested_hz metadata mismatch requested=$requested_normalized actual=$requested_hz"; return 1; }
  [[ "$expected_hz" == "$expected_normalized" ]] || { log "bridge expected_hz metadata mismatch requested=$expected_normalized actual=$expected_hz"; return 1; }
  [[ "$actual_hz" == "$expected_normalized" ]] || { log "bridge actual_hz metadata mismatch expected=$expected_normalized actual=$actual_hz"; return 1; }
  argv_hz="$(bridge_process_hz "$pid")" || { log "could not read bridge --hz from pid=$pid"; return 1; }
  [[ "$argv_hz" == "$expected_normalized" ]] || { log "bridge argv hz mismatch expected=$expected_normalized actual=$argv_hz"; return 1; }
  grep -Fq "pid=$pid local=" "$log_file" || { log "bridge log does not match pid=$pid"; return 1; }
  grep -Fq "requested_hz=$requested_normalized actual_hz=$argv_hz" "$log_file" || {
    log "bridge log hz mismatch requested=$requested_normalized actual=$argv_hz"
    return 1
  }
  log "bridge hz verified pid=$pid requested_hz=$requested_normalized actual_hz=$argv_hz"
}

verify_bridge_metrics() {
  local log_file="$LOG_DIR/ros_state_udp_bridge.log"
  local line
  local token
  line="$(grep -F "sent seq=" "$log_file" | tail -n 1)"
  for token in requested_hz= actual_hz= update_counts= source_age_ms= source_skew_ms= skipped=; do
    [[ "$line" == *"$token"* ]] || { log "bridge sent metrics missing $token"; return 1; }
  done
  for token in left_joints right_joints left_gripper right_gripper; do
    [[ "$line" == *"$token"* ]] || { log "bridge sent metrics missing source $token"; return 1; }
  done
  log "bridge sent metrics verified"
}

validate_hz_contract
trap cleanup_on_error ERR

if [[ "$EXECUTION" == "armed" ]]; then
  export JZ_UDP_EXECUTOR_ARMED="${JZ_UDP_EXECUTOR_ARMED:-1}"
else
  log "unsupported EXECUTION=$EXECUTION. start_replay.sh is for armed replay only."
  exit 2
fi

log "mode=replay"
log "ORIN_IP=$ORIN_IP X86_IP=$X86_IP STATE_PORT=$STATE_PORT COMMAND_PORT=$COMMAND_PORT"
log "profile=$JZ_STATE_HZ_PROFILE requested_hz=$(normalize_hz "$STATE_HZ") expected_hz=$(normalize_hz "$JZ_EXPECTED_STATE_HZ")"
log "max_source_age_ms=$MAX_SOURCE_AGE_MS max_source_skew_ms=$MAX_SOURCE_SKEW_MS require_all_sources_advanced=$REQUIRE_ALL_SOURCES_ADVANCED"
log "CONFIG=$CONFIG"
log "EXECUTION=$EXECUTION JZ_UDP_EXECUTOR_ARMED=${JZ_UDP_EXECUTOR_ARMED:-}"
log "PYTHON_CMD=$PYTHON_CMD"
log "READY_TIMEOUT_S=$READY_TIMEOUT_S"
log "BRIDGE_START_TIMEOUT_S=$BRIDGE_START_TIMEOUT_S"
log "STATE_WAIT_TIMEOUT_S=$STATE_WAIT_TIMEOUT_S"
log "STATE_READY_TIMEOUT_S=$STATE_READY_TIMEOUT_S"
log "EXECUTOR_READY_TIMEOUT_S=$EXECUTOR_READY_TIMEOUT_S"
log "logs: $LOG_DIR"
log "pids: $PID_DIR"

log "starting ROS state UDP bridge..."
ORIN_IP="$ORIN_IP" \
X86_IP="$X86_IP" \
STATE_PORT="$STATE_PORT" \
STATE_HZ="$STATE_HZ" \
JZ_STATE_HZ_PROFILE="$JZ_STATE_HZ_PROFILE" \
JZ_EXPECTED_STATE_HZ="$JZ_EXPECTED_STATE_HZ" \
JZ_TIMED_NON_30_STATE_HZ_CONFIRM="$JZ_TIMED_NON_30_STATE_HZ_CONFIRM" \
MAX_SOURCE_AGE_MS="$MAX_SOURCE_AGE_MS" \
MAX_SOURCE_SKEW_MS="$MAX_SOURCE_SKEW_MS" \
REQUIRE_ALL_SOURCES_ADVANCED="$REQUIRE_ALL_SOURCES_ADVANCED" \
STATE_WAIT_TIMEOUT_S="$STATE_WAIT_TIMEOUT_S" \
PYTHON_CMD="$PYTHON_CMD" \
AUTO_TAIL=0 \
  bash "$ORIN_ARM_DIR/start.sh"
sleep 0.5
ensure_alive "ros_state_udp_bridge" "$PID_DIR/ros_state_udp_bridge.pid" "$LOG_DIR/ros_state_udp_bridge.log"
verify_bridge_startup
wait_for_log_pattern \
  "ros_state_udp_bridge" \
  "$PID_DIR/ros_state_udp_bridge.pid" \
  "$LOG_DIR/ros_state_udp_bridge.log" \
  "local=" \
  "$BRIDGE_START_TIMEOUT_S"
wait_for_log_pattern \
  "ros_state_udp_bridge" \
  "$PID_DIR/ros_state_udp_bridge.pid" \
  "$LOG_DIR/ros_state_udp_bridge.log" \
  "sent seq=" \
  "$STATE_READY_TIMEOUT_S"
verify_bridge_metrics

log "starting Phase 3 command executor..."
CONFIG="$CONFIG" \
EXECUTION=armed \
JZ_UDP_EXECUTOR_ARMED=1 \
AUTO_TAIL=0 \
ORIN_IP="$ORIN_IP" \
X86_IP="$X86_IP" \
COMMAND_PORT="$COMMAND_PORT" \
PYTHON_CMD="$PYTHON_CMD" \
  bash "$ORIN_ARM_DIR/start_phase3_executor.sh"
sleep 0.5
ensure_alive "orin_phase3_command_executor" \
  "$PID_DIR/orin_phase3_command_executor.pid" \
  "$LOG_DIR/orin_phase3_command_executor.log"
wait_for_log_pattern \
  "orin_phase3_command_executor" \
  "$PID_DIR/orin_phase3_command_executor.pid" \
  "$LOG_DIR/orin_phase3_command_executor.log" \
  "PHASE3 COMMAND EXECUTOR ARMED" \
  "$EXECUTOR_READY_TIMEOUT_S"
wait_for_log_pattern \
  "orin_phase3_command_executor" \
  "$PID_DIR/orin_phase3_command_executor.pid" \
  "$LOG_DIR/orin_phase3_command_executor.log" \
  "port=$COMMAND_PORT" \
  "$EXECUTOR_READY_TIMEOUT_S"

log "replay services ready."
log "stop with: bash udp_test/all/stop_replay.sh"
trap - ERR

if [[ "$TAIL" == "1" ]]; then
  log "following logs now. Ctrl-C exits tail only; use stop_replay.sh to stop services."
  tail -f "$LOG_DIR/ros_state_udp_bridge.log" "$LOG_DIR/orin_phase3_command_executor.log"
fi
