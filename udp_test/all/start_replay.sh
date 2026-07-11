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
STATE_PROCESS_START_TIMEOUT_S="${STATE_PROCESS_START_TIMEOUT_S:-${BRIDGE_START_TIMEOUT_S}}"
STATE_WAIT_TIMEOUT_S="${STATE_WAIT_TIMEOUT_S:-${LEGACY_READY_TIMEOUT_S:-15}}"
STATE_READY_TIMEOUT_S="${STATE_READY_TIMEOUT_S:-${LEGACY_READY_TIMEOUT_S:-20}}"
EXECUTOR_READY_TIMEOUT_S="${EXECUTOR_READY_TIMEOUT_S:-${LEGACY_READY_TIMEOUT_S:-15}}"
FINAL_READY_TIMEOUT_S="${FINAL_READY_TIMEOUT_S:-${STATE_READY_TIMEOUT_S}}"
MIN_MEASURED_STATE_HZ_RATIO="${MIN_MEASURED_STATE_HZ_RATIO:-0.9}"
JZ_TIMED_MIN_MEASURED_STATE_HZ_RATIO_CONFIRM="${JZ_TIMED_MIN_MEASURED_STATE_HZ_RATIO_CONFIRM:-}"

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

validate_positive_integer() {
  local name="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    log "invalid $name=$value; expected a positive integer"
    return 2
  fi
}

validate_rate_ratio() {
  if [[ ! "$MIN_MEASURED_STATE_HZ_RATIO" =~ ^(0([.][0-9]+)?|1([.]0+)?)$ ]] || \
     ! awk -v ratio="$MIN_MEASURED_STATE_HZ_RATIO" 'BEGIN { exit !(ratio > 0 && ratio <= 1) }'; then
    log "invalid MIN_MEASURED_STATE_HZ_RATIO=$MIN_MEASURED_STATE_HZ_RATIO; expected 0 < ratio <= 1"
    return 2
  fi
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
      if ! awk -v ratio="$MIN_MEASURED_STATE_HZ_RATIO" 'BEGIN { exit !(ratio == 0.9) }' && \
         [[ "$JZ_TIMED_MIN_MEASURED_STATE_HZ_RATIO_CONFIRM" != "$MIN_MEASURED_STATE_HZ_RATIO" ]]; then
        log "timed profile refuses unconfirmed MIN_MEASURED_STATE_HZ_RATIO=$MIN_MEASURED_STATE_HZ_RATIO"
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
  for ((index = 1; index < ${#argv[@]}; index++)); do
    if [[ "${argv[index - 1]##*/}" == python* ]] && \
       [[ "${argv[index]}" == "udp_test/test_scripts/arm_side/orin_ros_state_udp_bridge.py" || \
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
  local profile="" requested_hz="" expected_hz="" configured_hz="" bridge_pid="" launcher_pid=""

  [[ -f "$startup_file" ]] || { log "missing bridge startup metadata: $startup_file"; return 1; }
  while IFS='=' read -r key value; do
    case "$key" in
      profile) profile="$value" ;;
      requested_hz) requested_hz="$value" ;;
      expected_hz) expected_hz="$value" ;;
      configured_hz) configured_hz="$value" ;;
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
  [[ "$configured_hz" == "$expected_normalized" ]] || { log "bridge configured_hz metadata mismatch expected=$expected_normalized actual=$configured_hz"; return 1; }
  argv_hz="$(bridge_process_hz "$pid")" || { log "could not read bridge --hz from pid=$pid"; return 1; }
  [[ "$argv_hz" == "$expected_normalized" ]] || { log "bridge argv hz mismatch expected=$expected_normalized actual=$argv_hz"; return 1; }
  grep -Fq "pid=$pid local=" "$log_file" || { log "bridge log does not match pid=$pid"; return 1; }
  grep -Fq "configured_hz=$argv_hz" "$log_file" || {
    log "bridge log configured hz mismatch requested=$requested_normalized configured=$argv_hz"
    return 1
  }
  log "bridge configured hz verified pid=$pid requested_hz=$requested_normalized configured_hz=$argv_hz"
}

verify_bridge_metrics() {
  local log_file="$LOG_DIR/ros_state_udp_bridge.log"
  local line
  local token
  line="$(grep -F "sent seq=" "$log_file" | tail -n 1 || true)"
  for token in configured_hz= measured_send_hz= rate_window_packets= update_counts= progress_modes= source_age_ms= source_skew_ms= skipped=; do
    [[ "$line" == *"$token"* ]] || { log "bridge sent metrics missing $token"; return 1; }
  done
  for token in left_joints right_joints left_gripper right_gripper; do
    [[ "$line" == *"$token"* ]] || { log "bridge sent metrics missing source $token"; return 1; }
  done
  log "bridge sent metrics verified"
}

metric_value() {
  local line="$1"
  local key="$2"
  local token
  for token in $line; do
    if [[ "$token" == "$key="* ]]; then
      printf '%s\n' "${token#*=}"
      return 0
    fi
  done
  return 1
}

BRIDGE_RATE_SEQ=0
BRIDGE_MEASURED_HZ=""

wait_for_bridge_rate_after() {
  local minimum_seq="$1"
  local timeout_s="$2"
  local pid_file="$PID_DIR/ros_state_udp_bridge.pid"
  local log_file="$LOG_DIR/ros_state_udp_bridge.log"
  local pid deadline line seq configured_hz measured_hz window_packets minimum_hz
  local last_measurement="missing"

  pid="$(cat "$pid_file")"
  deadline=$((SECONDS + timeout_s))
  minimum_hz="$(awk -v expected="$JZ_EXPECTED_STATE_HZ" -v ratio="$MIN_MEASURED_STATE_HZ_RATIO" \
    'BEGIN { printf "%.12g\n", expected * ratio }')"
  while ((SECONDS < deadline)); do
    if ! kill -0 "$pid" 2>/dev/null; then
      log "ros_state_udp_bridge exited while waiting for measured send rate"
      show_log_tail "$log_file"
      return 1
    fi
    line="$(grep -F "rate pid=$pid " "$log_file" | tail -n 1 || true)"
    if [[ -n "$line" ]]; then
      seq="$(metric_value "$line" seq || true)"
      configured_hz="$(metric_value "$line" configured_hz || true)"
      measured_hz="$(metric_value "$line" measured_send_hz || true)"
      window_packets="$(metric_value "$line" window_packets || true)"
      last_measurement="seq=${seq:-invalid} measured_send_hz=${measured_hz:-invalid}"
      if [[ "$seq" =~ ^[0-9]+$ ]] && ((seq > minimum_seq)) && \
         [[ "$window_packets" =~ ^[0-9]+$ ]] && ((window_packets >= 30)) && \
         [[ -n "$measured_hz" ]] && \
         awk -v configured="$configured_hz" -v expected="$JZ_EXPECTED_STATE_HZ" \
           -v measured="$measured_hz" -v minimum="$minimum_hz" \
           'BEGIN { exit !(configured == expected && measured >= minimum) }'; then
        BRIDGE_RATE_SEQ="$seq"
        BRIDGE_MEASURED_HZ="$measured_hz"
        log "bridge measured rate verified pid=$pid seq=$seq configured_hz=$configured_hz measured_send_hz=$measured_hz minimum_hz=$minimum_hz window_packets=$window_packets"
        return 0
      fi
    fi
    sleep 0.2
  done

  log "bridge measured send rate not ready within ${timeout_s}s: minimum_seq=$minimum_seq minimum_hz=$minimum_hz last=$last_measurement"
  show_log_tail "$log_file"
  return 1
}

validate_rate_ratio
validate_hz_contract
validate_positive_integer BRIDGE_START_TIMEOUT_S "$BRIDGE_START_TIMEOUT_S"
validate_positive_integer STATE_PROCESS_START_TIMEOUT_S "$STATE_PROCESS_START_TIMEOUT_S"
validate_positive_integer STATE_WAIT_TIMEOUT_S "$STATE_WAIT_TIMEOUT_S"
validate_positive_integer STATE_READY_TIMEOUT_S "$STATE_READY_TIMEOUT_S"
validate_positive_integer EXECUTOR_READY_TIMEOUT_S "$EXECUTOR_READY_TIMEOUT_S"
validate_positive_integer FINAL_READY_TIMEOUT_S "$FINAL_READY_TIMEOUT_S"
trap cleanup_on_error ERR

case "$EXECUTION" in
  armed)
    if [[ "${JZ_UDP_EXECUTOR_ARMED:-}" != "1" ]]; then
      log "armed replay requires explicit JZ_UDP_EXECUTOR_ARMED=1; actual=${JZ_UDP_EXECUTOR_ARMED:-missing}"
      exit 2
    fi
    export JZ_UDP_EXECUTOR_ARMED
    EXECUTOR_READY_PATTERN="PHASE3 COMMAND EXECUTOR ARMED"
    ;;
  dry_run)
    EXECUTOR_READY_PATTERN="PHASE3 COMMAND EXECUTOR DRY-RUN"
    ;;
  *)
    log "unsupported EXECUTION=$EXECUTION; expected armed or dry_run"
    exit 2
    ;;
esac

log "mode=replay"
log "ORIN_IP=$ORIN_IP X86_IP=$X86_IP STATE_PORT=$STATE_PORT COMMAND_PORT=$COMMAND_PORT"
log "profile=$JZ_STATE_HZ_PROFILE requested_hz=$(normalize_hz "$STATE_HZ") expected_hz=$(normalize_hz "$JZ_EXPECTED_STATE_HZ")"
log "max_source_age_ms=$MAX_SOURCE_AGE_MS max_source_skew_ms=$MAX_SOURCE_SKEW_MS require_all_sources_advanced=$REQUIRE_ALL_SOURCES_ADVANCED"
log "CONFIG=$CONFIG"
log "EXECUTION=$EXECUTION JZ_UDP_EXECUTOR_ARMED=${JZ_UDP_EXECUTOR_ARMED:-}"
log "PYTHON_CMD=$PYTHON_CMD"
log "READY_TIMEOUT_S=$READY_TIMEOUT_S"
log "BRIDGE_START_TIMEOUT_S=$BRIDGE_START_TIMEOUT_S"
log "STATE_PROCESS_START_TIMEOUT_S=$STATE_PROCESS_START_TIMEOUT_S"
log "STATE_WAIT_TIMEOUT_S=$STATE_WAIT_TIMEOUT_S"
log "STATE_READY_TIMEOUT_S=$STATE_READY_TIMEOUT_S"
log "EXECUTOR_READY_TIMEOUT_S=$EXECUTOR_READY_TIMEOUT_S"
log "FINAL_READY_TIMEOUT_S=$FINAL_READY_TIMEOUT_S"
log "MIN_MEASURED_STATE_HZ_RATIO=$MIN_MEASURED_STATE_HZ_RATIO"
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
STATE_PROCESS_START_TIMEOUT_S="$STATE_PROCESS_START_TIMEOUT_S" \
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
wait_for_bridge_rate_after 0 "$STATE_READY_TIMEOUT_S"
INITIAL_BRIDGE_RATE_SEQ="$BRIDGE_RATE_SEQ"
verify_bridge_metrics

log "starting Phase 3 command executor..."
CONFIG="$CONFIG" \
EXECUTION="$EXECUTION" \
JZ_UDP_EXECUTOR_ARMED="${JZ_UDP_EXECUTOR_ARMED:-}" \
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
  "$EXECUTOR_READY_PATTERN" \
  "$EXECUTOR_READY_TIMEOUT_S"
wait_for_log_pattern \
  "orin_phase3_command_executor" \
  "$PID_DIR/orin_phase3_command_executor.pid" \
  "$LOG_DIR/orin_phase3_command_executor.log" \
  "port=$COMMAND_PORT" \
  "$EXECUTOR_READY_TIMEOUT_S"

ensure_alive "ros_state_udp_bridge" "$PID_DIR/ros_state_udp_bridge.pid" "$LOG_DIR/ros_state_udp_bridge.log"
ensure_alive "orin_phase3_command_executor" \
  "$PID_DIR/orin_phase3_command_executor.pid" \
  "$LOG_DIR/orin_phase3_command_executor.log"
verify_bridge_startup
wait_for_bridge_rate_after "$INITIAL_BRIDGE_RATE_SEQ" "$FINAL_READY_TIMEOUT_S"
verify_bridge_metrics
ensure_alive "ros_state_udp_bridge" "$PID_DIR/ros_state_udp_bridge.pid" "$LOG_DIR/ros_state_udp_bridge.log"
ensure_alive "orin_phase3_command_executor" \
  "$PID_DIR/orin_phase3_command_executor.pid" \
  "$LOG_DIR/orin_phase3_command_executor.log"

log "replay services ready."
log "stop with: bash udp_test/all/stop_replay.sh"
trap - ERR

if [[ "$TAIL" == "1" ]]; then
  log "following logs now. Ctrl-C exits tail only; use stop_replay.sh to stop services."
  tail -f "$LOG_DIR/ros_state_udp_bridge.log" "$LOG_DIR/orin_phase3_command_executor.log"
fi
