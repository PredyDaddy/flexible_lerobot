#!/usr/bin/env bash
set -euo pipefail

# READONLY VR command topic probe.
# This script only runs ros2 topic list/info/hz/echo and writes logs.
# It must not publish ROS topics, send UDP commands, or call robot control APIs.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="$ROOT_DIR/udp_test/vr_test/logs"
mkdir -p "$LOG_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-$LOG_DIR/vr_command_probe_${STAMP}.log}"
HZ_DURATION_S="${HZ_DURATION_S:-3}"
ECHO_TIMEOUT_S="${ECHO_TIMEOUT_S:-2}"
KEYWORDS="${KEYWORDS:-command|cmd|joint|gripper|telecon|arm|hand|left|right|vel|twist|control|target|goal}"

log() {
  printf '%s\n' "$*" | tee -a "$LOG_FILE"
}

run_logged() {
  log ""
  log "===== $* ====="
  "$@" 2>&1 | tee -a "$LOG_FILE" || log "[probe] command failed exit=$?: $*"
}

require_ros2() {
  if ! command -v ros2 >/dev/null 2>&1; then
    log "[probe] ERROR: ros2 command not found. Source the ROS environment first."
    exit 2
  fi
}

topic_type() {
  local topic="$1"
  ros2 topic info "$topic" 2>/dev/null | awk -F': ' '/Type:/ {print $2; exit}'
}

main() {
  require_ros2

  log "VR COMMAND TOPIC PROBE - READONLY ONLY"
  log "This script only observes ROS topics."
  log "It does not publish, does not call send_action, and does not move the robot."
  log "log_file=$LOG_FILE"
  log "hz_duration_s=$HZ_DURATION_S echo_timeout_s=$ECHO_TIMEOUT_S"
  log "keywords=$KEYWORDS"
  log "started_at=$(date --iso-8601=seconds)"
  log "hostname=$(hostname)"
  log "user=$(id -un)"
  log "pwd=$(pwd)"

  run_logged ros2 topic list -t

  mapfile -t topics < <(ros2 topic list 2>/dev/null | sort)
  log ""
  log "===== topic info for all topics ====="
  for topic in "${topics[@]}"; do
    log ""
    log "--- topic: $topic ---"
    ros2 topic info "$topic" 2>&1 | tee -a "$LOG_FILE" || log "[probe] topic info failed: $topic"
  done

  mapfile -t candidate_topics < <(
    printf '%s\n' "${topics[@]}" | grep -E -i "$KEYWORDS" | sort -u || true
  )

  log ""
  log "===== candidate topics ====="
  if ((${#candidate_topics[@]} == 0)); then
    log "[probe] no candidate topics matched keywords"
  else
    printf '%s\n' "${candidate_topics[@]}" | tee -a "$LOG_FILE"
  fi

  log ""
  log "===== candidate hz samples ====="
  for topic in "${candidate_topics[@]}"; do
    log ""
    log "--- hz: $topic ---"
    timeout "${HZ_DURATION_S}s" ros2 topic hz "$topic" 2>&1 | tee -a "$LOG_FILE" || \
      log "[probe] hz sample ended or timed out: $topic"
  done

  log ""
  log "===== candidate echo samples ====="
  for topic in "${candidate_topics[@]}"; do
    type="$(topic_type "$topic" || true)"
    log ""
    log "--- echo once: $topic type=${type:-unknown} ---"
    timeout "${ECHO_TIMEOUT_S}s" ros2 topic echo --once "$topic" 2>&1 | tee -a "$LOG_FILE" || \
      log "[probe] echo sample ended or timed out: $topic"
  done

  log ""
  log "finished_at=$(date --iso-8601=seconds)"
  log "SUMMARY: READONLY probe complete log_file=$LOG_FILE candidates=${#candidate_topics[@]}"
}

main "$@"
