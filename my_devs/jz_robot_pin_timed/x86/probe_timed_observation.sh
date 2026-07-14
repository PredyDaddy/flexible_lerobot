#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"

CONDA_ENV="${CONDA_ENV:-${LEROBOT_CONDA_ENV:-lerobot_flex}}"
ORIN_IP="${ORIN_IP:-192.168.1.81}"
STATE_BIND_IP="${STATE_BIND_IP:-0.0.0.0}"
STATE_PORT="${STATE_PORT:-39010}"
PROBE_DURATION_S="${PROBE_DURATION_S:-10}"
PROBE_FPS="${PROBE_FPS:-30}"
CAMERAS="${CAMERAS:-}"
STATE_ONLY="${STATE_ONLY:-0}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_JSONL="${OUTPUT_JSONL:-${TIMED_LOG_DIR}/timed_observation_${RUN_STAMP}.jsonl}"

timed_make_python_cmd "${CONDA_ENV}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

ARGS=(
  --orin-ip "${ORIN_IP}"
  --bind-ip "${STATE_BIND_IP}"
  --state-port "${STATE_PORT}"
  --duration-s "${PROBE_DURATION_S}"
  --fps "${PROBE_FPS}"
)

case "${STATE_ONLY,,}" in
  1|true|yes) ARGS+=(--state-only) ;;
  0|false|no) ;;
  *) echo "[timed/x86/probe] STATE_ONLY must be true or false" >&2; exit 2 ;;
esac

if [[ -n "${CAMERAS}" ]]; then
  read -r -a CAMERA_NAMES <<<"${CAMERAS}"
  for camera in "${CAMERA_NAMES[@]}"; do
    ARGS+=(--camera "${camera}")
  done
fi

if [[ "${OUTPUT_JSONL}" != "none" ]]; then
  ARGS+=(--output-jsonl "${OUTPUT_JSONL}")
fi

echo "[timed/x86/probe] READ ONLY: receives state and direct ZMQ cameras; never calls send_action"
echo "[timed/x86/probe] state=udp://${STATE_BIND_IP}:${STATE_PORT} expected=${ORIN_IP}"
echo "[timed/x86/probe] duration_s=${PROBE_DURATION_S} fps=${PROBE_FPS} cameras=${CAMERAS:-all}"
echo "[timed/x86/probe] output_jsonl=${OUTPUT_JSONL} conda_env=${CONDA_ENV}"

cd "${REPO_ROOT}"
exec "${PYTHON_CMD[@]}" "${SCRIPT_DIR}/probe_timed_observation.py" "${ARGS[@]}" "$@"
