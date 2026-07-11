#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BASE_SCRIPT="${REPO_ROOT}/my_devs/jz_robot_pin/video_check/capture_rtsp_crf_comparison.sh"
CAMERA="${1:-${CAMERA:-right}}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/rtsp_${CAMERA}_${RUN_STAMP}}"

if [[ ! -f "${BASE_SCRIPT}" ]]; then
  echo "[timed/video_check] missing shared CRF capture script: ${BASE_SCRIPT}" >&2
  exit 1
fi

echo "[timed/video_check] read-only RTSP capture; output_dir=${OUTPUT_DIR}"
OUTPUT_DIR="${OUTPUT_DIR}" RUN_STAMP="${RUN_STAMP}" exec bash "${BASE_SCRIPT}" "$@"

