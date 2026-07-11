#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BASE_SCRIPT="${REPO_ROOT}/my_devs/jz_robot_pin/video_check/make_crf_comparison.sh"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/crf_comparison}"

if [[ ! -f "${BASE_SCRIPT}" ]]; then
  echo "[timed/video_check] missing shared CRF comparison script: ${BASE_SCRIPT}" >&2
  exit 1
fi

echo "[timed/video_check] reusing shared lossless-reference comparison; output_dir=${OUTPUT_DIR}"
OUTPUT_DIR="${OUTPUT_DIR}" exec bash "${BASE_SCRIPT}" "$@"

