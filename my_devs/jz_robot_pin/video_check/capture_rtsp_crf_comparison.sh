#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAMERA="${1:-${CAMERA:-right}}"
DURATION_S="${DURATION_S:-10}"
ORIN_IP="${ORIN_IP:-192.168.1.81}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/rtsp_${CAMERA}_${RUN_STAMP}}"

case "${CAMERA}" in
  head | left | right) ;;
  *)
    echo "[video_check] camera must be head, left, or right; got: ${CAMERA}" >&2
    exit 2
    ;;
esac

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "[video_check] ffmpeg is required" >&2
  exit 2
fi

RTSP_URL="rtsp://${ORIN_IP}:8554/robot_camera/camera_${CAMERA}"
SOURCE_VIDEO="${OUTPUT_DIR}/camera_${CAMERA}_source_copy.mkv"
COMPARISON_DIR="${OUTPUT_DIR}/crf_comparison"
mkdir -p "${OUTPUT_DIR}"

echo "[video_check] read-only RTSP capture"
echo "[video_check] url=${RTSP_URL} duration=${DURATION_S}s"
echo "[video_check] source_video=${SOURCE_VIDEO}"

# Stream copy preserves the RTSP H.264 payload and avoids introducing another
# encoder before the CRF comparison.
ffmpeg -hide_banner -loglevel warning -y -rtsp_transport tcp \
  -i "${RTSP_URL}" -t "${DURATION_S}" -map 0:v:0 -an -c:v copy \
  "${SOURCE_VIDEO}"

START_S=0 \
DURATION_S="${DURATION_S}" \
OUTPUT_DIR="${COMPARISON_DIR}" \
  bash "${SCRIPT_DIR}/make_crf_comparison.sh" "${SOURCE_VIDEO}"

echo "[video_check] source and CRF comparison completed"
echo "[video_check] output_dir=${OUTPUT_DIR}"

