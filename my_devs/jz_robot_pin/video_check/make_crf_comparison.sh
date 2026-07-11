#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

DEFAULT_INPUT="${REPO_ROOT}/tests/outputs/jz_robot_pin_real_20260710_180651/videos/observation.images.camera_right/chunk-000/file-000.mp4"
INPUT="${1:-${INPUT_VIDEO:-${DEFAULT_INPUT}}}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/crf_comparison_right}"
START_S="${START_S:-5}"
DURATION_S="${DURATION_S:-5}"
CRF_VALUES="${CRF_VALUES:-18 20 22 30}"
PRESET="${PRESET:-medium}"
GOP="${GOP:-2}"

if [[ ! -f "${INPUT}" ]]; then
  echo "[video_check] input video not found: ${INPUT}" >&2
  exit 2
fi
if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  echo "[video_check] ffmpeg and ffprobe are required" >&2
  exit 2
fi

read -r -a CRFS <<<"${CRF_VALUES}"
if [[ "${#CRFS[@]}" -ne 4 ]]; then
  echo "[video_check] CRF_VALUES must contain exactly four values for the 2x2 comparison" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"
REFERENCE="${OUTPUT_DIR}/reference_ffv1.mkv"
REPORT="${OUTPUT_DIR}/report.tsv"

echo "[video_check] input=${INPUT}"
echo "[video_check] segment=start:${START_S}s duration:${DURATION_S}s"
echo "[video_check] crf_values=${CRF_VALUES} preset=${PRESET} gop=${GOP}"
echo "[video_check] output_dir=${OUTPUT_DIR}"

# Decode the selected segment once into a lossless intermediate so every CRF
# candidate receives exactly the same frames.
ffmpeg -hide_banner -loglevel warning -y \
  -ss "${START_S}" -i "${INPUT}" -t "${DURATION_S}" -map 0:v:0 -an \
  -c:v ffv1 -level 3 -pix_fmt bgr0 \
  "${REFERENCE}"

printf 'crf\tbytes\tbit_rate_bps\tssim_all\tpsnr_average_db\tfile\n' >"${REPORT}"
VARIANTS=()

for crf in "${CRFS[@]}"; do
  output="${OUTPUT_DIR}/h264_crf${crf}_g${GOP}_yuv420p.mp4"
  VARIANTS+=("${output}")

  ffmpeg -hide_banner -loglevel warning -y \
    -i "${REFERENCE}" -map 0:v:0 -an \
    -c:v libx264 -preset "${PRESET}" -crf "${crf}" -g "${GOP}" \
    -pix_fmt yuv420p -movflags +faststart \
    "${output}"

  ssim="$({
    ffmpeg -hide_banner -i "${REFERENCE}" -i "${output}" \
      -lavfi '[0:v][1:v]ssim' -f null - 2>&1 || true
  } | sed -n 's/.* All:\([0-9.]*\) (.*/\1/p' | tail -n 1)"
  psnr="$({
    ffmpeg -hide_banner -i "${REFERENCE}" -i "${output}" \
      -lavfi '[0:v][1:v]psnr' -f null - 2>&1 || true
  } | sed -n 's/.* average:\([0-9.]*\) .*/\1/p' | tail -n 1)"
  bytes="$(stat -c '%s' "${output}")"
  bit_rate="$(ffprobe -v error -select_streams v:0 -show_entries stream=bit_rate \
    -of default=noprint_wrappers=1:nokey=1 "${output}")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${crf}" "${bytes}" "${bit_rate:-N/A}" "${ssim:-N/A}" "${psnr:-N/A}" "${output}" \
    >>"${REPORT}"
done

GRID="${OUTPUT_DIR}/compare_crf_18_20_22_30.mp4"
ffmpeg -hide_banner -loglevel warning -y \
  -i "${VARIANTS[0]}" -i "${VARIANTS[1]}" -i "${VARIANTS[2]}" -i "${VARIANTS[3]}" \
  -filter_complex "\
    [0:v]drawtext=text='CRF ${CRFS[0]}':x=16:y=16:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.65[v0];\
    [1:v]drawtext=text='CRF ${CRFS[1]}':x=16:y=16:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.65[v1];\
    [2:v]drawtext=text='CRF ${CRFS[2]}':x=16:y=16:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.65[v2];\
    [3:v]drawtext=text='CRF ${CRFS[3]}':x=16:y=16:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.65[v3];\
    [v0][v1][v2][v3]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[v]" \
  -map '[v]' -an -c:v libx264 -preset medium -crf 14 -g 30 -pix_fmt yuv420p -movflags +faststart \
  "${GRID}"

ffmpeg -hide_banner -loglevel warning -y -ss "$((DURATION_S / 2))" -i "${GRID}" \
  -frames:v 1 -update 1 -q:v 2 "${OUTPUT_DIR}/compare_crf_18_20_22_30.jpg"

echo "[video_check] completed"
echo "[video_check] individual videos are authoritative; the grid is re-encoded for convenient viewing"
echo "[video_check] report=${REPORT}"
echo "[video_check] grid=${GRID}"

