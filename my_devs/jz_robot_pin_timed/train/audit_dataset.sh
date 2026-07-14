#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../" && pwd)"
CONDA_ENV="${CONDA_ENV:-lerobot_flex}"
CONDA_BIN="${CONDA_BIN:-$(command -v conda 2>/dev/null || true)}"
if [[ -z "${CONDA_BIN}" ]]; then
  for candidate in "${HOME}/miniconda3/bin/conda" "${HOME}/anaconda3/bin/conda" /opt/conda/bin/conda; do
    if [[ -x "${candidate}" ]]; then
      CONDA_BIN="${candidate}"
      break
    fi
  done
fi
DATASET_NAME="${DATASET_NAME:-jz_robot_pin_timed_curated_42eps_20260713}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/tests/outputs/${DATASET_NAME}}"
FULL_VIDEO_DECODE="${FULL_VIDEO_DECODE:-1}"
REPORT_DIR="${REPORT_DIR:-${REPO_ROOT}/tests/outputs/audits/${DATASET_NAME}_$(date +%Y%m%d_%H%M%S)}"

if [[ ! -f "${DATASET_ROOT}/meta/info.json" ]]; then
  echo "[jz_pin_timed/audit] missing dataset: ${DATASET_ROOT}" >&2
  exit 2
fi
if [[ -z "${CONDA_BIN}" || ! -x "${CONDA_BIN}" ]]; then
  echo "[jz_pin_timed/audit] conda executable was not found" >&2
  exit 2
fi
if [[ "${FULL_VIDEO_DECODE}" != "0" && "${FULL_VIDEO_DECODE}" != "1" ]]; then
  echo "[jz_pin_timed/audit] FULL_VIDEO_DECODE must be 0 or 1" >&2
  exit 2
fi

read -r EPISODES FPS < <(
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" python - "${DATASET_ROOT}/meta/info.json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as stream:
    info = json.load(stream)
print(info["total_episodes"], info["fps"])
PY
)

mkdir -p "${REPORT_DIR}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

echo "[jz_pin_timed/audit] dataset=${DATASET_ROOT} reports=${REPORT_DIR}"
"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" python \
  my_devs/jz_robot_pin_timed/data_check/check_3_episodes.py \
  --dataset-root "${DATASET_ROOT}" \
  --expected-episodes "${EPISODES}" \
  --expected-episode-time-s 10 \
  --expected-fps "${FPS}" \
  --max-initial-joint-delta-rad 10 \
  --max-action-joint-step-rad 10 \
  --max-lag-p95-rad 0.05 \
  --report-json "${REPORT_DIR}/data_check_report.json"

"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" python \
  my_devs/jz_robot_pin_timed/data_check/check_timing.py \
  --dataset-root "${DATASET_ROOT}" \
  --expected-codec h264 \
  --expected-crf 18 \
  --expected-camera-fps "${FPS}" \
  --expected-camera-source-fps 30 \
  --min-camera-source-fps-ratio 0.9 \
  --expected-camera-protocol jz_realsense_zmq \
  --expected-command-mode armed \
  --expected-command-transport udp \
  --expected-action-key-count 18 \
  --require-source-timing \
  --max-source-age-ms 50 \
  --max-source-skew-ms 20 \
  --max-camera-age-ms 1000 \
  --max-camera-state-skew-ms 100 \
  --report-json "${REPORT_DIR}/timing_check_report.json"

"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" python \
  my_devs/jz_robot_pin_timed/data_check/check_training_projection.py \
  --dataset-root "${DATASET_ROOT}" \
  --report-json "${REPORT_DIR}/training_projection_report.json"

"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" python \
  "${SCRIPT_DIR}/check_training_readiness.py" \
  --dataset-root "${DATASET_ROOT}" \
  --report-json "${REPORT_DIR}/training_readiness.json"

if [[ "${FULL_VIDEO_DECODE}" == "1" ]]; then
  decoded=0
  while IFS= read -r -d '' video; do
    echo "[jz_pin_timed/audit] decode=${video}"
    ffmpeg -nostdin -v error -i "${video}" -f null -
    decoded=$((decoded + 1))
  done < <(find "${DATASET_ROOT}/videos" -type f -name '*.mp4' -print0)
  echo "decoded_video_files=${decoded}" | tee "${REPORT_DIR}/video_decode.txt"
fi

echo "status=PASS report_dir=${REPORT_DIR}"
