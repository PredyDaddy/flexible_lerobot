#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

infer_require_policy
infer_require_reference_dataset

DEVICE="${DEVICE:-cuda}"
SAMPLE_INDICES="${SAMPLE_INDICES:-first,middle,last}"
OUTPUT_JSON="${OUTPUT_JSON:-}"

CMD=(
  "${CONDA_PYTHON}"
  "${SCRIPT_DIR}/offline_infer.py"
  "--policy-path=${POLICY_PATH}"
  "--dataset-root=${REFERENCE_DATASET_ROOT}"
  "--dataset-repo-id=${REFERENCE_DATASET_REPO_ID}"
  "--sample-indices=${SAMPLE_INDICES}"
  "--device=${DEVICE}"
)
if [[ -n "${OUTPUT_JSON}" ]]; then
  CMD+=("--output-json=${OUTPUT_JSON}")
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

echo "[jz_pin_timed/offline-infer] policy=${POLICY_PATH}"
echo "[jz_pin_timed/offline-infer] dataset=${REFERENCE_DATASET_ROOT} samples=${SAMPLE_INDICES}"
echo "[jz_pin_timed/offline-infer] device=${DEVICE}"
exec "${CMD[@]}"
