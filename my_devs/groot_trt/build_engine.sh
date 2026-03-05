#!/usr/bin/env bash
set -euo pipefail

# Build GR00T TensorRT engines from ONNX exports using the TensorRT Python API.
#
# Why Python API:
# - Some dev machines do not have `trtexec` installed system-wide.
# - We want a repo-local, reproducible build path that works inside `lerobot_flex`.
#
# Required layout:
#   ${ONNX_DIR}/eagle2/vit_fp16.onnx
#   ${ONNX_DIR}/eagle2/llm_fp16.onnx
#   ${ONNX_DIR}/action_head/vlln_vl_self_attention.onnx
#   ${ONNX_DIR}/action_head/state_encoder.onnx
#   ${ONNX_DIR}/action_head/action_encoder.onnx
#   ${ONNX_DIR}/action_head/DiT_fp16.onnx
#   ${ONNX_DIR}/action_head/action_decoder.onnx
#
# Outputs:
#   ${ENGINE_DIR}/*.engine
#   ${ENGINE_DIR}/build_report.json

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

CONDA_ENV="${CONDA_ENV:-lerobot_flex}"
ONNX_DIR="${ONNX_DIR:-gr00t_onnx}"
ENGINE_DIR="${ENGINE_DIR:-gr00t_engine_api_trt1013}"

VIDEO_VIEWS="${VIDEO_VIEWS:-2}"  # Used only to choose defaults (1 or 2).
MAX_BATCH="${MAX_BATCH:-2}"
VIT_OPT_BATCH="${VIT_OPT_BATCH:-${VIDEO_VIEWS}}"
OPT_BATCH="${OPT_BATCH:-1}"

WORKSPACE_GB="${WORKSPACE_GB:-8}"

# Default profile lengths (override by exporting MIN_LEN/OPT_LEN/MAX_LEN).
if [[ -z "${MIN_LEN:-}" || -z "${OPT_LEN:-}" || -z "${MAX_LEN:-}" ]]; then
  if [[ "${VIDEO_VIEWS}" == "2" ]]; then
    MIN_LEN="${MIN_LEN:-80}"
    OPT_LEN="${OPT_LEN:-568}"
    MAX_LEN="${MAX_LEN:-600}"
  else
    MIN_LEN="${MIN_LEN:-80}"
    OPT_LEN="${OPT_LEN:-296}"
    MAX_LEN="${MAX_LEN:-300}"
  fi
fi

# Repo-local TensorRT install directory (pip --target ...).
TENSORRT_PY_DIR="${TENSORRT_PY_DIR:-/data/cqy_workspace/third_party/tensorrt_10_13_0_35}"

# Avoid writing temp files to `/tmp` (root filesystem may be full on some machines).
TMPDIR="${TMPDIR:-/data/cqy_workspace/tmp}"

echo "[INFO] REPO_ROOT: ${REPO_ROOT}"
echo "[INFO] CONDA_ENV: ${CONDA_ENV}"
echo "[INFO] ONNX_DIR: ${ONNX_DIR}"
echo "[INFO] ENGINE_DIR: ${ENGINE_DIR}"
echo "[INFO] VIDEO_VIEWS: ${VIDEO_VIEWS}"
echo "[INFO] MAX_BATCH: ${MAX_BATCH}"
echo "[INFO] VIT_OPT_BATCH: ${VIT_OPT_BATCH}"
echo "[INFO] OPT_BATCH: ${OPT_BATCH}"
echo "[INFO] MIN/OPT/MAX_LEN: ${MIN_LEN}/${OPT_LEN}/${MAX_LEN}"
echo "[INFO] WORKSPACE_GB: ${WORKSPACE_GB}"
echo "[INFO] TENSORRT_PY_DIR: ${TENSORRT_PY_DIR}"
echo "[INFO] TMPDIR: ${TMPDIR}"

mkdir -p "${ENGINE_DIR}"

cd "${REPO_ROOT}"

conda run -n "${CONDA_ENV}" env \
  TMPDIR="${TMPDIR}" \
  TENSORRT_PY_DIR="${TENSORRT_PY_DIR}" \
  python my_devs/groot_trt/build_engine.py \
    --onnx-dir "${ONNX_DIR}" \
    --engine-out-dir "${ENGINE_DIR}" \
    --max-batch "${MAX_BATCH}" \
    --vit-opt-batch "${VIT_OPT_BATCH}" \
    --opt-batch "${OPT_BATCH}" \
    --min-seq-len "${MIN_LEN}" \
    --opt-seq-len "${OPT_LEN}" \
    --max-seq-len "${MAX_LEN}" \
    --workspace-gb "${WORKSPACE_GB}"

echo "[OK] Engines saved in: ${ENGINE_DIR}/"

