#!/usr/bin/env bash
set -euo pipefail

# Build GR00T TensorRT engines from ONNX files.
#
# This script is adapted from upstream Isaac-GR00T deployment scripts, but:
# - supports configurable ONNX/engine directories
# - uses `set -euo pipefail` so failures don't silently pass
#
# Required layout:
#   ${ONNX_DIR}/eagle2/vit_${VIT_DTYPE}.onnx
#   ${ONNX_DIR}/eagle2/llm_${LLM_DTYPE}.onnx
#   ${ONNX_DIR}/action_head/vlln_vl_self_attention.onnx
#   ${ONNX_DIR}/action_head/state_encoder.onnx
#   ${ONNX_DIR}/action_head/action_encoder.onnx
#   ${ONNX_DIR}/action_head/DiT_${DIT_DTYPE}.onnx
#   ${ONNX_DIR}/action_head/action_decoder.onnx

TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
ONNX_DIR="${ONNX_DIR:-gr00t_onnx}"
ENGINE_DIR="${ENGINE_DIR:-gr00t_engine}"

VIDEO_VIEWS="${VIDEO_VIEWS:-1}"  # For profile defaults only. 1 or 2.
MAX_BATCH="${MAX_BATCH:-8}"

# Default profile lengths (override by exporting MIN_LEN/OPT_LEN/MAX_LEN)
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

VIT_DTYPE="${VIT_DTYPE:-fp16}"     # fp16|fp8
LLM_DTYPE="${LLM_DTYPE:-fp16}"     # fp16|nvfp4|nvfp4_full|fp8
DIT_DTYPE="${DIT_DTYPE:-fp16}"     # fp16|fp8

# Model dims (override if you export different ones)
BACKBONE_EMB_DIM="${BACKBONE_EMB_DIM:-2048}"
SA_EMB_DIM="${SA_EMB_DIM:-1536}"
SA_SEQ_LEN="${SA_SEQ_LEN:-49}"
MODEL_OUT_DIM="${MODEL_OUT_DIM:-1024}"
STATE_DIM="${STATE_DIM:-64}"
ACTION_HORIZON="${ACTION_HORIZON:-16}"
ACTION_DIM="${ACTION_DIM:-32}"

require_file() {
  local p="$1"
  if [[ ! -f "${p}" ]]; then
    echo "[ERROR] Missing file: ${p}" >&2
    exit 1
  fi
}

if [[ ! -x "${TRTEXEC}" ]]; then
  echo "[ERROR] trtexec not found or not executable: ${TRTEXEC}" >&2
  exit 1
fi

mkdir -p "${ENGINE_DIR}"

require_file "${ONNX_DIR}/action_head/vlln_vl_self_attention.onnx"
require_file "${ONNX_DIR}/action_head/state_encoder.onnx"
require_file "${ONNX_DIR}/action_head/action_encoder.onnx"
require_file "${ONNX_DIR}/action_head/DiT_${DIT_DTYPE}.onnx"
require_file "${ONNX_DIR}/action_head/action_decoder.onnx"

if [[ -f "${ONNX_DIR}/eagle2/vit_${VIT_DTYPE}.onnx" ]]; then
  HAS_BACKBONE=1
else
  HAS_BACKBONE=0
fi
if [[ -f "${ONNX_DIR}/eagle2/llm_${LLM_DTYPE}.onnx" ]]; then
  HAS_LLM=1
else
  HAS_LLM=0
fi

echo "[INFO] TRTEXEC: ${TRTEXEC}"
echo "[INFO] ONNX_DIR: ${ONNX_DIR}"
echo "[INFO] ENGINE_DIR: ${ENGINE_DIR}"
echo "[INFO] VIDEO_VIEWS: ${VIDEO_VIEWS}"
echo "[INFO] MAX_BATCH: ${MAX_BATCH}"
echo "[INFO] MIN/OPT/MAX_LEN: ${MIN_LEN}/${OPT_LEN}/${MAX_LEN}"
echo "[INFO] DTYPE: vit=${VIT_DTYPE} llm=${LLM_DTYPE} dit=${DIT_DTYPE}"
echo "[INFO] DIMS: backbone=${BACKBONE_EMB_DIM} sa=(${SA_SEQ_LEN},${SA_EMB_DIM}) model_out=${MODEL_OUT_DIM}"

echo "[INFO] Building vlln_vl_self_attention.engine"
"${TRTEXEC}" --useCudaGraph --verbose --stronglyTyped --separateProfileRun --noDataTransfers \
  --onnx="${ONNX_DIR}/action_head/vlln_vl_self_attention.onnx" \
  --saveEngine="${ENGINE_DIR}/vlln_vl_self_attention.engine" \
  --minShapes="backbone_features:1x${MIN_LEN}x${BACKBONE_EMB_DIM}" \
  --optShapes="backbone_features:1x${OPT_LEN}x${BACKBONE_EMB_DIM}" \
  --maxShapes="backbone_features:${MAX_BATCH}x${MAX_LEN}x${BACKBONE_EMB_DIM}" \
  > "${ENGINE_DIR}/vlln_vl_self_attention.log" 2>&1

echo "[INFO] Building DiT_${DIT_DTYPE}.engine"
"${TRTEXEC}" --useCudaGraph --verbose --stronglyTyped --separateProfileRun --noDataTransfers \
  --onnx="${ONNX_DIR}/action_head/DiT_${DIT_DTYPE}.onnx" \
  --saveEngine="${ENGINE_DIR}/DiT_${DIT_DTYPE}.engine" \
  --minShapes="sa_embs:1x${SA_SEQ_LEN}x${SA_EMB_DIM},vl_embs:1x${MIN_LEN}x${BACKBONE_EMB_DIM},timesteps_tensor:1" \
  --optShapes="sa_embs:1x${SA_SEQ_LEN}x${SA_EMB_DIM},vl_embs:1x${OPT_LEN}x${BACKBONE_EMB_DIM},timesteps_tensor:1" \
  --maxShapes="sa_embs:${MAX_BATCH}x${SA_SEQ_LEN}x${SA_EMB_DIM},vl_embs:${MAX_BATCH}x${MAX_LEN}x${BACKBONE_EMB_DIM},timesteps_tensor:${MAX_BATCH}" \
  > "${ENGINE_DIR}/DiT_${DIT_DTYPE}.log" 2>&1

echo "[INFO] Building state_encoder.engine"
"${TRTEXEC}" --useCudaGraph --verbose --stronglyTyped --separateProfileRun --noDataTransfers \
  --onnx="${ONNX_DIR}/action_head/state_encoder.onnx" \
  --saveEngine="${ENGINE_DIR}/state_encoder.engine" \
  --minShapes="state:1x1x${STATE_DIM},embodiment_id:1" \
  --optShapes="state:1x1x${STATE_DIM},embodiment_id:1" \
  --maxShapes="state:${MAX_BATCH}x1x${STATE_DIM},embodiment_id:${MAX_BATCH}" \
  > "${ENGINE_DIR}/state_encoder.log" 2>&1

echo "[INFO] Building action_encoder.engine"
"${TRTEXEC}" --useCudaGraph --verbose --stronglyTyped --separateProfileRun --noDataTransfers \
  --onnx="${ONNX_DIR}/action_head/action_encoder.onnx" \
  --saveEngine="${ENGINE_DIR}/action_encoder.engine" \
  --minShapes="actions:1x${ACTION_HORIZON}x${ACTION_DIM},timesteps_tensor:1,embodiment_id:1" \
  --optShapes="actions:1x${ACTION_HORIZON}x${ACTION_DIM},timesteps_tensor:1,embodiment_id:1" \
  --maxShapes="actions:${MAX_BATCH}x${ACTION_HORIZON}x${ACTION_DIM},timesteps_tensor:${MAX_BATCH},embodiment_id:${MAX_BATCH}" \
  > "${ENGINE_DIR}/action_encoder.log" 2>&1

echo "[INFO] Building action_decoder.engine"
"${TRTEXEC}" --useCudaGraph --verbose --stronglyTyped --separateProfileRun --noDataTransfers \
  --onnx="${ONNX_DIR}/action_head/action_decoder.onnx" \
  --saveEngine="${ENGINE_DIR}/action_decoder.engine" \
  --minShapes="model_output:1x${SA_SEQ_LEN}x${MODEL_OUT_DIM},embodiment_id:1" \
  --optShapes="model_output:1x${SA_SEQ_LEN}x${MODEL_OUT_DIM},embodiment_id:1" \
  --maxShapes="model_output:${MAX_BATCH}x${SA_SEQ_LEN}x${MODEL_OUT_DIM},embodiment_id:${MAX_BATCH}" \
  > "${ENGINE_DIR}/action_decoder.log" 2>&1

if [[ "${HAS_BACKBONE}" == "1" ]]; then
  echo "[INFO] Building vit_${VIT_DTYPE}.engine"
  "${TRTEXEC}" --useCudaGraph --verbose --stronglyTyped --separateProfileRun --noDataTransfers \
    --onnx="${ONNX_DIR}/eagle2/vit_${VIT_DTYPE}.onnx" \
    --saveEngine="${ENGINE_DIR}/vit_${VIT_DTYPE}.engine" \
    --minShapes="pixel_values:1x3x224x224,position_ids:1x256" \
    --optShapes="pixel_values:${VIDEO_VIEWS}x3x224x224,position_ids:${VIDEO_VIEWS}x256" \
    --maxShapes="pixel_values:${MAX_BATCH}x3x224x224,position_ids:${MAX_BATCH}x256" \
    > "${ENGINE_DIR}/vit_${VIT_DTYPE}.log" 2>&1
else
  echo "[WARN] Skipping backbone ViT engine (missing ${ONNX_DIR}/eagle2/vit_${VIT_DTYPE}.onnx)"
fi

if [[ "${HAS_LLM}" == "1" ]]; then
  echo "[INFO] Building llm_${LLM_DTYPE}.engine"
  if [[ ! "${LLM_DTYPE}" =~ ^(fp16|nvfp4|nvfp4_full|fp8)$ ]]; then
    echo "[ERROR] LLM_DTYPE must be fp16|nvfp4|nvfp4_full|fp8, got: ${LLM_DTYPE}" >&2
    exit 1
  fi

  if [[ "${LLM_DTYPE}" =~ ^nvfp4 ]]; then
    LLM_MAX_BATCH=1
  else
    LLM_MAX_BATCH="${MAX_BATCH}"
  fi

  "${TRTEXEC}" --useCudaGraph --verbose --stronglyTyped --separateProfileRun --noDataTransfers \
    --onnx="${ONNX_DIR}/eagle2/llm_${LLM_DTYPE}.onnx" \
    --saveEngine="${ENGINE_DIR}/llm_${LLM_DTYPE}.engine" \
    --minShapes="inputs_embeds:1x${MIN_LEN}x${BACKBONE_EMB_DIM},attention_mask:1x${MIN_LEN}" \
    --optShapes="inputs_embeds:1x${OPT_LEN}x${BACKBONE_EMB_DIM},attention_mask:1x${OPT_LEN}" \
    --maxShapes="inputs_embeds:${LLM_MAX_BATCH}x${MAX_LEN}x${BACKBONE_EMB_DIM},attention_mask:${LLM_MAX_BATCH}x${MAX_LEN}" \
    > "${ENGINE_DIR}/llm_${LLM_DTYPE}.log" 2>&1
else
  echo "[WARN] Skipping LLM engine (missing ${ONNX_DIR}/eagle2/llm_${LLM_DTYPE}.onnx)"
fi

echo "[OK] Engines saved in: ${ENGINE_DIR}/"

