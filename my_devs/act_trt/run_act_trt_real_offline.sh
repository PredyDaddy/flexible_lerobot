#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="/data/cqy_workspace/flexible_lerobot"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-lerobot_flex}"
RUN_NAME="${ACT_TRT_RUN_NAME:-20260305_190147_act_grasp_block_in_bin1_e15}"

CHECKPOINT="${CHECKPOINT:-$REPO_ROOT/outputs/train/$RUN_NAME/checkpoints/last/pretrained_model}"
DEPLOY_DIR="${DEPLOY_DIR:-$REPO_ROOT/outputs/deploy/act_trt/$RUN_NAME}"
ENGINE="${ENGINE:-$DEPLOY_DIR/act_single_fp32.plan}"
EXPORT_METADATA="${EXPORT_METADATA:-$DEPLOY_DIR/export_metadata.json}"
REPORT="${REPORT:-$DEPLOY_DIR/run_act_trt_real_report_manual.json}"

POLICY_DEVICE="${POLICY_DEVICE:-cuda:0}"
FPS="${FPS:-30}"
MAX_STEPS="${MAX_STEPS:-20}"
MOCK_CASE="${MOCK_CASE:-random}"
SEED="${SEED:-0}"
TASK="${TASK:-Put the block in the bin}"

if [[ ! -d "$REPO_ROOT" ]]; then
  echo "[ERROR] Repo root not found: $REPO_ROOT" >&2
  exit 1
fi

if [[ ! -d "$CHECKPOINT" ]]; then
  echo "[ERROR] Checkpoint directory not found: $CHECKPOINT" >&2
  exit 1
fi

if [[ ! -f "$ENGINE" ]]; then
  echo "[ERROR] TensorRT engine not found: $ENGINE" >&2
  exit 1
fi

if [[ ! -f "$EXPORT_METADATA" ]]; then
  echo "[ERROR] Export metadata not found: $EXPORT_METADATA" >&2
  exit 1
fi

cd "$REPO_ROOT"

CMD=(
  conda run -n "$CONDA_ENV_NAME"
  python my_devs/act_trt/run_act_trt_real.py
  --policy-path "$CHECKPOINT"
  --engine "$ENGINE"
  --export-metadata "$EXPORT_METADATA"
  --policy-device "$POLICY_DEVICE"
  --mock-observation true
  --mock-case "$MOCK_CASE"
  --seed "$SEED"
  --fps "$FPS"
  --max-steps "$MAX_STEPS"
  --task "$TASK"
  --report "$REPORT"
)

echo "[INFO] Repo root: $REPO_ROOT"
echo "[INFO] Conda env: $CONDA_ENV_NAME"
echo "[INFO] Checkpoint: $CHECKPOINT"
echo "[INFO] Engine: $ENGINE"
echo "[INFO] Export metadata: $EXPORT_METADATA"
echo "[INFO] Report: $REPORT"
echo "[INFO] Running offline ACT TensorRT inference..."

"${CMD[@]}" "$@"
