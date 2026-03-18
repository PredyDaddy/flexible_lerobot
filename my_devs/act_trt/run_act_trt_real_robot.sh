#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="/data/cqy_workspace/flexible_lerobot"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-lerobot_flex}"
RUN_NAME="${ACT_TRT_RUN_NAME:-20260305_190147_act_grasp_block_in_bin1_e15}"

CHECKPOINT="${CHECKPOINT:-$REPO_ROOT/outputs/train/$RUN_NAME/checkpoints/last/pretrained_model}"
DEPLOY_DIR="${DEPLOY_DIR:-$REPO_ROOT/outputs/deploy/act_trt/$RUN_NAME}"
ENGINE="${ENGINE:-$DEPLOY_DIR/act_single_fp32.plan}"
EXPORT_METADATA="${EXPORT_METADATA:-$DEPLOY_DIR/export_metadata.json}"

MODE="${MODE:-mock-send}"
REPORT_SUFFIX="${REPORT_SUFFIX:-$MODE}"
REPORT="${REPORT:-$DEPLOY_DIR/run_act_trt_real_report_${REPORT_SUFFIX}.json}"

ROBOT_ID="${ROBOT_ID:-my_so101}"
ROBOT_TYPE="${ROBOT_TYPE:-so101_follower}"
CALIB_DIR="${CALIB_DIR:-/home/cqy/.cache/huggingface/lerobot/calibration/robots/so101_follower}"
ROBOT_PORT="${ROBOT_PORT:-/dev/ttyACM0}"

TOP_CAM_INDEX="${TOP_CAM_INDEX:-4}"
WRIST_CAM_INDEX="${WRIST_CAM_INDEX:-6}"
IMG_WIDTH="${IMG_WIDTH:-640}"
IMG_HEIGHT="${IMG_HEIGHT:-480}"
FPS="${FPS:-30}"

POLICY_DEVICE="${POLICY_DEVICE:-cuda:0}"
TASK="${TASK:-Put the block in the bin}"
RUN_TIME_S="${RUN_TIME_S:-30}"
MAX_STEPS="${MAX_STEPS:-0}"
LOG_INTERVAL="${LOG_INTERVAL:-30}"

MOCK_CASE="${MOCK_CASE:-random}"
SEED="${SEED:-0}"
COMPARE_TORCH="${COMPARE_TORCH:-false}"
TORCH_POLICY_DEVICE="${TORCH_POLICY_DEVICE:-cpu}"
ALLOW_LIVE_SEND="${ALLOW_LIVE_SEND:-0}"

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
  --robot-id "$ROBOT_ID"
  --robot-type "$ROBOT_TYPE"
  --calib-dir "$CALIB_DIR"
  --robot-port "$ROBOT_PORT"
  --top-cam-index "$TOP_CAM_INDEX"
  --wrist-cam-index "$WRIST_CAM_INDEX"
  --img-width "$IMG_WIDTH"
  --img-height "$IMG_HEIGHT"
  --fps "$FPS"
  --policy-path "$CHECKPOINT"
  --engine "$ENGINE"
  --export-metadata "$EXPORT_METADATA"
  --policy-device "$POLICY_DEVICE"
  --task "$TASK"
  --run-time-s "$RUN_TIME_S"
  --max-steps "$MAX_STEPS"
  --log-interval "$LOG_INTERVAL"
  --report "$REPORT"
)

case "$MODE" in
  dry-run)
    CMD+=(--dry-run true)
    ;;
  mock-observation)
    CMD+=(--mock-observation true --mock-case "$MOCK_CASE" --seed "$SEED")
    ;;
  mock-send)
    CMD+=(--mock-send true)
    ;;
  live)
    if [[ "$ALLOW_LIVE_SEND" != "1" ]]; then
      echo "[ERROR] MODE=live will send actions to the robot. Re-run with ALLOW_LIVE_SEND=1 to confirm." >&2
      exit 1
    fi
    ;;
  *)
    echo "[ERROR] Unsupported MODE=$MODE. Supported values: dry-run, mock-observation, mock-send, live" >&2
    exit 1
    ;;
esac

if [[ "$COMPARE_TORCH" == "true" ]]; then
  CMD+=(--compare-torch true --torch-policy-device "$TORCH_POLICY_DEVICE")
fi

echo "[INFO] Repo root: $REPO_ROOT"
echo "[INFO] Conda env: $CONDA_ENV_NAME"
echo "[INFO] Mode: $MODE"
echo "[INFO] Robot id: $ROBOT_ID"
echo "[INFO] Robot type: $ROBOT_TYPE"
echo "[INFO] Robot port: $ROBOT_PORT"
echo "[INFO] Top camera index: $TOP_CAM_INDEX"
echo "[INFO] Wrist camera index: $WRIST_CAM_INDEX"
echo "[INFO] Checkpoint: $CHECKPOINT"
echo "[INFO] Engine: $ENGINE"
echo "[INFO] Export metadata: $EXPORT_METADATA"
echo "[INFO] Policy device: $POLICY_DEVICE"
echo "[INFO] Report: $REPORT"
if [[ "$MODE" == "live" ]]; then
  echo "[WARN] Live mode enabled. Robot actions will be sent."
elif [[ "$MODE" == "mock-send" ]]; then
  echo "[INFO] Safe bring-up mode: robot connects and inference runs, but actions are not sent."
fi

"${CMD[@]}" "$@"
