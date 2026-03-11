#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Usage:
#   bash vis.sh
#   bash vis.sh <dataset_name>
#   bash vis.sh <dataset_name> <episode_index>
#
# Examples:
#   bash vis.sh
#   bash vis.sh demo_pick
#   bash vis.sh demo_pick 0

# 运行环境名字。
CONDA_ENV="${CONDA_ENV:-lerobot_flex}"
# 数据集名字。命令第 1 个参数。
DATASET_NAME="${1:-agilex_record_demo_video}"
# LeRobot 内部使用的数据集 repo_id，一般不用改。
DATASET_REPO_ID="${DATASET_REPO_ID:-dummy/${DATASET_NAME}}"
# 数据集根目录。默认就是当前目录下的 outputs。
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${SCRIPT_DIR}/outputs}"
# 当前要读取的数据集目录。
DATASET_ROOT="${DATASET_ROOT:-${HF_LEROBOT_HOME}/${DATASET_REPO_ID}}"
# 可视化第几个 episode。命令第 2 个参数，从 0 开始。
EPISODE_INDEX="${2:-0}"

# 显示模式。local 本机显示，distant 远程显示。
MODE="${MODE:-local}"
# 是否把可视化结果保存成 .rrd 文件。0 不保存，1 保存。
SAVE="${SAVE:-0}"
# 保存 .rrd 文件的目录。
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/viz_outputs}"
# dataloader batch 大小。
BATCH_SIZE="${BATCH_SIZE:-8}"
# dataloader worker 数量。
NUM_WORKERS="${NUM_WORKERS:-0}"
# rerun web 端口。
WEB_PORT="${WEB_PORT:-9090}"
# rerun websocket 端口。
WS_PORT="${WS_PORT:-9087}"
# 时间戳对齐容差。
TOLERANCE_S="${TOLERANCE_S:-0.0001}"
# 是否显示压缩后的图像。
DISPLAY_COMPRESSED_IMAGES="${DISPLAY_COMPRESSED_IMAGES:-false}"

mkdir -p "${OUTPUT_DIR}"

export DISPLAY="${DISPLAY:-:0}"
export HF_LEROBOT_HOME

cd "${REPO_ROOT}"

if [[ "${CONDA_DEFAULT_ENV:-}" == "${CONDA_ENV}" ]]; then
  cmd=(
    lerobot-dataset-viz
    --repo-id "${DATASET_REPO_ID}"
    --episode-index "${EPISODE_INDEX}"
    --root "${DATASET_ROOT}"
    --mode "${MODE}"
    --save "${SAVE}"
    --output-dir "${OUTPUT_DIR}"
    --batch-size "${BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --web-port "${WEB_PORT}"
    --ws-port "${WS_PORT}"
    --tolerance-s "${TOLERANCE_S}"
    --display-compressed-images "${DISPLAY_COMPRESSED_IMAGES}"
  )
else
  cmd=(
    conda run --no-capture-output -n "${CONDA_ENV}" lerobot-dataset-viz
    --repo-id "${DATASET_REPO_ID}"
    --episode-index "${EPISODE_INDEX}"
    --root "${DATASET_ROOT}"
    --mode "${MODE}"
    --save "${SAVE}"
    --output-dir "${OUTPUT_DIR}"
    --batch-size "${BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --web-port "${WEB_PORT}"
    --ws-port "${WS_PORT}"
    --tolerance-s "${TOLERANCE_S}"
    --display-compressed-images "${DISPLAY_COMPRESSED_IMAGES}"
  )
fi

echo "Dataset input=${DATASET_ROOT}"
echo "Running command:"
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
