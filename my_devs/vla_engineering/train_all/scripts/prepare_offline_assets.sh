#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/data/cqy_workspace/flexible_lerobot"
TRAIN_ALL_ROOT="${REPO_ROOT}/my_devs/vla_engineering/train_all"

cd "${REPO_ROOT}"

mkdir -p google \
  "${TRAIN_ALL_ROOT}/outputs" \
  "${TRAIN_ALL_ROOT}/logs" \
  "${TRAIN_ALL_ROOT}/tmp" \
  "${TRAIN_ALL_ROOT}/hf_home" \
  "${TRAIN_ALL_ROOT}/reports"

ln -sfn \
  "${REPO_ROOT}/assets/modelscope/google/paligemma-3b-pt-224" \
  "${REPO_ROOT}/google/paligemma-3b-pt-224"

test -f "${REPO_ROOT}/assets/modelscope/lerobot/pi05_base/model.safetensors"
test -f "${REPO_ROOT}/assets/modelscope/lerobot/pi05_base/config.json"
test -f "${REPO_ROOT}/datasets/desk_cleanup_v1/eraser_cup_multi_task/meta/info.json"

echo "Offline assets are ready:"
echo "  tokenizer: ${REPO_ROOT}/google/paligemma-3b-pt-224"
echo "  pi05 base: ${REPO_ROOT}/assets/modelscope/lerobot/pi05_base"
echo "  dataset:   ${REPO_ROOT}/datasets/desk_cleanup_v1/eraser_cup_multi_task"
echo "  train_all: ${TRAIN_ALL_ROOT}"
