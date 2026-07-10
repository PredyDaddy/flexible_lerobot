#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/data/cqy_workspace/flexible_lerobot"
cd "${REPO_ROOT}"

mkdir -p google my_devs/vla_engineering/train/{outputs,logs,tmp,hf_home}

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
