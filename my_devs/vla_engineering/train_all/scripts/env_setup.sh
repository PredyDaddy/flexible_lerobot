#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/data/cqy_workspace/flexible_lerobot"
VLASH_ROOT="${REPO_ROOT}/my_devs/vla_engineering/vlash-main"
ENV_NAME="${ENV_NAME:-vlash_train}"

source /home/cqy/miniconda3/etc/profile.d/conda.sh

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" python=3.10
fi

conda activate "${ENV_NAME}"

conda install -y -c conda-forge ffmpeg=7.1.1
python -m pip install --upgrade pip

python -m pip install -e "${REPO_ROOT}[feetech,smolvla]"
python -m pip install -e "${VLASH_ROOT}" --no-deps

python -m pip install \
  "accelerate>=1.12.0" \
  "transformers==4.53.3" \
  "peft==0.18.1" \
  "bitsandbytes==0.48.2" \
  "termcolor" \
  "torchcodec"

bash "${REPO_ROOT}/my_devs/vla_engineering/train_all/scripts/prepare_offline_assets.sh"

python - <<'PY'
import sys

import accelerate
import lerobot
import peft
import torch
import transformers
import vlash

print("python", sys.executable)
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("accelerate", accelerate.__version__)
print("peft", peft.__version__)
print("lerobot", lerobot.__version__)
print("vlash", vlash.__file__)
PY

echo "train_all ${ENV_NAME} environment is ready."
