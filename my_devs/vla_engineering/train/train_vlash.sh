#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/data/cqy_workspace/flexible_lerobot"
TRAIN_ROOT="${REPO_ROOT}/my_devs/vla_engineering/train"
VLASH_ROOT="${REPO_ROOT}/my_devs/vla_engineering/vlash-main"
ENV_NAME="${ENV_NAME:-vlash_train}"
BASE_CONFIG="${TRAIN_ROOT}/pi05_so101_vlash_lora_10epochs_r16_bs1_accum8.yaml"
CONFIG_PATH="${TRAIN_ROOT}/pi05_so101_vlash_lora_10epochs_r192_bs1_accum8.yaml"

source /home/cqy/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME}"

cd "${REPO_ROOT}"

export PYTHONPATH="${VLASH_ROOT}:${REPO_ROOT}/src:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false

python - <<'PY'
from pathlib import Path

import yaml

src = Path("/data/cqy_workspace/flexible_lerobot/my_devs/vla_engineering/train/pi05_so101_vlash_lora_10epochs_r16_bs1_accum8.yaml")
dst = Path("/data/cqy_workspace/flexible_lerobot/my_devs/vla_engineering/train/pi05_so101_vlash_lora_10epochs_r192_bs1_accum8.yaml")

cfg = yaml.safe_load(src.read_text())

cfg["output_dir"] = "/data/cqy_workspace/flexible_lerobot/my_devs/vla_engineering/train/outputs/pi05_so101_vlash_lora_10epochs_r192_bs1_accum8"
cfg["job_name"] = "pi05_so101_vlash_lora_10epochs_r192_bs1_accum8"
cfg["batch_size"] = 1
cfg["grad_accum_steps"] = 8
cfg["steps"] = 66550
cfg["save_freq"] = 6655
cfg["log_freq"] = 50
cfg["num_workers"] = 2
cfg["lora"]["r"] = 192
cfg["lora"]["alpha"] = 192
cfg["lora"]["dropout"] = 0.05
cfg["wandb"] = {"enable": False}

dst.write_text(
    "# Generated for >=500M trainable params on 2026-07-03.\n"
    + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
)

print(f"wrote {dst}")
print("expected trainable params: >500M")
PY

python -m vlash.train --config_path="${CONFIG_PATH}"
