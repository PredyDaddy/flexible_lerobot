#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/data/cqy_workspace/flexible_lerobot"
TRAIN_ROOT="${REPO_ROOT}/my_devs/vla_engineering/train"
TRAIN_ALL_ROOT="${REPO_ROOT}/my_devs/vla_engineering/train_all"
VLASH_ROOT="${REPO_ROOT}/my_devs/vla_engineering/vlash-main"
CONDA_ROOT="${CONDA_ROOT:-/home/cqy/miniconda3}"
POLICY_ENV="${POLICY_ENV:-vlash_train}"

DEFAULT_POLICY_PATH="${TRAIN_ALL_ROOT}/outputs/pi05_so101_delay_lora_r192_delay8_bs8_accum1_5epochs/checkpoints/last/pretrained_model"
POLICY_PATH="${POLICY_PATH:-${DEFAULT_POLICY_PATH}}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8005}"
DEVICE="${DEVICE:-cuda}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-2}"
TASK="${TASK:-Put the eraser into the small box}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${TRAIN_ALL_ROOT}/logs/${RUN_ID}_pi05_delay_lora_http_smoke"

mkdir -p "${LOG_DIR}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${POLICY_ENV}"

cd "${REPO_ROOT}"

export PYTHONPATH="${VLASH_ROOT}:${REPO_ROOT}/src:${PYTHONPATH:-}"
export HF_HOME="${TRAIN_ALL_ROOT}/hf_home"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "Starting temporary train_all VLASH HTTP server for smoke inference"
echo "POLICY_PATH=${POLICY_PATH}"
echo "HOST=${HOST}"
echo "PORT=${PORT}"
echo "DEVICE=${DEVICE}"
echo "NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS}"
echo "TASK=${TASK}"
echo "LOG_DIR=${LOG_DIR}"

python "${TRAIN_ROOT}/serve_vlash_policy.py" \
  --policy-path "${POLICY_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --default-prompt "${TASK}" \
  --device "${DEVICE}" \
  --num-inference-steps "${NUM_INFERENCE_STEPS}" \
  > "${LOG_DIR}/server.log" 2>&1 &
SERVER_PID=$!

cleanup() {
  kill "${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

python - "${HOST}" "${PORT}" "${SERVER_PID}" "${LOG_DIR}/server.log" <<'PY'
import json
import os
import signal
import sys
import time
import urllib.request

host, port, pid_str, log_path = sys.argv[1:5]
pid = int(pid_str)
url = f"http://{host}:{port}/metadata"
last_error = None
for _ in range(180):
    try:
        os.kill(pid, 0)
    except OSError as exc:
        print(f"[ERROR] Server process exited before becoming ready: {exc}", file=sys.stderr)
        if os.path.exists(log_path):
            print(open(log_path, encoding="utf-8", errors="replace").read()[-4000:], file=sys.stderr)
        raise SystemExit(1)
    try:
        with urllib.request.urlopen(url, timeout=2.0) as response:
            metadata = json.loads(response.read().decode("utf-8"))
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        raise SystemExit(0)
    except Exception as exc:
        last_error = exc
        time.sleep(1)

print(f"[ERROR] Timed out waiting for {url}: {last_error}", file=sys.stderr)
if os.path.exists(log_path):
    print(open(log_path, encoding="utf-8", errors="replace").read()[-4000:], file=sys.stderr)
raise SystemExit(1)
PY

python "${TRAIN_ALL_ROOT}/scripts/smoke_http_inference.py" \
  --host "${HOST}" \
  --port "${PORT}" \
  --task "${TASK}" \
  --timeout-s 120

echo "train_all VLASH HTTP server smoke inference passed."
echo "LOG_DIR=${LOG_DIR}"
