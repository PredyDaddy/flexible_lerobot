#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

if [[ "${CONDA_DEFAULT_ENV:-}" != "lerobot_flex" ]]; then
  echo "[ERROR] Run this script from the lerobot_flex conda environment." >&2
  echo "[ERROR] Example: conda run -n lerobot_flex bash ${BASH_SOURCE[0]}" >&2
  exit 1
fi

UV_PREFIX="${GR00T17_ROOT}/tools/uv"
UV_BIN="${UV_PREFIX}/bin/uv"
require_path_within_root "${UV_PREFIX}"
require_path_within_root "${GR00T17_ENV}"

if [[ ! -x "${UV_BIN}" ]]; then
  echo "[SETUP] Installing uv under ${UV_PREFIX}"
  python -m pip install --prefix "${UV_PREFIX}" uv
fi

echo "[SETUP] uv version: $("${UV_BIN}" --version)"

if [[ ! -x "${GR00T17_ENV}/bin/python" ]]; then
  echo "[SETUP] Creating local virtual environment: ${GR00T17_ENV}"
  "${UV_BIN}" venv "${GR00T17_ENV}" --python "$(command -v python)" --seed
fi

FLASH_ATTN_WHEEL="${GR00T17_ROOT}/tools/wheels/flash_attn-2.7.4.post1-cp310-cp310-linux_x86_64.whl"
if [[ ! -f "${FLASH_ATTN_WHEEL}" ]]; then
  echo "[ERROR] Host-compatible flash-attn wheel is missing: ${FLASH_ATTN_WHEEL}" >&2
  echo "[ERROR] Run scripts/build_flash_attn.sh after installing the bootstrap dependencies." >&2
  exit 1
fi

enable_offline_mode
echo "[SETUP] Synchronizing the N1.7 lockfile"
UV_PROJECT_ENVIRONMENT="${GR00T17_ENV}" \
  "${UV_BIN}" sync \
  --project "${GR00T17_WORKSPACE}" \
  --locked \
  --offline \
  --all-extras

REPORT="${GR00T17_ROOT}/reports/environment.json"
"${GR00T17_ENV}/bin/python" - "${REPORT}" "${GR00T17_WORKSPACE}/uv.lock" <<'PY'
import json
import hashlib
from pathlib import Path
import platform
import subprocess
import sys

import torch

report_path = Path(sys.argv[1])
lock_path = Path(sys.argv[2])
report = {
    "schema_version": 1,
    "status": "passed",
    "python": sys.version,
    "executable": sys.executable,
    "platform": platform.platform(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "gpu_memory_bytes": (
        torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None
    ),
    "uv_lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
    "packages": subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True
    ).splitlines(),
}
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps({key: value for key, value in report.items() if key != "packages"}, indent=2))
PY

echo "[OK] Local environment ready: ${GR00T17_ENV}"
echo "[OK] Environment report: ${REPORT}"
