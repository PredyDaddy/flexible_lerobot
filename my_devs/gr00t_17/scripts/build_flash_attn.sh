#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
require_lerobot_flex_host
require_local_env

VERSION="2.7.4.post1"
CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-80}"
MAX_BUILD_JOBS="${MAX_JOBS:-8}"
BUILD_ID="$(date +%Y%m%d_%H%M%S)"
BUILD_ROOT="${GR00T17_ROOT}/tmp/flash-attn-build-${BUILD_ID}"
WHEELHOUSE="${BUILD_ROOT}/wheelhouse"
BUILD_LOG="${GR00T17_ROOT}/logs/flash_attn_build_${BUILD_ID}.log"
REPORT="${GR00T17_ROOT}/reports/flash_attn_build.json"

require_path_within_root "${BUILD_ROOT}"
require_path_within_root "${WHEELHOUSE}"
require_path_within_root "${BUILD_LOG}"
require_path_within_root "${REPORT}"

if compgen -G "${GR00T17_ROOT}/tools/wheels/flash_attn-${VERSION}-cp310-cp310-linux_x86_64.whl" >/dev/null; then
  echo "[OK] Host-built flash-attn wheel already exists."
  exit 0
fi
if [[ -e "${REPORT}" ]]; then
  echo "[ERROR] Build report exists but the final wheel is missing: ${REPORT}" >&2
  exit 1
fi

mkdir -p "${WHEELHOUSE}" "$(dirname "${BUILD_LOG}")" "${GR00T17_ROOT}/tools/wheels"

"${GR00T17_ENV}/bin/python" - <<'PY'
from torch.utils.cpp_extension import is_ninja_available

if not is_ninja_available():
    raise RuntimeError("ninja must be available on PATH for parallel flash-attn compilation")
print("[BUILD] ninja_available=true")
PY

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.6}"
export FLASH_ATTN_CUDA_ARCHS="${CUDA_ARCHS}"
export MAX_JOBS="${MAX_BUILD_JOBS}"
export NVCC_THREADS="${NVCC_THREADS:-2}"
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export TMPDIR="${BUILD_ROOT}"

echo "[BUILD] flash-attn=${VERSION} arches=${FLASH_ATTN_CUDA_ARCHS} jobs=${MAX_JOBS}"
"${GR00T17_ENV}/bin/python" -m pip wheel "flash-attn==${VERSION}" \
  --no-build-isolation \
  --no-deps \
  --no-binary flash-attn \
  --wheel-dir "${WHEELHOUSE}" 2>&1 | tee "${BUILD_LOG}"

mapfile -t BUILT_WHEELS < <(find "${WHEELHOUSE}" -maxdepth 1 -type f -name 'flash_attn-*.whl')
if [[ "${#BUILT_WHEELS[@]}" -ne 1 ]]; then
  echo "[ERROR] Expected exactly one built wheel, found ${#BUILT_WHEELS[@]}" >&2
  exit 1
fi
FINAL_WHEEL="${GR00T17_ROOT}/tools/wheels/$(basename "${BUILT_WHEELS[0]}")"
if [[ -e "${FINAL_WHEEL}" ]]; then
  echo "[ERROR] Refusing to overwrite final wheel: ${FINAL_WHEEL}" >&2
  exit 1
fi
mv "${BUILT_WHEELS[0]}" "${FINAL_WHEEL}"

"${GR00T17_ENV}/bin/python" - "${FINAL_WHEEL}" "${REPORT}" "${CUDA_ARCHS}" <<'PY'
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import torch

wheel = Path(sys.argv[1]).resolve(strict=True)
report_path = Path(sys.argv[2]).resolve()
arches = sys.argv[3].split(";")
digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
report = {
    "schema_version": 1,
    "status": "passed",
    "version": "2.7.4.post1",
    "wheel": str(wheel),
    "size": wheel.stat().st_size,
    "sha256": digest,
    "cuda_arches": arches,
    "cuda_home": str(Path("/usr/local/cuda-12.6").resolve()),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "torch_cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
    "host_glibc": platform.libc_ver(),
    "nvcc": subprocess.check_output(
        ["/usr/local/cuda-12.6/bin/nvcc", "--version"], text=True
    ).strip(),
}
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
PY

echo "[OK] Built wheel: ${FINAL_WHEEL}"
echo "[OK] Build report: ${REPORT}"
