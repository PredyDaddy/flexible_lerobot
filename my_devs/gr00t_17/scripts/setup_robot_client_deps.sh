#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
require_lerobot_flex_host

TARGET="${GR00T17_ROOT}/tools/robot_client_deps"
UV_BIN="${GR00T17_ROOT}/tools/uv/bin/uv"
require_path_within_root "${TARGET}"

if [[ ! -x "${UV_BIN}" ]]; then
  echo "[ERROR] Project-local uv is missing: ${UV_BIN}" >&2
  exit 1
fi

mkdir -p "${TARGET}"
"${UV_BIN}" pip install \
  --target "${TARGET}" \
  --offline \
  --no-deps \
  "pyzmq==27.0.1"

PYTHONPATH="${TARGET}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${CONDA_PREFIX}/bin/python" -c \
  'import zmq; assert zmq.__version__ == "27.0.1"; print(f"[OK] pyzmq {zmq.__version__}: {zmq.__file__}")'
