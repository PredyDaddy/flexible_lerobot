#!/usr/bin/env bash
set -euo pipefail

BACKEND_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-${BACKEND_ROOT}/server/configs/so101_mock.yaml}"
PORT="${PORT:-18080}"
PYTHON="${PYTHON:-${BACKEND_ROOT}/.venv/bin/python}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing server Python: ${PYTHON}" >&2
  echo "Run: ${BACKEND_ROOT}/scripts/create_server_venv.sh" >&2
  exit 1
fi

export PYTHONPATH="${BACKEND_ROOT}/server:${PYTHONPATH:-}"

exec "${PYTHON}" "${BACKEND_ROOT}/server/infer_server.py" \
  --config "${CONFIG}" \
  --port "${PORT}" \
  "$@"

