#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

cd "${REPO_ROOT}"

exec conda run --no-capture-output -n lerobot_flex \
  uvicorn my_devs.agilex_web_collection:app \
  --host "${HOST}" \
  --port "${PORT}"
