#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../../../.." && pwd)

cd "${REPO_ROOT}"

conda run -n lerobot_flex python \
  my_devs/train/groot/agilex/async/run_groot_remote_async_robot_client.py \
  "$@"
