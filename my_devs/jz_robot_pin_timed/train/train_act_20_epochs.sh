#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EPOCHS=20
export DATASET_NAME="${DATASET_NAME:-jz_robot_pin_timed_curated_42eps_20260713}"
export EXPECTED_TOTAL_EPISODES="${EXPECTED_TOTAL_EPISODES:-42}"
export EXPECTED_TOTAL_FRAMES="${EXPECTED_TOTAL_FRAMES:-8370}"
export EXPECTED_FPS="${EXPECTED_FPS:-20}"

exec bash "${SCRIPT_DIR}/train_act.sh"
