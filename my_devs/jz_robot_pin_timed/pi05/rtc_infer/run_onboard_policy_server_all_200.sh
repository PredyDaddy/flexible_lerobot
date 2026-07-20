#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

[[ "$#" -eq 0 ]] \
  || rtc_die "run_onboard_policy_server_all_200.sh accepts no CLI arguments"
[[ "${JZ_PI05_ALL_200_007320_CONFIRMED:-}" == "1" ]] \
  || rtc_die "all_200 checkpoint 007320 requires JZ_PI05_ALL_200_007320_CONFIRMED=1"

POLICY_PATH="${REPO_ROOT}/outputs/pi05_output/pi05_jz_robot_pin_timed_all_200eps_20260719_e15_b32_20260719_134823/checkpoints/007320/pretrained_model"

exec env \
  POLICY_PATH="${POLICY_PATH}" \
  REQUIRE_COMPLETE_STEP=false \
  JZ_PI05_ALL_170_047320_CONFIRMED=0 \
  JZ_PI05_ALL_200_007320_CONFIRMED=1 \
  bash "${SCRIPT_DIR}/run_onboard_policy_server.sh"
