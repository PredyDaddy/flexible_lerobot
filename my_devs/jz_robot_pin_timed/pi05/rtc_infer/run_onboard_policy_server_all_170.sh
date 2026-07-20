#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

[[ "$#" -eq 0 ]] \
  || rtc_die "run_onboard_policy_server_all_170.sh accepts no CLI arguments"
[[ "${JZ_PI05_ALL_170_047320_CONFIRMED:-}" == "1" ]] \
  || rtc_die "all_170 checkpoint 047320 requires JZ_PI05_ALL_170_047320_CONFIRMED=1"

POLICY_PATH="${REPO_ROOT}/outputs/pi05_output/pi05_jz_robot_pin_timed_all_170eps_20260717_e15_b8_20260717_134540/checkpoints/047320/pretrained_model"

exec env \
  POLICY_PATH="${POLICY_PATH}" \
  REQUIRE_COMPLETE_STEP=false \
  JZ_PI05_ALL_170_047320_CONFIRMED=1 \
  JZ_PI05_ALL_200_007320_CONFIRMED=0 \
  bash "${SCRIPT_DIR}/run_onboard_policy_server.sh"
