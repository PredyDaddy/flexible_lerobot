#!/usr/bin/env python

from __future__ import annotations

import os
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TIMED_ROOT = REPO_ROOT / "my_devs" / "jz_robot_pin_timed"


REQUIRED_ENTRYPOINTS = (
    "record.sh",
    "replay.sh",
    "start_record.sh",
    "start_replay.sh",
    "start_teleop.sh",
    "stop_teleop.sh",
    "check_timed_observation.sh",
    "recv_vr_udp.py",
    "data_check/record_and_check_3.sh",
    "data_check/check_3_episodes.py",
    "data_check/check_timing.py",
    "edge/start_pin_state.sh",
    "edge/start_pin_replay.sh",
    "edge/status_pin_replay.sh",
    "edge/stop_pin_replay.sh",
    "x86/start_pin_control.sh",
    "x86/start_pin_joystick.sh",
    "x86/start_pin_record.sh",
    "x86/start_pin_replay.sh",
    "x86/start_pin_teleop.sh",
    "x86/stop_pin_teleop.sh",
    "x86/probe_timed_observation.sh",
    "x86/probe_timed_observation.py",
)


@pytest.mark.parametrize("relative_path", REQUIRED_ENTRYPOINTS)
def test_timed_operational_entrypoints_exist_and_are_executable(relative_path: str) -> None:
    path = TIMED_ROOT / relative_path

    assert path.is_file(), relative_path
    assert os.access(path, os.X_OK), relative_path


def test_timed_record_uses_new_robot_and_explicit_crf() -> None:
    script = (TIMED_ROOT / "record.sh").read_text(encoding="utf-8")

    assert "--robot.type=jz_robot_pin_timed" in script
    assert 'VIDEO_CRF="${VIDEO_CRF:-18}"' in script
    assert '--dataset.video_crf="${VIDEO_CRF}"' in script
    assert "--robot.timing_sidecar=" in script


def test_three_episode_wrapper_records_timed_data_then_runs_both_checks() -> None:
    script = (TIMED_ROOT / "data_check/record_and_check_3.sh").read_text(encoding="utf-8")

    assert "NUM_EPISODES=3" in script
    assert "RESUME=false" in script
    assert "VIDEO_CRF=\"${VIDEO_CRF}\"" in script
    assert 'MAX_INITIAL_JOINT_DELTA_RAD="${MAX_INITIAL_JOINT_DELTA_RAD:-10.0}"' in script
    assert 'MAX_JOINT_STEP_RAD="${MAX_JOINT_STEP_RAD:-10.0}"' in script
    assert '"${SCRIPT_DIR}/check_3_episodes.py"' in script
    assert '"${SCRIPT_DIR}/check_timing.py"' in script


def test_timed_stop_scope_does_not_match_other_timed_workflows() -> None:
    script = (TIMED_ROOT / "x86/stop_pin_teleop.sh").read_text(encoding="utf-8")

    assert "--robot.id=jz_robot_pin_timed_control" in script
    assert "--robot\\.type=jz_robot_pin_timed" not in script
    assert "jz_robot_pin_timed_record" not in script
    assert "jz_robot_pin_timed_replay" not in script


def test_observation_probe_is_explicitly_read_only() -> None:
    probe = (TIMED_ROOT / "x86/probe_timed_observation.py").read_text(encoding="utf-8")

    assert 'send_action_transport="local"' in probe
    assert 'send_action_execution="dry_run"' in probe
    assert ".send_action(" not in probe


def test_control_record_and_replay_select_expected_robot_type() -> None:
    for relative_path in ("x86/start_pin_control.sh", "record.sh", "replay.sh"):
        script = (TIMED_ROOT / relative_path).read_text(encoding="utf-8")
        assert "--robot.type=jz_robot_pin_timed" in script, relative_path


@pytest.mark.parametrize("relative_path", ("edge/start_pin_state.sh", "edge/start_pin_replay.sh"))
def test_timed_edge_defaults_state_stream_to_recording_rate(relative_path: str) -> None:
    script = (TIMED_ROOT / relative_path).read_text(encoding="utf-8")

    assert 'STATE_HZ="${STATE_HZ:-30}"' in script
    assert 'env STATE_HZ="${STATE_HZ}"' in script
