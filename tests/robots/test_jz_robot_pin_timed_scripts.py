#!/usr/bin/env python

from __future__ import annotations

import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TIMED_ROOT = REPO_ROOT / "my_devs" / "jz_robot_pin_timed"
PIN_ROOT = REPO_ROOT / "my_devs" / "jz_robot_pin"


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


def _write_executable(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content), encoding="utf-8")
    path.chmod(0o755)


def _make_fake_edge_repo(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    root = tmp_path / "fake_repo"
    timed_script = root / "my_devs/jz_robot_pin_timed/edge/start_pin_replay.sh"
    legacy_script = root / "my_devs/jz_robot_pin/edge/start_pin_replay.sh"
    capture_file = root / "captured_state_hz_env.txt"

    timed_script.parent.mkdir(parents=True)
    legacy_script.parent.mkdir(parents=True)
    shutil.copy2(TIMED_ROOT / "edge/start_pin_replay.sh", timed_script)
    shutil.copy2(PIN_ROOT / "edge/start_pin_replay.sh", legacy_script)
    _write_executable(
        root / "my_devs/jz_robot_pin/lib/common.sh",
        """
        #!/usr/bin/env bash
        PIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
        REPO_ROOT="$(cd "${PIN_ROOT}/../.." && pwd)"
        """,
    )
    _write_executable(
        root / "udp_test/all/start_replay.sh",
        """
        #!/usr/bin/env bash
        set -euo pipefail
        {
          printf 'STATE_HZ=%s\n' "${STATE_HZ:-missing}"
          printf 'JZ_STATE_HZ_PROFILE=%s\n' "${JZ_STATE_HZ_PROFILE:-missing}"
          printf 'JZ_EXPECTED_STATE_HZ=%s\n' "${JZ_EXPECTED_STATE_HZ:-missing}"
          printf 'JZ_TIMED_NON_30_STATE_HZ_CONFIRM=%s\n' "${JZ_TIMED_NON_30_STATE_HZ_CONFIRM-missing}"
        } > "${PWD}/captured_state_hz_env.txt"
        echo "[fake downstream] called"
        """,
    )
    return root, timed_script, legacy_script, capture_file


def _edge_env(**overrides: str) -> dict[str, str]:
    env = os.environ.copy()
    for key in (
        "STATE_HZ",
        "JZ_STATE_HZ_PROFILE",
        "JZ_EXPECTED_STATE_HZ",
        "JZ_TIMED_NON_30_STATE_HZ_CONFIRM",
    ):
        env.pop(key, None)
    env.update({"EXECUTION": "armed", "JZ_UDP_EXECUTOR_ARMED": "1"})
    env.update(overrides)
    return env


def _run_edge(script: Path, root: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(script)],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )


def _read_captured_env(capture_file: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in capture_file.read_text(encoding="utf-8").splitlines())


def test_timed_edge_defaults_state_stream_to_30_hz(tmp_path: Path) -> None:
    root, timed_script, _legacy_script, capture_file = _make_fake_edge_repo(tmp_path)

    result = _run_edge(timed_script, root, _edge_env())

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert _read_captured_env(capture_file) == {
        "STATE_HZ": "30",
        "JZ_STATE_HZ_PROFILE": "timed",
        "JZ_EXPECTED_STATE_HZ": "30",
        "JZ_TIMED_NON_30_STATE_HZ_CONFIRM": "",
    }
    assert "requested_hz=30 expected_hz=30 non_30_override_confirmed=false" in output
    assert "[fake downstream] called" in output


def test_timed_edge_rejects_inherited_20_hz_without_calling_downstream(tmp_path: Path) -> None:
    root, timed_script, _legacy_script, capture_file = _make_fake_edge_repo(tmp_path)

    result = _run_edge(timed_script, root, _edge_env(STATE_HZ="20"))

    output = result.stdout + result.stderr
    assert result.returncode == 2, output
    assert "refusing non-30 timed state rate STATE_HZ=20" in output
    assert "JZ_TIMED_NON_30_STATE_HZ_CONFIRM=20" in output
    assert "[fake downstream] called" not in output
    assert not capture_file.exists()


def test_timed_edge_allows_only_an_exact_confirmed_non_30_override(tmp_path: Path) -> None:
    root, timed_script, _legacy_script, capture_file = _make_fake_edge_repo(tmp_path)

    mismatch = _run_edge(
        timed_script,
        root,
        _edge_env(STATE_HZ="25", JZ_TIMED_NON_30_STATE_HZ_CONFIRM="20"),
    )
    assert mismatch.returncode == 2, mismatch.stdout + mismatch.stderr
    assert not capture_file.exists()

    confirmed = _run_edge(
        timed_script,
        root,
        _edge_env(STATE_HZ="25", JZ_TIMED_NON_30_STATE_HZ_CONFIRM="25"),
    )

    output = confirmed.stdout + confirmed.stderr
    assert confirmed.returncode == 0, output
    assert _read_captured_env(capture_file) == {
        "STATE_HZ": "25",
        "JZ_STATE_HZ_PROFILE": "timed",
        "JZ_EXPECTED_STATE_HZ": "25",
        "JZ_TIMED_NON_30_STATE_HZ_CONFIRM": "25",
    }
    assert "non_30_override_confirmed=true" in output
    assert "[fake downstream] called" in output


def test_legacy_edge_keeps_20_hz_default(tmp_path: Path) -> None:
    root, _timed_script, legacy_script, capture_file = _make_fake_edge_repo(tmp_path)

    result = _run_edge(legacy_script, root, _edge_env())

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert _read_captured_env(capture_file) == {
        "STATE_HZ": "20",
        "JZ_STATE_HZ_PROFILE": "legacy",
        "JZ_EXPECTED_STATE_HZ": "20",
        "JZ_TIMED_NON_30_STATE_HZ_CONFIRM": "",
    }
    assert "profile=legacy requested_hz=20 expected_hz=20" in output
    assert "[fake downstream] called" in output
