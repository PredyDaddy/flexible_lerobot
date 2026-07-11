from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ALL_DIR = REPO_ROOT / "udp_test" / "all"
ORIN_ARM_DIR = REPO_ROOT / "udp_test" / "server_bash" / "orin_arm"
PIN_EDGE_DIR = REPO_ROOT / "my_devs" / "jz_robot_pin" / "edge"
TIMED_EDGE_DIR = REPO_ROOT / "my_devs" / "jz_robot_pin_timed" / "edge"


def test_udp_all_scripts_exist_are_executable_and_have_valid_bash_syntax() -> None:
    scripts = [
        ALL_DIR / "start_record.sh",
        ALL_DIR / "stop_record.sh",
        ALL_DIR / "start_replay.sh",
        ALL_DIR / "stop_replay.sh",
        ORIN_ARM_DIR / "start.sh",
        ORIN_ARM_DIR / "stop.sh",
        PIN_EDGE_DIR / "start_pin_state.sh",
        PIN_EDGE_DIR / "start_pin_replay.sh",
        TIMED_EDGE_DIR / "start_pin_state.sh",
        TIMED_EDGE_DIR / "start_pin_replay.sh",
    ]

    for script in scripts:
        assert script.exists(), f"missing {script}"
        assert os.access(script, os.X_OK), f"{script} is not executable"
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_record_scripts_manage_state_and_target_action_bridges() -> None:
    start_record = (ALL_DIR / "start_record.sh").read_text()
    stop_record = (ALL_DIR / "stop_record.sh").read_text()

    assert "conda run --no-capture-output -n lerobot python" in start_record
    assert "AUTO_TAIL=0" in start_record
    assert "server_bash/orin_arm" in start_record
    assert "start.sh" in start_record
    assert "ros_state_udp_bridge.pid" in start_record
    assert "ros_state_udp_bridge.log" in start_record
    assert "orin_ros_target_action_udp_bridge.py" in start_record
    assert "192.168.1.106" in start_record
    assert "192.168.1.81" in start_record
    assert "39030" in start_record
    assert "ros_target_action_udp_bridge.pid" in start_record

    assert "orin_ros_target_action_udp_bridge.py" in stop_record
    assert "ros_target_action_udp_bridge.pid" in stop_record
    assert "ORIN_ARM_DIR" in stop_record
    assert "stop.sh" in stop_record


def test_replay_scripts_manage_state_bridge_and_armed_phase3_executor() -> None:
    start_replay = (ALL_DIR / "start_replay.sh").read_text()
    stop_replay = (ALL_DIR / "stop_replay.sh").read_text()

    assert "conda run --no-capture-output -n lerobot python" in start_replay
    assert "AUTO_TAIL=0" in start_replay
    assert "start_phase3_executor.sh" in start_replay
    assert "orin_phase3_executor_config_armed_hold.yaml" in start_replay
    assert 'EXECUTION="$EXECUTION"' in start_replay
    assert "JZ_UDP_EXECUTOR_ARMED=1" in start_replay

    edge_start = (PIN_EDGE_DIR / "start_pin_replay.sh").read_text()
    bridge_start = (ORIN_ARM_DIR / "start.sh").read_text()
    for variable, default in (
        ("BRIDGE_START_TIMEOUT_S", "30"),
        ("STATE_WAIT_TIMEOUT_S", "15"),
        ("STATE_READY_TIMEOUT_S", "20"),
        ("EXECUTOR_READY_TIMEOUT_S", "15"),
    ):
        expected = f'{variable}="${{{variable}:-${{LEGACY_READY_TIMEOUT_S:-{default}}}}}"'
        assert expected in edge_start
        assert f'{variable}="${{{variable}}}" \\' in edge_start
        assert variable in start_replay
    assert 'BRIDGE_START_TIMEOUT_S="${BRIDGE_START_TIMEOUT_S:-${LEGACY_READY_TIMEOUT_S:-30}}"' in start_replay
    assert 'STATE_WAIT_TIMEOUT_S="${STATE_WAIT_TIMEOUT_S:-${LEGACY_READY_TIMEOUT_S:-15}}"' in start_replay
    assert 'STATE_READY_TIMEOUT_S="${STATE_READY_TIMEOUT_S:-${LEGACY_READY_TIMEOUT_S:-20}}"' in start_replay
    assert (
        'EXECUTOR_READY_TIMEOUT_S="${EXECUTOR_READY_TIMEOUT_S:-${LEGACY_READY_TIMEOUT_S:-15}}"'
        in start_replay
    )
    local_readiness = '"local=" \\\n  "$BRIDGE_START_TIMEOUT_S"'
    assert local_readiness in start_replay
    assert 'wait_for_bridge_rate_after 0 "$STATE_READY_TIMEOUT_S"' in start_replay
    assert start_replay.index(local_readiness) < start_replay.index("wait_for_bridge_rate_after 0")
    assert 'JZ_UDP_EXECUTOR_ARMED="${JZ_UDP_EXECUTOR_ARMED:-}"' in start_replay
    assert '--wait-timeout-s "$STATE_WAIT_TIMEOUT_S"' in bridge_start

    assert "server_bash/orin_arm/stop_all.sh" in stop_replay


def _write_executable(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content), encoding="utf-8")
    path.chmod(0o755)


def _make_fake_replay_repo(tmp_path: Path) -> Path:
    root = tmp_path / "fake_repo"
    all_dir = root / "udp_test" / "all"
    arm_dir = root / "udp_test" / "server_bash" / "orin_arm"
    all_dir.mkdir(parents=True)
    arm_dir.mkdir(parents=True)
    shutil.copy2(ALL_DIR / "start_replay.sh", all_dir / "start_replay.sh")

    fake_bridge = root / "udp_test/test_scripts/arm_side/orin_ros_state_udp_bridge.py"
    _write_executable(
        fake_bridge,
        textwrap.dedent(
            """
            #!/usr/bin/env python3
            import os
            import signal
            import sys
            import time

            signal.signal(signal.SIGTERM, lambda *_args: sys.exit(0))
            hz_index = sys.argv.index("--hz")
            argv_hz = sys.argv[hz_index + 1]
            configured_hz = os.environ.get("FAKE_LOG_CONFIGURED_HZ", argv_hz)
            measured_hz = os.environ.get("FAKE_MEASURED_HZ", argv_hz)
            log_path = os.environ["FAKE_BRIDGE_LOG_FILE"]
            with open(log_path, "a", buffering=1, encoding="utf-8") as log:
                print(
                    f"[fake bridge] pid={os.getpid()} local=127.0.0.1:40000 "
                    f"configured_hz={configured_hz}",
                    file=log,
                )
                time.sleep(float(os.environ.get("FAKE_BRIDGE_SENT_DELAY_S", "0")))
                if os.environ.get("FAKE_COMPLETE_METRICS", "1") == "1":
                    print(
                        f"[fake bridge] sent seq=30 configured_hz={configured_hz} "
                        f"measured_send_hz={measured_hz} rate_window_packets=30 "
                        "update_counts={left_joints:10,right_joints:10,left_gripper:5,right_gripper:5} "
                        "progress_modes={left_joints:header_stamp,right_joints:header_stamp,"
                        "left_gripper:generation,right_gripper:generation} "
                        "source_age_ms={left_joints:1,right_joints:1,left_gripper:2,right_gripper:2} "
                        "source_skew_ms=1 skipped={total:0,stale:0,skew:0,not_advanced:0}",
                        file=log,
                    )
                else:
                    print(
                        f"[fake bridge] sent seq=30 configured_hz={configured_hz} "
                        f"measured_send_hz={measured_hz} rate_window_packets=30",
                        file=log,
                    )
                print(
                    f"[fake bridge] rate pid={os.getpid()} seq=30 configured_hz={configured_hz} "
                    f"measured_send_hz={measured_hz} window_packets=30",
                    file=log,
                )
                if os.environ.get("FAKE_BRIDGE_EXIT_AFTER_FIRST_RATE", "0") == "1":
                    time.sleep(float(os.environ.get("FAKE_BRIDGE_EXIT_DELAY_S", "0.6")))
                    sys.exit(0)
                if os.environ.get("FAKE_NO_SECOND_RATE", "0") != "1":
                    time.sleep(float(os.environ.get("FAKE_SECOND_RATE_DELAY_S", "0.4")))
                    print(
                        f"[fake bridge] sent seq=60 configured_hz={configured_hz} "
                        f"measured_send_hz={measured_hz} rate_window_packets=30 "
                        "update_counts={left_joints:40,right_joints:40,left_gripper:40,right_gripper:40} "
                        "progress_modes={left_joints:header_stamp,right_joints:header_stamp,"
                        "left_gripper:generation,right_gripper:generation} "
                        "source_age_ms={left_joints:1,right_joints:1,left_gripper:2,right_gripper:2} "
                        "source_skew_ms=1 skipped={total:0,stale:0,skew:0,not_advanced:0}",
                        file=log,
                    )
                    print(
                        f"[fake bridge] rate pid={os.getpid()} seq=60 configured_hz={configured_hz} "
                        f"measured_send_hz={measured_hz} window_packets=30",
                        file=log,
                    )
                while True:
                    time.sleep(1)
            """
        ),
    )
    _write_executable(
        root / "fake_executor.py",
        """
        #!/usr/bin/env python3
        import os
        import signal
        import sys
        import time

        signal.signal(signal.SIGTERM, lambda *_args: sys.exit(0))
        log_path, delay, port = sys.argv[1:4]
        time.sleep(float(delay))
        with open(log_path, "a", buffering=1, encoding="utf-8") as log:
            mode = os.environ.get("EXECUTION", "missing")
            marker = "ARMED" if mode == "armed" else "DRY-RUN"
            print(f"PHASE3 COMMAND EXECUTOR {marker}", file=log)
            print(f"port={port}", file=log)
            while True:
                time.sleep(1)
        """,
    )

    _write_executable(
        arm_dir / "start.sh",
        """
        #!/usr/bin/env bash
        set -euo pipefail
        ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
        LOG_DIR="$ROOT_DIR/udp_test/server_bash/orin_arm/logs"
        PID_DIR="$ROOT_DIR/udp_test/server_bash/orin_arm/pids"
        mkdir -p "$LOG_DIR" "$PID_DIR"
        echo "${STATE_WAIT_TIMEOUT_S:-missing}" > "$ROOT_DIR/captured_state_wait_timeout.txt"
        : > "$LOG_DIR/ros_state_udp_bridge.log"
        FAKE_BRIDGE_LOG_FILE="$LOG_DIR/ros_state_udp_bridge.log" \
          "$FAKE_PYTHON" "$ROOT_DIR/udp_test/test_scripts/arm_side/orin_ros_state_udp_bridge.py" \
          --hz "${FAKE_ARGV_HZ:-$STATE_HZ}" > /dev/null 2>&1 &
        bridge_pid=$!
        echo "$bridge_pid" > "$PID_DIR/ros_state_udp_bridge.pid"
        {
          printf 'profile=%q\n' "${FAKE_METADATA_PROFILE:-${JZ_STATE_HZ_PROFILE}}"
          printf 'requested_hz=%q\n' "${FAKE_METADATA_REQUESTED_HZ:-${STATE_HZ}}"
          printf 'expected_hz=%q\n' "${FAKE_METADATA_EXPECTED_HZ:-${JZ_EXPECTED_STATE_HZ}}"
          printf 'configured_hz=%q\n' "${FAKE_METADATA_CONFIGURED_HZ:-${FAKE_ARGV_HZ:-$STATE_HZ}}"
          printf 'bridge_pid=%q\n' "$bridge_pid"
          printf 'launcher_pid=%q\n' "$bridge_pid"
        } > "$PID_DIR/ros_state_udp_bridge.startup"
        """,
    )
    _write_executable(
        arm_dir / "start_phase3_executor.sh",
        """
        #!/usr/bin/env bash
        set -euo pipefail
        ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
        LOG_DIR="$ROOT_DIR/udp_test/server_bash/orin_arm/logs"
        PID_DIR="$ROOT_DIR/udp_test/server_bash/orin_arm/pids"
        mkdir -p "$LOG_DIR" "$PID_DIR"
        touch "$ROOT_DIR/executor_started"
        printf '%s\n' "${JZ_UDP_EXECUTOR_ARMED-missing}" > "$ROOT_DIR/captured_executor_armed.txt"
        "$FAKE_PYTHON" "$ROOT_DIR/fake_executor.py" \
          "$LOG_DIR/orin_phase3_command_executor.log" "${FAKE_EXECUTOR_DELAY_S:-0}" "${COMMAND_PORT}" \
          > /dev/null 2>&1 &
        echo "$!" > "$PID_DIR/orin_phase3_command_executor.pid"
        """,
    )
    _write_executable(
        arm_dir / "stop_all.sh",
        """
        #!/usr/bin/env bash
        ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
        PID_DIR="$ROOT_DIR/udp_test/server_bash/orin_arm/pids"
        for pid_file in "$PID_DIR"/*.pid; do
          [[ -e "$pid_file" ]] || continue
          kill "$(cat "$pid_file")" 2>/dev/null || true
          rm -f "$pid_file"
        done
        """,
    )
    _write_executable(
        all_dir / "stop_replay.sh",
        """
        #!/usr/bin/env bash
        set -euo pipefail
        ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
        bash "$ROOT_DIR/udp_test/server_bash/orin_arm/stop_all.sh"
        """,
    )

    return root


def _fake_replay_env(**overrides: str) -> dict[str, str]:
    env = os.environ.copy()
    for key in (
        "STATE_HZ",
        "JZ_STATE_HZ_PROFILE",
        "JZ_EXPECTED_STATE_HZ",
        "JZ_TIMED_NON_30_STATE_HZ_CONFIRM",
        "EXECUTION",
        "FAKE_ARGV_HZ",
        "FAKE_METADATA_PROFILE",
        "FAKE_METADATA_REQUESTED_HZ",
        "FAKE_METADATA_EXPECTED_HZ",
        "FAKE_METADATA_CONFIGURED_HZ",
        "FAKE_LOG_CONFIGURED_HZ",
        "FAKE_MEASURED_HZ",
        "FAKE_COMPLETE_METRICS",
        "FAKE_BRIDGE_EXIT_AFTER_FIRST_RATE",
        "FAKE_NO_SECOND_RATE",
        "JZ_UDP_EXECUTOR_ARMED",
        "MIN_MEASURED_STATE_HZ_RATIO",
        "JZ_TIMED_MIN_MEASURED_STATE_HZ_RATIO_CONFIRM",
    ):
        env.pop(key, None)
    env.update(
        {
            "STATE_HZ": "30",
            "JZ_STATE_HZ_PROFILE": "timed",
            "JZ_EXPECTED_STATE_HZ": "30",
            "EXECUTION": "armed",
            "READY_TIMEOUT_S": "1",
            "BRIDGE_START_TIMEOUT_S": "3",
            "STATE_WAIT_TIMEOUT_S": "7",
            "STATE_READY_TIMEOUT_S": "3",
            "EXECUTOR_READY_TIMEOUT_S": "3",
            "FINAL_READY_TIMEOUT_S": "3",
            "FAKE_ARGV_HZ": "30",
            "FAKE_LOG_CONFIGURED_HZ": "30",
            "FAKE_MEASURED_HZ": "30",
            "FAKE_COMPLETE_METRICS": "1",
            "FAKE_BRIDGE_SENT_DELAY_S": "0.7",
            "FAKE_SECOND_RATE_DELAY_S": "0.8",
            "FAKE_EXECUTOR_DELAY_S": "0.2",
            "FAKE_PYTHON": sys.executable,
            "JZ_UDP_EXECUTOR_ARMED": "1",
            "MIN_MEASURED_STATE_HZ_RATIO": "0.9",
            "TAIL": "0",
        }
    )
    env.update(overrides)
    return env


def _run_fake_replay(root: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["bash", str(root / "udp_test/all/start_replay.sh")],
            cwd=root,
            env=env,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    finally:
        subprocess.run(
            ["bash", str(root / "udp_test/all/stop_replay.sh")],
            cwd=root,
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )


def test_replay_readiness_verifies_metadata_argv_log_hz_and_complete_metrics(tmp_path: Path) -> None:
    root = _make_fake_replay_repo(tmp_path)

    result = _run_fake_replay(root, _fake_replay_env())

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert (root / "captured_state_wait_timeout.txt").read_text().strip() == "7"
    assert "bridge configured hz verified" in output
    assert "configured_hz=30 measured_send_hz=30" in output
    assert output.count("bridge measured rate verified") == 2
    assert "bridge sent metrics verified" in output
    assert output.index("confirmed ready from log pattern: local=") < output.index(
        "bridge measured rate verified"
    )
    assert (root / "executor_started").is_file()
    assert (root / "captured_executor_armed.txt").read_text().strip() == "1"
    assert "replay services ready." in output


@pytest.mark.parametrize(
    ("overrides", "expected_error"),
    (
        ({"FAKE_METADATA_CONFIGURED_HZ": "20"}, "bridge configured_hz metadata mismatch"),
        (
            {"FAKE_ARGV_HZ": "20", "FAKE_METADATA_CONFIGURED_HZ": "30"},
            "bridge argv hz mismatch",
        ),
        ({"FAKE_LOG_CONFIGURED_HZ": "20"}, "bridge log configured hz mismatch"),
        ({"FAKE_COMPLETE_METRICS": "0"}, "bridge sent metrics missing update_counts="),
    ),
)
def test_replay_hz_or_metrics_mismatch_never_starts_executor_or_prints_ready(
    tmp_path: Path,
    overrides: dict[str, str],
    expected_error: str,
) -> None:
    root = _make_fake_replay_repo(tmp_path)

    result = _run_fake_replay(root, _fake_replay_env(**overrides))

    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert expected_error in output
    assert not (root / "executor_started").exists()
    assert "replay services ready." not in output


@pytest.mark.parametrize("measured_hz", ("20", "24"))
def test_replay_rejects_low_measured_rate_before_starting_executor(tmp_path: Path, measured_hz: str) -> None:
    root = _make_fake_replay_repo(tmp_path)

    result = _run_fake_replay(root, _fake_replay_env(FAKE_MEASURED_HZ=measured_hz))

    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert "bridge measured send rate not ready" in output
    assert f"measured_send_hz={measured_hz}" in output
    assert not (root / "executor_started").exists()
    assert "replay services ready." not in output


def test_direct_timed_replay_refuses_unconfirmed_minimum_rate_ratio_override(
    tmp_path: Path,
) -> None:
    root = _make_fake_replay_repo(tmp_path)

    result = _run_fake_replay(root, _fake_replay_env(MIN_MEASURED_STATE_HZ_RATIO="0.5"))

    output = result.stdout + result.stderr
    assert result.returncode == 2, output
    assert "refuses unconfirmed MIN_MEASURED_STATE_HZ_RATIO=0.5" in output
    assert not (root / "executor_started").exists()


def test_direct_timed_replay_accepts_exactly_confirmed_minimum_rate_ratio_override(
    tmp_path: Path,
) -> None:
    root = _make_fake_replay_repo(tmp_path)
    env = _fake_replay_env(
        MIN_MEASURED_STATE_HZ_RATIO="0.8",
        JZ_TIMED_MIN_MEASURED_STATE_HZ_RATIO_CONFIRM="0.8",
    )

    result = _run_fake_replay(root, env)

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "MIN_MEASURED_STATE_HZ_RATIO=0.8" in output
    assert "replay services ready." in output


@pytest.mark.parametrize("armed_value", (None, "0"))
def test_replay_requires_explicit_armed_gate_without_starting_bridge(
    tmp_path: Path, armed_value: str | None
) -> None:
    root = _make_fake_replay_repo(tmp_path)
    env = _fake_replay_env()
    if armed_value is None:
        env.pop("JZ_UDP_EXECUTOR_ARMED")
    else:
        env["JZ_UDP_EXECUTOR_ARMED"] = armed_value

    result = _run_fake_replay(root, env)

    output = result.stdout + result.stderr
    assert result.returncode == 2, output
    assert "requires explicit JZ_UDP_EXECUTOR_ARMED=1" in output
    assert not (root / "executor_started").exists()
    assert not (root / "captured_state_wait_timeout.txt").exists()


def test_replay_dry_run_does_not_require_armed_gate_and_uses_dry_run_readiness(
    tmp_path: Path,
) -> None:
    root = _make_fake_replay_repo(tmp_path)
    env = _fake_replay_env(EXECUTION="dry_run")
    env.pop("JZ_UDP_EXECUTOR_ARMED")

    result = _run_fake_replay(root, env)

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "PHASE3 COMMAND EXECUTOR DRY-RUN" in output
    assert (root / "captured_executor_armed.txt").read_text() == "\n"
    assert "replay services ready." in output


def test_replay_never_reports_ready_if_bridge_dies_during_executor_start(tmp_path: Path) -> None:
    root = _make_fake_replay_repo(tmp_path)
    env = _fake_replay_env(
        FAKE_BRIDGE_EXIT_AFTER_FIRST_RATE="1",
        FAKE_BRIDGE_EXIT_DELAY_S="0.6",
        FAKE_EXECUTOR_DELAY_S="0.4",
    )

    result = _run_fake_replay(root, env)

    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert (root / "executor_started").exists()
    assert "ros_state_udp_bridge" in output
    assert "replay services ready." not in output


def test_replay_requires_a_fresh_rate_window_before_final_ready(tmp_path: Path) -> None:
    root = _make_fake_replay_repo(tmp_path)

    result = _run_fake_replay(root, _fake_replay_env(FAKE_NO_SECOND_RATE="1"))

    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert (root / "executor_started").exists()
    assert "minimum_seq=30" in output
    assert "replay services ready." not in output


def _make_fake_orin_start_repo(tmp_path: Path) -> Path:
    root = tmp_path / "fake_orin_start_repo"
    arm_dir = root / "udp_test/server_bash/orin_arm"
    bridge = root / "udp_test/test_scripts/arm_side/orin_ros_state_udp_bridge.py"
    arm_dir.mkdir(parents=True)
    shutil.copy2(ORIN_ARM_DIR / "start.sh", arm_dir / "start.sh")
    _write_executable(
        arm_dir / "stop.sh",
        """
        #!/usr/bin/env bash
        set -euo pipefail
        """,
    )
    _write_executable(
        bridge,
        """
        #!/usr/bin/env python3
        import os
        import signal
        import sys
        import time

        signal.signal(signal.SIGTERM, lambda *_args: sys.exit(0))
        hz = sys.argv[sys.argv.index("--hz") + 1]
        log_hz = os.environ.get("FAKE_BRIDGE_LOG_CONFIGURED_HZ", hz)
        print(
            f"[fake bridge] pid={os.getpid()} local=127.0.0.1:40000 "
            f"configured_hz={log_hz}",
            flush=True,
        )
        while True:
            time.sleep(1)
        """,
    )
    return root


def _fake_orin_start_env(**overrides: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "STATE_HZ": "30",
            "JZ_STATE_HZ_PROFILE": "timed",
            "JZ_EXPECTED_STATE_HZ": "30",
            "STATE_PROCESS_START_TIMEOUT_S": "2",
            "STATE_WAIT_TIMEOUT_S": "1",
            "RUN_READINESS": "0",
            "AUTO_TAIL": "0",
            "PYTHON_CMD": sys.executable,
        }
    )
    env.pop("JZ_TIMED_NON_30_STATE_HZ_CONFIRM", None)
    env.pop("FAKE_BRIDGE_LOG_CONFIGURED_HZ", None)
    env.update(overrides)
    return env


def test_orin_start_verifies_real_python_argv_and_matching_log_hz(tmp_path: Path) -> None:
    root = _make_fake_orin_start_repo(tmp_path)
    start_script = root / "udp_test/server_bash/orin_arm/start.sh"
    pid_file = root / "udp_test/server_bash/orin_arm/pids/ros_state_udp_bridge.pid"

    result = subprocess.run(
        ["bash", str(start_script)],
        cwd=root,
        env=_fake_orin_start_env(),
        text=True,
        capture_output=True,
        timeout=8,
        check=False,
    )
    try:
        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        pid = int(pid_file.read_text(encoding="utf-8"))
        cmdline = (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
        assert b"udp_test/test_scripts/arm_side/orin_ros_state_udp_bridge.py" in cmdline
        assert cmdline[cmdline.index(b"--hz") + 1] == b"30"
        startup = (pid_file.parent / "ros_state_udp_bridge.startup").read_text(encoding="utf-8")
        assert "requested_hz=30" in startup
        assert "configured_hz=30" in startup
        assert "verified bridge" in output
    finally:
        if pid_file.exists():
            os.kill(int(pid_file.read_text(encoding="utf-8")), signal.SIGTERM)


def test_orin_start_rejects_log_hz_mismatch_and_removes_startup_files(tmp_path: Path) -> None:
    root = _make_fake_orin_start_repo(tmp_path)
    start_script = root / "udp_test/server_bash/orin_arm/start.sh"
    pid_dir = root / "udp_test/server_bash/orin_arm/pids"

    result = subprocess.run(
        ["bash", str(start_script)],
        cwd=root,
        env=_fake_orin_start_env(FAKE_BRIDGE_LOG_CONFIGURED_HZ="20"),
        text=True,
        capture_output=True,
        timeout=8,
        check=False,
    )

    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert "bridge log hz verification failed requested_hz=30 configured_hz=30" in output
    assert not (pid_dir / "ros_state_udp_bridge.pid").exists()
    assert not (pid_dir / "ros_state_udp_bridge.startup").exists()
    match = re.search(r"pid=(\d+) local=", output)
    assert match is not None
    deadline = time.monotonic() + 2
    while Path("/proc", match.group(1)).exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not Path("/proc", match.group(1)).exists()


@pytest.mark.parametrize("timeout", ("0", "1.5", "abc"))
def test_orin_start_rejects_non_integer_process_timeout_before_launch(tmp_path: Path, timeout: str) -> None:
    root = _make_fake_orin_start_repo(tmp_path)

    result = subprocess.run(
        ["bash", str(root / "udp_test/server_bash/orin_arm/start.sh")],
        cwd=root,
        env=_fake_orin_start_env(STATE_PROCESS_START_TIMEOUT_S=timeout),
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 2, output
    assert "expected a positive integer" in output
    assert not (root / "udp_test/server_bash/orin_arm/pids/ros_state_udp_bridge.pid").exists()


def test_orin_stop_ignores_stale_pid_and_stops_matching_orphan(tmp_path: Path) -> None:
    root = tmp_path / "fake_orin_stop_repo"
    arm_dir = root / "udp_test/server_bash/orin_arm"
    bridge = root / "udp_test/test_scripts/arm_side/orin_ros_state_udp_bridge.py"
    arm_dir.mkdir(parents=True)
    shutil.copy2(ORIN_ARM_DIR / "stop.sh", arm_dir / "stop.sh")
    _write_executable(
        bridge,
        """
        #!/usr/bin/env python3
        import signal
        import sys
        import time

        signal.signal(signal.SIGTERM, lambda *_args: sys.exit(0))
        while True:
            time.sleep(1)
        """,
    )
    orphan = subprocess.Popen([sys.executable, str(bridge)])
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", str(bridge)])
    pid_dir = arm_dir / "pids"
    pid_dir.mkdir()
    (pid_dir / "ros_state_udp_bridge.pid").write_text(str(unrelated.pid), encoding="utf-8")
    (pid_dir / "ros_state_udp_bridge.startup").write_text(
        f"bridge_pid={orphan.pid}\nlauncher_pid={orphan.pid}\n", encoding="utf-8"
    )
    try:
        result = subprocess.run(
            ["bash", str(arm_dir / "stop.sh")],
            cwd=root,
            env={**os.environ, "STOP_GRACE_S": "5"},
            text=True,
            capture_output=True,
            timeout=8,
            check=False,
        )
        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        orphan.wait(timeout=2)
        assert unrelated.poll() is None
        assert f"ignoring pid_file pid={unrelated.pid}" in output
        assert not (pid_dir / "ros_state_udp_bridge.pid").exists()
        assert not (pid_dir / "ros_state_udp_bridge.startup").exists()
    finally:
        for process in (orphan, unrelated):
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)


def test_orin_stop_refuses_grace_shorter_than_executor_shutdown_window(tmp_path: Path) -> None:
    root = tmp_path / "fake_short_stop_repo"
    arm_dir = root / "udp_test/server_bash/orin_arm"
    arm_dir.mkdir(parents=True)
    shutil.copy2(ORIN_ARM_DIR / "stop.sh", arm_dir / "stop.sh")

    result = subprocess.run(
        ["bash", str(arm_dir / "stop.sh")],
        cwd=root,
        env={**os.environ, "STOP_GRACE_S": "4"},
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 2, output
    assert "expected integer >= 5" in output
