from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ALL_DIR = REPO_ROOT / "udp_test" / "all"
ORIN_ARM_DIR = REPO_ROOT / "udp_test" / "server_bash" / "orin_arm"
PIN_EDGE_DIR = REPO_ROOT / "my_devs" / "jz_robot_pin" / "edge"


def test_udp_all_scripts_exist_are_executable_and_have_valid_bash_syntax() -> None:
    scripts = [
        ALL_DIR / "start_record.sh",
        ALL_DIR / "stop_record.sh",
        ALL_DIR / "start_replay.sh",
        ALL_DIR / "stop_replay.sh",
        ORIN_ARM_DIR / "start.sh",
        PIN_EDGE_DIR / "start_pin_replay.sh",
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
    assert "EXECUTION=armed" in start_replay
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
    assert '"local=" \\\n  "$BRIDGE_START_TIMEOUT_S"' in start_replay
    assert '"sent seq=" \\\n  "$STATE_READY_TIMEOUT_S"' in start_replay
    assert start_replay.index('"local="') < start_replay.index('"sent seq="')
    assert '--wait-timeout-s "$STATE_WAIT_TIMEOUT_S"' in bridge_start

    assert "server_bash/orin_arm/stop_all.sh" in stop_replay


def _write_executable(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content), encoding="utf-8")
    path.chmod(0o755)


def test_replay_readiness_uses_separate_start_state_and_executor_windows(tmp_path: Path) -> None:
    root = tmp_path / "fake_repo"
    all_dir = root / "udp_test" / "all"
    arm_dir = root / "udp_test" / "server_bash" / "orin_arm"
    all_dir.mkdir(parents=True)
    arm_dir.mkdir(parents=True)
    shutil.copy2(ALL_DIR / "start_replay.sh", all_dir / "start_replay.sh")

    (root / "fake_service.py").write_text(
        textwrap.dedent(
            """
            import signal
            import sys
            import time

            mode, log_path = sys.argv[1:3]
            signal.signal(signal.SIGTERM, lambda *_args: sys.exit(0))
            with open(log_path, "a", buffering=1, encoding="utf-8") as log:
                if mode == "bridge":
                    delay = float(sys.argv[3])
                    time.sleep(delay)
                    print("[fake bridge] local=127.0.0.1:40000", file=log)
                    time.sleep(delay)
                    print("[fake bridge] sent seq=1", file=log)
                else:
                    time.sleep(float(sys.argv[3]))
                    print("PHASE3 COMMAND EXECUTOR ARMED", file=log)
                    print("port=39020", file=log)
                while True:
                    time.sleep(1)
            """
        ),
        encoding="utf-8",
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
        "$FAKE_PYTHON" "$ROOT_DIR/fake_service.py" bridge \
          "$LOG_DIR/ros_state_udp_bridge.log" "$FAKE_BRIDGE_STEP_S" \
          > /dev/null 2>&1 &
        echo "$!" > "$PID_DIR/ros_state_udp_bridge.pid"
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
        "$FAKE_PYTHON" "$ROOT_DIR/fake_service.py" executor \
          "$LOG_DIR/orin_phase3_command_executor.log" "$FAKE_EXECUTOR_DELAY_S" \
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

    env = os.environ.copy()
    env.update(
        {
            "READY_TIMEOUT_S": "1",
            "BRIDGE_START_TIMEOUT_S": "4",
            "STATE_WAIT_TIMEOUT_S": "7",
            "STATE_READY_TIMEOUT_S": "4",
            "EXECUTOR_READY_TIMEOUT_S": "3",
            "FAKE_BRIDGE_STEP_S": "2.2",
            "FAKE_EXECUTOR_DELAY_S": "1.2",
            "FAKE_PYTHON": sys.executable,
            "TAIL": "0",
        }
    )
    try:
        result = subprocess.run(
            ["bash", str(all_dir / "start_replay.sh")],
            cwd=root,
            env=env,
            text=True,
            capture_output=True,
            timeout=15,
            check=False,
        )
    finally:
        subprocess.run(["bash", str(all_dir / "stop_replay.sh")], cwd=root, check=False)

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert (root / "captured_state_wait_timeout.txt").read_text().strip() == "7"
    assert output.index("confirmed ready from log pattern: local=") < output.index(
        "confirmed ready from log pattern: sent seq="
    )
    assert "replay services ready." in output
