from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ALL_DIR = REPO_ROOT / "udp_test" / "all"


def test_udp_all_scripts_exist_are_executable_and_have_valid_bash_syntax() -> None:
    scripts = [
        ALL_DIR / "start_record.sh",
        ALL_DIR / "stop_record.sh",
        ALL_DIR / "start_replay.sh",
        ALL_DIR / "stop_replay.sh",
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

    assert "server_bash/orin_arm/stop_all.sh" in stop_replay
