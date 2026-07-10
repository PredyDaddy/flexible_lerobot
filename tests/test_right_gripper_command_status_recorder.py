#!/usr/bin/env python

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RECORDER = REPO_ROOT / "udp_test/local_test/record_right_gripper_command_status.py"


def test_right_gripper_command_status_recorder_help_does_not_require_rclpy() -> None:
    result = subprocess.run(
        [sys.executable, str(RECORDER), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Readonly recorder comparing right gripper status against command messages" in result.stdout
    assert "No module named 'rclpy'" not in result.stderr
