#!/usr/bin/env python

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ORIN_TIMING_PROBE = REPO_ROOT / "udp_test/diagnostics/orin_record_timing_probe.py"


def test_orin_record_timing_probe_help_does_not_require_rclpy() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"

    result = subprocess.run(
        [sys.executable, str(ORIN_TIMING_PROBE), "--help"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Readonly Orin-side timing probe" in result.stdout
    assert "No module named 'rclpy'" not in result.stderr
