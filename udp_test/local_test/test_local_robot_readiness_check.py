#!/usr/bin/env python3

from local_robot_readiness_check import CheckResult


def test_check_result_status_values() -> None:
    assert CheckResult(name="ok", ok=True, detail="").status == "PASS"
    assert CheckResult(name="warn", ok=True, detail="", warn=True).status == "WARN"
    assert CheckResult(name="fail", ok=False, detail="").status == "FAIL"
