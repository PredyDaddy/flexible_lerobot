from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_actuation_gate.py"
SPEC = importlib.util.spec_from_file_location("validate_actuation_gate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def make_report(tmp_path: Path) -> tuple[Path, Path, str]:
    checkpoint = tmp_path / "checkpoint-63600"
    checkpoint.mkdir()
    task = "Put the eraser into the small box"
    report = {
        "status": "passed",
        "level": "guarded_actuation",
        "actuation_performed": True,
        "sent_action_count": 5,
        "task": task,
        "checkpoint": {"path": str(checkpoint), "global_step": 63600},
        "actuation_safety": {
            "run_time_s": 1.0,
            "execution_horizon": 1,
            "control_hz": 5.0,
            "max_command_delta": 0.25,
            "max_relative_target": 0.25,
        },
    }
    report_path = tmp_path / "summary.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return report_path, checkpoint, task


def test_valid_guarded_smoke_passes(tmp_path: Path) -> None:
    report_path, checkpoint, task = make_report(tmp_path)
    result = MODULE.validate_gate(tmp_path, report_path, checkpoint, task)
    assert result["status"] == "passed"
    assert result["sent_action_count"] == 5


def test_excessive_smoke_delta_is_rejected(tmp_path: Path) -> None:
    report_path, checkpoint, task = make_report(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["actuation_safety"]["max_command_delta"] = 1.0
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(RuntimeError, match="max_command_delta"):
        MODULE.validate_gate(tmp_path, report_path, checkpoint, task)
