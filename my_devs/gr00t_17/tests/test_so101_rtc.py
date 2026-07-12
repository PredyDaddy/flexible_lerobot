from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "so101_rtc_robot_client.py"
SPEC = importlib.util.spec_from_file_location("so101_rtc_robot_client", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_rtc_formal_defaults_match_validated_unrestricted_run() -> None:
    args = MODULE.build_parser().parse_args(["--mode", "predict"])
    assert args.run_time_s == 120.0
    assert args.execution_horizon == 8
    assert args.control_hz == 30.0
    assert args.bounds_mode == "physical"
    assert args.max_command_delta == 200.0
    assert args.max_relative_target == 200.0
    assert args.expected_backend == "any"


def test_dry_run_does_not_require_actuation_gate() -> None:
    args = MODULE.build_parser().parse_args(["--mode", "dry-run", "--run-time-s", "1"])
    MODULE.validate_args(args)


def test_rtc_options_align_16_step_chunk() -> None:
    args = MODULE.build_parser().parse_args(["--mode", "predict"])
    options = MODULE.rtc_options(args, frozen_steps=2)
    assert options == {
        "rtc_enabled": True,
        "rtc_advance_steps": 8,
        "rtc_frozen_steps": 2,
        "rtc_ramp_rate": 2.0,
    }


def test_rtc_options_clamp_frozen_steps_to_overlap() -> None:
    args = MODULE.build_parser().parse_args(["--mode", "predict"])
    assert MODULE.rtc_options(args, frozen_steps=20)["rtc_frozen_steps"] == 8


def test_timeline_queue_discards_actions_elapsed_during_inference() -> None:
    queue = MODULE.TimelineActionQueue()
    first = np.arange(16 * 2, dtype=np.float32).reshape(16, 2)
    assert queue.merge(first, request_step=0, current_step=0) == 0
    np.testing.assert_array_equal(queue.get(8), first[8])

    replacement = first + 100
    assert queue.merge(replacement, request_step=8, current_step=10) == 2
    assert queue.get(9) is None
    np.testing.assert_array_equal(queue.get(10), replacement[2])
    assert queue.remaining(10) == 14


def test_timeline_queue_rejects_stale_chunk() -> None:
    queue = MODULE.TimelineActionQueue()
    chunk = np.zeros((16, 6), dtype=np.float32)
    try:
        queue.merge(chunk, request_step=0, current_step=16)
    except RuntimeError as exc:
        assert "stale" in str(exc)
    else:
        raise AssertionError("A fully stale RTC response was accepted")
