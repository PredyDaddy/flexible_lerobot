from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "so101_robot_client.py"
SPEC = importlib.util.spec_from_file_location("so101_robot_client", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_build_model_observation_shapes() -> None:
    observation = {key: float(index) for index, key in enumerate(MODULE.JOINT_KEYS)}
    observation["top"] = np.full((480, 640, 3), 10, dtype=np.uint8)
    observation["wrist"] = np.full((480, 640, 3), 20, dtype=np.uint8)
    result = MODULE.build_model_observation(observation, MODULE.KNOWN_TASKS[0])

    assert result["video"]["top"].shape == (1, 1, 480, 640, 3)
    assert result["video"]["wrist"].shape == (1, 1, 480, 640, 3)
    assert result["state"]["single_arm"].shape == (1, 1, 5)
    assert result["state"]["gripper"].shape == (1, 1, 1)
    assert result["language"][MODULE.LANGUAGE_KEY] == [[MODULE.KNOWN_TASKS[0]]]


def test_validate_action_chunk_and_sanitize() -> None:
    action = {
        "single_arm": np.ones((1, 16, 5), dtype=np.float32) * 20,
        "gripper": np.ones((1, 16, 1), dtype=np.float32) * 50,
    }
    chunk = MODULE.validate_action_chunk(action)
    assert chunk.shape == (16, 6)

    bounds = MODULE.ActionBounds(
        lower=np.asarray([-10, -10, -10, -10, -10, 0], dtype=np.float32),
        upper=np.asarray([10, 10, 10, 10, 10, 60], dtype=np.float32),
        source="test",
    )
    safe, details = MODULE.sanitize_action(
        chunk[0], np.zeros(6, dtype=np.float32), bounds, max_command_delta=2
    )
    np.testing.assert_array_equal(safe, np.asarray([2, 2, 2, 2, 2, 2], dtype=np.float32))
    assert details["hard_bound_clipped"] is True
    assert details["delta_clipped"] is True


def test_msgpack_numpy_roundtrip() -> None:
    source = {"value": np.arange(12, dtype=np.float32).reshape(3, 4)}
    packed = MODULE.msgpack.packb(source, default=MODULE.encode_msgpack)
    result = MODULE.msgpack.unpackb(packed, object_hook=MODULE.decode_msgpack)
    np.testing.assert_array_equal(result["value"], source["value"])


def test_actuation_requires_two_explicit_gates() -> None:
    parser = MODULE.build_parser()
    args = parser.parse_args(["--mode", "actuate"])
    assert args.bounds_mode == "physical"
    assert args.max_command_delta == 3.0
    assert args.max_relative_target == 3.0
    assert args.control_hz == 5.0
    try:
        MODULE.validate_args(args)
    except PermissionError as exc:
        assert MODULE.ACTUATION_CONFIRMATION in str(exc)
    else:
        raise AssertionError("Actuation unexpectedly passed without confirmation gates")


def test_actuation_rejects_non_finite_or_unbounded_runtime() -> None:
    parser = MODULE.build_parser()
    for runtime in (math.inf, math.nan, 601.0):
        args = parser.parse_args(
            [
                "--mode",
                "actuate",
                "--run-time-s",
                str(runtime),
                "--enable-actuation",
                "--confirm-actuation",
                MODULE.ACTUATION_CONFIRMATION,
            ]
        )
        try:
            MODULE.validate_args(args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Unsafe runtime unexpectedly passed: {runtime}")


def test_actuation_rejects_long_network_wait() -> None:
    parser = MODULE.build_parser()
    args = parser.parse_args(
        [
            "--mode",
            "actuate",
            "--request-timeout-s",
            "3",
            "--max-inference-s",
            "2",
            "--enable-actuation",
            "--confirm-actuation",
            MODULE.ACTUATION_CONFIRMATION,
        ]
    )
    try:
        MODULE.validate_args(args)
    except ValueError as exc:
        assert "request_timeout_s" in str(exc)
    else:
        raise AssertionError("Actuation unexpectedly accepted a network timeout above the inference limit")


def test_partial_robot_connection_cleanup_disables_torque() -> None:
    class FakeBus:
        is_connected = True

        def __init__(self) -> None:
            self.disable_torque = None

        def disconnect(self, disable_torque: bool) -> None:
            self.disable_torque = disable_torque
            self.is_connected = False

    class FakeCamera:
        is_connected = False

    class FakeRobot:
        def __init__(self) -> None:
            self.bus = FakeBus()
            self.cameras = {"failed_camera": FakeCamera()}

    robot = FakeRobot()
    errors = MODULE.disconnect_robot_safely(robot)
    assert errors == []
    assert robot.bus.disable_torque is True
