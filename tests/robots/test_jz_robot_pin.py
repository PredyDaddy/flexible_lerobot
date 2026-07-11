#!/usr/bin/env python

from __future__ import annotations

import os
import socket
import time
from pathlib import Path

import pytest

from lerobot.robots.config import RobotConfig
from lerobot.robots.jz_robot_pin import JZRobotPin, JZRobotPinConfig
from lerobot.robots.jz_robot_pin.protocol import (
    COMMAND_MODE_ARMED,
    PROTOCOL_VERSION,
    STATE_MESSAGE_TYPE,
    decode_jz_robot_udp_command_packet,
    make_jz_robot_udp_target_action_packet,
)
from lerobot.robots.utils import make_robot_from_config
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.jz_robot_pin_target_action import JZRobotPinTargetActionTeleopConfig
from lerobot.teleoperators.utils import make_teleoperator_from_config

REPO_ROOT = Path(__file__).resolve().parents[2]
PIN_ROOT = REPO_ROOT / "my_devs/jz_robot_pin"


def sample_state_packet(seq: int = 7) -> dict:
    return {
        "version": PROTOCOL_VERSION,
        "type": STATE_MESSAGE_TYPE,
        "robot": "robot1",
        "seq": seq,
        "stamp_ns": 123456789,
        "joints": {
            "left": {f"left_joint{i}": float(i) for i in range(1, 8)},
            "right": {f"right_joint{i}": float(i + 10) for i in range(1, 8)},
        },
        "grippers": {
            "left": {"width": 0.01, "force": 1.0},
            "right": {"width": 0.02, "force": 2.0},
        },
    }


def sample_action() -> dict[str, float]:
    return {
        **{f"left_left_joint{i}.pos": float(i) for i in range(1, 8)},
        **{f"right_right_joint{i}.pos": float(i + 10) for i in range(1, 8)},
        "left_gripper.width": 0.01,
        "left_gripper.force": 1.0,
        "right_gripper.width": 0.02,
        "right_gripper.force": 2.0,
    }


def sample_target_action_packet(seq: int = 1, stamp_ns: int | None = None) -> dict:
    action = sample_action()
    return make_jz_robot_udp_target_action_packet(
        robot="robot1",
        seq=seq,
        stamp_ns=time.time_ns() if stamp_ns is None else stamp_ns,
        actions={
            "left": {f"left_joint{i}": action[f"left_left_joint{i}.pos"] for i in range(1, 8)},
            "right": {f"right_joint{i}": action[f"right_right_joint{i}.pos"] for i in range(1, 8)},
            "grippers": {
                "left": {
                    "width": action["left_gripper.width"],
                    "force": action["left_gripper.force"],
                },
                "right": {
                    "width": action["right_gripper.width"],
                    "force": action["right_gripper.force"],
                },
            },
        },
    )


def make_config(**overrides) -> JZRobotPinConfig:
    values = {
        "id": "test_jz_robot_pin",
        "rtsp_cameras": {},
        "use_gripper": True,
        "state_timeout_s": 1.0,
        "connect_timeout_s": 0.01,
    }
    values.update(overrides)
    return JZRobotPinConfig(**values)


def test_jz_robot_pin_is_registered_as_independent_robot_type() -> None:
    cfg = make_config()
    robot = make_robot_from_config(cfg)

    assert RobotConfig.get_choice_name(JZRobotPinConfig) == "jz_robot_pin"
    assert isinstance(robot, JZRobotPin)
    assert robot.name == "jz_robot_pin"


def test_jz_robot_pin_defaults_are_safe_and_state_sender_alias_is_explicit() -> None:
    cfg = JZRobotPinConfig(allowed_state_sender_ip="10.0.0.2")

    assert cfg.allowed_state_sender_ip == "10.0.0.2"
    assert cfg.allowed_sender_ip == "10.0.0.2"
    assert cfg.send_action_transport == "local"
    assert cfg.send_action_execution == "dry_run"
    assert cfg.require_armed_env
    assert cfg.armed_env_var == "JZ_ROBOT_PIN_ARMED"
    assert cfg.max_initial_joint_delta_rad == 0.02
    assert cfg.max_joint_step_rad == 0.02


def test_jz_robot_pin_accepts_legacy_allowed_sender_alias() -> None:
    cfg = JZRobotPinConfig(allowed_sender_ip="10.0.0.3")

    assert cfg.allowed_sender_ip == "10.0.0.3"
    assert cfg.allowed_state_sender_ip == "10.0.0.3"


@pytest.mark.parametrize("field", ["max_initial_joint_delta_rad", "max_joint_step_rad"])
@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), float("-inf"), -0.01])
def test_jz_robot_pin_rejects_invalid_joint_safety_limits(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        make_config(**{field: value})


@pytest.mark.parametrize("field", ["max_initial_joint_delta_rad", "max_joint_step_rad"])
def test_jz_robot_pin_armed_requires_enabled_joint_safety_limits(field: str) -> None:
    with pytest.raises(ValueError, match="armed JZRobotPin requires"):
        make_config(send_action_execution=COMMAND_MODE_ARMED, **{field: 0.0})


def test_jz_robot_pin_armed_cannot_disable_environment_gate() -> None:
    with pytest.raises(ValueError, match="require_armed_env=true"):
        make_config(send_action_execution=COMMAND_MODE_ARMED, require_armed_env=False)


def test_jz_robot_pin_observation_features_and_cached_state() -> None:
    robot = JZRobotPin(make_config())
    robot._is_connected = True
    robot._state_cache.update(sample_state_packet(), sender=("192.168.1.81", 39010))

    obs = robot.get_observation()

    assert robot.observation_features["left_left_joint1.pos"] is float
    assert robot.action_features["right_right_joint7.pos"] is float
    assert obs["left_left_joint1.pos"] == 1.0
    assert obs["right_right_joint7.pos"] == 17.0
    assert obs["left_gripper.width"] == 0.01


def test_jz_robot_pin_refuses_armed_send_without_pin_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JZ_ROBOT_PIN_ARMED", raising=False)
    robot = JZRobotPin(make_config(send_action_execution=COMMAND_MODE_ARMED, send_action_transport="local"))
    robot._is_connected = True

    with pytest.raises(RuntimeError, match="JZ_ROBOT_PIN_ARMED"):
        robot.send_action(sample_action())


def test_jz_robot_pin_udp_armed_send_requires_env_and_sends_packet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JZ_ROBOT_PIN_ARMED", "1")
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    receiver.settimeout(1.0)
    _, port = receiver.getsockname()
    robot = JZRobotPin(
        make_config(
            command_target_ip="127.0.0.1",
            command_target_port=port,
            send_action_transport="udp",
            send_action_execution=COMMAND_MODE_ARMED,
        )
    )
    robot._is_connected = True
    robot._state_cache.update(sample_state_packet(), sender=("192.168.1.81", 39010))

    try:
        sent_action = robot.send_action(sample_action())
        data, sender = receiver.recvfrom(65535)
    finally:
        robot.disconnect()
        receiver.close()

    decoded = decode_jz_robot_udp_command_packet(data)
    assert sender[0] == "127.0.0.1"
    assert sent_action == sample_action()
    assert decoded["mode"] == COMMAND_MODE_ARMED
    assert decoded["type"] == "command"
    assert decoded["actions"]["left"]["left_joint1"] == 1.0


def test_jz_robot_pin_rejects_initial_joint_delta_from_latest_state() -> None:
    robot = JZRobotPin(make_config(max_initial_joint_delta_rad=0.01))
    robot._is_connected = True
    robot._state_cache.update(sample_state_packet(), sender=("192.168.1.81", 39010))
    action = sample_action()
    action["left_left_joint1.pos"] += 0.02

    with pytest.raises(ValueError, match="initial joint delta"):
        robot.send_action(action)


def test_jz_robot_pin_rejects_initial_action_without_state() -> None:
    robot = JZRobotPin(make_config())
    robot._is_connected = True

    with pytest.raises(TimeoutError, match="without a robot state packet"):
        robot.send_action(sample_action())


def test_jz_robot_pin_rejects_initial_action_from_unexpected_sender() -> None:
    robot = JZRobotPin(make_config())
    robot._is_connected = True
    robot._state_cache.update(sample_state_packet(), sender=("10.0.0.9", 39010))

    with pytest.raises(RuntimeError, match="unexpected sender"):
        robot.send_action(sample_action())


def test_jz_robot_pin_rejects_initial_action_for_wrong_robot() -> None:
    robot = JZRobotPin(make_config())
    robot._is_connected = True
    packet = sample_state_packet()
    packet["robot"] = "robot2"
    robot._state_cache.update(packet, sender=("192.168.1.81", 39010))

    with pytest.raises(RuntimeError, match="does not match"):
        robot.send_action(sample_action())


def test_jz_robot_pin_initial_joint_delta_boundary() -> None:
    robot = JZRobotPin(make_config(max_initial_joint_delta_rad=0.02))
    robot._is_connected = True
    robot._state_cache.update(sample_state_packet(), sender=("192.168.1.81", 39010))
    action = sample_action()
    action["left_left_joint1.pos"] += 0.02

    assert robot.send_action(action) == action


def test_jz_robot_pin_armed_rechecks_state_freshness_before_every_send(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JZ_ROBOT_PIN_ARMED", "1")
    robot = JZRobotPin(
        make_config(
            send_action_execution=COMMAND_MODE_ARMED,
            send_action_transport="local",
            state_timeout_s=0.001,
        )
    )
    robot._is_connected = True
    robot._state_cache.update(sample_state_packet(), sender=("192.168.1.81", 39010))
    robot.send_action(sample_action())
    time.sleep(0.01)

    with pytest.raises(TimeoutError, match="stale robot state"):
        robot.send_action(sample_action())


def test_jz_robot_pin_armed_rejects_nonadvancing_state_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JZ_ROBOT_PIN_ARMED", "1")
    robot = JZRobotPin(
        make_config(send_action_execution=COMMAND_MODE_ARMED, send_action_transport="local")
    )
    robot._is_connected = True
    robot._state_cache.update(sample_state_packet(seq=7), sender=("192.168.1.81", 39010))
    robot.send_action(sample_action())
    robot._state_cache.update(sample_state_packet(seq=6), sender=("192.168.1.81", 39010))

    with pytest.raises(TimeoutError, match="sequence did not advance"):
        robot.send_action(sample_action())


def test_jz_robot_pin_rejects_step_joint_delta_from_last_sent_action() -> None:
    robot = JZRobotPin(make_config(max_joint_step_rad=0.01, send_action_transport="local"))
    robot._is_connected = True
    robot._state_cache.update(sample_state_packet(), sender=("192.168.1.81", 39010))
    first = sample_action()
    second = sample_action()
    second["right_right_joint7.pos"] += 0.02

    robot.send_action(first)

    with pytest.raises(ValueError, match="step joint delta"):
        robot.send_action(second)


def test_jz_robot_pin_clamps_gripper_limits() -> None:
    robot = JZRobotPin(
        make_config(
            gripper_width_min=0.0,
            gripper_width_max=50.0,
            gripper_force_min=5.0,
            gripper_force_max=80.0,
            send_action_transport="local",
        )
    )
    robot._is_connected = True
    robot._state_cache.update(sample_state_packet(), sender=("192.168.1.81", 39010))
    action = sample_action()
    action["left_gripper.width"] = 100.0
    action["right_gripper.force"] = 1.0

    sent = robot.send_action(action)

    assert sent["left_gripper.width"] == 50.0
    assert sent["right_gripper.force"] == 5.0


def test_jz_robot_pin_target_action_teleop_is_registered_and_can_hold_current() -> None:
    cfg = JZRobotPinTargetActionTeleopConfig(target_action_port=0, stale_policy="hold_current")
    teleop = make_teleoperator_from_config(cfg)
    robot = JZRobotPin(make_config())
    observation = {key: float(index) for index, key in enumerate(robot.action_features)}

    assert (
        TeleoperatorConfig.get_choice_name(JZRobotPinTargetActionTeleopConfig)
        == "jz_robot_pin_target_action"
    )
    assert teleop.action_features == robot.action_features

    teleop.connect()
    try:
        action = teleop.get_action_from_observation(observation)
    finally:
        teleop.disconnect()

    assert action == observation


def test_jz_robot_pin_target_action_teleop_raise_policy_preserves_strict_timeout() -> None:
    cfg = JZRobotPinTargetActionTeleopConfig(target_action_port=0, stale_policy="raise")
    teleop = make_teleoperator_from_config(cfg)
    observation = {key: 0.0 for key in teleop.action_features}

    teleop.connect()
    try:
        with pytest.raises(TimeoutError):
            teleop.get_action_from_observation(observation)
    finally:
        teleop.disconnect()


def test_jz_robot_pin_target_action_accepts_same_cached_packet_more_than_once() -> None:
    cfg = JZRobotPinTargetActionTeleopConfig(target_action_port=0, stale_policy="raise")
    teleop = make_teleoperator_from_config(cfg)
    teleop._is_connected = True
    teleop._target_action_cache.update(
        sample_target_action_packet(seq=4), sender=("127.0.0.1", 39030)
    )

    first = teleop.get_action()
    second = teleop.get_action()

    assert first == second == sample_action()


def test_jz_robot_pin_target_action_rejects_nonadvancing_sequence() -> None:
    cfg = JZRobotPinTargetActionTeleopConfig(target_action_port=0, stale_policy="raise")
    teleop = make_teleoperator_from_config(cfg)
    teleop._is_connected = True
    teleop._target_action_cache.update(
        sample_target_action_packet(seq=4), sender=("127.0.0.1", 39030)
    )
    teleop.get_action()
    teleop._target_action_cache.update(
        sample_target_action_packet(seq=3), sender=("127.0.0.1", 39030)
    )

    with pytest.raises(TimeoutError, match="sequence did not advance"):
        teleop.get_action()


def test_jz_robot_pin_target_action_rejects_replayed_old_stamp() -> None:
    cfg = JZRobotPinTargetActionTeleopConfig(
        target_action_port=0,
        stale_policy="raise",
        packet_max_age_s=0.1,
    )
    teleop = make_teleoperator_from_config(cfg)
    teleop._is_connected = True
    teleop._target_action_cache.update(
        sample_target_action_packet(seq=1, stamp_ns=time.time_ns() - 1_000_000_000),
        sender=("127.0.0.1", 39030),
    )

    with pytest.raises(TimeoutError, match="stamp is stale"):
        teleop.get_action()


@pytest.mark.parametrize(
    "field",
    ["packet_max_age_s", "packet_max_future_skew_s", "seq_reset_timeout_s"],
)
@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), -0.01])
def test_jz_robot_pin_target_action_rejects_invalid_packet_safety_config(
    field: str, value: float
) -> None:
    with pytest.raises(ValueError):
        JZRobotPinTargetActionTeleopConfig(**{field: value})


def test_jz_robot_pin_record_defaults_to_strict_target_action_input() -> None:
    script = (PIN_ROOT / "record.sh").read_text(encoding="utf-8")

    assert 'TARGET_ACTION_CONNECT_TIMEOUT_S="${TARGET_ACTION_CONNECT_TIMEOUT_S:-5.0}"' in script
    assert 'TARGET_ACTION_STALE_POLICY="${TARGET_ACTION_STALE_POLICY:-raise}"' in script
    assert '--teleop.connect_timeout_s="${TARGET_ACTION_CONNECT_TIMEOUT_S}"' in script
    assert '--teleop.stale_policy="${TARGET_ACTION_STALE_POLICY}"' in script


def test_jz_robot_pin_data_check_wrapper_records_then_checks_exactly_three_episodes() -> None:
    script = (PIN_ROOT / "data_check/record_and_check_3.sh").read_text(encoding="utf-8")

    assert "NUM_EPISODES=3" in script
    assert "RESUME=false" in script
    assert "VIDEO_ENCODING_BATCH_SIZE=3" in script
    assert 'MAX_INITIAL_JOINT_DELTA_RAD="${MAX_INITIAL_JOINT_DELTA_RAD:-10.0}"' in script
    assert 'MAX_JOINT_STEP_RAD="${MAX_JOINT_STEP_RAD:-10.0}"' in script
    assert '--max-initial-joint-delta-rad "${MAX_INITIAL_JOINT_DELTA_RAD}"' in script
    assert '--max-action-joint-step-rad "${MAX_JOINT_STEP_RAD}"' in script
    assert 'bash "${PIN_ROOT}/record.sh"' in script
    assert '"${SCRIPT_DIR}/check_3_episodes.py"' in script


def test_jz_robot_pin_python_helpers_do_not_bypass_conda_with_python_override() -> None:
    common = (PIN_ROOT / "lib/common.sh").read_text(encoding="utf-8")
    joystick = (PIN_ROOT / "x86/start_pin_joystick.sh").read_text(encoding="utf-8")

    assert '${PYTHON:-}' not in common
    assert "exec env -u PYTHON" in joystick


def test_jz_robot_pin_joystick_continuously_publishes_for_recording() -> None:
    script = (PIN_ROOT / "x86/start_pin_joystick.sh").read_text(encoding="utf-8")

    assert 'PUBLISH_MODE="${PUBLISH_MODE:-always}"' in script
    assert 'VISUAL_DURATION_S="${VISUAL_DURATION_S:-86400}"' in script
    assert '--publish-mode "${PUBLISH_MODE}"' in script
    assert '--duration "${VISUAL_DURATION_S}"' in script
    assert 'VISUAL_FREQUENCY="${VISUAL_FREQUENCY:-90}"' in script
    assert 'DISPLAY_EVERY="${DISPLAY_EVERY:-3}"' in script
    assert 'VISUALIZE_WHOLE_ROBOT="${VISUALIZE_WHOLE_ROBOT:-true}"' in script
    assert 'SHOW_VR_DEBUG="${SHOW_VR_DEBUG:-false}"' in script
    assert 'VISUALIZE_WHOLE_ROBOT="${VISUALIZE_WHOLE_ROBOT}"' in script
    assert '--frequency "${VISUAL_FREQUENCY}"' in script
    assert '--display-every "${DISPLAY_EVERY}"' in script
    assert 'ROBOT_VISUAL_ARGS=(--no-arm-meshes-only)' in script
    assert '"${ROBOT_VISUAL_ARGS[@]}"' in script
    assert '"${VR_DEBUG_ARGS[@]}"' in script
    assert "unsupported VISUALIZE_WHOLE_ROBOT=" in script


@pytest.mark.parametrize(
    "rel_path",
    [
        "record.sh",
        "replay.sh",
        "reply.sh",
        "start_record.sh",
        "start_replay.sh",
        "start_teleop.sh",
        "stop_teleop.sh",
        "x86/start_pin_control.sh",
        "x86/start_pin_joystick.sh",
        "x86/start_pin_teleop.sh",
        "x86/stop_pin_teleop.sh",
        "x86/start_pin_record.sh",
        "x86/start_pin_replay.sh",
        "edge/start_pin_state.sh",
        "edge/start_pin_replay.sh",
        "edge/stop_pin_replay.sh",
        "edge/status_pin_replay.sh",
        "lib/common.sh",
        "data_check/record_and_check_3.sh",
        "data_check/check_3_episodes.py",
    ],
)
def test_jz_robot_pin_scripts_live_under_requested_folder(rel_path: str) -> None:
    script = PIN_ROOT / rel_path

    assert script.is_file()
    assert os.access(script, os.X_OK)
