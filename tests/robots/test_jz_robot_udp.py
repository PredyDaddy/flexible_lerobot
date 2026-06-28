#!/usr/bin/env python

from __future__ import annotations

import ast
import math
import socket
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

from lerobot.robots.jz_robot_udp import JZRobotUDP, JZRobotUDPConfig
from lerobot.robots.jz_robot_udp.protocol import (
    COMMAND_MESSAGE_TYPE,
    PROTOCOL_VERSION,
    STATE_MESSAGE_TYPE,
    decode_jz_robot_udp_command_packet,
    decode_state_packet,
    encode_jz_robot_udp_command_packet,
    encode_state_packet,
    make_jz_robot_udp_command_packet,
)
from lerobot.robots.jz_robot_udp.state_cache import StateCache

REPO_ROOT = Path(__file__).resolve().parents[2]
ORIN_COMMAND_RECEIVER = REPO_ROOT / "udp_test/test_scripts/arm_side/orin_udp_command_receiver.py"


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
    action = {
        **{f"left_left_joint{i}.pos": float(i) for i in range(1, 8)},
        **{f"right_right_joint{i}.pos": float(i + 10) for i in range(1, 8)},
        "left_gripper.width": 0.01,
        "left_gripper.force": 1.0,
        "right_gripper.width": 0.02,
        "right_gripper.force": 2.0,
    }
    return action


def sample_command_actions() -> dict:
    return {
        "left": {f"left_joint{i}": float(i) for i in range(1, 8)},
        "right": {f"right_joint{i}": float(i + 10) for i in range(1, 8)},
        "grippers": {
            "left": {"width": 0.01, "force": 1.0},
            "right": {"width": 0.02, "force": 2.0},
        },
    }


def make_config(**overrides) -> JZRobotUDPConfig:
    values = {
        "id": "test_jz_robot_udp",
        "rtsp_cameras": {},
        "use_gripper": True,
        "state_timeout_s": 1.0,
        "connect_timeout_s": 0.01,
    }
    values.update(overrides)
    return JZRobotUDPConfig(
        **values,
    )


def test_jz_robot_udp_command_config_defaults_are_safe() -> None:
    cfg = JZRobotUDPConfig()

    assert cfg.command_target_ip == "192.168.1.81"
    assert cfg.command_target_port == 39020
    assert cfg.send_action_transport == "local"
    assert cfg.send_action_execution == "dry_run"
    assert cfg.command_robot == "robot1"
    assert cfg.command_timeout_s == 0.2


@pytest.mark.parametrize("execution", ["active", "armed", "execute", "publish", ""])
def test_jz_robot_udp_command_config_rejects_non_dry_run_execution(execution: str) -> None:
    with pytest.raises(ValueError, match="dry_run"):
        JZRobotUDPConfig(send_action_execution=execution)


@pytest.mark.parametrize("transport", ["active", "armed", "execute", "publish", ""])
def test_jz_robot_udp_command_config_rejects_unknown_transport(transport: str) -> None:
    with pytest.raises(ValueError, match="send_action_transport"):
        JZRobotUDPConfig(send_action_transport=transport)


def test_state_packet_round_trip_validates_schema() -> None:
    encoded = encode_state_packet(sample_state_packet())
    decoded = decode_state_packet(encoded)

    assert decoded["version"] == 1
    assert decoded["type"] == "state"
    assert decoded["seq"] == 7
    assert decoded["joints"]["left"]["left_joint1"] == 1.0
    assert decoded["grippers"]["right"]["force"] == 2.0


def test_command_packet_round_trip_validates_schema() -> None:
    packet = make_jz_robot_udp_command_packet(
        robot="robot1",
        seq=1,
        stamp_ns=123,
        mode="dry_run",
        actions=sample_command_actions(),
    )

    decoded = decode_jz_robot_udp_command_packet(encode_jz_robot_udp_command_packet(packet))

    assert decoded["version"] == PROTOCOL_VERSION
    assert decoded["type"] == COMMAND_MESSAGE_TYPE
    assert decoded["robot"] == "robot1"
    assert decoded["seq"] == 1
    assert decoded["stamp_ns"] == 123
    assert decoded["mode"] == "dry_run"
    assert decoded["actions"]["left"]["left_joint1"] == 1.0
    assert decoded["actions"]["grippers"]["right"]["force"] == 2.0


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda packet: packet.__setitem__("type", "state"), "message type"),
        (lambda packet: packet.pop("actions"), "actions"),
        (lambda packet: packet.pop("stamp_ns"), "stamp_ns"),
        (lambda packet: packet.__setitem__("seq", True), "seq"),
        (lambda packet: packet.__setitem__("stamp_ns", False), "stamp_ns"),
        (lambda packet: packet.__setitem__("mode", "execute"), "dry_run"),
        (lambda packet: packet["actions"]["grippers"]["left"].__setitem__("velocity", 0.1), "velocity"),
        (lambda packet: packet["actions"]["left"].__setitem__("left_joint1", math.nan), "finite"),
        (lambda packet: packet["actions"]["right"].__setitem__("right_joint1", math.inf), "finite"),
    ],
)
def test_command_packet_rejects_invalid_cases(mutation, match: str) -> None:
    packet = make_jz_robot_udp_command_packet(
        robot="robot1",
        seq=1,
        stamp_ns=123,
        mode="dry_run",
        actions=sample_command_actions(),
    )
    mutation(packet)

    with pytest.raises(Exception, match=match):
        decode_jz_robot_udp_command_packet(encode_jz_robot_udp_command_packet(packet))


def test_state_cache_waits_for_latest_state() -> None:
    cache = StateCache()
    assert cache.latest() is None

    cache.update(sample_state_packet(seq=1), sender=("192.168.1.81", 39010))

    latest = cache.wait(timeout_s=0.01)
    assert latest is not None
    assert latest.packet["seq"] == 1
    assert latest.sender == ("192.168.1.81", 39010)


def test_jz_robot_udp_observation_features_match_jz_robot_style() -> None:
    robot = JZRobotUDP(make_config())

    assert robot.observation_features["left_left_joint1.pos"] is float
    assert robot.observation_features["right_right_joint7.pos"] is float
    assert robot.observation_features["left_gripper.width"] is float
    assert robot.observation_features["right_gripper.force"] is float
    assert robot.action_features["left_left_joint1.pos"] is float
    assert robot.action_features["right_right_joint7.pos"] is float
    assert robot.action_features["left_gripper.width"] is float
    assert robot.action_features["right_gripper.force"] is float


def test_jz_robot_udp_get_observation_from_cached_state() -> None:
    robot = JZRobotUDP(make_config())
    robot._is_connected = True
    robot._state_cache.update(sample_state_packet(), sender=("192.168.1.81", 39010))

    obs = robot.get_observation()

    assert obs["left_left_joint1.pos"] == 1.0
    assert obs["right_right_joint7.pos"] == 17.0
    assert obs["left_gripper.width"] == 0.01
    assert obs["right_gripper.force"] == 2.0


def test_jz_robot_udp_local_dry_run_send_action_returns_float_copy_without_udp_sender() -> None:
    robot = JZRobotUDP(make_config(send_action_transport="local", send_action_execution="dry_run"))
    robot._is_connected = True

    returned = robot.send_action(sample_action())

    assert returned == sample_action()
    assert all(isinstance(value, float) for value in returned.values())
    assert robot._command_seq == 1
    assert robot._command_sender is None


def test_jz_robot_udp_send_action_rejects_missing_or_extra_keys() -> None:
    robot = JZRobotUDP(make_config(send_action_transport="local", send_action_execution="dry_run"))
    robot._is_connected = True
    missing_action = sample_action()
    del missing_action["left_left_joint1.pos"]
    extra_action = {**sample_action(), "left_left_joint99.pos": 99.0}

    with pytest.raises(ValueError, match="missing"):
        robot.send_action(missing_action)
    with pytest.raises(ValueError, match="unexpected"):
        robot.send_action(extra_action)


def test_jz_robot_udp_send_action_rejects_bool_values() -> None:
    robot = JZRobotUDP(make_config(send_action_transport="local", send_action_execution="dry_run"))
    robot._is_connected = True
    action = sample_action()
    action["left_left_joint1.pos"] = True

    with pytest.raises(ValueError, match="numeric"):
        robot.send_action(action)


def test_jz_robot_udp_udp_dry_run_send_action_sends_command_packet() -> None:
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    receiver.settimeout(1.0)
    _, port = receiver.getsockname()
    robot = JZRobotUDP(
        make_config(
            command_target_ip="127.0.0.1",
            command_target_port=port,
            send_action_transport="udp",
            send_action_execution="dry_run",
        )
    )
    robot._is_connected = True

    try:
        returned = robot.send_action(sample_action())
        data, sender = receiver.recvfrom(65535)
    finally:
        robot.disconnect()
        receiver.close()

    decoded = decode_jz_robot_udp_command_packet(data)
    assert returned == sample_action()
    assert sender[0] == "127.0.0.1"
    assert decoded["type"] == "command"
    assert decoded["mode"] == "dry_run"
    assert decoded["robot"] == "robot1"
    assert decoded["seq"] == 1
    assert decoded["actions"]["left"]["left_joint1"] == 1.0
    assert decoded["actions"]["right"]["right_joint7"] == 17.0
    assert decoded["actions"]["grippers"]["left"]["width"] == 0.01


def test_orin_command_receiver_has_no_ros_publish_path() -> None:
    tree = ast.parse(ORIN_COMMAND_RECEIVER.read_text())
    forbidden_import_roots = {"rclpy"}
    forbidden_import_parts = {"cmd_vel"}
    forbidden_attributes = {"create_publisher", "publish"}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", maxsplit=1)[0]
                parts = set(alias.name.lower().split("."))
                assert root not in forbidden_import_roots
                assert not (parts & forbidden_import_parts)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root = module.split(".", maxsplit=1)[0]
            parts = set(module.lower().split("."))
            assert root not in forbidden_import_roots
            assert not (parts & forbidden_import_parts)
        elif isinstance(node, ast.Attribute):
            assert node.attr not in forbidden_attributes


def test_jz_robot_udp_stale_state_fails() -> None:
    robot = JZRobotUDP(make_config(state_timeout_s=0.01))
    robot._is_connected = True
    robot._state_cache.update(sample_state_packet(), sender=("192.168.1.81", 39010))
    time.sleep(0.02)

    with pytest.raises(TimeoutError, match="stale"):
        robot.get_observation()


def test_state_packet_requires_gripper_fields_when_grippers_are_present() -> None:
    packet = sample_state_packet()
    del packet["grippers"]["left"]["force"]

    with pytest.raises(Exception, match="force"):
        decode_state_packet(encode_state_packet(packet))


def test_connect_waits_for_fresh_state_after_receiver_start() -> None:
    robot = JZRobotUDP(make_config(connect_timeout_s=0.01))
    robot._state_cache.update(sample_state_packet(), sender=("192.168.1.81", 39010))
    robot._receiver = Mock()
    robot._receiver.start.side_effect = lambda: None
    robot._receiver.stop.side_effect = lambda: None

    with pytest.raises(TimeoutError, match="first JZRobot UDP state packet"):
        robot.connect()


def test_jz_robot_udp_sender_filter_rejects_unexpected_sender() -> None:
    robot = JZRobotUDP(make_config(allowed_sender_ip="192.168.1.81"))
    robot._is_connected = True
    robot._state_cache.update(sample_state_packet(), sender=("192.168.1.200", 39010))

    with pytest.raises(RuntimeError, match="unexpected sender"):
        robot.get_observation()
