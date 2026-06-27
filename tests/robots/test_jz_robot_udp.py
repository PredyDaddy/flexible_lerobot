#!/usr/bin/env python

from __future__ import annotations

import time
from unittest.mock import Mock

import pytest

from lerobot.robots.jz_robot_udp import JZRobotUDP, JZRobotUDPConfig
from lerobot.robots.jz_robot_udp.protocol import (
    PROTOCOL_VERSION,
    STATE_MESSAGE_TYPE,
    decode_state_packet,
    encode_state_packet,
)
from lerobot.robots.jz_robot_udp.state_cache import StateCache


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


def test_state_packet_round_trip_validates_schema() -> None:
    encoded = encode_state_packet(sample_state_packet())
    decoded = decode_state_packet(encoded)

    assert decoded["version"] == 1
    assert decoded["type"] == "state"
    assert decoded["seq"] == 7
    assert decoded["joints"]["left"]["left_joint1"] == 1.0
    assert decoded["grippers"]["right"]["force"] == 2.0


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
    assert robot.action_features == {}


def test_jz_robot_udp_get_observation_from_cached_state() -> None:
    robot = JZRobotUDP(make_config())
    robot._is_connected = True
    robot._state_cache.update(sample_state_packet(), sender=("192.168.1.81", 39010))

    obs = robot.get_observation()

    assert obs["left_left_joint1.pos"] == 1.0
    assert obs["right_right_joint7.pos"] == 17.0
    assert obs["left_gripper.width"] == 0.01
    assert obs["right_gripper.force"] == 2.0


def test_jz_robot_udp_rejects_send_action_without_network_side_effects() -> None:
    robot = JZRobotUDP(make_config())
    robot._is_connected = True

    with pytest.raises(NotImplementedError, match="readonly"):
        robot.send_action({"left_left_joint1.pos": 0.0})


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
