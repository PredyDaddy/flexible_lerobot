#!/usr/bin/env python

from __future__ import annotations

import pytest

from lerobot.robots.jz_robot_udp.protocol import make_jz_robot_udp_target_action_packet
from lerobot.robots.jz_robot_udp.state_cache import StateCache
from lerobot.teleoperators.jz_robot_udp_target_action import (
    JZRobotUDPTargetActionTeleop,
    JZRobotUDPTargetActionTeleopConfig,
)
from lerobot.teleoperators.utils import make_teleoperator_from_config


def sample_actions() -> dict:
    return {
        "left": {f"left_joint{i}": float(i) for i in range(1, 8)},
        "right": {f"right_joint{i}": float(i + 10) for i in range(1, 8)},
        "grippers": {
            "left": {"width": 0.01, "force": 1.0},
            "right": {"width": 0.02, "force": 2.0},
        },
    }


def make_teleop(**overrides) -> JZRobotUDPTargetActionTeleop:
    cfg = JZRobotUDPTargetActionTeleopConfig(
        id="test_jz_robot_udp_target_action",
        bind_ip="127.0.0.1",
        target_action_port=0,
        connect_timeout_s=0.0,
        **overrides,
    )
    return JZRobotUDPTargetActionTeleop(cfg)


def test_target_action_teleop_maps_latest_packet_to_lerobot_action() -> None:
    teleop = make_teleop()
    cache = StateCache()
    packet = make_jz_robot_udp_target_action_packet(
        robot="robot1",
        seq=1,
        stamp_ns=123,
        actions=sample_actions(),
    )
    cache.update(packet, sender=("192.168.1.81", 39030))
    teleop._target_action_cache = cache
    teleop._is_connected = True

    action = teleop.get_action()

    assert action["left_left_joint1.pos"] == 1.0
    assert action["left_left_joint7.pos"] == 7.0
    assert action["right_right_joint1.pos"] == 11.0
    assert action["right_right_joint7.pos"] == 17.0
    assert action["left_gripper.width"] == 0.01
    assert action["right_gripper.force"] == 2.0


def test_target_action_teleop_factory_registration() -> None:
    cfg = JZRobotUDPTargetActionTeleopConfig(
        id="factory_jz_robot_udp_target_action",
        bind_ip="127.0.0.1",
        target_action_port=0,
        connect_timeout_s=0.0,
    )

    teleop = make_teleoperator_from_config(cfg)

    assert isinstance(teleop, JZRobotUDPTargetActionTeleop)
    assert teleop.action_features["left_left_joint1.pos"] is float
    assert teleop.action_features["right_right_joint7.pos"] is float


def test_target_action_teleop_rejects_stale_or_missing_packets() -> None:
    teleop = make_teleop(target_action_timeout_s=0.0)
    teleop._is_connected = True

    with pytest.raises(TimeoutError, match="No JZRobot UDP target action packet"):
        teleop.get_action()

    teleop._target_action_cache.update(
        make_jz_robot_udp_target_action_packet(
            robot="robot1",
            seq=1,
            stamp_ns=123,
            actions=sample_actions(),
        ),
        sender=("192.168.1.81", 39030),
    )

    with pytest.raises(TimeoutError, match="stale"):
        teleop.get_action()
