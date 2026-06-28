#!/usr/bin/env python

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.robots.jz_robot_udp.config_jz_robot_udp import (
    DEFAULT_LEFT_JOINT_NAMES,
    DEFAULT_RIGHT_JOINT_NAMES,
)

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("jz_robot_udp_hold")
@dataclass
class JZRobotUDPHoldTeleopConfig(TeleoperatorConfig):
    """Observation-aware hold action source for JZRobotUDP recording."""

    left_joint_names: list[str] = field(default_factory=lambda: DEFAULT_LEFT_JOINT_NAMES.copy())
    right_joint_names: list[str] = field(default_factory=lambda: DEFAULT_RIGHT_JOINT_NAMES.copy())
    use_gripper: bool = True

    def __post_init__(self) -> None:
        if not self.left_joint_names:
            raise ValueError("left_joint_names must not be empty")
        if not self.right_joint_names:
            raise ValueError("right_joint_names must not be empty")
