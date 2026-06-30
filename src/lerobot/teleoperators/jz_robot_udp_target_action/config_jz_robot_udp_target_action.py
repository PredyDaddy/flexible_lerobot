#!/usr/bin/env python

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.robots.jz_robot_udp.config_jz_robot_udp import (
    DEFAULT_LEFT_JOINT_NAMES,
    DEFAULT_RIGHT_JOINT_NAMES,
)

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("jz_robot_udp_target_action")
@dataclass
class JZRobotUDPTargetActionTeleopConfig(TeleoperatorConfig):
    """UDP target-action source for recording commanded JZRobot actions."""

    bind_ip: str = "0.0.0.0"
    target_action_port: int = 39030
    receive_buffer_size: int = 65535
    allowed_sender_ip: str | None = "192.168.1.81"

    left_joint_names: list[str] = field(default_factory=lambda: DEFAULT_LEFT_JOINT_NAMES.copy())
    right_joint_names: list[str] = field(default_factory=lambda: DEFAULT_RIGHT_JOINT_NAMES.copy())
    use_gripper: bool = True

    connect_timeout_s: float = 5.0
    target_action_timeout_s: float = 0.5

    def __post_init__(self) -> None:
        if not self.left_joint_names:
            raise ValueError("left_joint_names must not be empty")
        if not self.right_joint_names:
            raise ValueError("right_joint_names must not be empty")
        if self.target_action_port < 0 or self.target_action_port > 65535:
            raise ValueError(f"target_action_port must be in 0..65535, got {self.target_action_port}")
        if self.receive_buffer_size <= 0:
            raise ValueError("receive_buffer_size must be positive")
        if self.connect_timeout_s < 0:
            raise ValueError("connect_timeout_s must be non-negative")
        if self.target_action_timeout_s < 0:
            raise ValueError("target_action_timeout_s must be non-negative")

