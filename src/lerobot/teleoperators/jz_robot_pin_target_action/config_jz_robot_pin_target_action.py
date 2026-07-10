#!/usr/bin/env python

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Real

from lerobot.robots.jz_robot_pin.config_jz_robot_pin import (
    DEFAULT_LEFT_JOINT_NAMES,
    DEFAULT_RIGHT_JOINT_NAMES,
)

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("jz_robot_pin_target_action")
@dataclass
class JZRobotPinTargetActionTeleopConfig(TeleoperatorConfig):
    """UDP target-action source for the pin VR/joystick control path."""

    bind_ip: str = "0.0.0.0"
    target_action_port: int = 39030
    receive_buffer_size: int = 65535
    allowed_sender_ip: str | None = "127.0.0.1"

    left_joint_names: list[str] = field(default_factory=lambda: DEFAULT_LEFT_JOINT_NAMES.copy())
    right_joint_names: list[str] = field(default_factory=lambda: DEFAULT_RIGHT_JOINT_NAMES.copy())
    use_gripper: bool = True

    connect_timeout_s: float = 0.0
    target_action_timeout_s: float = 0.5
    stale_policy: str = "hold_current"
    packet_max_age_s: float = 1.0
    packet_max_future_skew_s: float = 0.25
    seq_reset_timeout_s: float = 1.0

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
        if self.stale_policy not in ("raise", "hold_current"):
            raise ValueError("stale_policy must be 'raise' or 'hold_current'")
        self._validate_nonnegative_finite("packet_max_age_s", self.packet_max_age_s)
        self._validate_nonnegative_finite(
            "packet_max_future_skew_s", self.packet_max_future_skew_s
        )
        self._validate_nonnegative_finite("seq_reset_timeout_s", self.seq_reset_timeout_s)
        if self.seq_reset_timeout_s <= 0:
            raise ValueError("seq_reset_timeout_s must be positive")

    @staticmethod
    def _validate_nonnegative_finite(name: str, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
            raise ValueError(f"{name} must be a finite numeric value")
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
