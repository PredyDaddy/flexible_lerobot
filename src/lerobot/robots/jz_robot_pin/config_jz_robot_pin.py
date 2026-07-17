#!/usr/bin/env python

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

from lerobot.robots.config import RobotConfig
from lerobot.robots.jz_robot_udp.config_jz_robot_udp import (
    DEFAULT_LEFT_JOINT_NAMES,
    DEFAULT_RIGHT_JOINT_NAMES,
    JZRobotUDPConfig,
    RTSPCameraConfig,
)
from lerobot.robots.jz_robot_udp.protocol import COMMAND_MODE_ARMED


@RobotConfig.register_subclass("jz_robot_pin")
@dataclass
class JZRobotPinConfig(JZRobotUDPConfig):
    """x86-side JZ pin robot client for the current edge UDP replay/control stack."""

    # Prefer the explicit pin name in new configs, while keeping inherited
    # allowed_sender_ip as a compatibility alias for JZRobotUDP-style args.
    allowed_sender_ip: str | None = "192.168.1.81"
    allowed_state_sender_ip: str | None = None

    send_action_transport: str = "local"
    send_action_execution: str = "dry_run"

    armed_env_var: str = "JZ_ROBOT_PIN_ARMED"
    require_armed_env: bool = True

    max_initial_joint_delta_rad: float = 0.02
    max_joint_step_rad: float = 0.02
    allow_armed_joint_delta_bypass: bool = False
    state_seq_reset_timeout_s: float = 1.0
    gripper_width_min: float | None = None
    gripper_width_max: float | None = None
    gripper_force_min: float | None = None
    gripper_force_max: float | None = None

    def __post_init__(self) -> None:
        if self.allowed_state_sender_ip is None:
            self.allowed_state_sender_ip = self.allowed_sender_ip
        else:
            # The pin-specific name is canonical when supplied. This also keeps
            # JZRobotUDP-style allowed_sender_ip callers working as an alias.
            self.allowed_sender_ip = self.allowed_state_sender_ip

        super().__post_init__()

        if not isinstance(self.armed_env_var, str) or not self.armed_env_var:
            raise ValueError("armed_env_var must be a non-empty string")
        self._validate_nonnegative_finite("max_initial_joint_delta_rad", self.max_initial_joint_delta_rad)
        self._validate_nonnegative_finite("max_joint_step_rad", self.max_joint_step_rad)
        if not isinstance(self.allow_armed_joint_delta_bypass, bool):
            raise ValueError("allow_armed_joint_delta_bypass must be a boolean")
        self._validate_nonnegative_finite("state_seq_reset_timeout_s", self.state_seq_reset_timeout_s)
        if self.state_seq_reset_timeout_s <= 0:
            raise ValueError("state_seq_reset_timeout_s must be positive")
        if self.send_action_execution == COMMAND_MODE_ARMED:
            if not self.require_armed_env:
                raise ValueError("armed JZRobotPin requires require_armed_env=true")
            if self.allow_armed_joint_delta_bypass:
                if self.max_initial_joint_delta_rad != 0 or self.max_joint_step_rad != 0:
                    raise ValueError("armed joint delta bypass requires both joint delta limits to equal 0")
            else:
                if self.max_initial_joint_delta_rad <= 0:
                    raise ValueError("armed JZRobotPin requires max_initial_joint_delta_rad > 0")
                if self.max_joint_step_rad <= 0:
                    raise ValueError("armed JZRobotPin requires max_joint_step_rad > 0")
        self._validate_optional_range("gripper_width", self.gripper_width_min, self.gripper_width_max)
        self._validate_optional_range("gripper_force", self.gripper_force_min, self.gripper_force_max)

    @staticmethod
    def _validate_nonnegative_finite(name: str, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
            raise ValueError(f"{name} must be a finite numeric value")
        if value < 0:
            raise ValueError(f"{name} must be non-negative")

    @staticmethod
    def _validate_optional_range(name: str, minimum: float | None, maximum: float | None) -> None:
        for suffix, value in {"min": minimum, "max": maximum}.items():
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value))
            ):
                raise ValueError(f"{name}_{suffix} must be finite numeric or None")
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError(f"{name}_min must be <= {name}_max")


__all__ = [
    "DEFAULT_LEFT_JOINT_NAMES",
    "DEFAULT_RIGHT_JOINT_NAMES",
    "JZRobotPinConfig",
    "RTSPCameraConfig",
]
