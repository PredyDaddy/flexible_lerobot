#!/usr/bin/env python

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

from lerobot.robots.config import RobotConfig
from lerobot.robots.jz_robot_pin.config_jz_robot_pin import JZRobotPinConfig


@RobotConfig.register_subclass("jz_robot_pin_timed")
@dataclass
class JZRobotPinTimedConfig(JZRobotPinConfig):
    """Pin robot variant with timestamp-preserving PyAV RTSP receivers."""

    camera_buffer_size: int = 8
    camera_reconnect_delay_ms: int = 250
    max_camera_state_receive_skew_ms: float = 100.0
    enforce_camera_state_receive_skew: bool = True
    reject_reused_camera_frames: bool = False
    timing_log_every_n: int = 30
    timing_sidecar: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        for field_name in (
            "enforce_camera_state_receive_skew",
            "reject_reused_camera_frames",
            "timing_sidecar",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"{field_name} must be a boolean")
        if isinstance(self.camera_buffer_size, bool) or not isinstance(self.camera_buffer_size, int):
            raise ValueError("camera_buffer_size must be an integer")
        if self.camera_buffer_size <= 0:
            raise ValueError("camera_buffer_size must be positive")
        if isinstance(self.camera_reconnect_delay_ms, bool) or not isinstance(
            self.camera_reconnect_delay_ms, int
        ):
            raise ValueError("camera_reconnect_delay_ms must be an integer")
        if self.camera_reconnect_delay_ms < 0:
            raise ValueError("camera_reconnect_delay_ms must be non-negative")
        if isinstance(self.timing_log_every_n, bool) or not isinstance(self.timing_log_every_n, int):
            raise ValueError("timing_log_every_n must be an integer")
        if self.timing_log_every_n < 0:
            raise ValueError("timing_log_every_n must be non-negative")
        if (
            isinstance(self.max_camera_state_receive_skew_ms, bool)
            or not isinstance(self.max_camera_state_receive_skew_ms, Real)
            or not math.isfinite(float(self.max_camera_state_receive_skew_ms))
            or self.max_camera_state_receive_skew_ms <= 0
        ):
            raise ValueError("max_camera_state_receive_skew_ms must be a positive finite number")


__all__ = ["JZRobotPinTimedConfig"]
