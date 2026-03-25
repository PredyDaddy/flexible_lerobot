#!/usr/bin/env python

from __future__ import annotations

from dataclasses import dataclass

from ..configs import CameraConfig, ColorMode

__all__ = ["ROS2TopicCameraConfig", "ColorMode"]

_VALID_RELIABILITY = {"best_effort", "reliable"}
_VALID_DURABILITY = {"volatile", "transient_local"}


@CameraConfig.register_subclass("ros2_topic")
@dataclass
class ROS2TopicCameraConfig(CameraConfig):
    image_topic: str
    color_mode: ColorMode = ColorMode.RGB
    timeout_ms: int = 5000
    qos_depth: int = 1
    reliability: str = "reliable"
    durability: str = "volatile"
    warmup_s: float = 0.0

    def __post_init__(self) -> None:
        if not self.image_topic:
            raise ValueError("`image_topic` cannot be empty.")

        if self.color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"`color_mode` is expected to be {ColorMode.RGB.value} or {ColorMode.BGR.value}, but {self.color_mode} is provided."
            )

        if self.timeout_ms <= 0:
            raise ValueError(f"`timeout_ms` must be positive, but {self.timeout_ms} is provided.")

        if self.qos_depth <= 0:
            raise ValueError(f"`qos_depth` must be positive, but {self.qos_depth} is provided.")

        self.reliability = self.reliability.lower()
        if self.reliability not in _VALID_RELIABILITY:
            raise ValueError(
                f"`reliability` must be one of {_VALID_RELIABILITY}, but {self.reliability} is provided."
            )

        self.durability = self.durability.lower()
        if self.durability not in _VALID_DURABILITY:
            raise ValueError(
                f"`durability` must be one of {_VALID_DURABILITY}, but {self.durability} is provided."
            )

        if self.warmup_s < 0:
            raise ValueError(f"`warmup_s` must be >= 0, but {self.warmup_s} is provided.")
