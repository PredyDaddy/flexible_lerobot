#!/usr/bin/env python

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import RobotConfig

DEFAULT_LEFT_JOINT_NAMES = [
    "left_joint1",
    "left_joint2",
    "left_joint3",
    "left_joint4",
    "left_joint5",
    "left_joint6",
    "left_joint7",
]

DEFAULT_RIGHT_JOINT_NAMES = [
    "right_joint1",
    "right_joint2",
    "right_joint3",
    "right_joint4",
    "right_joint5",
    "right_joint6",
    "right_joint7",
]


@dataclass
class RTSPCameraConfig:
    url: str
    fps: int = 30
    width: int = 640
    height: int = 480
    timeout_ms: int = 5000
    warmup_frames: int = 1
    color_mode: str = "rgb"

    def __post_init__(self) -> None:
        if not self.url.startswith("rtsp://"):
            raise ValueError(f"RTSP camera url must start with rtsp://, got {self.url}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("RTSP camera width and height must be positive")
        if self.color_mode not in ("rgb", "bgr"):
            raise ValueError("RTSP camera color_mode must be 'rgb' or 'bgr'")


@RobotConfig.register_subclass("jz_robot_udp")
@dataclass
class JZRobotUDPConfig(RobotConfig):
    """Readonly UDP client configuration for the remote JZRobot interface."""

    bind_ip: str = "0.0.0.0"
    state_port: int = 39010
    receive_buffer_size: int = 65535
    allowed_sender_ip: str | None = "192.168.1.81"

    left_joint_names: list[str] = field(default_factory=lambda: DEFAULT_LEFT_JOINT_NAMES.copy())
    right_joint_names: list[str] = field(default_factory=lambda: DEFAULT_RIGHT_JOINT_NAMES.copy())
    use_gripper: bool = True

    connect_timeout_s: float = 5.0
    state_timeout_s: float = 0.5

    rtsp_cameras: dict[str, RTSPCameraConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.left_joint_names:
            raise ValueError("left_joint_names must not be empty")
        if not self.right_joint_names:
            raise ValueError("right_joint_names must not be empty")
        if self.state_port <= 0 or self.state_port > 65535:
            raise ValueError(f"state_port must be in 1..65535, got {self.state_port}")
        if self.receive_buffer_size <= 0:
            raise ValueError("receive_buffer_size must be positive")
