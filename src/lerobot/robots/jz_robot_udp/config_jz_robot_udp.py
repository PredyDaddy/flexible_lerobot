#!/usr/bin/env python

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import RobotConfig
from .protocol import COMMAND_MODES

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
    transport: str = "tcp"

    def __post_init__(self) -> None:
        if not self.url.startswith("rtsp://"):
            raise ValueError(f"RTSP camera url must start with rtsp://, got {self.url}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("RTSP camera width and height must be positive")
        if self.color_mode not in ("rgb", "bgr"):
            raise ValueError("RTSP camera color_mode must be 'rgb' or 'bgr'")
        if self.transport not in ("tcp", "udp"):
            raise ValueError("RTSP camera transport must be 'tcp' or 'udp'")


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

    command_target_ip: str = "192.168.1.81"
    command_target_port: int = 39020
    send_action_transport: str = "local"
    send_action_execution: str = "dry_run"
    command_robot: str = "robot1"
    command_timeout_s: float = 0.2

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
        if isinstance(self.command_target_port, bool) or not isinstance(self.command_target_port, int):
            raise ValueError(f"command_target_port must be an integer in 1..65535, got {self.command_target_port}")
        if self.command_target_port <= 0 or self.command_target_port > 65535:
            raise ValueError(f"command_target_port must be in 1..65535, got {self.command_target_port}")
        if self.send_action_transport not in ("local", "udp"):
            raise ValueError("send_action_transport must be 'local' or 'udp'")
        if self.send_action_execution not in COMMAND_MODES:
            raise ValueError(f"send_action_execution must be one of {COMMAND_MODES}")
        if not isinstance(self.command_robot, str) or not self.command_robot:
            raise ValueError("command_robot must be a non-empty string")
        if isinstance(self.command_timeout_s, bool) or not isinstance(self.command_timeout_s, int | float):
            raise ValueError("command_timeout_s must be positive")
        if self.command_timeout_s <= 0:
            raise ValueError("command_timeout_s must be positive")
