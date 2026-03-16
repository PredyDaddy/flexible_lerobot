from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.robots.config import RobotConfig

from .common import ARM_CHOICES, DEFAULT_ARM

LEGACY_CAMERA_KEY_ALIASES = {
    "cam_high": "camera_front",
    "cam_left_wrist": "camera_left",
    "cam_right_wrist": "camera_right",
}


def _normalize_camera_key(key: str) -> str:
    return LEGACY_CAMERA_KEY_ALIASES.get(key, key)


@RobotConfig.register_subclass("single_arm_agilex")
@dataclass(kw_only=True)
class SingleArmAgileXRobotConfig(RobotConfig):
    arm: str = DEFAULT_ARM
    control_mode: str = "passive_follow"
    state_left_topic: str = "/puppet/joint_left"
    state_right_topic: str = "/puppet/joint_right"
    command_left_topic: str = "/master/joint_left"
    command_right_topic: str = "/master/joint_right"
    front_camera_topic: str = "/camera_f/color/image_raw"
    left_camera_topic: str = "/camera_l/color/image_raw"
    right_camera_topic: str = "/camera_r/color/image_raw"
    front_camera_key: str = "camera_front"
    left_camera_key: str = "camera_left"
    right_camera_key: str = "camera_right"
    image_height: int = 480
    image_width: int = 640
    observation_timeout_s: float = 2.0
    queue_size: int = 1
    action_smoothing_alpha: float | None = None
    max_joint_step_rad: float | None = None
    joint_names: list[str] = field(default_factory=lambda: [f"joint{i}" for i in range(7)])

    def __post_init__(self):
        super().__post_init__()
        if self.arm not in ARM_CHOICES:
            raise ValueError(f"Unsupported arm: {self.arm}")
        if self.control_mode not in {"passive_follow", "command_master"}:
            raise ValueError(f"Unsupported control_mode: {self.control_mode}")
        if len(self.joint_names) != 7:
            raise ValueError("SingleArmAgileXRobot expects exactly 7 joint names per arm")
        if self.action_smoothing_alpha is not None and not 0.0 < self.action_smoothing_alpha <= 1.0:
            raise ValueError(
                "action_smoothing_alpha must be within (0, 1], "
                f"got {self.action_smoothing_alpha}"
            )
        if self.max_joint_step_rad is not None and self.max_joint_step_rad <= 0.0:
            raise ValueError(f"max_joint_step_rad must be positive, got {self.max_joint_step_rad}")

        self.front_camera_key = _normalize_camera_key(self.front_camera_key)
        self.left_camera_key = _normalize_camera_key(self.left_camera_key)
        self.right_camera_key = _normalize_camera_key(self.right_camera_key)

        camera_keys = (self.front_camera_key, self.left_camera_key, self.right_camera_key)
        if len(set(camera_keys)) != len(camera_keys):
            raise ValueError("Camera keys must be distinct")
        if self.front_camera_key != "camera_front":
            raise ValueError(f"front_camera_key must resolve to 'camera_front', got {self.front_camera_key}")
        if self.left_camera_key != "camera_left":
            raise ValueError(f"left_camera_key must resolve to 'camera_left', got {self.left_camera_key}")
        if self.right_camera_key != "camera_right":
            raise ValueError(f"right_camera_key must resolve to 'camera_right', got {self.right_camera_key}")
