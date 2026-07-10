from __future__ import annotations

from dataclasses import MISSING
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
from pathlib import Path
from typing import Any
from typing import get_type_hints

import yaml


@dataclass(frozen=True)
class ClientConfig:
    infer_url: str = "http://127.0.0.1:18080"
    endpoint: str = "/infer"
    timeout_s: float = 60.0
    run_time_s: float = 0.0
    log_interval: int = 10
    task: str = "Put the eraser into the small box"


@dataclass(frozen=True)
class RobotConfig:
    robot_id: str = "hfy_follower"
    robot_port: str = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00"
    calib_dir: str = "/home/cqy/.cache/huggingface/lerobot/calibration/robots/so_follower"
    max_relative_target: float | None = 10.0
    top_cam: str = "/dev/video4"
    wrist_cam: str = "/dev/video6"
    top_cam_fourcc: str = "YUYV"
    wrist_cam_fourcc: str = "MJPG"
    img_width: int = 640
    img_height: int = 480
    fps: int = 30
    motor_io_retries: int = 10


@dataclass(frozen=True)
class ExecutorConfig:
    execute_actions: bool = False
    action_chunk_steps: int = 30
    control_fps: float = 30.0
    policy_fps: float = 30.0
    interpolate_actions: bool = False
    boundary_blend_steps: int = 0
    max_action_delta_per_step: float = 0.0
    action_ema_alpha: float = 1.0
    dry_run: bool = False
    async_prefetch: bool = True
    prefetch_after_steps: int = 25


@dataclass(frozen=True)
class Config:
    client: ClientConfig = field(default_factory=ClientConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    executor: ExecutorConfig = field(default_factory=ExecutorConfig)


def _dict_to_dataclass(cls: type, data: dict[str, Any], prefix: str = "") -> Any:
    known_names = {f.name for f in fields(cls)}
    extra = [k for k in data if k not in known_names]
    if extra:
        raise KeyError(f"Unknown config keys under {prefix or 'root'}: {extra}")

    type_hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    missing: list[str] = []
    for f in fields(cls):
        if f.name not in data:
            if f.default is not MISSING:
                kwargs[f.name] = f.default
                continue
            if f.default_factory is not MISSING:
                kwargs[f.name] = f.default_factory()
                continue
            missing.append(f.name)
            continue
        value = data[f.name]
        field_type = type_hints.get(f.name, f.type)
        if hasattr(field_type, "__dataclass_fields__") and isinstance(value, dict):
            kwargs[f.name] = _dict_to_dataclass(field_type, value, prefix=f"{prefix}{f.name}.")
        else:
            kwargs[f.name] = value
    if missing:
        raise KeyError(f"Missing config keys under {prefix or 'root'}: {missing}")
    return cls(**kwargs)


def load_config(config_path: str | Path) -> Config:
    with Path(config_path).expanduser().open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    return _dict_to_dataclass(Config, data)
