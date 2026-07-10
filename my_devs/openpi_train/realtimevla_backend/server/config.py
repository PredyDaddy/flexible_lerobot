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
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 18080
    endpoint: str = "/infer"
    access_log: bool = False


@dataclass(frozen=True)
class ModelConfig:
    adapter: str = "openpi_so101"
    checkpoint_dir: str = ""
    default_prompt: str = "Put the eraser into the small box"
    dataset_format: str = "v21"
    asset_id: str | None = "desk_cleanup_v1/eraser_cup_multi_task_v21_full"
    pytorch_device: str | None = "cuda"
    action_horizon: int = 50
    valid_action_num: int = 30
    state_dim: int = 6
    action_dim: int = 6


@dataclass(frozen=True)
class InferenceConfig:
    optimizer: str = "pass_through"
    include_raw_actions: bool = True
    warmup_on_start: bool = False
    warmup_image_width: int = 640
    warmup_image_height: int = 480


@dataclass(frozen=True)
class Config:
    server: ServerConfig = field(default_factory=ServerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


def _coerce_sequence(value: Any, target_type: Any) -> Any:
    origin = getattr(target_type, "__origin__", None)
    if origin not in {list, tuple}:
        return value
    if value is None:
        return value
    if origin is tuple:
        return tuple(value)
    return list(value)


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
            kwargs[f.name] = _coerce_sequence(value, field_type)

    if missing:
        raise KeyError(f"Missing config keys under {prefix or 'root'}: {missing}")
    return cls(**kwargs)


def load_config(config_path: str | Path) -> Config:
    with Path(config_path).expanduser().open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    return _dict_to_dataclass(Config, data)
