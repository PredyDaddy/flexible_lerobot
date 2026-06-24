from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RobotEndpointConfig:
    name: str
    namespace: str
    orin_ip: str = ""
    collector_ip: str = ""
    ros_domain_id: int | None = None


@dataclass(frozen=True)
class RecordingConfig:
    output_root: str
    sample_rate_hz: int = 20
    image_format: str = "jpg"
    image_quality: int = 95
    max_camera_delta_ms: float = 25.0
    max_state_delta_ms: float = 10.0
    max_action_delta_ms: float = 10.0


@dataclass(frozen=True)
class CameraConfig:
    rtsp_url: str
    width: int
    height: int
    fps: int
    transport: str = "tcp"


@dataclass(frozen=True)
class ActionSourceConfig:
    id: str
    topic: str
    type: str
    names: list[str]
    include_names: list[str] | None = None

    @property
    def exported_names(self) -> list[str]:
        return list(self.include_names if self.include_names is not None else self.names)


@dataclass(frozen=True)
class VectorTopicConfig:
    topic: str
    names: list[str]
    sources: list[ActionSourceConfig] = field(default_factory=list)


@dataclass(frozen=True)
class BridgeCaptureConfig:
    robot: RobotEndpointConfig
    recording: RecordingConfig
    cameras: dict[str, CameraConfig]
    state: VectorTopicConfig
    action: VectorTopicConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BridgeCaptureConfig":
        with Path(path).open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream) or {}
        config = cls.from_dict(raw)
        config.validate()
        return config

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "BridgeCaptureConfig":
        robot = RobotEndpointConfig(**raw["robot"])
        recording = RecordingConfig(**raw["recording"])
        cameras = {name: CameraConfig(**value) for name, value in (raw.get("cameras") or {}).items()}
        state = _vector_topic_from_dict(raw["state"])
        action = _vector_topic_from_dict(raw["action"])
        return cls(robot=robot, recording=recording, cameras=cameras, state=state, action=action)

    def validate(self) -> None:
        if self.recording.sample_rate_hz <= 0:
            raise ValueError("recording.sample_rate_hz must be positive")
        _validate_vector_topic(self.state, "state", require_sources=False)
        _validate_vector_topic(self.action, "action", require_sources=True)


def _vector_topic_from_dict(raw: dict[str, Any]) -> VectorTopicConfig:
    sources = [ActionSourceConfig(**source) for source in raw.get("sources", [])]
    return VectorTopicConfig(topic=raw["topic"], names=list(raw["names"]), sources=sources)


def _validate_vector_topic(config: VectorTopicConfig, label: str, *, require_sources: bool) -> None:
    if not config.topic:
        raise ValueError(f"{label}.topic must not be empty")
    if len(config.names) == 0:
        raise ValueError(f"{label}.names must not be empty")
    if len(set(config.names)) != len(config.names):
        raise ValueError(f"{label}.names contains duplicates")
    if require_sources and len(config.sources) == 0:
        raise ValueError(f"{label}.sources must not be empty")

    supplied: set[str] = set()
    source_ids: set[str] = set()
    for source in config.sources:
        if source.id in source_ids:
            raise ValueError(f"{label}.sources contains duplicate id: {source.id}")
        source_ids.add(source.id)
        missing_include = set(source.exported_names) - set(source.names)
        if missing_include:
            raise ValueError(f"{label}.source {source.id} include_names not present in names: {sorted(missing_include)}")
        supplied.update(source.exported_names)

    missing = set(config.names) - supplied
    if (require_sources or config.sources) and missing:
        raise ValueError(f"{label}.names are not supplied by sources: {sorted(missing)}")
