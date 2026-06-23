#!/usr/bin/env python

"""Artifact metadata helpers for PI0.5 split TensorRT deployments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PI05SplitTRTArtifactMetadata:
    """Human- and script-readable description of a split TensorRT artifact set."""

    policy_path: str
    prefix_engine_path: str
    denoise_engine_path: str
    precision: str = "fp32"
    model_dtype: str = "float32"
    num_layers: int = 18
    batch_size: int = 1
    chunk_size: int = 50
    max_action_dim: int = 32
    tokenizer_max_length: int = 200
    image_resolution: list[int] = field(default_factory=lambda: [224, 224])
    cameras: list[str] = field(default_factory=lambda: ["top", "wrist"])
    validation: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: str | Path) -> "PI05SplitTRTArtifactMetadata":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def find_metadata_path(artifact_dir: str | Path) -> Path:
    path = Path(artifact_dir).expanduser()
    if path.is_file():
        return path
    return path / "metadata.json"
