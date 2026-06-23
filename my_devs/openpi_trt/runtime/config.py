#!/usr/bin/env python

"""Runtime configuration objects for LeRobot PI0.5 split TensorRT backends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from runtime.protocol import PI05_PREFIX_CACHE_LAYERS

DEFAULT_PREFIX_ENGINE_PATH = Path("my_devs/openpi_trt/artifacts/pi05_so101_prefix_cache_b1_fp32.engine")
DEFAULT_DENOISE_ENGINE_PATH = Path("my_devs/openpi_trt/artifacts/pi05_so101_denoise_step_b1_fp32.engine")


@dataclass(frozen=True)
class PI05SplitTRTConfig:
    """Configuration for the SO101 PI0.5 prefix-cache + denoise TensorRT runtime.

    The defaults intentionally mirror the engine artifacts that are already used
    by the existing scripts and by `my_devs/vla_engineering`.
    """

    prefix_engine_path: Path
    denoise_engine_path: Path
    num_layers: int = PI05_PREFIX_CACHE_LAYERS
    batch_size: int = 1
    chunk_size: int = 50
    max_action_dim: int = 32
    camera_count: int = 2

    @classmethod
    def from_paths(
        cls,
        prefix_engine_path: str | Path,
        denoise_engine_path: str | Path,
        *,
        num_layers: int = PI05_PREFIX_CACHE_LAYERS,
        batch_size: int = 1,
        chunk_size: int = 50,
        max_action_dim: int = 32,
        camera_count: int = 2,
    ) -> "PI05SplitTRTConfig":
        return cls(
            prefix_engine_path=Path(prefix_engine_path).expanduser(),
            denoise_engine_path=Path(denoise_engine_path).expanduser(),
            num_layers=num_layers,
            batch_size=batch_size,
            chunk_size=chunk_size,
            max_action_dim=max_action_dim,
            camera_count=camera_count,
        )

    def validate_files(self) -> None:
        if not self.prefix_engine_path.is_file():
            raise FileNotFoundError(f"Prefix TensorRT engine not found: {self.prefix_engine_path}")
        if not self.denoise_engine_path.is_file():
            raise FileNotFoundError(f"Denoise TensorRT engine not found: {self.denoise_engine_path}")

    def to_dict(self) -> dict:
        return {
            "prefix_engine_path": str(self.prefix_engine_path),
            "denoise_engine_path": str(self.denoise_engine_path),
            "num_layers": self.num_layers,
            "batch_size": self.batch_size,
            "chunk_size": self.chunk_size,
            "max_action_dim": self.max_action_dim,
            "camera_count": self.camera_count,
        }
