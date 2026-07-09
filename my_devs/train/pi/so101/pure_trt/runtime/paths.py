from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


def resolve_repo_root(start: Path) -> Path:
    resolved = start.expanduser().resolve()
    for candidate in (resolved, *resolved.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "lerobot").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate repository root from {start}")


REPO_ROOT = resolve_repo_root(Path(__file__))
PURE_TRT_ROOT = REPO_ROOT / "my_devs" / "train" / "pi" / "so101" / "pure_trt"
LEGACY_OPENPI_TRT_ROOT = REPO_ROOT / "my_devs" / "openpi_trt"
LEGACY_ARTIFACT_DIR = LEGACY_OPENPI_TRT_ROOT / "artifacts"


@dataclass(frozen=True, slots=True)
class PI05PureTRTPaths:
    root: Path = PURE_TRT_ROOT
    artifact_dir: Path = PURE_TRT_ROOT / "artifacts"
    runtime_assets_dir: Path = PURE_TRT_ROOT / "artifacts" / "pi05_runtime_assets"
    prefix_onnx: Path = PURE_TRT_ROOT / "artifacts" / "pi05_so101_prefix_cache_b1_fp32.onnx"
    denoise_onnx: Path = PURE_TRT_ROOT / "artifacts" / "pi05_so101_denoise_step_b1_fp32.onnx"
    prefix_fp32_engine: Path = PURE_TRT_ROOT / "artifacts" / "pi05_so101_prefix_cache_b1_fp32.engine"
    prefix_fp16_constrained_engine: Path = (
        PURE_TRT_ROOT / "artifacts" / "pi05_so101_prefix_cache_b1_fp16_constrained.engine"
    )
    denoise_fp32_engine: Path = PURE_TRT_ROOT / "artifacts" / "pi05_so101_denoise_step_b1_fp32.engine"
    denoise_fp16_constrained_engine: Path = (
        PURE_TRT_ROOT / "artifacts" / "pi05_so101_denoise_step_b1_fp16_constrained.engine"
    )

    def prefix_engine(self, profile: str) -> Path:
        if profile == "fp32":
            return self.prefix_fp32_engine
        if profile == "fp16_constrained":
            return self.prefix_fp16_constrained_engine
        raise ValueError(f"Unsupported prefix profile: {profile}")

    def denoise_engine(self, profile: str) -> Path:
        if profile == "fp32":
            return self.denoise_fp32_engine
        if profile == "fp16_constrained":
            return self.denoise_fp16_constrained_engine
        raise ValueError(f"Unsupported denoise profile: {profile}")

    def to_dict(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}


def default_paths() -> PI05PureTRTPaths:
    return PI05PureTRTPaths()


def legacy_artifact(name: str) -> Path:
    return LEGACY_ARTIFACT_DIR / name
