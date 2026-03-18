from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import combine_feature_dicts
from lerobot.policies.factory import get_policy_class
from lerobot.processor import PolicyAction, PolicyProcessorPipeline, make_default_processors
from lerobot.processor.converters import (
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.utils.import_utils import register_third_party_plugins

TOKENIZER_SUBDIR = Path("google") / "paligemma-3b-pt-224"
SUPPORTED_ROBOT_TYPES = frozenset({"so100_follower", "so101_follower"})


@dataclass(slots=True)
class DatasetFeatureArtifacts:
    dataset_features: dict[str, dict[str, Any]]
    robot_action_processor: Any
    robot_observation_processor: Any


@dataclass(slots=True)
class PolicyLoadArtifacts:
    repo_root: Path
    policy_path: Path
    tokenizer_dir: Path
    policy_config: PreTrainedConfig
    policy_class: type[Any]
    policy: Any
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]]
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction]


def resolve_repo_root(script_path: str | Path) -> Path:
    resolved_path = Path(script_path).expanduser().resolve()
    search_roots = (resolved_path,) + tuple(resolved_path.parents)
    for candidate in search_roots:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src/lerobot").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate repository root from path: {script_path}")


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value

    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {value}")


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    return default if raw is None else parse_bool(raw)


def maybe_path(path_str: str | os.PathLike[str] | None) -> Path | None:
    if path_str is None:
        return None
    path_text = os.fspath(path_str)
    return None if path_text == "" else Path(path_text).expanduser()


def get_local_tokenizer_dir(repo_root: str | Path) -> Path:
    return Path(repo_root).expanduser().resolve() / TOKENIZER_SUBDIR


def ensure_local_tokenizer_dir(repo_root: str | Path) -> Path:
    local_tok = get_local_tokenizer_dir(repo_root)
    if not local_tok.is_dir():
        raise FileNotFoundError(
            "Missing local tokenizer directory for offline PI0.5 inference.\n"
            f"Expected: {local_tok}\n"
            "Fix:\n"
            "  - Download tokenizer files to that directory (or create a symlink), then retry.\n"
            "  - In our setup this is a symlink to ModelScope cache:\n"
            "      assets/modelscope/google/paligemma-3b-pt-224\n"
        )
    return local_tok


def resolve_policy_path(policy_path: str | Path) -> Path:
    resolved = Path(policy_path).expanduser()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Policy path does not exist: {resolved}")
    return resolved


def load_pre_post_processors(
    policy_path: str | Path,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    checkpoint_path = resolve_policy_path(policy_path)
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=str(checkpoint_path),
        config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=str(checkpoint_path),
        config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return preprocessor, postprocessor


def ensure_policy_registry() -> None:
    register_third_party_plugins()
    from lerobot import policies as _policies  # noqa: F401


def load_policy_config(policy_path: str | Path, *, device_override: str | None = None) -> PreTrainedConfig:
    checkpoint_path = resolve_policy_path(policy_path)
    ensure_policy_registry()
    policy_cfg = PreTrainedConfig.from_pretrained(str(checkpoint_path))
    policy_cfg.pretrained_path = checkpoint_path
    if device_override is not None:
        policy_cfg.device = device_override
    return policy_cfg


def build_so101_robot_config(
    *,
    robot_id: str,
    robot_type: str,
    calib_dir: str | os.PathLike[str] | None,
    robot_port: str,
    top_cam_index: int,
    wrist_cam_index: int,
    img_width: int,
    img_height: int,
    camera_fps: int,
) -> SOFollowerRobotConfig:
    if robot_type not in SUPPORTED_ROBOT_TYPES:
        raise ValueError(
            f"Unsupported robot_type={robot_type!r}. "
            "This runtime currently supports so100_follower/so101_follower."
        )

    cameras = {
        "top": OpenCVCameraConfig(
            index_or_path=top_cam_index,
            width=img_width,
            height=img_height,
            fps=camera_fps,
        ),
        "wrist": OpenCVCameraConfig(
            index_or_path=wrist_cam_index,
            width=img_width,
            height=img_height,
            fps=camera_fps,
        ),
    }
    return SOFollowerRobotConfig(
        id=robot_id,
        calibration_dir=maybe_path(calib_dir),
        port=robot_port,
        cameras=cameras,
    )


def build_dataset_features(
    *,
    action_features: dict[str, Any],
    observation_features: dict[str, Any],
) -> DatasetFeatureArtifacts:
    _, robot_action_processor, robot_observation_processor = make_default_processors()
    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=robot_action_processor,
            initial_features=create_initial_features(action=action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=observation_features),
            use_videos=True,
        ),
    )
    return DatasetFeatureArtifacts(
        dataset_features=dataset_features,
        robot_action_processor=robot_action_processor,
        robot_observation_processor=robot_observation_processor,
    )


def load_policy_and_processors(
    policy_path: str | Path,
    *,
    repo_root: str | Path | None = None,
    strict: bool = False,
    device_override: str | None = None,
) -> PolicyLoadArtifacts:
    resolved_repo_root = resolve_repo_root(Path(__file__)) if repo_root is None else Path(repo_root).expanduser().resolve()
    tokenizer_dir = ensure_local_tokenizer_dir(resolved_repo_root)
    checkpoint_path = resolve_policy_path(policy_path)
    policy_cfg = load_policy_config(checkpoint_path, device_override=device_override)
    policy_class = get_policy_class(policy_cfg.type)
    policy = policy_class.from_pretrained(str(checkpoint_path), strict=strict)
    policy.to(policy_cfg.device)
    preprocessor, postprocessor = load_pre_post_processors(checkpoint_path)
    return PolicyLoadArtifacts(
        repo_root=resolved_repo_root,
        policy_path=checkpoint_path,
        tokenizer_dir=tokenizer_dir,
        policy_config=policy_cfg,
        policy_class=policy_class,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
    )


def run_offline_load_smoke(
    policy_path: str | Path,
    *,
    repo_root: str | Path | None = None,
    strict: bool = False,
    device_override: str | None = None,
) -> PolicyLoadArtifacts:
    return load_policy_and_processors(
        policy_path,
        repo_root=repo_root,
        strict=strict,
        device_override=device_override,
    )
