from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def resolve_repo_root(script_path: Path) -> Path:
    resolved_path = script_path.resolve()
    for candidate in resolved_path.parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src/lerobot").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate repository root from script path: {script_path}")


REPO_ROOT = resolve_repo_root(Path(__file__))
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())


from lerobot import policies  # noqa: F401  # Register policy config classes.
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.policies.factory import get_policy_class
from lerobot.processor import PolicyAction, PolicyProcessorPipeline, make_default_processors
from lerobot.processor.converters import (
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.utils.constants import OBS_STATE, OBS_STR
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging


DEFAULT_POLICY_PATH = (
    "/data/cqy_workspace/flexible_lerobot/outputs/train/"
    "groot_grasp_block_in_bin1_repro_20260302_223413/bs32_20260302_223447/"
    "checkpoints/last/pretrained_model"
)
DEFAULT_SERVER_ADDRESS = "127.0.0.1:8080"
DEFAULT_LOG_DIR = REPO_ROOT / "logs" / "groot_async_infer"
GROOT_MAX_ACTIONS_PER_CHUNK = 16

AGGREGATE_FUNCTIONS: dict[str, Callable[[Any, Any], Any]] = {
    "latest_only": lambda old, new: new,
    "weighted_average": lambda old, new: 0.3 * old + 0.7 * new,
    "average": lambda old, new: 0.5 * old + 0.5 * new,
    "conservative": lambda old, new: 0.7 * old + 0.3 * new,
}


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


def maybe_path(path_str: str | None) -> Path | None:
    return None if not path_str else Path(path_str).expanduser()


def clamp_actions_per_chunk(actions_per_chunk: int, logger: logging.Logger | None = None) -> int:
    if actions_per_chunk <= 0:
        raise ValueError(f"actions_per_chunk must be positive, got {actions_per_chunk}")

    if actions_per_chunk > GROOT_MAX_ACTIONS_PER_CHUNK:
        if logger is not None:
            logger.warning(
                "GR00T action horizon is capped at %s. Clamping actions_per_chunk from %s to %s.",
                GROOT_MAX_ACTIONS_PER_CHUNK,
                actions_per_chunk,
                GROOT_MAX_ACTIONS_PER_CHUNK,
            )
        return GROOT_MAX_ACTIONS_PER_CHUNK

    return actions_per_chunk


def get_aggregate_function(name: str) -> Callable[[Any, Any], Any]:
    if name not in AGGREGATE_FUNCTIONS:
        raise ValueError(f"Unknown aggregate function '{name}'. Available: {sorted(AGGREGATE_FUNCTIONS)}")
    return AGGREGATE_FUNCTIONS[name]


def configure_logging(name: str, level: str = "INFO") -> logging.Logger:
    DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = DEFAULT_LOG_DIR / f"{name}_{int(time.time())}.log"
    init_logging(log_file=log_file, display_pid=False, console_level=level.upper())
    logger = logging.getLogger(name)
    logger.setLevel(level.upper())
    return logger


@dataclass
class TimedObservation:
    timestamp: float
    timestep: int
    observation: dict[str, Any]
    must_go: bool = False

    def get_timestamp(self) -> float:
        return self.timestamp

    def get_timestep(self) -> int:
        return self.timestep

    def get_observation(self) -> dict[str, Any]:
        return self.observation


@dataclass
class TimedAction:
    timestamp: float
    timestep: int
    action: Any

    def get_timestamp(self) -> float:
        return self.timestamp

    def get_timestep(self) -> int:
        return self.timestep

    def get_action(self) -> Any:
        return self.action


@dataclass
class RemoteGrootPolicySetup:
    policy_path: str
    task: str
    robot_type: str
    actions_per_chunk: int
    backend: str = "pytorch"
    trt_engine_path: str | None = None
    vit_dtype: str = "fp16"
    llm_dtype: str = "fp16"
    dit_dtype: str = "fp16"
    trt_action_head_only: bool = False


@dataclass
class FPSTracker:
    target_fps: float
    first_timestamp: float | None = None
    total_obs_count: int = 0

    def calculate_fps_metrics(self, current_timestamp: float) -> dict[str, float]:
        self.total_obs_count += 1
        if self.first_timestamp is None:
            self.first_timestamp = current_timestamp

        total_duration = current_timestamp - self.first_timestamp
        avg_fps = (self.total_obs_count - 1) / total_duration if total_duration > 1e-6 else 0.0
        return {"avg_fps": avg_fps, "target_fps": self.target_fps}

    def reset(self) -> None:
        self.first_timestamp = None
        self.total_obs_count = 0


def build_robot_config(args: argparse.Namespace) -> SOFollowerRobotConfig:
    cameras = {
        "top": OpenCVCameraConfig(
            index_or_path=args.top_cam_index,
            width=args.img_width,
            height=args.img_height,
            fps=args.fps,
        ),
        "wrist": OpenCVCameraConfig(
            index_or_path=args.wrist_cam_index,
            width=args.img_width,
            height=args.img_height,
            fps=args.fps,
        ),
    }
    return SOFollowerRobotConfig(
        id=args.robot_id,
        calibration_dir=maybe_path(args.calib_dir),
        port=args.robot_port,
        cameras=cameras,
    )


def build_robot_runtime_helpers(robot: Any) -> tuple[Any, Any, dict[str, Any]]:
    _, robot_action_processor, robot_observation_processor = make_default_processors()
    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=robot_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True,
        ),
    )
    return robot_action_processor, robot_observation_processor, dataset_features


def build_observation_frame(
    observation: dict[str, Any],
    dataset_features: dict[str, Any],
    robot_observation_processor: Any,
) -> dict[str, Any]:
    processed_observation = robot_observation_processor(observation)
    return build_dataset_frame(dataset_features, processed_observation, prefix=OBS_STR)


def load_pre_post_processors(
    policy_path: Path,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=str(policy_path),
        config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=str(policy_path),
        config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return preprocessor, postprocessor


def load_groot_policy_bundle(
    setup: RemoteGrootPolicySetup,
) -> tuple[
    PreTrainedConfig,
    Any,
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    register_third_party_plugins()

    policy_path = Path(setup.policy_path).expanduser()
    if not policy_path.is_dir():
        raise FileNotFoundError(f"Policy path does not exist: {policy_path}")

    policy_cfg = PreTrainedConfig.from_pretrained(str(policy_path))
    policy_cfg.pretrained_path = policy_path
    if policy_cfg.type != "groot":
        raise ValueError(f"Expected a groot checkpoint, got policy type {policy_cfg.type!r}")

    policy_class = get_policy_class(policy_cfg.type)
    policy = policy_class.from_pretrained(str(policy_path), strict=False)
    policy.to(policy_cfg.device)

    if setup.backend == "tensorrt":
        if not setup.trt_engine_path:
            raise ValueError("--trt-engine-path is required when backend=tensorrt")
        from lerobot.policies.groot.trt_runtime.patch import setup_tensorrt_engines

        setup_tensorrt_engines(
            policy._groot_model,
            setup.trt_engine_path,
            vit_dtype=setup.vit_dtype,
            llm_dtype=setup.llm_dtype,
            dit_dtype=setup.dit_dtype,
            action_head_only=setup.trt_action_head_only,
        )

    preprocessor, postprocessor = load_pre_post_processors(policy_path)
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    return policy_cfg, policy, preprocessor, postprocessor


def observation_state_vector(observation: dict[str, Any]) -> Any:
    import torch

    state = observation[OBS_STATE]
    if isinstance(state, torch.Tensor):
        return state.detach().to(dtype=torch.float32).reshape(-1)
    return torch.as_tensor(state, dtype=torch.float32).reshape(-1)


def observations_similar(obs1: TimedObservation, obs2: TimedObservation, atol: float = 1.0) -> bool:
    import torch

    state1 = observation_state_vector(obs1.get_observation())
    state2 = observation_state_vector(obs2.get_observation())
    return bool(torch.linalg.norm(state1 - state2) < atol)
