#!/usr/bin/env python

from __future__ import annotations

import argparse
import logging
import os
import pickle  # nosec
import socket
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


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
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.processor import PolicyAction, PolicyProcessorPipeline, make_default_processors
from lerobot.processor.converters import (
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.processor.device_processor import DeviceProcessorStep
from lerobot.robots.agilex import AgileXRobotConfig
from lerobot.robots.agilex.agilex_ros_bridge import ACTION_FEATURE_NAMES, CAMERA_KEYS, POSITION_FEATURE_NAMES
from lerobot.utils.constants import OBS_STR
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import auto_select_torch_device, init_logging

DEFAULT_SERVER_HOST = "10.1.26.37"
DEFAULT_SERVER_PORT = 5560
DEFAULT_SOCKET_TIMEOUT_S = 30.0
DEFAULT_RECONNECT_RETRIES = 5
DEFAULT_RECONNECT_RETRY_DELAY_S = 1.0
DEFAULT_MAX_MESSAGE_BYTES = 64 * 1024 * 1024
DEFAULT_LOG_DIR = REPO_ROOT / "logs" / "groot_agilex_remote_sync"

HEADER_STRUCT = struct.Struct("!Q")
OBSERVATION_STATE_KEY = "observation.state"
OBSERVATION_IMAGE_PREFIX = "observation.images."
LEFT_ARM = "left"
RIGHT_ARM = "right"
BOTH_ARMS = "both"
AUTO_ARM = "auto"


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


def normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"", "none", "null"}:
        return None
    return value


def resolve_policy_device(requested: str | None, config_device: str | None) -> str:
    requested = normalize_optional_text(requested)
    if config_device is None or str(config_device).strip().lower() in {"", "none", "null"}:
        config_device = auto_select_torch_device().type

    if requested is None:
        return str(config_device)
    if requested == "auto":
        return auto_select_torch_device().type
    return requested


def configure_logging(name: str, level: str = "INFO", log_dir: Path | None = None) -> logging.Logger:
    log_dir = DEFAULT_LOG_DIR if log_dir is None else log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{name}_{int(time.time())}.log"
    init_logging(log_file=log_file, display_pid=False, console_level=level.upper())
    logger = logging.getLogger(name)
    logger.setLevel(level.upper())
    return logger


def ensure_groot_checkpoint_assets(policy_path: Path, *, require_weights: bool = True) -> None:
    required_files = [
        policy_path / "config.json",
        policy_path / "policy_preprocessor.json",
        policy_path / "policy_postprocessor.json",
    ]
    if require_weights and not any(policy_path.glob("model*.safetensors")):
        required_files.append(policy_path / "model.safetensors")

    missing_files = [path for path in required_files if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(f"Missing checkpoint assets: {missing_files}")


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


def sync_processor_device(pipeline: PolicyProcessorPipeline[Any, Any], device: str) -> None:
    for step in pipeline.steps:
        if isinstance(step, DeviceProcessorStep):
            step.device = device
            step.__post_init__()


def build_agilex_robot_config(args: argparse.Namespace) -> AgileXRobotConfig:
    return AgileXRobotConfig(
        id=args.robot_id,
        control_mode=args.control_mode,
        state_left_topic=args.state_left_topic,
        state_right_topic=args.state_right_topic,
        command_left_topic=args.command_left_topic,
        command_right_topic=args.command_right_topic,
        front_camera_topic=args.front_camera_topic,
        left_camera_topic=args.left_camera_topic,
        right_camera_topic=args.right_camera_topic,
        image_height=args.image_height,
        image_width=args.image_width,
        observation_timeout_s=args.observation_timeout_s,
        queue_size=args.queue_size,
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


def build_agilex_policy_runtime_helpers(
    contract: "AgileXPolicyContract",
    *,
    image_height: int,
    image_width: int,
) -> tuple[Any, dict[str, Any]]:
    _, _, robot_observation_processor = make_default_processors()
    observation_features: dict[str, type | tuple[int, int, int]] = {
        key: float for key in contract.state_feature_names
    }
    observation_features.update(
        {
            key: (image_height, image_width, 3)
            for key in contract.camera_feature_keys
        }
    )
    dataset_features = aggregate_pipeline_dataset_features(
        pipeline=robot_observation_processor,
        initial_features=create_initial_features(observation=observation_features),
        use_videos=True,
    )
    return robot_observation_processor, dataset_features


def build_observation_frame(
    observation: dict[str, Any],
    dataset_features: dict[str, Any],
    robot_observation_processor: Any,
) -> dict[str, Any]:
    processed_observation = robot_observation_processor(observation)
    return build_dataset_frame(dataset_features, processed_observation, prefix=OBS_STR)


def select_agilex_policy_observation(
    observation: dict[str, Any],
    contract: "AgileXPolicyContract",
) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for key in (*contract.state_feature_names, *contract.camera_feature_keys):
        if key not in observation:
            raise KeyError(f"Missing required observation key for contract: {key}")
        selected[key] = observation[key]
    return selected


def build_agilex_policy_observation_frame(
    observation: dict[str, Any],
    contract: "AgileXPolicyContract",
    dataset_features: dict[str, Any],
    robot_observation_processor: Any,
) -> dict[str, Any]:
    selected_observation = select_agilex_policy_observation(observation, contract)
    return build_observation_frame(selected_observation, dataset_features, robot_observation_processor)


def validate_live_observation(observation: dict[str, Any], image_height: int, image_width: int) -> None:
    required_keys = [*POSITION_FEATURE_NAMES, *CAMERA_KEYS]
    missing_keys = [key for key in required_keys if key not in observation]
    if missing_keys:
        raise KeyError(f"Missing AgileX observation keys: {missing_keys}")

    expected_image_shape = (image_height, image_width, 3)
    for key in POSITION_FEATURE_NAMES:
        value = observation[key]
        if not np.isscalar(value):
            raise TypeError(f"Expected scalar joint value for {key}, got {type(value)}")

    for key in CAMERA_KEYS:
        value = observation[key]
        if not isinstance(value, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray image for {key}, got {type(value)}")
        if value.shape != expected_image_shape:
            raise ValueError(
                f"Unexpected live image shape for {key}: expected {expected_image_shape}, got {value.shape}"
            )


def summarize_observation(observation: dict[str, Any]) -> str:
    state_vector = np.asarray([float(observation[key]) for key in POSITION_FEATURE_NAMES], dtype=np.float32)
    image_summary = ", ".join(
        f"{key}={observation[key].shape}/{observation[key].dtype}" for key in CAMERA_KEYS if key in observation
    )
    return (
        f"state[min={state_vector.min():.4f}, max={state_vector.max():.4f}, mean={state_vector.mean():.4f}] | "
        f"images[{image_summary}]"
    )


def summarize_action_tensor(action: Any) -> str:
    if isinstance(action, torch.Tensor):
        action_tensor = action.detach().to(dtype=torch.float32, device="cpu").reshape(-1)
    else:
        action_tensor = torch.as_tensor(action, dtype=torch.float32).reshape(-1)
    return (
        f"action[min={action_tensor.min().item():.4f}, max={action_tensor.max().item():.4f}, "
        f"mean={action_tensor.mean().item():.4f}, std={action_tensor.std(unbiased=False).item():.4f}]"
    )


def summarize_policy_contract(contract: "AgileXPolicyContract") -> str:
    return (
        f"scope={contract.control_scope} | "
        f"state_dim={contract.state_dim} | "
        f"action_dim={contract.action_dim} | "
        f"state_keys={list(contract.state_feature_names)} | "
        f"camera_keys={list(contract.camera_feature_keys)} | "
        f"action_keys={list(contract.action_feature_names)}"
    )


def recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    bytes_remaining = size
    while bytes_remaining > 0:
        chunk = sock.recv(bytes_remaining)
        if not chunk:
            raise EOFError("Socket closed while receiving data")
        chunks.append(chunk)
        bytes_remaining -= len(chunk)
    return b"".join(chunks)


def send_message(sock: socket.socket, message: Any) -> int:
    data = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)  # nosec
    sock.sendall(HEADER_STRUCT.pack(len(data)))
    sock.sendall(data)
    return len(data)


def receive_message(sock: socket.socket, max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES) -> Any:
    header = recv_exact(sock, HEADER_STRUCT.size)
    payload_size = HEADER_STRUCT.unpack(header)[0]
    if payload_size > max_message_bytes:
        raise ValueError(
            f"Incoming payload size {payload_size} exceeds max_message_bytes={max_message_bytes}"
        )
    payload = recv_exact(sock, payload_size)
    return pickle.loads(payload)  # nosec


def open_client_socket(host: str, port: int, timeout_s: float) -> socket.socket:
    sock = socket.create_connection((host, port), timeout=timeout_s)
    sock.settimeout(timeout_s)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return sock


@dataclass(eq=True)
class RemoteGrootPolicyConfig:
    policy_path: str | None = None
    task: str | None = None
    robot_type: str | None = None
    backend: str | None = None
    trt_engine_path: str | None = None
    vit_dtype: str | None = None
    llm_dtype: str | None = None
    dit_dtype: str | None = None
    trt_action_head_only: bool | None = None
    policy_device: str | None = None
    control_arm: str | None = None
    reset_policy_state: bool = True


@dataclass(eq=True, frozen=True)
class AgileXPolicyContract:
    control_scope: str
    state_feature_names: tuple[str, ...]
    camera_feature_keys: tuple[str, ...]
    action_feature_names: tuple[str, ...]
    state_dim: int
    action_dim: int


@dataclass
class ObservationPacket:
    step: int
    timestamp: float
    observation: dict[str, Any]


@dataclass
class ActionPacket:
    step: int
    timestamp: float
    action: Any
    server_received_at: float
    server_sent_at: float
    predict_latency_ms: float
    total_latency_ms: float


@dataclass
class RemoteMessage:
    kind: str
    payload: Any = None


@dataclass
class RemoteError:
    message: str
    traceback: str | None = None


@dataclass
class RemoteReady:
    policy_path: str
    task: str
    robot_type: str
    backend: str
    policy_device: str
    policy_contract: AgileXPolicyContract


@dataclass
class LoopMetrics:
    total_steps: int = 0
    started_at: float = 0.0

    def start(self) -> None:
        if self.started_at == 0.0:
            self.started_at = time.perf_counter()

    def tick(self) -> float:
        self.start()
        self.total_steps += 1
        elapsed = max(time.perf_counter() - self.started_at, 1e-6)
        return self.total_steps / elapsed


def resolve_remote_policy_config(
    incoming: RemoteGrootPolicyConfig,
    *,
    default_policy_path: str | None,
    default_task: str,
    default_robot_type: str,
    default_backend: str,
    default_trt_engine_path: str | None,
    default_vit_dtype: str,
    default_llm_dtype: str,
    default_dit_dtype: str,
    default_trt_action_head_only: bool,
    default_policy_device: str | None,
    default_control_arm: str | None,
) -> RemoteGrootPolicyConfig:
    policy_path = normalize_optional_text(incoming.policy_path) or normalize_optional_text(default_policy_path)
    if policy_path is None:
        raise ValueError("policy_path is required either on the server or in the client handshake")

    return RemoteGrootPolicyConfig(
        policy_path=policy_path,
        task=normalize_optional_text(incoming.task) or default_task,
        robot_type=normalize_optional_text(incoming.robot_type) or default_robot_type,
        backend=normalize_optional_text(incoming.backend) or default_backend,
        trt_engine_path=normalize_optional_text(incoming.trt_engine_path)
        or normalize_optional_text(default_trt_engine_path),
        vit_dtype=normalize_optional_text(incoming.vit_dtype) or default_vit_dtype,
        llm_dtype=normalize_optional_text(incoming.llm_dtype) or default_llm_dtype,
        dit_dtype=normalize_optional_text(incoming.dit_dtype) or default_dit_dtype,
        trt_action_head_only=(
            default_trt_action_head_only if incoming.trt_action_head_only is None else incoming.trt_action_head_only
        ),
        policy_device=normalize_optional_text(incoming.policy_device)
        or normalize_optional_text(default_policy_device),
        control_arm=normalize_control_arm(incoming.control_arm) or normalize_control_arm(default_control_arm),
        reset_policy_state=incoming.reset_policy_state,
    )


def model_cache_key(config: RemoteGrootPolicyConfig) -> tuple[Any, ...]:
    return (
        config.policy_path,
        config.backend,
        config.trt_engine_path,
        config.vit_dtype,
        config.llm_dtype,
        config.dit_dtype,
        config.trt_action_head_only,
        config.policy_device,
    )


def register_plugins_once() -> None:
    register_third_party_plugins()


def normalize_control_arm(control_arm: str | None) -> str | None:
    normalized = normalize_optional_text(control_arm)
    if normalized is None:
        return None
    lowered = normalized.lower()
    if lowered not in {AUTO_ARM, LEFT_ARM, RIGHT_ARM, BOTH_ARMS}:
        raise ValueError(
            f"Unsupported control_arm={control_arm!r}. Expected one of: {AUTO_ARM}, {LEFT_ARM}, {RIGHT_ARM}, {BOTH_ARMS}"
        )
    return lowered


def _feature_shape(feature: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in getattr(feature, "shape", ()))


def _single_arm_feature_names(control_arm: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if control_arm == LEFT_ARM:
        return (
            tuple(f"{LEFT_ARM}_joint{i}.pos" for i in range(7)),
            tuple((f"camera_front", f"camera_left")),
        )
    if control_arm == RIGHT_ARM:
        return (
            tuple(f"{RIGHT_ARM}_joint{i}.pos" for i in range(7)),
            tuple((f"camera_front", f"camera_right")),
        )
    raise ValueError(f"Expected a single arm, got {control_arm!r}")


def resolve_agilex_policy_contract(
    policy_cfg: PreTrainedConfig,
    requested_control_arm: str | None = None,
) -> AgileXPolicyContract:
    requested_control_arm = normalize_control_arm(requested_control_arm)
    state_feature = policy_cfg.input_features.get(OBSERVATION_STATE_KEY)
    action_feature = policy_cfg.output_features.get("action")
    if state_feature is None:
        raise ValueError(f"Checkpoint is missing required input feature {OBSERVATION_STATE_KEY!r}")
    if action_feature is None:
        raise ValueError("Checkpoint is missing required output feature 'action'")

    state_shape = _feature_shape(state_feature)
    action_shape = _feature_shape(action_feature)
    if len(state_shape) != 1 or len(action_shape) != 1:
        raise ValueError(
            "AgileX remote runtime expects flat state/action features, "
            f"got state_shape={state_shape}, action_shape={action_shape}"
        )

    state_dim = state_shape[0]
    action_dim = action_shape[0]
    camera_feature_keys = tuple(
        key.removeprefix(OBSERVATION_IMAGE_PREFIX)
        for key in sorted(policy_cfg.input_features)
        if key.startswith(OBSERVATION_IMAGE_PREFIX)
    )

    if state_dim == len(POSITION_FEATURE_NAMES) and action_dim == len(ACTION_FEATURE_NAMES):
        if requested_control_arm not in {None, AUTO_ARM, BOTH_ARMS}:
            raise ValueError(
                "Dual-arm AgileX checkpoint requires control_arm to be unset, 'auto', or 'both', "
                f"got {requested_control_arm!r}"
            )
        return AgileXPolicyContract(
            control_scope=BOTH_ARMS,
            state_feature_names=tuple(POSITION_FEATURE_NAMES),
            camera_feature_keys=camera_feature_keys,
            action_feature_names=tuple(ACTION_FEATURE_NAMES),
            state_dim=state_dim,
            action_dim=action_dim,
        )

    if state_dim != 7 or action_dim != 7:
        raise ValueError(
            "Unsupported AgileX checkpoint contract. Expected 7/7 single-arm or 14/14 dual-arm, "
            f"got state_dim={state_dim}, action_dim={action_dim}, cameras={camera_feature_keys}"
        )

    inferred_control_arm = requested_control_arm
    if inferred_control_arm in {None, AUTO_ARM}:
        if "camera_right" in camera_feature_keys and "camera_left" not in camera_feature_keys:
            inferred_control_arm = RIGHT_ARM
        elif "camera_left" in camera_feature_keys and "camera_right" not in camera_feature_keys:
            inferred_control_arm = LEFT_ARM
        else:
            raise ValueError(
                "Cannot infer single-arm AgileX checkpoint ownership from camera keys. "
                f"Please set control_arm explicitly. cameras={camera_feature_keys}"
            )

    if inferred_control_arm not in {LEFT_ARM, RIGHT_ARM}:
        raise ValueError(f"Single-arm AgileX checkpoint requires control_arm to be left/right, got {inferred_control_arm!r}")

    state_feature_names, default_camera_feature_keys = _single_arm_feature_names(inferred_control_arm)
    resolved_camera_feature_keys = camera_feature_keys if camera_feature_keys else default_camera_feature_keys
    return AgileXPolicyContract(
        control_scope=inferred_control_arm,
        state_feature_names=state_feature_names,
        camera_feature_keys=resolved_camera_feature_keys,
        action_feature_names=state_feature_names,
        state_dim=state_dim,
        action_dim=action_dim,
    )


def decode_agilex_policy_action(
    action: Any,
    contract: AgileXPolicyContract,
) -> dict[str, float]:
    action_tensor = torch.as_tensor(action, dtype=torch.float32).reshape(-1).to("cpu")
    if action_tensor.numel() != contract.action_dim:
        raise ValueError(
            f"Unexpected action size {action_tensor.numel()} for contract action_dim={contract.action_dim}"
        )
    return {
        name: float(action_tensor[index])
        for index, name in enumerate(contract.action_feature_names)
    }


def build_agilex_hold_action(observation: dict[str, Any]) -> dict[str, float]:
    return {name: float(observation[name]) for name in ACTION_FEATURE_NAMES}


def merge_agilex_action(
    hold_action: dict[str, float],
    predicted_action: dict[str, float],
) -> dict[str, float]:
    merged = dict(hold_action)
    merged.update(predicted_action)
    return merged
