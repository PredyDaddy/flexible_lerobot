from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from lerobot import policies  # noqa: F401
from lerobot.configs.policies import PreTrainedConfig
from lerobot.utils.constants import HF_LEROBOT_HOME

EXPECTED_STATE_KEY = "observation.state"
EXPECTED_FRONT_CAMERA_KEY = "observation.images.camera_front"
EXPECTED_RIGHT_CAMERA_KEY = "observation.images.camera_right"
EXPECTED_ACTION_KEY = "action"

RIGHT_PREFIX = "right"
ARM_CHOICES = (RIGHT_PREFIX,)
DEFAULT_ARM = RIGHT_PREFIX
DEFAULT_TASK = "Execute the trained Agilex right-arm GROOT task"

MVP_ACTIONS_PER_CHUNK = 1
MVP_CHUNK_SIZE_THRESHOLD = 1.0
MVP_AGGREGATE_FN_NAME = "latest_only"
ONE_STEP_TIMESTEP_STRATEGY = "latest_action + 1"

REQUIRED_TOKENIZER_ASSET_FILES = (
    "processor_config.json",
    "preprocessor_config.json",
    "image_processing_eagle2_5_vl_fast.py",
)


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


def feature_shape(feature: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in getattr(feature, "shape", ()))


def _is_local_path_reference(raw: str) -> bool:
    return raw.startswith(("/", ".", "~"))


def ensure_checkpoint_assets(policy_path: Path) -> list[Path]:
    required_files = [
        policy_path / "config.json",
        policy_path / "policy_preprocessor.json",
        policy_path / "policy_postprocessor.json",
    ]
    missing_files = [path for path in required_files if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(f"Missing checkpoint assets: {missing_files}")

    weight_files = sorted(path for path in policy_path.glob("model*.safetensors") if path.is_file())
    if not weight_files:
        raise FileNotFoundError(
            "Missing GROOT model weights. Expected model.safetensors or model*.safetensors shards "
            f"under {policy_path}."
        )
    return weight_files


def load_policy_config(policy_path: Path) -> PreTrainedConfig:
    if not policy_path.is_dir():
        raise FileNotFoundError(f"Policy path does not exist: {policy_path}")
    ensure_checkpoint_assets(policy_path)
    return PreTrainedConfig.from_pretrained(str(policy_path))


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _find_processor_step(config_path: Path, registry_name: str) -> dict[str, Any]:
    config = _load_json(config_path)
    steps = config.get("steps", [])
    for step in steps:
        if step.get("registry_name") == registry_name:
            return step
    raise ValueError(f"Could not find processor step {registry_name!r} in {config_path}")


def _resolve_local_asset_path(raw: str) -> Path | None:
    candidate = Path(raw).expanduser()
    return candidate if candidate.exists() else None


def validate_local_base_model_path(base_model_path: str) -> tuple[str, Path | str]:
    resolved = Path(base_model_path).expanduser()
    if resolved.exists():
        return "local_path", resolved.resolve()
    if _is_local_path_reference(base_model_path):
        raise FileNotFoundError(f"base_model_path does not exist: {resolved}")
    return "repo_id", base_model_path


def validate_tokenizer_assets(tokenizer_assets_repo: str) -> tuple[str, Path]:
    local_asset_path = _resolve_local_asset_path(tokenizer_assets_repo)
    if local_asset_path is not None:
        missing = [name for name in REQUIRED_TOKENIZER_ASSET_FILES if not (local_asset_path / name).exists()]
        if missing:
            raise FileNotFoundError(
                "Tokenizer assets path is missing required files: "
                f"path={local_asset_path}, missing={missing}"
            )
        return "local_path", local_asset_path

    cache_dir = HF_LEROBOT_HOME / tokenizer_assets_repo
    missing = [name for name in REQUIRED_TOKENIZER_ASSET_FILES if not (cache_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Tokenizer assets cache is missing required files: "
            f"path={cache_dir}, missing={missing}"
        )
    return "hf_lerobot_cache", cache_dir


def validate_groot_checkpoint_for_mvp(policy_path: Path, arm: str) -> dict[str, Any]:
    if arm != RIGHT_PREFIX:
        raise ValueError(f"Agilex GROOT MVP only supports arm={RIGHT_PREFIX!r}, got {arm!r}")

    weight_files = ensure_checkpoint_assets(policy_path)
    policy_cfg = load_policy_config(policy_path)

    if getattr(policy_cfg, "type", None) != "groot":
        raise ValueError(f"Expected GROOT policy, got {getattr(policy_cfg, 'type', None)!r}")

    expected_input_shapes = {
        EXPECTED_STATE_KEY: (7,),
        EXPECTED_FRONT_CAMERA_KEY: (3, 480, 640),
        EXPECTED_RIGHT_CAMERA_KEY: (3, 480, 640),
    }
    expected_output_shapes = {EXPECTED_ACTION_KEY: (7,)}

    actual_input_keys = set(policy_cfg.input_features)
    actual_output_keys = set(policy_cfg.output_features)
    missing_inputs = sorted(set(expected_input_shapes) - actual_input_keys)
    extra_inputs = sorted(actual_input_keys - set(expected_input_shapes))
    missing_outputs = sorted(set(expected_output_shapes) - actual_output_keys)
    extra_outputs = sorted(actual_output_keys - set(expected_output_shapes))
    if missing_inputs or extra_inputs or missing_outputs or extra_outputs:
        raise ValueError(
            "Checkpoint schema does not match the Agilex GROOT right-arm MVP contract. "
            f"missing_inputs={missing_inputs}, extra_inputs={extra_inputs}, "
            f"missing_outputs={missing_outputs}, extra_outputs={extra_outputs}"
        )

    for key, expected_shape in expected_input_shapes.items():
        actual_shape = feature_shape(policy_cfg.input_features[key])
        if actual_shape != expected_shape:
            raise ValueError(f"Unexpected shape for {key}: expected {expected_shape}, got {actual_shape}")

    for key, expected_shape in expected_output_shapes.items():
        actual_shape = feature_shape(policy_cfg.output_features[key])
        if actual_shape != expected_shape:
            raise ValueError(f"Unexpected shape for {key}: expected {expected_shape}, got {actual_shape}")

    preprocessor_step = _find_processor_step(policy_path / "policy_preprocessor.json", "groot_pack_inputs_v3")
    preprocessor_cfg = preprocessor_step.get("config", {})
    language_key = preprocessor_cfg.get("language_key")
    if language_key != "task":
        raise ValueError(f"Expected language_key='task', got {language_key!r}")

    action_horizon = preprocessor_cfg.get("action_horizon")
    policy_n_action_steps = getattr(policy_cfg, "n_action_steps", None)
    if action_horizon is not None and policy_n_action_steps is not None and action_horizon != policy_n_action_steps:
        raise ValueError(
            "GROOT preprocessor action_horizon does not match config n_action_steps: "
            f"{action_horizon} != {policy_n_action_steps}"
        )

    postprocessor_step = _find_processor_step(
        policy_path / "policy_postprocessor.json",
        "groot_action_unpack_unnormalize_v1",
    )
    postprocessor_cfg = postprocessor_step.get("config", {})
    env_action_dim = postprocessor_cfg.get("env_action_dim")
    if env_action_dim != 7:
        raise ValueError(f"Expected env_action_dim=7, got {env_action_dim!r}")

    base_model_mode, base_model_location = validate_local_base_model_path(str(policy_cfg.base_model_path))
    tokenizer_mode, tokenizer_location = validate_tokenizer_assets(str(policy_cfg.tokenizer_assets_repo))

    return {
        "policy_cfg": policy_cfg,
        "weight_files": weight_files,
        "language_key": language_key,
        "action_horizon": action_horizon,
        "env_action_dim": env_action_dim,
        "base_model_mode": base_model_mode,
        "base_model_location": base_model_location,
        "tokenizer_mode": tokenizer_mode,
        "tokenizer_location": tokenizer_location,
    }


def format_check_summary(*, policy_path: Path, arm: str, report: dict[str, Any], policy_device: str) -> str:
    policy_cfg = report["policy_cfg"]
    lines = [
        "[INFO] Agilex GROOT async MVP preflight passed.",
        f"[INFO] Arm: {arm}",
        f"[INFO] Policy path: {policy_path}",
        f"[INFO] Policy type: {policy_cfg.type}",
        f"[INFO] Policy device: {policy_device}",
        f"[INFO] chunk_size: {getattr(policy_cfg, 'chunk_size', None)}",
        f"[INFO] n_action_steps: {getattr(policy_cfg, 'n_action_steps', None)}",
        f"[INFO] Input state shape: {feature_shape(policy_cfg.input_features[EXPECTED_STATE_KEY])}",
        (
            "[INFO] Input images: "
            f"camera_front={feature_shape(policy_cfg.input_features[EXPECTED_FRONT_CAMERA_KEY])}, "
            f"camera_right={feature_shape(policy_cfg.input_features[EXPECTED_RIGHT_CAMERA_KEY])}"
        ),
        f"[INFO] Output action shape: {feature_shape(policy_cfg.output_features[EXPECTED_ACTION_KEY])}",
        f"[INFO] language_key: {report['language_key']}",
        f"[INFO] action_horizon: {report['action_horizon']}",
        f"[INFO] env_action_dim: {report['env_action_dim']}",
        f"[INFO] weight_files: {[path.name for path in report['weight_files']]}",
        f"[INFO] base_model_path ({report['base_model_mode']}): {report['base_model_location']}",
        f"[INFO] tokenizer_assets ({report['tokenizer_mode']}): {report['tokenizer_location']}",
    ]
    return "\n".join(lines)


def format_client_runtime_summary(
    *,
    arm: str,
    policy_path: str,
    server_address: str,
    policy_device: str,
    control_mode: str,
    task: str,
    run_time_s: float,
    front_camera_key: str,
    side_camera_key: str,
) -> str:
    lines = [
        "[INFO] Agilex GROOT async MVP client configuration:",
        f"[INFO] Arm: {arm}",
        f"[INFO] Policy path (server-visible): {policy_path}",
        "[INFO] Policy type: groot",
        f"[INFO] Policy device: {policy_device}",
        f"[INFO] Server address: {server_address}",
        f"[INFO] Control mode: {control_mode}",
        f"[INFO] actions_per_chunk: {MVP_ACTIONS_PER_CHUNK}",
        f"[INFO] chunk_size_threshold: {MVP_CHUNK_SIZE_THRESHOLD}",
        f"[INFO] aggregate_fn_name: {MVP_AGGREGATE_FN_NAME}",
        f"[INFO] one_step_timestep_strategy: {ONE_STEP_TIMESTEP_STRATEGY}",
        f"[INFO] camera keys: [{front_camera_key}, {side_camera_key}]",
        f"[INFO] task: {task}",
        f"[INFO] run_time_s: {run_time_s} (<=0 means until Ctrl+C)",
    ]
    return "\n".join(lines)
