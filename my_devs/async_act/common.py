from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from lerobot import policies  # noqa: F401
from lerobot.configs.policies import PreTrainedConfig
from lerobot.utils.utils import auto_select_torch_device

EXPECTED_STATE_KEY = "observation.state"
EXPECTED_ACTION_KEY = "action"

LEFT_PREFIX = "left"
RIGHT_PREFIX = "right"
ARM_CHOICES = (LEFT_PREFIX, RIGHT_PREFIX)
DEFAULT_ARM = RIGHT_PREFIX
DEFAULT_TASK = "Execute the trained Agilex single-arm ACT task"

ARM_TO_STATE_NAMES = {
    LEFT_PREFIX: tuple(f"{LEFT_PREFIX}_joint{i}.pos" for i in range(7)),
    RIGHT_PREFIX: tuple(f"{RIGHT_PREFIX}_joint{i}.pos" for i in range(7)),
}
ARM_TO_ACTION_NAMES = ARM_TO_STATE_NAMES
ARM_TO_LIVE_CAMERA_KEY = {
    LEFT_PREFIX: "camera_left",
    RIGHT_PREFIX: "camera_right",
}
ARM_TO_POLICY_CAMERA_KEY = {
    LEFT_PREFIX: "observation.images.camera_left",
    RIGHT_PREFIX: "observation.images.camera_right",
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


def parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"", "none", "null"}:
        return None
    return float(value)


def parse_optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"", "none", "null"}:
        return None
    return int(value)


def feature_shape(feature: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in getattr(feature, "shape", ()))


def ensure_checkpoint_assets(policy_path: Path) -> None:
    required_files = [
        policy_path / "config.json",
        policy_path / "policy_preprocessor.json",
        policy_path / "policy_postprocessor.json",
    ]
    missing_files = [path for path in required_files if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(f"Missing checkpoint assets: {missing_files}")


def load_policy_config(policy_path: Path) -> PreTrainedConfig:
    if not policy_path.is_dir():
        raise FileNotFoundError(f"Policy path does not exist: {policy_path}")
    ensure_checkpoint_assets(policy_path)
    return PreTrainedConfig.from_pretrained(str(policy_path))


def validate_single_arm_checkpoint_schema(policy_cfg: PreTrainedConfig, arm: str) -> tuple[int, int]:
    if getattr(policy_cfg, "type", None) != "act":
        raise ValueError(f"Expected ACT policy, got {getattr(policy_cfg, 'type', None)!r}")

    expected_visual_keys = {
        "observation.images.camera_front": (3, 480, 640),
        ARM_TO_POLICY_CAMERA_KEY[arm]: (3, 480, 640),
    }
    expected_input_shapes = {EXPECTED_STATE_KEY: (7,), **expected_visual_keys}
    expected_output_shapes = {EXPECTED_ACTION_KEY: (7,)}

    actual_input_keys = set(policy_cfg.input_features)
    actual_output_keys = set(policy_cfg.output_features)
    missing_inputs = sorted(set(expected_input_shapes) - actual_input_keys)
    extra_inputs = sorted(actual_input_keys - set(expected_input_shapes))
    missing_outputs = sorted(set(expected_output_shapes) - actual_output_keys)
    extra_outputs = sorted(actual_output_keys - set(expected_output_shapes))
    if missing_inputs or extra_inputs or missing_outputs or extra_outputs:
        raise ValueError(
            "Checkpoint schema does not match the requested Agilex single-arm deployment contract. "
            f"arm={arm}, missing_inputs={missing_inputs}, extra_inputs={extra_inputs}, "
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

    _, image_height, image_width = expected_visual_keys["observation.images.camera_front"]
    return image_height, image_width


def resolve_policy_device(requested: str | None, config_device: str | None) -> str:
    if requested is not None and requested.strip().lower() in {"", "none", "null"}:
        requested = None
    if config_device is None or str(config_device).lower() in {"", "none", "null"}:
        config_device = auto_select_torch_device().type
    if requested is None:
        return str(config_device)
    if requested == "auto":
        return auto_select_torch_device().type
    return requested


def resolve_actions_per_chunk(
    *,
    chunk_size: int,
    explicit_actions_per_chunk: int | None,
    policy_n_action_steps: int | None,
) -> int:
    resolved = explicit_actions_per_chunk
    if resolved is None:
        resolved = policy_n_action_steps
    if resolved is None:
        resolved = int(chunk_size)
    if not 1 <= int(resolved) <= int(chunk_size):
        raise ValueError(f"actions_per_chunk must be within [1, {chunk_size}], got {resolved}")
    return int(resolved)


def resolve_temporal_ensemble_settings(
    *,
    chunk_size: int,
    explicit_actions_per_chunk: int | None,
    policy_n_action_steps: int | None,
    policy_temporal_ensemble_coeff: float | None,
    requested_chunk_size_threshold: float,
    aggregate_fn_name: str,
) -> tuple[int, float, list[str]]:
    notes: list[str] = []
    if policy_temporal_ensemble_coeff is None:
        return (
            resolve_actions_per_chunk(
                chunk_size=chunk_size,
                explicit_actions_per_chunk=explicit_actions_per_chunk,
                policy_n_action_steps=policy_n_action_steps,
            ),
            requested_chunk_size_threshold,
            notes,
        )

    if policy_n_action_steps not in {None, 1}:
        raise ValueError(
            "--policy-n-action-steps must be 1 when temporal ensemble mode is enabled. "
            f"Got {policy_n_action_steps}."
        )
    effective_actions_per_chunk = chunk_size
    if explicit_actions_per_chunk not in {None, chunk_size}:
        notes.append(
            "Temporal ensemble mode overrides actions_per_chunk to checkpoint chunk_size "
            f"({explicit_actions_per_chunk} -> {chunk_size})."
        )
    effective_threshold = 1.0
    if requested_chunk_size_threshold != effective_threshold:
        notes.append(
            "Temporal ensemble mode overrides chunk_size_threshold to 1.0 "
            f"({requested_chunk_size_threshold} -> 1.0) to send observations every step."
        )
    notes.append(
        "Temporal ensemble mode ignores aggregate_fn_name="
        f"{aggregate_fn_name!r} and uses ACT-aware overlap aggregation."
    )
    return effective_actions_per_chunk, effective_threshold, notes


def reject_temporal_ensemble(policy_temporal_ensemble_coeff: float | None) -> None:
    if policy_temporal_ensemble_coeff is not None:
        raise ValueError(
            "--policy-temporal-ensemble-coeff is no longer rejected globally. "
            "Use resolve_temporal_ensemble_settings() to compute the effective async runtime."
        )


def format_checkpoint_summary(
    *,
    arm: str,
    policy_path: Path,
    policy_cfg: PreTrainedConfig,
    actions_per_chunk: int,
    chunk_size_threshold: float,
    aggregate_fn_name: str,
    server_address: str,
    control_mode: str,
    run_time_s: float,
    runtime_mode: str,
    temporal_ensemble_coeff: float | None,
    action_smoothing_alpha: float | None = None,
    max_joint_step_rad: float | None = None,
    runtime_notes: list[str] | None = None,
) -> str:
    side_policy_key = ARM_TO_POLICY_CAMERA_KEY[arm]
    lines = [
        f"[INFO] Arm: {arm}",
        f"[INFO] Control mode: {control_mode}",
        f"[INFO] Policy path: {policy_path}",
        f"[INFO] Policy type: {policy_cfg.type}",
        f"[INFO] Policy device: {policy_cfg.device}",
        f"[INFO] Server address: {server_address}",
        (
            "[INFO] ACT config: "
            f"chunk_size={policy_cfg.chunk_size}, "
            f"n_action_steps={getattr(policy_cfg, 'n_action_steps', None)}, "
            f"temporal_ensemble_coeff={getattr(policy_cfg, 'temporal_ensemble_coeff', None)}"
        ),
        f"[INFO] runtime_mode: {runtime_mode}",
        f"[INFO] temporal_ensemble_coeff: {temporal_ensemble_coeff}",
        f"[INFO] actions_per_chunk: {actions_per_chunk}",
        f"[INFO] chunk_size_threshold: {chunk_size_threshold}",
        f"[INFO] aggregate_fn_name: {aggregate_fn_name}",
        f"[INFO] action_smoothing_alpha: {action_smoothing_alpha}",
        f"[INFO] max_joint_step_rad: {max_joint_step_rad}",
        f"[INFO] run_time_s: {run_time_s} (<=0 means until Ctrl+C)",
        f"[INFO] Input state shape: {feature_shape(policy_cfg.input_features[EXPECTED_STATE_KEY])}",
        (
            "[INFO] Input images: "
            f"camera_front={feature_shape(policy_cfg.input_features['observation.images.camera_front'])}, "
            f"{side_policy_key}={feature_shape(policy_cfg.input_features[side_policy_key])}"
        ),
        f"[INFO] Output action shape: {feature_shape(policy_cfg.output_features[EXPECTED_ACTION_KEY])}",
    ]
    for note in runtime_notes or []:
        lines.append(f"[INFO] {note}")
    return "\n".join(lines)
