#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from safetensors import safe_open


DEFAULT_POLICY_PATH = (
    "/data/cqy_workspace/flexible_lerobot/outputs/pi05_eraser_cup_multi_task_runs/"
    "20260602_200955/checkpoints/last/pretrained_model"
)
DEFAULT_TASK = "First put the eraser into the small box, then move the cup back to the upper-right corner"
DEFAULT_ROBOT_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00"
DEFAULT_CALIB_DIR = "/home/cqy/.cache/huggingface/lerobot/calibration/robots/so_follower"
KNOWN_TASKS = {
    "eraser_to_box": "Put the eraser into the small box",
    "cup_to_upper_right": "Move the cup back to the upper-right corner",
    "eraser_then_cup": "First put the eraser into the small box, then move the cup back to the upper-right corner",
}


def resolve_repo_root(script_path: Path) -> Path:
    resolved_path = script_path.resolve()
    for candidate in resolved_path.parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src/lerobot").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate repository root from script path: {script_path}")


def ensure_repo_on_path(repo_root: Path) -> None:
    repo_str = repo_root.as_posix()
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


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


def parse_camera(value: str) -> int | Path:
    if value.isdecimal():
        return int(value)
    return Path(value).expanduser()


def optional_float(value: str | None) -> float | None:
    if value is None or value.strip().lower() in {"", "none", "null"}:
        return None
    return float(value)


def optional_json_dict(value: str | None) -> dict[str, float] | None:
    if value is None or value.strip().lower() in {"", "none", "null"}:
        return None

    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("Expected a JSON object mapping names to numeric values.")

    result: dict[str, float] = {}
    for key, item in parsed.items():
        if not isinstance(key, str):
            raise argparse.ArgumentTypeError("Expected all JSON object keys to be strings.")
        if not isinstance(item, int | float):
            raise argparse.ArgumentTypeError(f"Expected numeric value for {key!r}, got {item!r}.")
        result[key] = float(item)
    return result


def ensure_local_tokenizer_dir(repo_root: Path) -> None:
    local_tok = repo_root / "google" / "paligemma-3b-pt-224"
    if not local_tok.is_dir():
        raise FileNotFoundError(
            "Missing local tokenizer directory for offline PI0.5 inference.\n"
            f"Expected: {local_tok}\n"
            "Fix:\n"
            "  - Download tokenizer files to that directory or create a symlink.\n"
            "  - In this workspace it is usually a symlink to:\n"
            "      assets/modelscope/google/paligemma-3b-pt-224\n"
        )


def summarize_safetensors(path: Path, limit: int = 8) -> None:
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        print(f"[INFO] {path.name}: {path.stat().st_size} bytes, tensors={len(keys)}")
        for key in keys[:limit]:
            tensor_slice = handle.get_slice(key)
            print(
                f"[INFO]   {key}: "
                f"shape={tuple(tensor_slice.get_shape())} dtype={tensor_slice.get_dtype()}"
            )
        if len(keys) > limit:
            print("[INFO]   ...")


def validate_policy_artifacts(
    policy_path: Path,
    *,
    expected_state_dim: int | None = 6,
    expected_action_dim: int | None = 6,
    expected_image_shape: list[int] | None = None,
    summarize: bool = True,
) -> dict:
    required_files = [
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_preprocessor_step_2_normalizer_processor.safetensors",
        "policy_postprocessor.json",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        "train_config.json",
    ]
    missing = [name for name in required_files if not (policy_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Policy checkpoint is missing required files: {missing}")

    with (policy_path / "config.json").open() as f:
        config = json.load(f)

    expected_image_shape = expected_image_shape or [3, 480, 640]
    expected_inputs = {
        "observation.images.top": expected_image_shape,
        "observation.images.wrist": expected_image_shape,
    }
    if expected_state_dim is not None:
        expected_inputs["observation.state"] = [expected_state_dim]

    for key, expected_shape in expected_inputs.items():
        feature = config.get("input_features", {}).get(key)
        if feature is None or feature.get("shape") != expected_shape:
            raise ValueError(
                f"Unexpected input feature for {key}: {feature}. Expected shape {expected_shape}."
            )

    if expected_action_dim is not None:
        action_feature = config.get("output_features", {}).get("action")
        if action_feature is None or action_feature.get("shape") != [expected_action_dim]:
            raise ValueError(
                f"Unexpected action feature: {action_feature}. Expected shape [{expected_action_dim}]."
            )

    print("[INFO] Policy artifacts found and feature shapes match expected SO top/wrist setup.")
    print(
        "[INFO] Policy config: "
        f"type={config.get('type')} dtype={config.get('dtype')} "
        f"chunk_size={config.get('chunk_size')} n_action_steps={config.get('n_action_steps')}"
    )
    if summarize:
        summarize_safetensors(policy_path / "model.safetensors")
        summarize_safetensors(policy_path / "policy_preprocessor_step_2_normalizer_processor.safetensors")
        summarize_safetensors(policy_path / "policy_postprocessor_step_0_unnormalizer_processor.safetensors")
    return config
