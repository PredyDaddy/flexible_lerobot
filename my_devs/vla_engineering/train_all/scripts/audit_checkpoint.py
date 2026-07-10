#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def file_info(path: Path) -> dict[str, Any]:
    return {
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a train_all/VLASH checkpoint artifact contract.")
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--expected-state-dim", type=int, default=6)
    parser.add_argument("--expected-action-dim", type=int, default=6)
    parser.add_argument("--expected-image-height", type=int, default=480)
    parser.add_argument("--expected-image-width", type=int, default=640)
    args = parser.parse_args()

    policy_path = Path(args.policy_path).expanduser().resolve()
    required = ["config.json", "model.safetensors", "train_config.json"]
    optional = [
        "policy_preprocessor.json",
        "policy_preprocessor_step_2_normalizer_processor.safetensors",
        "policy_postprocessor.json",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    ]
    missing = [name for name in required if not (policy_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required checkpoint files in {policy_path}: {missing}")

    config = read_json(policy_path / "config.json")
    train_config = read_json(policy_path / "train_config.json")
    input_features = config.get("input_features", {})
    output_features = config.get("output_features", {})
    expected_image_shape = [3, args.expected_image_height, args.expected_image_width]
    expected_inputs = {
        "observation.images.top": expected_image_shape,
        "observation.images.wrist": expected_image_shape,
        "observation.state": [args.expected_state_dim],
    }

    schema_errors: list[str] = []
    for key, expected_shape in expected_inputs.items():
        feature = input_features.get(key)
        if feature is None:
            schema_errors.append(f"missing input feature: {key}")
        elif feature.get("shape") != expected_shape:
            schema_errors.append(f"{key} shape {feature.get('shape')} != {expected_shape}")

    action_feature = output_features.get("action")
    if action_feature is None:
        schema_errors.append("missing output feature: action")
    elif action_feature.get("shape") != [args.expected_action_dim]:
        schema_errors.append(f"action shape {action_feature.get('shape')} != {[args.expected_action_dim]}")

    if schema_errors:
        raise ValueError("Checkpoint schema mismatch: " + "; ".join(schema_errors))

    lora_dir = policy_path / "lora_adapters"
    summary = {
        "policy_path": str(policy_path),
        "required_files": {name: file_info(policy_path / name) for name in required},
        "optional_lerobot_bundle_files": {name: file_info(policy_path / name) for name in optional},
        "lora_adapters": {
            "exists": lora_dir.is_dir(),
            "adapter_config": file_info(lora_dir / "adapter_config.json"),
            "adapter_model": file_info(lora_dir / "adapter_model.safetensors"),
        },
        "policy": {
            "type": config.get("type"),
            "chunk_size": config.get("chunk_size"),
            "n_action_steps": config.get("n_action_steps"),
            "num_inference_steps": config.get("num_inference_steps"),
            "dtype": config.get("dtype"),
            "state_cond": config.get("state_cond"),
            "normalization_mapping": config.get("normalization_mapping"),
        },
        "train": {
            "job_name": train_config.get("job_name"),
            "output_dir": train_config.get("output_dir"),
            "max_delay_steps": train_config.get("max_delay_steps"),
            "shared_observation": train_config.get("shared_observation"),
            "dataset": train_config.get("dataset"),
            "lora": train_config.get("lora"),
        },
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
