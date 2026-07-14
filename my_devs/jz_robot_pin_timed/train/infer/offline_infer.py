#!/usr/bin/env python

"""Run ACT inference on immutable dataset samples without connecting to a robot."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.robots.jz_robot_pin_timed.training_schema import (
    JZPinTrainingSchema,
    load_training_schema_from_local_checkpoint,
)

CAMERA_PREFIX = "observation.images."


def parse_indices(raw: str, dataset_length: int) -> list[int]:
    aliases = {
        "first": 0,
        "middle": dataset_length // 2,
        "last": dataset_length - 1,
    }
    indices: list[int] = []
    for token in raw.split(","):
        token = token.strip().lower()
        if not token:
            continue
        index = aliases.get(token, int(token) if token.lstrip("-").isdigit() else None)
        if index is None:
            raise ValueError(f"Invalid sample index token: {token!r}")
        if index < 0:
            index += dataset_length
        if not 0 <= index < dataset_length:
            raise IndexError(f"Sample index {index} is outside [0, {dataset_length})")
        if index not in indices:
            indices.append(index)
    if not indices:
        raise ValueError("At least one sample index is required")
    return indices


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def synchronize(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--sample-indices", default="first,middle,last")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    policy_path = args.policy_path.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA inference was requested but torch.cuda.is_available() is false")

    required = (
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "train_config.json",
    )
    missing = [name for name in required if not (policy_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Checkpoint is missing required files: {missing}")

    schema = load_training_schema_from_local_checkpoint(policy_path)
    if schema is None:
        raise ValueError("Checkpoint has no serialized JZ raw18/model16 training schema")
    schema.ensure_trainable()

    config_json = load_json(policy_path / "config.json")
    if config_json["input_features"]["observation.state"]["shape"] != [16]:
        raise ValueError("Checkpoint observation.state is not model16")
    if config_json["output_features"]["action"]["shape"] != [16]:
        raise ValueError("Checkpoint action is not model16")

    policy_cfg = PreTrainedConfig.from_pretrained(str(policy_path), local_files_only=True)
    if policy_cfg.type != "act":
        raise ValueError(f"Expected an ACT checkpoint, got {policy_cfg.type!r}")
    policy_cfg.pretrained_path = policy_path
    policy_cfg.device = args.device
    policy_class = get_policy_class(policy_cfg.type)
    policy = policy_class.from_pretrained(
        str(policy_path),
        config=policy_cfg,
        strict=False,
        local_files_only=True,
    ).to(args.device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(policy_path),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    dataset = LeRobotDataset(args.dataset_repo_id, root=dataset_root, video_backend="pyav")
    schema.validate_raw_features(dataset.meta.features)
    indices = parse_indices(args.sample_indices, len(dataset))

    info = load_json(dataset_root / "meta/info.json")
    raw_action_names = info["features"]["action"]["names"]
    results = []
    for index in indices:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()
        raw_sample = dataset[index]

        synchronize(args.device)
        start = time.perf_counter()
        with torch.inference_mode():
            processed = preprocessor(raw_sample)
            model_action = policy.select_action(processed)
            raw_action = postprocessor(model_action)
        synchronize(args.device)
        latency_ms = (time.perf_counter() - start) * 1000

        if raw_sample["observation.state"].shape[-1] != 18:
            raise ValueError(f"Dataset sample {index} state is not raw18")
        if model_action.shape[-1] != 16 or raw_action.shape[-1] != 18:
            raise ValueError(f"Inference boundary is {model_action.shape} -> {raw_action.shape}")
        if not torch.isfinite(model_action).all() or not torch.isfinite(raw_action).all():
            raise ValueError(f"Inference sample {index} produced non-finite action values")

        camera_shapes = {
            key: list(value.shape)
            for key, value in processed.items()
            if key.startswith(CAMERA_PREFIX)
        }
        if len(camera_shapes) != 3:
            raise ValueError(f"Expected three processed cameras, got {sorted(camera_shapes)}")

        raw_vector = raw_action.detach().cpu().reshape(-1, 18)[0]
        model_vector = model_action.detach().cpu().reshape(-1, 16)[0]
        gripper_out_of_range = {
            name: raw_vector[position].item()
            for name, position in (("left_gripper.width", 14), ("right_gripper.width", 16))
            if not 0.0 <= raw_vector[position].item() <= 100.0
        }
        result = {
            "sample_index": index,
            "episode_index": int(raw_sample["episode_index"].item()),
            "frame_index": int(raw_sample["frame_index"].item()),
            "latency_ms": latency_ms,
            "raw_state_shape": list(raw_sample["observation.state"].shape),
            "model_action_shape": list(model_action.shape),
            "raw_action_shape": list(raw_action.shape),
            "processed_camera_shapes": camera_shapes,
            "model16_action": model_vector.tolist(),
            "raw18_action": dict(zip(raw_action_names, raw_vector.tolist(), strict=True)),
            "gripper_out_of_range_before_robot_clamp": gripper_out_of_range,
        }
        results.append(result)
        print(
            f"sample={index} episode={result['episode_index']} frame={result['frame_index']} "
            f"latency_ms={latency_ms:.3f} model_action={tuple(model_action.shape)} "
            f"raw_action={tuple(raw_action.shape)}"
        )
        print(
            "  grippers="
            f"left_width={raw_vector[14].item():.6f} left_force={raw_vector[15].item():.6f} "
            f"right_width={raw_vector[16].item():.6f} right_force={raw_vector[17].item():.6f}"
        )
        if gripper_out_of_range:
            print(
                "  WARNING gripper width is outside [0,100] before the live Robot boundary clamp: "
                f"{gripper_out_of_range}"
            )

    report = {
        "status": "PASS",
        "policy_path": str(policy_path),
        "dataset_root": str(dataset_root),
        "device": args.device,
        "policy_type": policy_cfg.type,
        "chunk_size": policy_cfg.chunk_size,
        "n_action_steps": policy_cfg.n_action_steps,
        "schema_sources": schema.observation_sources,
        "samples": results,
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"status=PASS samples={len(results)} policy={policy_path} "
        f"raw18->model16->raw18 device={args.device}"
    )


if __name__ == "__main__":
    main()
