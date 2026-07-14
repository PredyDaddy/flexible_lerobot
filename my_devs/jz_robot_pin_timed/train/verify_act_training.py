#!/usr/bin/env python

"""Verify a JZ ACT checkpoint, including one offline raw18 inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.robots.jz_robot_pin_timed.training_schema import JZPinTrainingDatasetView, JZPinTrainingSchema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--expected-steps", type=int, required=True)
    parser.add_argument("--expected-epochs", type=int)
    parser.add_argument("--steps-per-epoch", type=int, required=True)
    parser.add_argument("--expected-resize", required=True)
    parser.add_argument("--expected-schema", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def main() -> None:
    args = parse_args()
    model_dir = args.output_dir / "checkpoints" / "last" / "pretrained_model"
    required = [
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "train_config.json",
    ]
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing final checkpoint files: {missing}")

    train_config = load_json(model_dir / "train_config.json")
    if train_config["steps"] != args.expected_steps:
        raise ValueError(f"Expected {args.expected_steps} steps, got {train_config['steps']}")
    if args.expected_epochs is not None:
        epochs = args.expected_steps / args.steps_per_epoch
        if epochs != args.expected_epochs:
            raise ValueError(f"Expected {args.expected_epochs} epochs, computed {epochs}")

    config = load_json(model_dir / "config.json")
    if config["output_features"]["action"]["shape"] != [16]:
        raise ValueError("Final policy action feature is not model16")
    if config["input_features"]["observation.state"]["shape"] != [16]:
        raise ValueError("Final policy state feature is not model16")
    image_features = [key for key in config["input_features"] if key.startswith("observation.images.")]
    if len(image_features) != 3:
        raise ValueError(f"Expected three image features, got {image_features}")

    resize = [int(value) for value in args.expected_resize.replace("x", ",").split(",")]
    expected_schema = load_json(args.expected_schema)
    preprocessor_json = load_json(model_dir / "policy_preprocessor.json")
    resize_steps = [
        step
        for step in preprocessor_json["steps"]
        if step.get("registry_name") == "image_crop_resize_processor"
    ]
    if len(resize_steps) != 1 or resize_steps[0]["config"]["resize_size"] != resize:
        raise ValueError(f"Expected one serialized resize step {resize}, got {resize_steps}")
    projection_steps = [
        step
        for step in preprocessor_json["steps"]
        if step.get("class", "").endswith(".JZPinRaw18ToTraining16ProcessorStep")
    ]
    if len(projection_steps) != 1 or projection_steps[0]["config"]["schema"] != expected_schema:
        raise ValueError("Serialized raw18-to-model16 step is missing or has the wrong schema")
    normalizer_index = next(
        index
        for index, step in enumerate(preprocessor_json["steps"])
        if step.get("registry_name") == "normalizer_processor"
    )
    if preprocessor_json["steps"].index(projection_steps[0]) >= normalizer_index:
        raise ValueError("raw18-to-model16 projection must run before normalization")

    postprocessor_json = load_json(model_dir / "policy_postprocessor.json")
    expansion_steps = [
        step
        for step in postprocessor_json["steps"]
        if step.get("class", "").endswith(".JZPinTraining16ToRaw18ActionProcessorStep")
    ]
    if len(expansion_steps) != 1 or expansion_steps[0]["config"]["schema"] != expected_schema:
        raise ValueError("Serialized model16-to-raw18 step is missing or has the wrong schema")
    unnormalizer_index = next(
        index
        for index, step in enumerate(postprocessor_json["steps"])
        if step.get("registry_name") == "unnormalizer_processor"
    )
    if postprocessor_json["steps"].index(expansion_steps[0]) <= unnormalizer_index:
        raise ValueError("model16-to-raw18 expansion must run after unnormalization")

    schema = JZPinTrainingSchema(expected_schema)
    raw_dataset = LeRobotDataset(args.dataset_repo_id, root=args.dataset_root, video_backend="pyav")
    projected_dataset = JZPinTrainingDatasetView(raw_dataset, schema)
    policy = ACTPolicy.from_pretrained(model_dir, local_files_only=True).to(args.device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(model_dir),
    )

    raw_sample = raw_dataset[0]
    with torch.inference_mode():
        processed = preprocessor(raw_sample)
        model_action = policy.select_action(processed)
        raw_action = postprocessor(model_action)

    if tuple(raw_sample["observation.state"].shape) != (18,):
        raise ValueError(f"Raw sample is not 18D: {raw_sample['observation.state'].shape}")
    if tuple(projected_dataset[0]["observation.state"].shape) != (16,):
        raise ValueError("Dataset view is not model16")
    if model_action.shape[-1] != 16 or raw_action.shape[-1] != 18:
        raise ValueError(f"Inference boundary has shapes {model_action.shape} -> {raw_action.shape}")
    if not torch.isfinite(model_action).all() or not torch.isfinite(raw_action).all():
        raise ValueError("Offline inference produced non-finite output")
    for key in image_features:
        if tuple(processed[key].shape[-3:]) != (3, *resize):
            raise ValueError(f"Processed image {key} has shape {processed[key].shape}")

    print(f"status=PASS output_dir={args.output_dir}")
    print(
        f"steps={args.expected_steps} steps_per_epoch={args.steps_per_epoch} "
        f"epochs={args.expected_epochs if args.expected_epochs is not None else 'smoke'}"
    )
    print(f"features=raw18->3_cameras+model16->model16_action->raw18 resize={resize}")
    print(f"inference=model_action{tuple(model_action.shape)} raw_action{tuple(raw_action.shape)} finite=true")
    print(f"checkpoint={model_dir}")


if __name__ == "__main__":
    main()
