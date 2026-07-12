#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import random
from collections.abc import Mapping
from pathlib import Path

import gr00t.model.gr00t_n1d7.setup  # noqa: F401
import torch
from gr00t.configs.base_config import get_default_config
from gr00t.data.dataset.factory import DatasetFactory
from transformers import AutoProcessor


def ensure_within(path: Path, root: Path, *, must_exist: bool = False) -> Path:
    resolved = path.expanduser().resolve(strict=must_exist)
    allowed = root.expanduser().resolve(strict=True)
    try:
        resolved.relative_to(allowed)
    except ValueError as exc:
        raise ValueError(f"Path escapes allowed root: {resolved}") from exc
    return resolved


def load_modality_config(path: Path) -> None:
    spec = importlib.util.spec_from_file_location("gr00t17_so101_validation", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import modality config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def tensor_summary(tensor: torch.Tensor) -> dict:
    summary = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
    }
    if tensor.is_floating_point():
        summary["finite"] = bool(torch.isfinite(tensor).all())
        summary["min"] = float(tensor.min())
        summary["max"] = float(tensor.max())
    return summary


def flatten_tensors(value, prefix: str = "") -> dict[str, dict]:
    if torch.is_tensor(value):
        return {prefix: tensor_summary(value)}
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten_tensors(item, child_prefix))
        return result
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and validate one real N1.7 training batch.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--backbone-assets", type=Path, required=True)
    parser.add_argument("--modality-config", type=Path, required=True)
    parser.add_argument("--allowed-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    allowed_root = args.allowed_root.expanduser().resolve(strict=True)
    dataset_path = ensure_within(args.dataset, allowed_root, must_exist=True)
    model_dir = ensure_within(args.model_dir, allowed_root, must_exist=True)
    backbone_assets = ensure_within(args.backbone_assets, allowed_root, must_exist=True)
    modality_path = ensure_within(args.modality_config, allowed_root, must_exist=True)
    report_path = ensure_within(args.report, allowed_root)
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite batch validation report: {report_path}")

    load_modality_config(modality_path)
    random.seed(42)
    torch.manual_seed(42)
    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": [str(dataset_path)],
                        "mix_ratio": 1.0,
                        "embodiment_tag": "new_embodiment",
                    }
                ],
            }
        }
    )
    config.data.shard_size = 64
    config.data.episode_sampling_rate = 0.1
    config.data.num_shards_per_epoch = 2
    config.model.model_name = str(backbone_assets)
    config.model.state_dropout_prob = 0.0
    config.model.use_relative_action = True

    model_config = config.model
    processor = AutoProcessor.from_pretrained(
        model_dir,
        modality_configs=config.data.modality_configs,
        use_percentiles=model_config.use_percentiles,
        image_crop_size=model_config.image_crop_size,
        image_target_size=model_config.image_target_size,
        random_rotation_angle=model_config.random_rotation_angle,
        color_jitter_params=model_config.color_jitter_params,
        model_name=model_config.model_name,
        model_type=model_config.backbone_model_type,
        formalize_language=model_config.formalize_language,
        apply_sincos_state_encoding=model_config.apply_sincos_state_encoding,
        max_action_horizon=model_config.action_horizon,
        use_albumentations=model_config.use_albumentations_transforms,
        extra_augmentation_config=model_config.extra_augmentation_config,
        shortest_image_edge=model_config.shortest_image_edge,
        crop_fraction=model_config.crop_fraction,
        use_alternate_vl_dit=model_config.use_alternate_vl_dit,
        use_relative_action=model_config.use_relative_action,
        exclude_state=model_config.exclude_state,
        state_dropout_prob=model_config.state_dropout_prob,
        use_mean_std=model_config.use_mean_std,
        transformers_loading_kwargs={"trust_remote_code": True, "local_files_only": True},
        local_files_only=True,
        trust_remote_code=True,
    )
    training_dataset, _ = DatasetFactory(config).build(processor)
    sample = next(iter(training_dataset))
    batch = processor.collator([sample])
    tensors = flatten_tensors(batch)

    expected_shapes = {
        "inputs.action_mask": [1, 40, 132],
        "inputs.embodiment_id": [1],
        "inputs.image_grid_thw": [2, 3],
        "inputs.action": [1, 40, 132],
        "inputs.state": [1, 1, 132],
    }
    for key, expected_shape in expected_shapes.items():
        if key not in tensors:
            raise KeyError(f"Training batch is missing tensor: {key}")
        if tensors[key]["shape"] != expected_shape:
            raise RuntimeError(f"Unexpected shape for {key}: {tensors[key]['shape']} != {expected_shape}")
    nonfinite = [key for key, value in tensors.items() if value.get("finite") is False]
    if nonfinite:
        raise RuntimeError(f"Non-finite tensors in training batch: {nonfinite}")

    inputs = batch["inputs"]
    action_mask_active = int(inputs["action_mask"].sum().item())
    if action_mask_active != 16 * 6:
        raise RuntimeError(f"Expected 96 active action values, got {action_mask_active}")
    prompt = processor.processor.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)

    single_dataset = training_dataset.datasets[0]
    report = {
        "schema_version": 1,
        "status": "passed",
        "dataset": str(dataset_path),
        "model_dir": str(model_dir),
        "backbone_assets": str(backbone_assets),
        "shard_size": config.data.shard_size,
        "episode_sampling_rate": config.data.episode_sampling_rate,
        "state_dropout_prob": config.model.state_dropout_prob,
        "shard_count": len(single_dataset),
        "effective_training_steps": int(sum(single_dataset.shard_lengths)),
        "action_mask_active_values": action_mask_active,
        "prompt": prompt,
        "tensors": tensors,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
