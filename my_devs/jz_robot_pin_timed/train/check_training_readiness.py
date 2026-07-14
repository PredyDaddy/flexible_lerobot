#!/usr/bin/env python

"""Check LeRobot runtime readability and the JZ training boundary offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.jz_robot_pin_timed.training_schema import (
    JZPinTrainingDatasetView,
    JZPinTrainingSchema,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--report-json", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def status_from(path: Path) -> str | None:
    if not path.is_file():
        return None
    return load_json(path).get("status")


def main() -> None:
    args = parse_args()
    root = args.dataset_root.resolve()
    info = load_json(root / "meta/info.json")
    schema_path = root / "meta/jz_pin_training_schema.json"
    schema = JZPinTrainingSchema.from_file(schema_path)
    schema.ensure_trainable()
    schema.validate_raw_features(info["features"])

    repo_id = f"local/{root.name}"
    dataset = LeRobotDataset(repo_id, root=root, video_backend="pyav")
    if len(dataset) != info["total_frames"]:
        raise ValueError(f"Dataset length {len(dataset)} != metadata {info['total_frames']}")
    if dataset.num_episodes != info["total_episodes"]:
        raise ValueError(
            f"Dataset episode count {dataset.num_episodes} != metadata {info['total_episodes']}"
        )

    view = JZPinTrainingDatasetView(dataset, schema)
    sample_indices = sorted({0, len(view) // 2, len(view) - 1})
    sample_shapes = []
    for index in sample_indices:
        sample = view[index]
        state = sample["observation.state"]
        action = sample["action"]
        if tuple(state.shape) != (16,) or action.shape[-1] != 16:
            raise ValueError(f"Projected sample {index} has state/action {state.shape}/{action.shape}")
        if not torch.isfinite(state).all() or not torch.isfinite(action).all():
            raise ValueError(f"Projected sample {index} contains non-finite values")
        camera_shapes = {
            key: list(value.shape)
            for key, value in sample.items()
            if key.startswith("observation.images.")
        }
        if len(camera_shapes) != 3:
            raise ValueError(f"Projected sample {index} has cameras {sorted(camera_shapes)}")
        sample_shapes.append(
            {
                "index": index,
                "state": list(state.shape),
                "action": list(action.shape),
                "cameras": camera_shapes,
            }
        )

    curation_status = status_from(root / "meta/jz_pin_curation_report.json")
    color_status = status_from(root / "color_review.json")
    if curation_status not in (None, "MERGE_PASS"):
        raise ValueError(f"Unexpected curation status: {curation_status}")
    if color_status not in (None, "PASS", "PASS_WITH_REVIEW"):
        raise ValueError(f"Color review is not approved: {color_status}")

    report = {
        "status": "PASS",
        "dataset_root": str(root),
        "repo_id": repo_id,
        "episodes": dataset.num_episodes,
        "frames": len(dataset),
        "fps": dataset.fps,
        "raw_dimension": 18,
        "model_dimension": 16,
        "camera_keys": list(dataset.meta.camera_keys),
        "schema": str(schema_path),
        "observation_sources": schema.observation_sources,
        "curation_status": curation_status,
        "color_review_status": color_status,
        "sample_shapes": sample_shapes,
    }
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
