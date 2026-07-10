#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path("/data/cqy_workspace/flexible_lerobot")
VLASH_ROOT = REPO_ROOT / "my_devs/vla_engineering/vlash-main"
for path in (VLASH_ROOT, REPO_ROOT / "src"):
    if path.as_posix() not in sys.path:
        sys.path.insert(0, path.as_posix())

from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.configs.policies import PreTrainedConfig

from vlash.datasets import VLASHDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe local LeRobot/VLASH dataset compatibility.")
    parser.add_argument("--dataset-root", default=str(REPO_ROOT / "datasets/desk_cleanup_v1/eraser_cup_multi_task"))
    parser.add_argument("--repo-id", default="desk_cleanup_v1/eraser_cup_multi_task")
    parser.add_argument("--policy-path", default=str(REPO_ROOT / "assets/modelscope/lerobot/pi05_base"))
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--max-delay-steps", type=int, default=1)
    args = parser.parse_args()

    cfg = PreTrainedConfig.from_pretrained(
        args.policy_path,
        cli_overrides=[
            "--device=cpu",
            "--state_cond=true",
        ],
    )
    meta = LeRobotDatasetMetadata(args.repo_id, root=args.dataset_root)
    delta_timestamps = resolve_delta_timestamps(cfg, meta)
    dataset = VLASHDataset(
        args.repo_id,
        root=args.dataset_root,
        delta_timestamps=delta_timestamps,
        video_backend=args.video_backend,
        max_delay_steps=args.max_delay_steps,
    )
    item = dataset[0]

    summary = {
        "repo_id": args.repo_id,
        "root": str(args.dataset_root),
        "num_frames": dataset.num_frames,
        "num_episodes": dataset.num_episodes,
        "camera_keys": dataset.meta.camera_keys,
        "delta_timestamps": delta_timestamps,
        "sample": {},
    }
    for key, value in item.items():
        if hasattr(value, "shape"):
            summary["sample"][key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        else:
            summary["sample"][key] = repr(value)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
