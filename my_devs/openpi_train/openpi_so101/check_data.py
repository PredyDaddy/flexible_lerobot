from __future__ import annotations

import argparse
import json
import os

import numpy as np

from openpi_so101 import config as so101_config
from openpi_so101 import dataset_v3
from openpi_so101 import patches
from openpi_so101 import runtime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check SO101 dataset and OpenPI transformed batch.")
    parser.add_argument("--max-frames", type=int, default=int(os.environ.get("OPENPI_SO101_CHECK_MAX_FRAMES", "64")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("OPENPI_SO101_BATCH_SIZE", "2")))
    parser.add_argument("--decode-images", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _summarize_tree(x):
    if isinstance(x, dict):
        return {key: _summarize_tree(value) for key, value in x.items()}
    arr = np.asarray(x)
    return {"shape": list(arr.shape), "dtype": str(arr.dtype)}


def main() -> None:
    runtime.bootstrap()
    args = build_parser().parse_args()
    root = runtime.dataset_root()
    meta = dataset_v3.LeRobotV3Metadata(os.environ["OPENPI_SO101_REPO_ID"], root)
    print(json.dumps(
        {
            "root": str(root),
            "repo_id": meta.repo_id,
            "codebase_version": meta.info.get("codebase_version"),
            "total_episodes": meta.info.get("total_episodes"),
            "total_frames": meta.info.get("total_frames"),
            "fps": meta.fps,
            "video_keys": meta.video_keys,
            "tasks": meta.tasks,
        },
        ensure_ascii=False,
        indent=2,
    ))

    ds = dataset_v3.SO101LeRobotV3Dataset(
        meta.repo_id,
        root,
        action_horizon=50,
        max_frames=args.max_frames,
        decode_images=args.decode_images,
    )
    sample = ds[0]
    print("Raw adapter sample:")
    print(json.dumps(_summarize_tree(sample), ensure_ascii=False, indent=2))

    patches.patch_openpi_data_loader(max_frames=args.max_frames, decode_images=args.decode_images)
    from openpi.training import data_loader as _data_loader

    config = so101_config.make_config(exp_name="check_data", batch_size=args.batch_size, num_train_steps=1)
    data = _data_loader.create_data_loader(config, num_batches=1, skip_norm_stats=True)
    batch = next(iter(data))
    observation, actions = batch
    print("OpenPI transformed batch:")
    print(json.dumps(
        {
            "observation": _summarize_tree(observation.to_dict()),
            "actions": _summarize_tree(actions),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
