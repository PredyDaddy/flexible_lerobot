from __future__ import annotations

import argparse
import os

import numpy as np
import tqdm

from openpi_so101 import config as so101_config
from openpi_so101 import patches
from openpi_so101 import runtime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute OpenPI norm stats for the SO101 LeRobot v3 dataset.")
    parser.add_argument("--max-frames", type=int, default=int(os.environ.get("OPENPI_SO101_NORM_MAX_FRAMES", "2048")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("OPENPI_SO101_BATCH_SIZE", "32")))
    parser.add_argument("--action-horizon", type=int, default=int(os.environ.get("OPENPI_SO101_ACTION_HORIZON", "50")))
    parser.add_argument("--dataset-format", choices=("v3", "v21"), default=os.environ.get("OPENPI_SO101_DATASET_FORMAT", "v3"))
    parser.add_argument("--asset-id", default=os.environ.get("OPENPI_SO101_ASSET_ID"))
    parser.add_argument("--decode-images", action=argparse.BooleanOptionalAction, default=False)
    return parser


def main() -> None:
    runtime.bootstrap()
    args = build_parser().parse_args()
    patches.patch_openpi_data_loader(
        max_frames=args.max_frames,
        decode_images=args.decode_images,
        dataset_format=args.dataset_format,
    )

    from openpi.shared import normalize
    from openpi.training import data_loader as _data_loader
    from scripts.compute_norm_stats import RemoveStrings

    config = so101_config.make_config(
        exp_name="norm_stats",
        batch_size=args.batch_size,
        action_horizon=args.action_horizon,
        num_train_steps=1,
        prompt_from_task=args.dataset_format == "v21",
        asset_id=args.asset_id or ("desk_cleanup_v1/eraser_cup_multi_task_v21_pilot" if args.dataset_format == "v21" else None),
    )
    so101_config.register_config(config)
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            RemoveStrings(),
        ],
    )
    num_batches = max(1, min(len(dataset), args.max_frames) // args.batch_size)
    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=args.batch_size,
        num_workers=0,
        shuffle=False,
        num_batches=num_batches,
    )

    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    for batch in tqdm.tqdm(loader, total=num_batches, desc="Computing SO101 stats"):
        for key in stats:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: value.get_statistics() for key, value in stats.items()}
    output_path = config.assets_dirs / data_config.asset_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    main()
