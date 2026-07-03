from __future__ import annotations

import argparse

import numpy as np

from openpi_so101 import policy
from openpi_so101 import runtime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local checkpoint inference on one SO101 dataset sample.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--prompt", default="Put the eraser into the small box")
    parser.add_argument("--dataset-format", choices=("v3", "v21"), default="v3")
    parser.add_argument("--asset-id", default=None)
    return parser


def main() -> None:
    runtime.bootstrap()
    args = build_parser().parse_args()

    from openpi.policies import policy_config
    from openpi_so101 import config as so101_config
    from openpi_so101 import dataset_v3
    from openpi_so101 import patches

    config = so101_config.make_config(
        exp_name="smoke_infer",
        asset_id=args.asset_id
        or ("desk_cleanup_v1/eraser_cup_multi_task_v21_pilot" if args.dataset_format == "v21" else None),
    )
    so101_config.register_config(config)
    if args.dataset_format == "v21":
        ds = patches.LeRobotDatasetV21Compat(
            config.data.repo_id,
            runtime.converted_dataset_root(),
            action_horizon=config.model.action_horizon,
            max_frames=8,
        )
    else:
        ds = dataset_v3.SO101LeRobotV3Dataset(
            config.data.repo_id,
            runtime.dataset_root(),
            action_horizon=config.model.action_horizon,
            max_frames=8,
            decode_images=True,
        )
    sample = ds[0]
    example = {
        "observation.images.top": sample["observation.images.top"],
        "observation.images.wrist": sample["observation.images.wrist"],
        "observation.state": sample["observation.state"],
        "prompt": args.prompt,
    }
    trained_policy = policy_config.create_trained_policy(
        config,
        args.checkpoint_dir,
        repack_transforms=policy.SO101_INFERENCE_REPACK_TRANSFORMS,
        default_prompt=args.prompt,
    )
    result = trained_policy.infer(example)
    actions = np.asarray(result["actions"])
    print(f"actions shape={actions.shape} dtype={actions.dtype}")
    print(actions[:2])


if __name__ == "__main__":
    main()
