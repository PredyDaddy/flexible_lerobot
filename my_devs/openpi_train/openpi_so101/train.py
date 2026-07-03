from __future__ import annotations

import argparse
import os
from pathlib import Path

from openpi_so101 import config as so101_config
from openpi_so101 import patches
from openpi_so101 import runtime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run OpenPI pi05 LoRA fine-tuning on the SO101 dataset.")
    parser.add_argument("--exp-name", default=os.environ.get("OPENPI_SO101_EXP_NAME", "smoke"))
    parser.add_argument("--num-train-steps", type=int, default=int(os.environ.get("OPENPI_SO101_STEPS", "10")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("OPENPI_SO101_BATCH_SIZE", "1")))
    parser.add_argument("--max-frames", type=int, default=int(os.environ.get("OPENPI_SO101_MAX_FRAMES", "256")))
    parser.add_argument("--action-horizon", type=int, default=int(os.environ.get("OPENPI_SO101_ACTION_HORIZON", "50")))
    parser.add_argument("--learning-rate", type=float, default=float(os.environ.get("OPENPI_SO101_LR", "5e-5")))
    parser.add_argument("--save-interval", type=int, default=int(os.environ.get("OPENPI_SO101_SAVE_INTERVAL", "10")))
    parser.add_argument("--log-interval", type=int, default=int(os.environ.get("OPENPI_SO101_LOG_INTERVAL", "1")))
    parser.add_argument("--dataset-root", default=os.environ.get("OPENPI_SO101_DATASET_ROOT"))
    parser.add_argument("--dataset-format", choices=("v3", "v21"), default=os.environ.get("OPENPI_SO101_DATASET_FORMAT", "v3"))
    parser.add_argument("--converted-root", default=os.environ.get("OPENPI_SO101_V21_ROOT"))
    parser.add_argument("--asset-id", default=os.environ.get("OPENPI_SO101_ASSET_ID"))
    parser.add_argument("--base-params", default=os.environ.get("OPENPI_PI05_BASE_PARAMS"))
    parser.add_argument("--overwrite", action="store_true", default=os.environ.get("OPENPI_SO101_OVERWRITE") == "1")
    parser.add_argument("--resume", action="store_true", default=os.environ.get("OPENPI_SO101_RESUME") == "1")
    parser.add_argument("--wandb", action="store_true", default=os.environ.get("OPENPI_SO101_WANDB") == "1")
    parser.add_argument("--decode-images", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    runtime.bootstrap()
    args = build_parser().parse_args()

    if args.dataset_root:
        os.environ["OPENPI_SO101_DATASET_ROOT"] = str(Path(args.dataset_root).expanduser().resolve())
    if args.converted_root:
        os.environ["OPENPI_SO101_V21_ROOT"] = str(Path(args.converted_root).expanduser().resolve())
    if args.base_params:
        os.environ["OPENPI_PI05_BASE_PARAMS"] = str(Path(args.base_params).expanduser().resolve())

    patches.patch_openpi_data_loader(
        max_frames=args.max_frames,
        decode_images=args.decode_images,
        dataset_format=args.dataset_format,
    )
    config = so101_config.make_config(
        exp_name=args.exp_name,
        num_train_steps=args.num_train_steps,
        batch_size=args.batch_size,
        action_horizon=args.action_horizon,
        learning_rate=args.learning_rate,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        wandb_enabled=args.wandb,
        overwrite=args.overwrite,
        resume=args.resume,
        prompt_from_task=args.dataset_format == "v21",
        asset_id=args.asset_id
        or os.environ.get("OPENPI_SO101_V21_ASSET_ID")
        or ("desk_cleanup_v1/eraser_cup_multi_task_v21_pilot" if args.dataset_format == "v21" else None),
    )
    so101_config.register_config(config)

    from scripts import train as openpi_train

    openpi_train.main(config)


if __name__ == "__main__":
    main()
