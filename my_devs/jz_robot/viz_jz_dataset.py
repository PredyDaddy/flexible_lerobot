#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.lerobot_dataset_viz import visualize_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize a recorded JZRobot dataset episode with Rerun."
    )
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--mode", choices=["local", "distant"], default="local")
    parser.add_argument("--web-port", type=int, default=9090)
    parser.add_argument("--ws-port", type=int, default=9087)
    parser.add_argument("--save-rrd", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--display-compressed-images", action="store_true")
    parser.add_argument("--tolerance-s", type=float, default=1e-4)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    root = Path(args.dataset_root).expanduser() if args.dataset_root else None
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else None

    dataset = LeRobotDataset(
        args.dataset_repo_id,
        episodes=[args.episode],
        root=root,
        tolerance_s=args.tolerance_s,
    )

    print(
        "[INFO] dataset loaded: "
        f"repo_id={args.dataset_repo_id}, root={root}, episode={args.episode}, "
        f"camera_keys={dataset.meta.camera_keys}"
    )

    rrd_path = visualize_dataset(
        dataset=dataset,
        episode_index=args.episode,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mode=args.mode,
        web_port=args.web_port,
        ws_port=args.ws_port,
        save=args.save_rrd,
        output_dir=output_dir,
        display_compressed_images=args.display_compressed_images,
    )

    if rrd_path is not None:
        print(f"[INFO] Saved Rerun file: {rrd_path}")


if __name__ == "__main__":
    main()
