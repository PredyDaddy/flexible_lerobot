from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm

from openpi_so101 import dataset_v3
from openpi_so101 import paths
from openpi_so101 import runtime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert a LeRobot v3 SO101 dataset to LeRobot v2.1.")
    parser.add_argument("--source-root", default=os.environ.get("OPENPI_SO101_DATASET_ROOT", str(paths.DEFAULT_SOURCE_DATASET)))
    parser.add_argument("--output-root", default=os.environ.get("OPENPI_SO101_V21_ROOT", str(paths.DEFAULT_CONVERTED_DATA_ROOT)))
    parser.add_argument("--repo-id", default=os.environ.get("OPENPI_SO101_V21_REPO_ID", paths.DEFAULT_REPO_ID))
    parser.add_argument("--episodes", type=int, default=int(os.environ.get("OPENPI_SO101_V21_EPISODES", "3")))
    parser.add_argument(
        "--allow-more-episodes",
        action="store_true",
        help="Allow converting more than 3 episodes. Use this only for an intentional full conversion.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--video-backend", default="pyav")
    return parser


def _v21_features(v3_meta: dataset_v3.LeRobotV3Metadata) -> dict:
    features = {}
    for key in ("action", "observation.state"):
        ft = dict(v3_meta.features[key])
        ft["shape"] = tuple(ft["shape"])
        features[key] = ft
    for key in v3_meta.video_keys:
        height, width, channels = v3_meta.features[key]["shape"]
        features[key] = {
            "dtype": "video",
            "shape": (channels, height, width),
            "names": ["channels", "height", "width"],
        }
    return features


def _episode_rows(source_root: Path, episode_index: int) -> pd.DataFrame:
    data_paths = sorted((source_root / "data").glob("chunk-*/*.parquet"))
    if not data_paths:
        raise FileNotFoundError(f"No source parquet files under {source_root / 'data'}")
    frames = [pd.read_parquet(path, filters=[("episode_index", "=", episode_index)]) for path in data_paths]
    rows = pd.concat(frames, ignore_index=True)
    if rows.empty:
        raise ValueError(f"Episode {episode_index} has no rows in {source_root}")
    return rows.sort_values("frame_index").reset_index(drop=True)


def _validate_output_root(source_root: Path, output_root: Path) -> None:
    allowed_roots = [
        (paths.PROJECT_ROOT / "data").resolve(),
        (paths.PROJECT_ROOT / "easy_use" / "data").resolve(),
    ]
    source_root = source_root.resolve()
    output_root = output_root.resolve()

    if not any(output_root.is_relative_to(root) for root in allowed_roots):
        allowed = ", ".join(str(root) for root in allowed_roots)
        raise ValueError(f"Refusing to write outside allowed data roots ({allowed}): {output_root}")
    if output_root in allowed_roots:
        raise ValueError(f"Refusing to overwrite a whole project data directory: {output_root}")
    if output_root == source_root or output_root.is_relative_to(source_root) or source_root.is_relative_to(output_root):
        raise ValueError(f"Refusing unsafe overlap between source and output: {source_root} vs {output_root}")


def main() -> None:
    runtime.bootstrap()
    args = build_parser().parse_args()
    source_root = Path(args.source_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    _validate_output_root(source_root, output_root)
    if args.episodes < 1:
        raise ValueError("--episodes must be at least 1")
    if args.episodes > 3 and not args.allow_more_episodes:
        raise ValueError("Pilot conversion is capped at 3 episodes. Pass --allow-more-episodes for an intentional larger conversion.")

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_root} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output_root)

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    v3_meta = dataset_v3.LeRobotV3Metadata(args.repo_id, source_root)
    out = LeRobotDataset.create(
        args.repo_id,
        fps=v3_meta.fps,
        root=output_root,
        robot_type=v3_meta.info.get("robot_type", "so_follower"),
        features=_v21_features(v3_meta),
        use_videos=True,
        video_backend=args.video_backend,
    )
    src = dataset_v3.SO101LeRobotV3Dataset(
        args.repo_id,
        source_root,
        action_horizon=1,
        max_frames=None,
        decode_images=True,
    )

    episode_indices = list(range(args.episodes))
    for episode_index in tqdm.tqdm(episode_indices, desc="Converting episodes"):
        rows = _episode_rows(source_root, episode_index)
        for row in rows.itertuples(index=False):
            global_index = int(getattr(row, "index"))
            sample = src[global_index]
            out.add_frame(
                {
                    "action": np.asarray(sample["action"][0], dtype=np.float32),
                    "observation.state": np.asarray(sample["observation.state"], dtype=np.float32),
                    "observation.images.top": np.asarray(sample["observation.images.top"], dtype=np.uint8),
                    "observation.images.wrist": np.asarray(sample["observation.images.wrist"], dtype=np.uint8),
                    "task": str(sample["prompt"]),
                }
            )
        out.save_episode()

    print(f"Wrote LeRobot v2.1 dataset: {output_root}")
    print(f"Episodes: {episode_indices}")


if __name__ == "__main__":
    main()
