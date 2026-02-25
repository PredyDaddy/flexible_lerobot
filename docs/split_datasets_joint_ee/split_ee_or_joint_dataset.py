#!/usr/bin/env python3

"""
python docs/split_datasets_joint_ee/split_ee_or_joint_dataset.py \
    --input-root ~/.cache/huggingface/lerobot/cqy/agilex_vla_demo_ee_pinocchio_Dual_Arm_Simultaneous_Execution_clean \
    --output-base outputs/split_datasets_custom \
    --spaces both \
    --video-mode auto \
    --overwrite
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from datasets import Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
from lerobot.datasets.utils import (
    DATA_DIR,
    EPISODES_DIR,
    get_hf_features_from_features,
    load_info,
    load_stats,
    write_info,
)
from lerobot.datasets.utils import write_stats as write_stats_json


@dataclass(frozen=True)
class SplitSpec:
    space: str
    observation_indices: list[int]
    observation_names: list[str]
    action_mode: str
    action_indices: list[int] | None
    action_names: list[str]


def _sorted_parquet_files(root: Path, relative_dir: str) -> list[Path]:
    parquet_dir = root / relative_dir
    paths = sorted(parquet_dir.glob("chunk-*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files found under: {parquet_dir}")
    return paths


def _write_data_parquet(df: pd.DataFrame, path: Path, features: dict) -> None:
    hf_features = get_hf_features_from_features(features)
    ds = Dataset.from_dict(df.to_dict(orient="list"), features=hf_features, split="train")
    table = ds.with_format("arrow")[:]
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(path, schema=table.schema, compression="snappy", use_dictionary=True)
    writer.write_table(table)
    writer.close()


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if mode == "copy":
        shutil.copy2(src, dst)
        return

    if mode == "symlink":
        if dst.exists():
            return
        os.symlink(src, dst)
        return

    if mode == "hardlink":
        if dst.exists():
            return
        os.link(src, dst)
        return

    if mode == "auto":
        try:
            _link_or_copy(src, dst, mode="hardlink")
            return
        except OSError:
            try:
                _link_or_copy(src, dst, mode="symlink")
                return
            except OSError:
                _link_or_copy(src, dst, mode="copy")
                return

    raise ValueError(f"Unknown video mode: {mode}")


def _copy_videos(src_root: Path, dst_root: Path, video_mode: str) -> None:
    src_videos = src_root / "videos"
    if not src_videos.exists():
        return

    for src_path in src_videos.rglob("*"):
        if src_path.is_dir():
            continue
        rel = src_path.relative_to(src_root)
        dst_path = dst_root / rel
        _link_or_copy(src_path, dst_path, mode=video_mode)


def _prepare_split_specs(info: dict, spaces: set[str]) -> dict[str, SplitSpec]:
    obs_ft = info["features"].get("observation.state")
    act_ft = info["features"].get("action")
    if obs_ft is None or act_ft is None:
        raise ValueError("Dataset must include both 'observation.state' and 'action' features.")

    obs_names = list(obs_ft.get("names") or [])
    act_names = list(act_ft.get("names") or [])

    if not obs_names:
        raise ValueError("Missing names for 'observation.state' in meta/info.json.")
    if not act_names:
        raise ValueError("Missing names for 'action' in meta/info.json.")

    obs_name_to_idx = {name: i for i, name in enumerate(obs_names)}

    joint_indices_in_obs = []
    for name in act_names:
        if name not in obs_name_to_idx:
            raise ValueError(f"Action name '{name}' not found in observation.state names.")
        joint_indices_in_obs.append(obs_name_to_idx[name])

    ee_indices_in_obs = [
        i for i, name in enumerate(obs_names) if name.startswith("left_ee.") or name.startswith("right_ee.")
    ]
    ee_names = [obs_names[i] for i in ee_indices_in_obs]

    specs: dict[str, SplitSpec] = {}

    if "joint" in spaces:
        specs["joint"] = SplitSpec(
            space="joint",
            observation_indices=joint_indices_in_obs,
            observation_names=[obs_names[i] for i in joint_indices_in_obs],
            action_mode="keep",
            action_indices=None,
            action_names=act_names,
        )

    if "ee" in spaces:
        if not ee_indices_in_obs:
            raise ValueError(
                "Requested EE split, but no EE names found in observation.state. "
                "Did you record with record_ee_pose=true?"
            )
        specs["ee"] = SplitSpec(
            space="ee",
            observation_indices=ee_indices_in_obs,
            observation_names=ee_names,
            action_mode="shift_from_observation",
            action_indices=None,
            action_names=ee_names,
        )

    return specs


def _build_output_info(src_info: dict, spec: SplitSpec) -> dict:
    dst_info = deepcopy(src_info)
    dst_info["features"]["observation.state"] = {
        **dst_info["features"]["observation.state"],
        "shape": [len(spec.observation_indices)],
        "names": spec.observation_names,
    }

    if spec.action_mode == "keep":
        return dst_info

    if spec.action_mode == "shift_from_observation":
        dst_info["features"]["action"] = {
            **dst_info["features"]["action"],
            "shape": [len(spec.observation_indices)],
            "names": spec.action_names,
        }
        return dst_info

    raise ValueError(f"Unknown action_mode: {spec.action_mode}")


def _split_data_joint(src_root: Path, dst_root: Path, dst_features: dict, observation_indices: list[int]) -> None:
    for src_path in _sorted_parquet_files(src_root, DATA_DIR):
        df = pd.read_parquet(src_path).reset_index(drop=True)
        obs = np.stack(df["observation.state"].to_list())
        obs = obs[:, observation_indices]
        df["observation.state"] = list(obs)

        rel = src_path.relative_to(src_root)
        dst_path = dst_root / rel
        _write_data_parquet(df, dst_path, dst_features)


def _split_data_ee(
    src_root: Path, dst_root: Path, dst_features: dict, observation_indices: list[int]
) -> None:
    pending_df: pd.DataFrame | None = None
    pending_dst_path: Path | None = None
    pending_last_episode: int | None = None

    for src_path in _sorted_parquet_files(src_root, DATA_DIR):
        df = pd.read_parquet(src_path).reset_index(drop=True)
        df = df.sort_values("index").reset_index(drop=True)

        obs = np.stack(df["observation.state"].to_list())
        ee_obs = obs[:, observation_indices]
        ep = df["episode_index"].to_numpy()

        action = np.empty_like(ee_obs)
        action[:-1] = ee_obs[1:]
        boundary = ep[:-1] != ep[1:]
        if np.any(boundary):
            action[:-1][boundary] = ee_obs[:-1][boundary]
        action[-1] = ee_obs[-1]

        df["observation.state"] = list(ee_obs)
        df["action"] = list(action)

        rel = src_path.relative_to(src_root)
        dst_path = dst_root / rel

        if pending_df is not None and pending_dst_path is not None and pending_last_episode is not None:
            first_ep = int(df["episode_index"].iloc[0])
            if first_ep == pending_last_episode:
                pending_df.at[len(pending_df) - 1, "action"] = df["observation.state"].iloc[0]
            _write_data_parquet(pending_df, pending_dst_path, dst_features)

        pending_df = df
        pending_dst_path = dst_path
        pending_last_episode = int(df["episode_index"].iloc[-1])

    if pending_df is not None and pending_dst_path is not None:
        _write_data_parquet(pending_df, pending_dst_path, dst_features)


def _compute_episode_stats_for_state_action(
    dataset_root: Path,
    features: dict,
) -> tuple[dict[int, dict], dict]:
    data_files = _sorted_parquet_files(dataset_root, DATA_DIR)

    feature_subset = {k: features[k] for k in ("observation.state", "action")}

    per_episode: dict[int, dict] = {}
    current_ep: int | None = None
    buffers: dict[str, list[np.ndarray]] = {"observation.state": [], "action": []}

    for path in data_files:
        df = pd.read_parquet(path, columns=["episode_index", "observation.state", "action"])
        for ep_idx, obs, act in zip(
            df["episode_index"].to_list(),
            df["observation.state"].to_list(),
            df["action"].to_list(),
            strict=True,
        ):
            ep_idx = int(ep_idx)
            if current_ep is None:
                current_ep = ep_idx

            if ep_idx != current_ep:
                episode_data = {k: np.stack(v) for k, v in buffers.items()}
                per_episode[current_ep] = compute_episode_stats(episode_data, feature_subset)
                buffers = {"observation.state": [], "action": []}
                current_ep = ep_idx

            buffers["observation.state"].append(obs)
            buffers["action"].append(act)

    if current_ep is not None:
        episode_data = {k: np.stack(v) for k, v in buffers.items()}
        per_episode[current_ep] = compute_episode_stats(episode_data, feature_subset)

    aggregated = aggregate_stats([per_episode[k] for k in sorted(per_episode)])
    return per_episode, aggregated


def _update_episode_metadata_stats(
    dst_root: Path,
    per_episode_stats: dict[int, dict],
) -> None:
    def _ensure_1d(value: object) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return np.atleast_1d(value)
        return np.atleast_1d(np.asarray(value))

    for ep_path in _sorted_parquet_files(dst_root, EPISODES_DIR):
        df = pd.read_parquet(ep_path).reset_index(drop=True)
        for row_idx, ep_idx in enumerate(df["episode_index"].to_list()):
            ep_idx = int(ep_idx)
            if ep_idx not in per_episode_stats:
                raise KeyError(f"Missing stats for episode_index={ep_idx}")
            stats = per_episode_stats[ep_idx]
            for feature_key in ("observation.state", "action"):
                for stat_name, value in stats[feature_key].items():
                    df.at[row_idx, f"stats/{feature_key}/{stat_name}"] = value

        # Pandas may coerce 1-element arrays to 0-dim ndarrays when assigning into cells.
        # Parquet writing then fails because it expects 1D array values for list-like columns.
        for col in ("stats/observation.state/count", "stats/action/count"):
            if col in df.columns:
                df[col] = df[col].apply(_ensure_1d)

        ep_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(ep_path)


def _copy_meta_episodes(src_root: Path, dst_root: Path) -> None:
    src_dir = src_root / EPISODES_DIR
    if not src_dir.exists():
        raise FileNotFoundError(f"Missing episodes metadata dir: {src_dir}")

    for src_path in _sorted_parquet_files(src_root, EPISODES_DIR):
        rel = src_path.relative_to(src_root)
        dst_path = dst_root / rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)


def _copy_tasks(src_root: Path, dst_root: Path) -> None:
    src_tasks = src_root / "meta/tasks.parquet"
    if not src_tasks.exists():
        raise FileNotFoundError(f"Missing tasks file: {src_tasks}")
    dst_tasks = dst_root / "meta/tasks.parquet"
    dst_tasks.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_tasks, dst_tasks)


def _ensure_empty_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output dir already exists: {path} (use --overwrite to replace)")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)


def _split_one_dataset(
    src_root: Path,
    dst_root: Path,
    spec: SplitSpec,
    video_mode: str,
    overwrite: bool,
) -> None:
    src_info = load_info(src_root)
    dst_info = _build_output_info(src_info, spec)

    _ensure_empty_dir(dst_root, overwrite=overwrite)
    write_info(dst_info, dst_root)
    _copy_tasks(src_root, dst_root)
    _copy_meta_episodes(src_root, dst_root)

    dst_features = dst_info["features"]
    if spec.space == "joint":
        _split_data_joint(src_root, dst_root, dst_features, observation_indices=spec.observation_indices)
    elif spec.space == "ee":
        _split_data_ee(src_root, dst_root, dst_features, observation_indices=spec.observation_indices)
    else:
        raise ValueError(f"Unknown space: {spec.space}")

    _copy_videos(src_root, dst_root, video_mode=video_mode)

    per_episode_stats, aggregated = _compute_episode_stats_for_state_action(dst_root, dst_features)
    _update_episode_metadata_stats(dst_root, per_episode_stats)

    src_stats = load_stats(src_root)
    if src_stats is None:
        raise FileNotFoundError(f"Missing meta/stats.json in: {src_root}")

    dst_stats = deepcopy(src_stats)
    dst_stats["observation.state"] = aggregated["observation.state"]
    dst_stats["action"] = aggregated["action"]
    write_stats_json(dst_stats, dst_root)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Split a v3.0 LeRobot dataset into joint-space and/or EE-space variants.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Dataset root containing meta/, data/, videos/ (e.g. ~/.cache/huggingface/lerobot/<repo_id>).",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=REPO_ROOT / "outputs/split_datasets",
        help="Base directory to write the split datasets into.",
    )
    parser.add_argument(
        "--spaces",
        nargs="+",
        choices=["joint", "ee", "both"],
        default=["both"],
        help="Which split(s) to generate.",
    )
    parser.add_argument(
        "--video-mode",
        choices=["auto", "hardlink", "symlink", "copy"],
        default="auto",
        help="How to place videos into the output dataset(s).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output directories if they exist.")

    args = parser.parse_args()

    src_root: Path = args.input_root.expanduser().resolve()
    if not (src_root / "meta/info.json").exists():
        raise FileNotFoundError(f"Not a dataset root (missing meta/info.json): {src_root}")

    spaces = set(args.spaces)
    if "both" in spaces:
        spaces = {"joint", "ee"}

    src_info = load_info(src_root)
    specs = _prepare_split_specs(src_info, spaces)

    dataset_name = src_root.name
    out_base = args.output_base.expanduser().resolve()
    out_base.mkdir(parents=True, exist_ok=True)

    for space, spec in specs.items():
        dst_root = out_base / f"{dataset_name}_{space}"
        print(f"[split] {space}: {src_root} -> {dst_root}")
        _split_one_dataset(
            src_root=src_root,
            dst_root=dst_root,
            spec=spec,
            video_mode=args.video_mode,
            overwrite=args.overwrite,
        )
        print(f"[done]  {space}: {dst_root}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise
