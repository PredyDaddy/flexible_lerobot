#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EXPECTED_JOINT_NAMES = [
    *[f"left_left_joint{i}.pos" for i in range(1, 8)],
    *[f"right_right_joint{i}.pos" for i in range(1, 8)],
]
EXPECTED_GRIPPER_NAMES = [
    "left_gripper.width",
    "left_gripper.force",
    "right_gripper.width",
    "right_gripper.force",
]
EXPECTED_NAMES = [*EXPECTED_JOINT_NAMES, *EXPECTED_GRIPPER_NAMES]
REQUIRED_CAMERA_KEYS = [
    "observation.images.camera_head",
    "observation.images.camera_left",
    "observation.images.camera_right",
]
EXPECTED_CAMERA_SHAPES = {
    "observation.images.camera_head": [720, 1280, 3],
    "observation.images.camera_left": [480, 640, 3],
    "observation.images.camera_right": [480, 640, 3],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate three jz_robot_pin LeRobot episodes.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, default=3)
    parser.add_argument("--expected-dim", type=int, default=18)
    parser.add_argument("--joint-dim", type=int, default=14)
    parser.add_argument("--expected-fps", type=int, default=30)
    parser.add_argument("--expected-episode-time-s", type=float, default=10.0)
    parser.add_argument("--min-frame-ratio", type=float, default=0.9)
    parser.add_argument("--max-initial-joint-delta-rad", type=float, default=10.0)
    parser.add_argument("--max-action-joint-step-rad", type=float, default=10.0)
    parser.add_argument("--lag-min", type=int, default=1)
    parser.add_argument("--lag-max", type=int, default=6)
    parser.add_argument("--max-lag-mae-rad", type=float, default=0.01)
    parser.add_argument("--max-lag-p95-rad", type=float, default=0.03)
    parser.add_argument("--moving-step-threshold-rad", type=float, default=0.001)
    parser.add_argument("--expected-robot-type", default="jz_robot_pin")
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument("--allow-no-video", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def read_parquet_tree(root: Path, pattern: str) -> pd.DataFrame:
    files = sorted(root.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquet files matched {root / pattern}")
    return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)


def stack_vector_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        raise KeyError(f"Dataset is missing vector column {column!r}")
    values = np.stack(frame[column].to_numpy())
    if values.ndim != 2:
        raise ValueError(f"Dataset column {column!r} must be rank 2 after stacking, got {values.shape}")
    return values


def add_error(report: dict[str, Any], message: str) -> None:
    report["errors"].append(message)


def add_warning(report: dict[str, Any], message: str) -> None:
    report["warnings"].append(message)


def validate_feature_schema(
    info: dict[str, Any], expected_dim: int, report: dict[str, Any]
) -> tuple[list[str], list[str]]:
    features = info.get("features", {})
    action_feature = features.get("action", {})
    state_feature = features.get("observation.state", {})
    action_shape = action_feature.get("shape")
    state_shape = state_feature.get("shape")
    action_names = list(action_feature.get("names") or [])
    state_names = list(state_feature.get("names") or [])

    if action_shape != [expected_dim]:
        add_error(report, f"action metadata shape must be [{expected_dim}], got {action_shape}")
    if state_shape != [expected_dim]:
        add_error(report, f"observation.state metadata shape must be [{expected_dim}], got {state_shape}")
    if action_names != state_names:
        add_error(report, "action and observation.state metadata names/order differ")
    if expected_dim == len(EXPECTED_NAMES) and action_names != EXPECTED_NAMES:
        add_error(report, f"18-dimensional action names/order differ from expected: {action_names}")

    return action_names, state_names


def validate_video_files(
    root: Path, info: dict[str, Any], episodes_meta: pd.DataFrame, report: dict[str, Any]
) -> None:
    features = info.get("features", {})
    for camera_key in REQUIRED_CAMERA_KEYS:
        feature = features.get(camera_key)
        if feature is None:
            add_error(report, f"missing required camera feature {camera_key}")
            continue
        if feature.get("dtype") != "video":
            add_error(report, f"camera feature {camera_key} must use video dtype")
        if feature.get("shape") != EXPECTED_CAMERA_SHAPES[camera_key]:
            add_error(
                report,
                f"camera feature {camera_key} shape must be {EXPECTED_CAMERA_SHAPES[camera_key]}, "
                f"got {feature.get('shape')}",
            )

        chunk_column = f"videos/{camera_key}/chunk_index"
        file_column = f"videos/{camera_key}/file_index"
        if chunk_column not in episodes_meta or file_column not in episodes_meta:
            add_error(report, f"episode metadata is missing video location for {camera_key}")
            continue

        for chunk_index, file_index in zip(
            episodes_meta[chunk_column], episodes_meta[file_column], strict=True
        ):
            chunk = int(np.asarray(chunk_index).reshape(-1)[0])
            file = int(np.asarray(file_index).reshape(-1)[0])
            video_path = root / "videos" / camera_key / f"chunk-{chunk:03d}" / f"file-{file:03d}.mp4"
            if not video_path.is_file() or video_path.stat().st_size <= 0:
                add_error(report, f"missing or empty video file: {video_path}")


def lag_metrics(
    action_joints: np.ndarray,
    state_joints: np.ndarray,
    moving_mask: np.ndarray,
    lag_min: int,
    lag_max: int,
) -> list[dict[str, float | int]]:
    metrics: list[dict[str, float | int]] = []
    for lag in range(lag_min, lag_max + 1):
        if len(action_joints) <= lag:
            continue
        mask = moving_mask[:-lag]
        if not np.any(mask):
            continue
        absolute_error = np.abs(action_joints[:-lag][mask] - state_joints[lag:][mask])
        metrics.append(
            {
                "lag_frames": lag,
                "pairs": int(mask.sum()),
                "mae_rad": float(absolute_error.mean()),
                "p95_rad": float(np.quantile(absolute_error, 0.95)),
                "max_rad": float(absolute_error.max()),
            }
        )
    return metrics


def validate_episode(
    episode_index: int,
    episode: pd.DataFrame,
    args: argparse.Namespace,
    report: dict[str, Any],
) -> dict[str, Any]:
    episode = episode.sort_values("frame_index").reset_index(drop=True)
    metrics: dict[str, Any] = {"episode_index": episode_index, "frames": int(len(episode))}
    minimum_frames = math.floor(
        args.expected_fps * args.expected_episode_time_s * args.min_frame_ratio
    )
    if len(episode) < minimum_frames:
        add_error(
            report,
            f"episode {episode_index} has {len(episode)} frames; expected at least {minimum_frames}",
        )

    frame_index = episode["frame_index"].to_numpy(dtype=np.int64)
    expected_frame_index = np.arange(len(episode), dtype=np.int64)
    if not np.array_equal(frame_index, expected_frame_index):
        add_error(report, f"episode {episode_index} frame_index is not contiguous from zero")

    timestamp = episode["timestamp"].to_numpy(dtype=np.float64)
    expected_timestamp = expected_frame_index / args.expected_fps
    timestamp_error = float(np.max(np.abs(timestamp - expected_timestamp))) if len(timestamp) else 0.0
    metrics["max_timestamp_error_s"] = timestamp_error
    if timestamp_error > 1e-3:
        add_error(report, f"episode {episode_index} timestamp grid error is {timestamp_error:.6f}s")

    action = stack_vector_column(episode, "action")
    state = stack_vector_column(episode, "observation.state")
    metrics["action_shape"] = list(action.shape)
    metrics["state_shape"] = list(state.shape)
    if action.shape[1] != args.expected_dim or state.shape[1] != args.expected_dim:
        add_error(
            report,
            f"episode {episode_index} expected {args.expected_dim}D action/state, "
            f"got {action.shape[1]}D/{state.shape[1]}D",
        )
        return metrics
    if action.dtype != np.float32:
        add_error(report, f"episode {episode_index} action dtype must be float32, got {action.dtype}")
    if state.dtype != np.float32:
        add_error(
            report,
            f"episode {episode_index} observation.state dtype must be float32, got {state.dtype}",
        )
    if not np.isfinite(action).all() or not np.isfinite(state).all():
        add_error(report, f"episode {episode_index} contains non-finite action/state values")

    action_joints = action[:, : args.joint_dim].astype(np.float64)
    state_joints = state[:, : args.joint_dim].astype(np.float64)
    initial_delta = float(np.max(np.abs(action_joints[0] - state_joints[0])))
    metrics["initial_max_joint_delta_rad"] = initial_delta
    if initial_delta > args.max_initial_joint_delta_rad + 1e-9:
        add_error(
            report,
            f"episode {episode_index} initial joint delta {initial_delta:.6f}rad exceeds "
            f"{args.max_initial_joint_delta_rad:.6f}rad",
        )

    if len(action_joints) > 1:
        action_steps = np.max(np.abs(np.diff(action_joints, axis=0)), axis=1)
        max_action_step = float(action_steps.max())
        moving_mask = np.concatenate(
            ([False], action_steps > args.moving_step_threshold_rad)
        )
    else:
        max_action_step = 0.0
        moving_mask = np.zeros(len(action_joints), dtype=bool)
    metrics["max_action_joint_step_rad"] = max_action_step
    metrics["moving_frames"] = int(moving_mask.sum())
    if max_action_step > args.max_action_joint_step_rad + 1e-6:
        add_error(
            report,
            f"episode {episode_index} action step {max_action_step:.6f}rad exceeds "
            f"{args.max_action_joint_step_rad:.6f}rad",
        )
    if int(moving_mask.sum()) < max(5, int(0.01 * len(episode))):
        add_error(report, f"episode {episode_index} contains too few moving joint frames")

    same_frame_error = np.max(np.abs(action_joints - state_joints), axis=1)
    metrics["hold_current_fraction_1e_3"] = float(np.mean(same_frame_error <= 1e-3))

    per_lag = lag_metrics(
        action_joints,
        state_joints,
        moving_mask,
        args.lag_min,
        args.lag_max,
    )
    metrics["lag_metrics"] = per_lag
    if not per_lag:
        add_error(report, f"episode {episode_index} has no moving samples for lag validation")
    else:
        best = min(per_lag, key=lambda item: (item["mae_rad"], item["p95_rad"]))
        metrics["best_lag"] = best
        if best["mae_rad"] > args.max_lag_mae_rad:
            add_error(
                report,
                f"episode {episode_index} best lag MAE {best['mae_rad']:.6f}rad exceeds "
                f"{args.max_lag_mae_rad:.6f}rad",
            )
        if best["p95_rad"] > args.max_lag_p95_rad:
            add_error(
                report,
                f"episode {episode_index} best lag P95 {best['p95_rad']:.6f}rad exceeds "
                f"{args.max_lag_p95_rad:.6f}rad",
            )

    grippers = action[:, args.joint_dim :]
    if grippers.size and (float(grippers.min()) < -1e-6 or float(grippers.max()) > 100.000001):
        add_error(report, f"episode {episode_index} action gripper values are outside [0, 100]")
    state_grippers = state[:, args.joint_dim :]
    if state_grippers.size and (
        float(state_grippers.min()) < -1e-6 or float(state_grippers.max()) > 100.000001
    ):
        add_error(
            report,
            f"episode {episode_index} observation gripper values are outside [0, 100]",
        )
    if metrics["hold_current_fraction_1e_3"] > 0.5:
        add_warning(
            report,
            f"episode {episode_index} has {metrics['hold_current_fraction_1e_3']:.1%} "
            "same-frame hold-like samples",
        )
    return metrics


def run_check(args: argparse.Namespace) -> dict[str, Any]:
    root = args.dataset_root.expanduser().resolve()
    report: dict[str, Any] = {
        "dataset_root": str(root),
        "status": "FAIL",
        "errors": [],
        "warnings": [],
        "episodes": [],
        "thresholds": {
            "expected_episodes": args.expected_episodes,
            "expected_dim": args.expected_dim,
            "joint_dim": args.joint_dim,
            "max_initial_joint_delta_rad": args.max_initial_joint_delta_rad,
            "max_action_joint_step_rad": args.max_action_joint_step_rad,
            "lag_min": args.lag_min,
            "lag_max": args.lag_max,
            "max_lag_mae_rad": args.max_lag_mae_rad,
            "max_lag_p95_rad": args.max_lag_p95_rad,
        },
    }
    required_files = [
        root / "meta/info.json",
        root / "meta/stats.json",
        root / "meta/tasks.parquet",
    ]
    for required in required_files:
        if not required.is_file():
            add_error(report, f"missing required dataset file: {required}")
    if report["errors"]:
        return report

    info = read_json(root / "meta/info.json")
    report["robot_type"] = info.get("robot_type")
    report["fps"] = info.get("fps")
    report["total_episodes"] = info.get("total_episodes")
    report["total_frames"] = info.get("total_frames")
    if info.get("robot_type") != args.expected_robot_type:
        add_error(
            report,
            f"robot_type must be {args.expected_robot_type!r}, got {info.get('robot_type')!r}",
        )
    if info.get("fps") != args.expected_fps:
        add_error(report, f"dataset fps must be {args.expected_fps}, got {info.get('fps')}")
    if info.get("total_episodes") != args.expected_episodes:
        add_error(
            report,
            f"dataset must contain exactly {args.expected_episodes} episodes, "
            f"got {info.get('total_episodes')}",
        )
    validate_feature_schema(info, args.expected_dim, report)

    try:
        data = read_parquet_tree(root, "data/chunk-*/file-*.parquet")
        episodes_meta = read_parquet_tree(root, "meta/episodes/chunk-*/file-*.parquet")
    except (FileNotFoundError, OSError, ValueError) as exc:
        add_error(report, str(exc))
        return report

    required_data_columns = {
        "action",
        "observation.state",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
    }
    missing_data_columns = sorted(required_data_columns - set(data.columns))
    if missing_data_columns:
        add_error(report, f"dataset parquet is missing columns: {missing_data_columns}")
        return report
    required_episode_columns = {"episode_index", "length"}
    missing_episode_columns = sorted(required_episode_columns - set(episodes_meta.columns))
    if missing_episode_columns:
        add_error(report, f"episode metadata is missing columns: {missing_episode_columns}")
        return report

    if int(info.get("total_frames", -1)) != len(data):
        add_error(
            report,
            f"info total_frames={info.get('total_frames')} does not match parquet rows={len(data)}",
        )
    if "index" not in data:
        add_error(report, "dataset is missing global index column")
    else:
        global_index = data.sort_values("index")["index"].to_numpy(dtype=np.int64)
        if not np.array_equal(global_index, np.arange(len(data), dtype=np.int64)):
            add_error(report, "global dataset index is not contiguous from zero")
    episode_ids = sorted(int(value) for value in data["episode_index"].unique())
    expected_episode_ids = list(range(args.expected_episodes))
    if episode_ids != expected_episode_ids:
        add_error(report, f"episode ids must be {expected_episode_ids}, got {episode_ids}")
    meta_episode_ids = sorted(int(value) for value in episodes_meta["episode_index"].unique())
    if meta_episode_ids != expected_episode_ids:
        add_error(report, f"episode metadata ids must be {expected_episode_ids}, got {meta_episode_ids}")

    if not args.allow_no_video:
        validate_video_files(root, info, episodes_meta, report)

    for episode_index in expected_episode_ids:
        episode = data[data["episode_index"] == episode_index]
        if episode.empty:
            add_error(report, f"episode {episode_index} has no data rows")
            continue
        episode_metrics = validate_episode(episode_index, episode, args, report)
        report["episodes"].append(episode_metrics)
        meta_rows = episodes_meta[episodes_meta["episode_index"] == episode_index]
        if len(meta_rows) == 1 and int(meta_rows.iloc[0]["length"]) != len(episode):
            add_error(
                report,
                f"episode {episode_index} metadata length={int(meta_rows.iloc[0]['length'])} "
                f"does not match data rows={len(episode)}",
            )

    report["status"] = "PASS" if not report["errors"] else "FAIL"
    return report


def main() -> int:
    args = parse_args()
    report = run_check(args)
    report_path = args.report_json or args.dataset_root / "data_check_report.json"
    report_path = report_path.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")

    print(f"dataset_root={report['dataset_root']}")
    print(f"status={report['status']}")
    for episode in report["episodes"]:
        best = episode.get("best_lag")
        lag_one = next(
            (item for item in episode.get("lag_metrics", []) if item["lag_frames"] == 1),
            None,
        )
        lag_one_text = "none" if lag_one is None else (
            f"mae={lag_one['mae_rad']:.6f}rad p95={lag_one['p95_rad']:.6f}rad"
        )
        best_text = "none" if best is None else (
            f"lag={best['lag_frames']} mae={best['mae_rad']:.6f}rad p95={best['p95_rad']:.6f}rad"
        )
        print(
            f"episode={episode['episode_index']} frames={episode['frames']} "
            f"initial_delta={episode.get('initial_max_joint_delta_rad')} "
            f"max_step={episode.get('max_action_joint_step_rad')} lag1={lag_one_text} "
            f"best_lag={best_text}"
        )
    for warning in report["warnings"]:
        print(f"WARNING: {warning}")
    for error in report["errors"]:
        print(f"ERROR: {error}")
    print(f"report_json={report_path}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
