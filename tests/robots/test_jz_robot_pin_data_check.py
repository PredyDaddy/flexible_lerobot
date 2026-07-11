from __future__ import annotations

import json
from argparse import Namespace

import numpy as np
import pandas as pd

from my_devs.jz_robot_pin.data_check.check_3_episodes import (
    EXPECTED_CAMERA_SHAPES,
    EXPECTED_NAMES,
    REQUIRED_CAMERA_KEYS,
    main,
    run_check,
)


def make_args(dataset_root) -> Namespace:
    return Namespace(
        dataset_root=dataset_root,
        expected_episodes=3,
        expected_dim=18,
        joint_dim=14,
        expected_fps=10,
        expected_episode_time_s=1.0,
        min_frame_ratio=0.9,
        max_initial_joint_delta_rad=10.0,
        max_action_joint_step_rad=10.0,
        lag_min=1,
        lag_max=3,
        max_lag_mae_rad=0.01,
        max_lag_p95_rad=0.03,
        moving_step_threshold_rad=0.001,
        expected_robot_type="jz_robot_pin",
        report_json=None,
        allow_no_video=False,
    )


def write_synthetic_dataset(root, *, action_dim: int = 18) -> None:
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)

    frames = []
    episode_rows = []
    global_index = 0
    for episode_index in range(3):
        action = np.zeros((10, action_dim), dtype=np.float32)
        state = np.zeros((10, action_dim), dtype=np.float32)
        for frame_index in range(10):
            action[frame_index, :14] = frame_index * 0.005
            action[frame_index, 14:] = 50.0
            state[frame_index, 14:] = 50.0
            if frame_index > 0:
                state[frame_index, :14] = action[frame_index - 1, :14]
        for frame_index in range(10):
            frames.append(
                {
                    "action": action[frame_index],
                    "observation.state": state[frame_index],
                    "timestamp": frame_index / 10,
                    "frame_index": frame_index,
                    "episode_index": episode_index,
                    "index": global_index,
                    "task_index": 0,
                }
            )
            global_index += 1
        episode_row = {
            "episode_index": episode_index,
            "length": 10,
            "dataset_from_index": episode_index * 10,
            "dataset_to_index": (episode_index + 1) * 10,
        }
        for camera_key in REQUIRED_CAMERA_KEYS:
            episode_row[f"videos/{camera_key}/chunk_index"] = 0
            episode_row[f"videos/{camera_key}/file_index"] = 0
        episode_rows.append(episode_row)

    pd.DataFrame(frames).to_parquet(root / "data/chunk-000/file-000.parquet")
    pd.DataFrame(episode_rows).to_parquet(root / "meta/episodes/chunk-000/file-000.parquet")
    pd.DataFrame({"task_index": [0], "task": ["test"]}).to_parquet(root / "meta/tasks.parquet")
    (root / "meta/stats.json").write_text("{}\n", encoding="utf-8")
    features = {
        "action": {"dtype": "float32", "shape": [action_dim], "names": EXPECTED_NAMES[:action_dim]},
        "observation.state": {
            "dtype": "float32",
            "shape": [action_dim],
            "names": EXPECTED_NAMES[:action_dim],
        },
    }
    for camera_key in REQUIRED_CAMERA_KEYS:
        features[camera_key] = {
            "dtype": "video",
            "shape": EXPECTED_CAMERA_SHAPES[camera_key],
            "names": ["height", "width", "channels"],
        }
        video_path = root / "videos" / camera_key / "chunk-000/file-000.mp4"
        video_path.parent.mkdir(parents=True)
        video_path.write_bytes(b"synthetic-video-placeholder")
    info = {
        "robot_type": "jz_robot_pin",
        "total_episodes": 3,
        "total_frames": 30,
        "fps": 10,
        "features": features,
    }
    (root / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")


def set_initial_joint_delta(root, delta: float) -> None:
    data_path = root / "data/chunk-000/file-000.parquet"
    data = pd.read_parquet(data_path)
    actions = []
    states = []
    for _, row in data.iterrows():
        action = np.asarray(row["action"], dtype=np.float32).copy()
        state = np.asarray(row["observation.state"], dtype=np.float32).copy()
        action[0] += delta
        if row["frame_index"] > 0:
            state[0] += delta
        actions.append(action)
        states.append(state)
    data["action"] = actions
    data["observation.state"] = states
    data.to_parquet(data_path)


def test_three_episode_checker_accepts_valid_18d_dataset(tmp_path) -> None:
    root = tmp_path / "valid"
    write_synthetic_dataset(root)

    report = run_check(make_args(root))

    assert report["status"] == "PASS"
    assert report["errors"] == []
    assert len(report["episodes"]) == 3
    assert all(episode["best_lag"]["lag_frames"] == 1 for episode in report["episodes"])


def test_three_episode_checker_uses_effectively_open_development_delta_limit(tmp_path) -> None:
    root = tmp_path / "initial_delta"
    write_synthetic_dataset(root)
    set_initial_joint_delta(root, 1.0)

    open_report = run_check(make_args(root))

    assert open_report["status"] == "PASS"

    strict_args = make_args(root)
    strict_args.max_initial_joint_delta_rad = 0.15
    strict_report = run_check(strict_args)

    assert strict_report["status"] == "FAIL"
    assert any("initial joint delta" in error for error in strict_report["errors"])


def test_three_episode_checker_rejects_wrong_vector_dimension(tmp_path) -> None:
    root = tmp_path / "wrong_dim"
    write_synthetic_dataset(root, action_dim=17)

    report = run_check(make_args(root))

    assert report["status"] == "FAIL"
    assert any("shape must be [18]" in error or "expected 18D" in error for error in report["errors"])


def test_three_episode_checker_reports_lag_threshold_failure(tmp_path) -> None:
    root = tmp_path / "bad_lag"
    write_synthetic_dataset(root)
    data_path = root / "data/chunk-000/file-000.parquet"
    data = pd.read_parquet(data_path)
    data["observation.state"] = data["observation.state"].map(
        lambda value: np.asarray(value, dtype=np.float32) + 0.1
    )
    data.to_parquet(data_path)

    report = run_check(make_args(root))

    assert report["status"] == "FAIL"
    assert any("best lag MAE" in error for error in report["errors"])


def test_three_episode_checker_cli_writes_json_report(tmp_path, monkeypatch) -> None:
    root = tmp_path / "cli"
    report_path = tmp_path / "report.json"
    write_synthetic_dataset(root)
    monkeypatch.setattr(
        "sys.argv",
        [
            "check_3_episodes.py",
            "--dataset-root",
            str(root),
            "--expected-fps",
            "10",
            "--expected-episode-time-s",
            "1",
            "--lag-max",
            "3",
            "--report-json",
            str(report_path),
        ],
    )

    assert main() == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert report["thresholds"]["max_initial_joint_delta_rad"] == 10.0
    assert report["thresholds"]["max_action_joint_step_rad"] == 10.0
