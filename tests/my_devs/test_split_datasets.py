from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from my_devs.split_datasets.split_bimanual_lerobot_dataset import SplitConfig, split_dataset
from my_devs.split_datasets.validate_split_dataset import DatasetSpec, validate_dataset


ACTION_KEY = "action"
STATE_KEY = "observation.state"
FRONT_CAMERA_KEY = "observation.images.camera_front"
LEFT_CAMERA_KEY = "observation.images.camera_left"
RIGHT_CAMERA_KEY = "observation.images.camera_right"


def make_source_dataset(root: Path, repo_id: str) -> Path:
    features = {
        ACTION_KEY: {
            "dtype": "float32",
            "shape": (14,),
            "names": [f"left_joint{i}.pos" for i in range(7)] + [f"right_joint{i}.pos" for i in range(7)],
        },
        STATE_KEY: {
            "dtype": "float32",
            "shape": (14,),
            "names": [f"left_joint{i}.pos" for i in range(7)] + [f"right_joint{i}.pos" for i in range(7)],
        },
        FRONT_CAMERA_KEY: {"dtype": "image", "shape": (4, 5, 3), "names": ["height", "width", "channels"]},
        LEFT_CAMERA_KEY: {"dtype": "image", "shape": (4, 5, 3), "names": ["height", "width", "channels"]},
        RIGHT_CAMERA_KEY: {"dtype": "image", "shape": (4, 5, 3), "names": ["height", "width", "channels"]},
    }
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=5,
        features=features,
        root=root / repo_id,
        robot_type="agilex",
        use_videos=False,
        vcodec="h264",
    )
    try:
        for episode in range(2):
            states = []
            for frame in range(3):
                base = episode * 100 + frame * 10
                states.append(np.arange(base, base + 14, dtype=np.float32))
            for frame in range(3):
                action = states[min(frame + 1, 2)]
                state = states[frame]
                image = np.full((4, 5, 3), fill_value=episode * 20 + frame, dtype=np.uint8)
                dataset.add_frame(
                    {
                        "task": "synthetic task",
                        ACTION_KEY: action,
                        STATE_KEY: state,
                        FRONT_CAMERA_KEY: image,
                        LEFT_CAMERA_KEY: image + 1,
                        RIGHT_CAMERA_KEY: image + 2,
                    }
                )
            dataset.episode_buffer["timestamp"] = [frame / 5 for frame in range(3)]
            dataset.save_episode()
    finally:
        dataset.finalize()
    return root / repo_id


def test_split_dataset_and_validate(tmp_path: Path):
    source_root = tmp_path / "source_root"
    target_root = tmp_path / "target_root"
    source_repo_id = "dummy/source_bimanual"
    left_repo_id = "dummy/source_left"
    right_repo_id = "dummy/source_right"

    make_source_dataset(source_root, source_repo_id)

    left_path, right_path = split_dataset(
        SplitConfig(
            source_root=source_root,
            source_repo_id=source_repo_id,
            target_root=target_root,
            left_repo_id=left_repo_id,
            right_repo_id=right_repo_id,
            overwrite=False,
            task_mode="copy",
            task_text=None,
            left_task_text=None,
            right_task_text=None,
            episode_indices=None,
            vcodec="h264",
        )
    )

    assert left_path.exists()
    assert right_path.exists()

    left_info = json.loads((left_path / "meta" / "info.json").read_text())
    right_info = json.loads((right_path / "meta" / "info.json").read_text())
    assert left_info["features"][ACTION_KEY]["shape"] == [7]
    assert right_info["features"][ACTION_KEY]["shape"] == [7]
    assert sorted(k for k in left_info["features"] if k.startswith("observation.images.")) == [
        FRONT_CAMERA_KEY,
        LEFT_CAMERA_KEY,
    ]
    assert sorted(k for k in right_info["features"] if k.startswith("observation.images.")) == [
        FRONT_CAMERA_KEY,
        RIGHT_CAMERA_KEY,
    ]

    left_result = validate_dataset(
        source_dir=source_root / source_repo_id,
        target_dir=left_path,
        spec=DatasetSpec(
            side="left",
            repo_id=left_repo_id,
            vector_slice=slice(0, 7),
            camera_keys=(FRONT_CAMERA_KEY, LEFT_CAMERA_KEY),
        ),
        video_backend="pyav",
    )
    right_result = validate_dataset(
        source_dir=source_root / source_repo_id,
        target_dir=right_path,
        spec=DatasetSpec(
            side="right",
            repo_id=right_repo_id,
            vector_slice=slice(7, 14),
            camera_keys=(FRONT_CAMERA_KEY, RIGHT_CAMERA_KEY),
        ),
        video_backend="pyav",
    )

    assert left_result.lag1_max == 0.0
    assert right_result.lag1_max == 0.0
