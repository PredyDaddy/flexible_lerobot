import json
from pathlib import Path

import numpy as np

from lerobot.robots.jz_robot.bridge_capture.convert_raw_to_lerobot import (
    RawToLeRobotConverter,
    build_lerobot_features,
)


class FakeLeRobotDataset:
    created_kwargs = None

    def __init__(self):
        self.frames = []
        self.saved_episodes = 0
        self.finalized = False

    @classmethod
    def create(cls, **kwargs):
        cls.created_kwargs = kwargs
        return cls()

    def add_frame(self, frame):
        self.frames.append(frame)

    def save_episode(self):
        self.saved_episodes += 1

    def finalize(self):
        self.finalized = True


def test_build_lerobot_features_uses_raw_metadata_camera_shapes():
    metadata = {
        "state_names": ["s1", "s2"],
        "action_names": ["a1"],
        "cameras": {
            "camera_head": {"height": 720, "width": 1280},
            "camera_left": {"height": 480, "width": 640},
        },
    }

    features = build_lerobot_features(metadata)

    assert features["observation.state"] == {
        "dtype": "float32",
        "shape": (2,),
        "names": ["s1", "s2"],
    }
    assert features["action"] == {"dtype": "float32", "shape": (1,), "names": ["a1"]}
    assert features["observation.images.camera_head"] == {
        "dtype": "image",
        "shape": (720, 1280, 3),
        "names": ["height", "width", "channel"],
    }


def test_converter_skips_invalid_samples_and_writes_lerobot_frames(tmp_path: Path):
    episode = _write_raw_episode(tmp_path, "000001")
    output_root = tmp_path / "lerobot"

    result = RawToLeRobotConverter(dataset_factory=FakeLeRobotDataset).convert(
        raw_episodes=[episode],
        repo_id="local/jz_test",
        output_root=output_root,
    )

    assert result.episodes_written == 1
    assert result.frames_written == 1
    assert FakeLeRobotDataset.created_kwargs["repo_id"] == "local/jz_test"
    assert FakeLeRobotDataset.created_kwargs["root"] == output_root
    assert FakeLeRobotDataset.created_kwargs["fps"] == 20

    dataset = result.dataset
    assert dataset.saved_episodes == 1
    assert dataset.finalized is True
    assert len(dataset.frames) == 1
    frame = dataset.frames[0]
    assert frame["task"] == "pick cube"
    assert frame["observation.state"].dtype == np.float32
    assert frame["observation.state"].tolist() == [0.1, 0.2]
    assert frame["action"].tolist() == [0.3]
    assert frame["observation.images.camera_head"].shape == (4, 5, 3)


def _write_raw_episode(root: Path, episode_id: str) -> Path:
    import cv2

    episode = root / f"episode_{episode_id}"
    frame_dir = episode / "frames" / "camera_head"
    frame_dir.mkdir(parents=True)
    image = np.full((4, 5, 3), 127, dtype=np.uint8)
    assert cv2.imwrite(str(frame_dir / "000000.jpg"), image)

    metadata = {
        "task": "pick cube",
        "sample_rate_hz": 20,
        "state_names": ["s1", "s2"],
        "action_names": ["a1"],
        "robot": {"name": "jz_dual_arm"},
        "cameras": {"camera_head": {"height": 4, "width": 5}},
    }
    (episode / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    samples = [
        {
            "sample_index": 0,
            "capture_time_ns": 0,
            "state": [0.1, 0.2],
            "action": [0.3],
            "camera_head_frame_path": "frames/camera_head/000000.jpg",
            "valid": True,
        },
        {
            "sample_index": 1,
            "capture_time_ns": 50_000_000,
            "state": [0.4, 0.5],
            "action": [0.6],
            "valid": False,
            "invalid_reason": "camera_head_missing",
        },
    ]
    (episode / "samples.jsonl").write_text(
        "\n".join(json.dumps(sample) for sample in samples),
        encoding="utf-8",
    )
    (episode / "events.jsonl").write_text("", encoding="utf-8")
    return episode
