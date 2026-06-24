import json
from pathlib import Path

from lerobot.robots.jz_robot.bridge_capture.validate_raw import validate_episode


def test_validate_episode_reports_dimension_and_validity(tmp_path: Path):
    episode = tmp_path / "episode_000001"
    episode.mkdir()
    (episode / "metadata.json").write_text(
        json.dumps(
            {
                "state_names": ["s1", "s2"],
                "action_names": ["a1"],
                "cameras": {},
            }
        ),
        encoding="utf-8",
    )
    (episode / "samples.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"sample_index": 0, "state": [0.1, 0.2], "action": [0.3], "valid": True}),
                json.dumps({"sample_index": 1, "state": [0.1], "action": [0.3], "valid": True}),
                json.dumps({"sample_index": 2, "state": [0.1, 0.2], "action": [], "valid": False}),
            ]
        ),
        encoding="utf-8",
    )

    report = validate_episode(episode)

    assert report.sample_count == 3
    assert report.valid_count == 2
    assert report.errors == ["sample 1 state dimension 1 != expected 2"]


def test_validate_episode_reports_missing_valid_camera_frame(tmp_path: Path):
    episode = tmp_path / "episode_000002"
    episode.mkdir()
    (episode / "metadata.json").write_text(
        json.dumps(
            {
                "state_names": ["s1"],
                "action_names": ["a1"],
                "cameras": {"camera_head": {"height": 4, "width": 5}},
            }
        ),
        encoding="utf-8",
    )
    (episode / "samples.jsonl").write_text(
        json.dumps({"sample_index": 0, "state": [0.1], "action": [0.2], "valid": True}),
        encoding="utf-8",
    )

    report = validate_episode(episode)

    assert report.valid_count == 1
    assert report.camera_frame_counts == {"camera_head": 0}
    assert report.errors == ["sample 0 missing frame path for camera: camera_head"]
