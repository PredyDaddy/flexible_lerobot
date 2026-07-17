from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from my_devs.jz_robot_pin_timed.web_collection_system.server import (
    ApiError,
    DatasetInfo,
    build_record_environment,
    build_replay_environment,
    discover_datasets,
    load_dataset,
    validate_record_root,
)

WEB_ROOT = Path(__file__).resolve().parents[1] / "my_devs" / "jz_robot_pin_timed" / "web_collection_system"


def write_dataset(
    root: Path, *, episodes: int = 3, frames: int = 60, robot_type: str = "jz_robot_pin_timed"
) -> None:
    meta = root / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps(
            {
                "robot_type": robot_type,
                "total_episodes": episodes,
                "total_frames": frames,
                "fps": 20,
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize("episode_count", [1, 5, 6, 17])
def test_record_environment_uses_requested_episode_count_as_encoding_batch(
    tmp_path: Path, episode_count: int
) -> None:
    dataset_root = tmp_path / "custom_recording"
    resolved = validate_record_root(dataset_root)
    env = build_record_environment(
        resolved,
        {
            "NUM_EPISODES": str(episode_count),
            "EPISODE_TIME_S": "12",
            "RECORD_FPS": "25",
            "MAX_INITIAL_JOINT_DELTA_RAD": "8.0",
        },
    )

    assert env["DATASET_NAME"] == "custom_recording"
    assert env["DATASET_ROOT"] == str(dataset_root)
    assert env["DATASET_REPO_ID"] == "local/custom_recording"
    assert env["NUM_EPISODES"] == str(episode_count)
    assert env["VIDEO_ENCODING_BATCH_SIZE"] == str(episode_count)
    assert env["EPISODE_TIME_S"] == "12"
    assert env["RECORD_FPS"] == "25"
    assert env["RESUME"] == "false"
    assert env["EXECUTION"] == "armed"
    assert env["ZMQ_PRESET"] == "jz_three_zmq"
    assert env["RTSP_PRESET"] == "none"
    assert env["MAX_CAMERA_STATE_RECEIVE_SKEW_MS"] == "200.0"


def test_record_environment_accepts_camera_state_skew_override(tmp_path: Path) -> None:
    dataset_root = tmp_path / "custom_recording"
    env = build_record_environment(
        dataset_root,
        {"MAX_CAMERA_STATE_RECEIVE_SKEW_MS": "150.5"},
    )

    assert env["MAX_CAMERA_STATE_RECEIVE_SKEW_MS"] == "150.5"


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "not-a-number"])
def test_record_environment_rejects_invalid_camera_state_skew(tmp_path: Path, value: str) -> None:
    dataset_root = tmp_path / "custom_recording"
    with pytest.raises(ApiError):
        build_record_environment(dataset_root, {"MAX_CAMERA_STATE_RECEIVE_SKEW_MS": value})


@pytest.mark.parametrize("value", ["0", "-1", "5.5", "not-a-number"])
def test_record_environment_rejects_invalid_episode_count(tmp_path: Path, value: str) -> None:
    dataset_root = tmp_path / f"recording_{value.replace('.', '_')}"
    with pytest.raises(ApiError):
        build_record_environment(dataset_root, {"NUM_EPISODES": value})


def test_record_root_must_be_absolute_new_and_have_safe_name(tmp_path: Path) -> None:
    with pytest.raises(ApiError, match="绝对路径"):
        validate_record_root("relative/dataset")
    with pytest.raises(ApiError, match="只能包含"):
        validate_record_root(tmp_path / "bad name")

    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(ApiError, match="已存在"):
        validate_record_root(existing)


def test_dataset_discovery_filters_non_timed_and_empty_datasets(tmp_path: Path) -> None:
    valid = tmp_path / "valid"
    write_dataset(valid, episodes=4, frames=80)
    write_dataset(tmp_path / "wrong_robot", robot_type="other")
    write_dataset(tmp_path / "empty", episodes=0, frames=0)

    datasets = discover_datasets(tmp_path)

    assert [item.name for item in datasets] == ["valid"]
    assert datasets[0].total_episodes == 4
    assert datasets[0].total_frames == 80


def test_arbitrary_absolute_replay_dataset_loads_outside_discovery_root(tmp_path: Path) -> None:
    discovery_root = tmp_path / "outputs"
    discovery_root.mkdir()
    external = tmp_path / "data" / "collection_system_verify" / "test1"
    write_dataset(external, episodes=6, frames=120)

    assert discover_datasets(discovery_root) == []
    dataset = load_dataset(external)

    assert dataset.path == str(external)
    assert dataset.total_episodes == 6
    assert dataset.total_frames == 120

    with pytest.raises(ApiError, match="绝对路径"):
        load_dataset(Path("data/collection_system_verify/test1"))


def test_replay_environment_selects_dataset_and_episode(tmp_path: Path) -> None:
    root = tmp_path / "replay_source"
    write_dataset(root, episodes=10, frames=2000)
    dataset = load_dataset(root)

    env = build_replay_environment(
        dataset,
        5,
        {
            "REPLAY_FPS": "20",
            "MAX_INITIAL_JOINT_DELTA_RAD": "0.5",
            "MAX_JOINT_STEP_RAD": "0.05",
            "PLAY_SOUNDS": "true",
        },
    )

    assert env["DATASET_NAME"] == "replay_source"
    assert env["DATASET_ROOT"] == str(root)
    assert env["DATASET_REPO_ID"] == "local/replay_source"
    assert env["EPISODE"] == "5"
    assert env["EXECUTION"] == "armed"
    assert env["SEND_ACTION_TRANSPORT"] == "udp"


def test_replay_rejects_episode_outside_dataset(tmp_path: Path) -> None:
    dataset = DatasetInfo(
        name="source",
        path=str(tmp_path / "source"),
        total_episodes=2,
        total_frames=40,
        fps=20,
        modified_at=0,
    )

    with pytest.raises(ApiError, match=r"0\.\.1"):
        build_replay_environment(dataset, 2)


def test_frontend_has_complete_chinese_english_language_switch() -> None:
    html = (WEB_ROOT / "static" / "index.html").read_text(encoding="utf-8")
    javascript = (WEB_ROOT / "static" / "app.js").read_text(encoding="utf-8")

    assert 'id="language-toggle"' in html
    assert 'id="record-episodes"' in html
    assert 'id="record-camera-skew"' in html
    assert 'id="replay-custom-root"' in html
    assert 'id="replay-custom-load"' in html
    assert "async function loadCustomDataset()" in javascript
    assert "MAX_CAMERA_STATE_RECEIVE_SKEW_MS" in javascript
    assert "FIXED 10 EPISODES" not in html
    assert "数采系统" in html
    assert "素材系统" not in html
    assert 'pageTitle: "JZ Timed Data Collection System"' in javascript
    assert 'localStorage.setItem("jz_web_language"' in javascript
    assert 'document.documentElement.lang = state.language === "en" ? "en" : "zh-CN"' in javascript

    html_keys = set(re.findall(r'data-i18n(?:-title|-aria-label)?="([A-Za-z0-9]+)"', html))
    zh_block, en_and_rest = javascript.split("  en: {", maxsplit=1)
    en_block = en_and_rest.split("  },\n};", maxsplit=1)[0]
    zh_keys = set(re.findall(r"^    ([A-Za-z0-9]+):", zh_block, flags=re.MULTILINE))
    en_keys = set(re.findall(r"^    ([A-Za-z0-9]+):", en_block, flags=re.MULTILINE))

    assert html_keys <= zh_keys
    assert html_keys <= en_keys
    assert zh_keys == en_keys
