from __future__ import annotations

import importlib
from dataclasses import MISSING, fields
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.robots import make_robot_from_config
from lerobot.robots.agilex import AgileXRobotConfig
from lerobot.robots.agilex.agilex_ros_bridge import (
    ACTION_FEATURE_NAMES,
    CAMERA_KEYS,
    EFFORT_FEATURE_NAMES,
    POSITION_FEATURE_NAMES,
    VELOCITY_FEATURE_NAMES,
)
from lerobot.scripts.lerobot_calibrate import CalibrateConfig, calibrate
from lerobot.scripts.lerobot_record import DatasetRecordConfig, RecordConfig, record
from lerobot.scripts.lerobot_replay import DatasetReplayConfig, ReplayConfig, replay
from lerobot.scripts.lerobot_teleoperate import TeleoperateConfig, teleoperate
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.teleoperators.agilex_teleoperator import AgileXTeleoperatorConfig
from lerobot.utils.constants import ACTION
from tests.fixtures.constants import DUMMY_REPO_ID


def test_cli_scripts_import_with_agilex_core():
    module_names = [
        "lerobot.scripts.lerobot_calibrate",
        "lerobot.scripts.lerobot_teleoperate",
        "lerobot.scripts.lerobot_record",
        "lerobot.scripts.lerobot_replay",
    ]

    for module_name in module_names:
        module = importlib.import_module(module_name)
        assert importlib.reload(module).__name__ == module_name


def _build_config(config_cls, tmp_path: Path, **overrides):
    defaults = {
        "id": "agilex-test",
        "calibration_dir": tmp_path / "calibration",
        "control_mode": "passive_follow",
        "state_left_topic": "/puppet/joint_left",
        "state_right_topic": "/puppet/joint_right",
        "command_left_topic": "/master/joint_left",
        "command_right_topic": "/master/joint_right",
        "action_left_topic": "/master/joint_left",
        "action_right_topic": "/master/joint_right",
        "front_camera_topic": "/camera_f/color/image_raw",
        "left_camera_topic": "/camera_l/color/image_raw",
        "right_camera_topic": "/camera_r/color/image_raw",
        "front_camera_key": "cam_high",
        "left_camera_key": "cam_left_wrist",
        "right_camera_key": "cam_right_wrist",
        "image_height": 480,
        "image_width": 640,
        "observation_timeout_s": 0.01,
        "queue_size": 1,
    }
    values = {}
    for field in fields(config_cls):
        if field.name in overrides:
            values[field.name] = overrides[field.name]
        elif field.default is not MISSING or field.default_factory is not MISSING:  # type: ignore[attr-defined]
            continue
        elif field.name in defaults:
            values[field.name] = defaults[field.name]
        else:
            raise AssertionError(f"Unhandled required config field: {config_cls.__name__}.{field.name}")

    return config_cls(**values)


def _state_features(offset: float = 0.0) -> dict[str, float]:
    features = {}
    for idx, key in enumerate(POSITION_FEATURE_NAMES):
        features[key] = float(idx) + offset
    for idx, key in enumerate(VELOCITY_FEATURE_NAMES):
        features[key] = float(idx) + offset + 100.0
    for idx, key in enumerate(EFFORT_FEATURE_NAMES):
        features[key] = float(idx) + offset + 200.0
    return features


def _action_features(offset: float = 0.0) -> dict[str, float]:
    return {key: float(idx) + offset for idx, key in enumerate(ACTION_FEATURE_NAMES)}


def _image_features() -> dict[str, np.ndarray]:
    return {
        key: np.zeros((480, 640, 3), dtype=np.uint8) + idx
        for idx, key in enumerate(CAMERA_KEYS)
    }


class FakeBridge:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.is_connected = False
        self.connect_calls: list[tuple[str, bool]] = []
        self.wait_calls: list[tuple[float, bool]] = []
        self.published_actions: list[dict[str, float]] = []

    def connect(self, *, node_name: str, needs_publishers: bool) -> None:
        self.is_connected = True
        self.connect_calls.append((node_name, needs_publishers))

    def wait_for_ready(self, *, timeout_s: float, require_images: bool) -> None:
        self.wait_calls.append((timeout_s, require_images))

    def get_state_features(self) -> dict[str, float]:
        return _state_features()

    def get_images(self) -> dict[str, np.ndarray]:
        return _image_features()

    def get_action_features(self) -> dict[str, float]:
        return _action_features(offset=10.0)

    def publish_action(self, action: dict[str, float]) -> None:
        self.published_actions.append(action.copy())

    def disconnect(self) -> None:
        self.is_connected = False


class FakePipeline:
    def __init__(self, *, tuple_mode: bool = False):
        self.tuple_mode = tuple_mode
        self.reset_calls = 0

    def __call__(self, value):
        if self.tuple_mode:
            return value[0]
        return value

    def transform_features(self, features):
        return features

    def reset(self):
        self.reset_calls += 1


class FakeEpisodeFrames:
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int):
        return self.rows[index]

    def select_columns(self, key: str):
        return FakeEpisodeFrames([{key: row[key]} for row in self.rows])


class FakeReplayDataset:
    def __init__(self, action_names: list[str]):
        action_array = np.array([float(i) for i in range(len(action_names))], dtype=np.float32)
        self.hf_dataset = SimpleNamespace(
            filter=lambda fn: FakeEpisodeFrames(
                [{"episode_index": 0, ACTION: action_array}] if fn({"episode_index": 0}) else []
            )
        )
        self.features = {ACTION: {"names": action_names}}
        self.fps = 30


class FakeRecordDataset:
    def __init__(self, features: dict, root: Path):
        self.features = features
        self.root = root
        self.fps = 30
        self.meta = SimpleNamespace(
            stats={},
            video_keys=[],
            total_episodes=0,
            total_frames=0,
            total_tasks=0,
        )
        self.episodes_since_last_encoding = 0
        self.num_episodes = 0
        self.num_frames = 0
        self.finalize_calls = 0

    def save_episode(self):
        self.num_episodes += 1
        self.meta.total_episodes = self.num_episodes

    def clear_episode_buffer(self):
        return None

    def finalize(self):
        self.finalize_calls += 1


class FakeVideoEncodingManager:
    def __init__(self, dataset):
        self.dataset = dataset

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def test_agilex_factory_feature_alignment_and_send_action(monkeypatch, tmp_path):
    import lerobot.robots.agilex.agilex as agilex_robot_module
    import lerobot.teleoperators.agilex_teleoperator.agilex_teleoperator as agilex_teleop_module

    monkeypatch.setattr(agilex_robot_module, "AgileXRosBridge", FakeBridge)
    monkeypatch.setattr(agilex_teleop_module, "AgileXRosBridge", FakeBridge)

    passive_robot_cfg = _build_config(AgileXRobotConfig, tmp_path, control_mode="passive_follow")
    command_robot_cfg = _build_config(AgileXRobotConfig, tmp_path, control_mode="command_master")
    teleop_cfg = _build_config(AgileXTeleoperatorConfig, tmp_path)

    passive_robot = make_robot_from_config(passive_robot_cfg)
    command_robot = make_robot_from_config(command_robot_cfg)
    teleop_device = make_teleoperator_from_config(teleop_cfg)

    assert set(passive_robot.action_features) == set(ACTION_FEATURE_NAMES)
    assert set(passive_robot.action_features) == set(teleop_device.action_features)

    teleop_device.connect()
    teleop_action = teleop_device.get_action()
    assert teleop_action == _action_features(offset=10.0)
    teleop_device.disconnect()

    action = _action_features(offset=1.0)

    passive_robot.connect()
    sent_action = passive_robot.send_action(action)
    assert sent_action == action
    assert passive_robot._bridge.published_actions == []
    passive_robot.disconnect()

    command_robot.connect()
    sent_action = command_robot.send_action(action)
    assert sent_action == action
    assert command_robot._bridge.published_actions == [action]
    command_robot.disconnect()


def test_agilex_script_object_calls(monkeypatch, tmp_path):
    import lerobot.robots.agilex.agilex as agilex_robot_module
    import lerobot.scripts.lerobot_record as record_script
    import lerobot.scripts.lerobot_replay as replay_script
    import lerobot.scripts.lerobot_teleoperate as teleop_script
    import lerobot.teleoperators.agilex_teleoperator.agilex_teleoperator as agilex_teleop_module

    monkeypatch.setattr(agilex_robot_module, "AgileXRosBridge", FakeBridge)
    monkeypatch.setattr(agilex_teleop_module, "AgileXRosBridge", FakeBridge)

    action_pipeline = FakePipeline(tuple_mode=True)
    robot_action_pipeline = FakePipeline(tuple_mode=True)
    observation_pipeline = FakePipeline()
    replay_pipeline = FakePipeline(tuple_mode=True)

    monkeypatch.setattr(
        teleop_script,
        "make_default_processors",
        lambda: (action_pipeline, robot_action_pipeline, observation_pipeline),
    )
    monkeypatch.setattr(
        record_script,
        "make_default_processors",
        lambda: (action_pipeline, robot_action_pipeline, observation_pipeline),
    )
    monkeypatch.setattr(replay_script, "make_default_robot_action_processor", lambda: replay_pipeline)
    monkeypatch.setattr(record_script, "init_keyboard_listener", lambda: (None, {"exit_early": False, "rerecord_episode": False, "stop_recording": False}))
    monkeypatch.setattr(record_script, "record_loop", lambda **kwargs: None)
    monkeypatch.setattr(record_script, "sanity_check_dataset_name", lambda *args, **kwargs: None)
    monkeypatch.setattr(record_script, "sanity_check_dataset_robot_compatibility", lambda *args, **kwargs: None)
    monkeypatch.setattr(record_script, "VideoEncodingManager", FakeVideoEncodingManager)
    monkeypatch.setattr(record_script, "log_say", lambda *args, **kwargs: None)
    monkeypatch.setattr(replay_script, "log_say", lambda *args, **kwargs: None)
    monkeypatch.setattr(record_script, "is_headless", lambda: True)
    monkeypatch.setattr(replay_script, "precise_sleep", lambda *args, **kwargs: None)
    monkeypatch.setattr(teleop_script, "precise_sleep", lambda *args, **kwargs: None)

    monkeypatch.setattr(
        record_script.LeRobotDataset,
        "create",
        lambda repo_id, fps, root, robot_type, features, **kwargs: FakeRecordDataset(features, root),
    )
    monkeypatch.setattr(
        replay_script,
        "LeRobotDataset",
        lambda repo_id, root=None, episodes=None: FakeReplayDataset(ACTION_FEATURE_NAMES),
    )

    robot_cfg = _build_config(AgileXRobotConfig, tmp_path, control_mode="command_master")
    teleop_cfg = _build_config(AgileXTeleoperatorConfig, tmp_path)

    calibrate(CalibrateConfig(robot=robot_cfg))
    calibrate(CalibrateConfig(teleop=teleop_cfg))

    teleoperate(TeleoperateConfig(robot=robot_cfg, teleop=teleop_cfg, teleop_time_s=0.0))

    record_cfg = RecordConfig(
        robot=robot_cfg,
        teleop=teleop_cfg,
        dataset=DatasetRecordConfig(
            repo_id=DUMMY_REPO_ID,
            single_task="agilex smoke",
            root=tmp_path / "record",
            num_episodes=1,
            episode_time_s=0.1,
            reset_time_s=0,
            push_to_hub=False,
        ),
        play_sounds=False,
    )
    dataset = record(record_cfg)
    assert isinstance(dataset, FakeRecordDataset)
    assert dataset.num_episodes == 1
    assert dataset.finalize_calls >= 1

    replay(
        ReplayConfig(
            robot=robot_cfg,
            dataset=DatasetReplayConfig(repo_id=DUMMY_REPO_ID, episode=0, root=tmp_path / "record"),
            play_sounds=False,
        )
    )

    assert action_pipeline.reset_calls == 0
    assert robot_action_pipeline.reset_calls == 0
    assert observation_pipeline.reset_calls == 0
    assert replay_pipeline.reset_calls == 0

