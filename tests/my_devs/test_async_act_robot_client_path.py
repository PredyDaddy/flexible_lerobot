from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.async_inference.helpers import raw_observation_to_observation
from my_devs.async_act.config_single_arm_agilex_robot import SingleArmAgileXRobotConfig


class FakeBridge:
    def __init__(self, *, topics, joint_names, queue_size):  # noqa: ANN001
        self.topics = topics
        self.joint_names = joint_names
        self.queue_size = queue_size
        self.is_connected = False

    def connect(self, *, node_name, needs_publishers):  # noqa: ANN001
        self.is_connected = True

    def wait_for_ready(self, *, timeout_s, require_images):  # noqa: ANN001
        return None

    def disconnect(self):
        self.is_connected = False

    def get_state_features(self):
        state = {}
        for arm, offset in (("left", 0.0), ("right", 100.0)):
            for idx in range(7):
                state[f"{arm}_joint{idx}.pos"] = offset + idx
        return state

    def get_action_features(self):
        return self.get_state_features()

    def get_images(self):
        return {
            "camera_front": np.zeros((480, 640, 3), dtype=np.uint8),
            "camera_right": np.full((480, 640, 3), 7, dtype=np.uint8),
        }

    def publish_action(self, action):  # noqa: ANN001
        return None


def test_robot_client_single_arm_lerobot_features_and_observation_path(monkeypatch):
    pytest.importorskip("grpc")
    try:
        from lerobot.async_inference.configs import RobotClientConfig
        from lerobot.async_inference.robot_client import RobotClient
    except RuntimeError as exc:
        pytest.skip(f"grpc runtime not compatible in this environment: {exc}")

    module = importlib.import_module("my_devs.async_act.single_arm_agilex_robot")
    monkeypatch.setattr(module, "AgileXRosBridge", FakeBridge)

    client = RobotClient(
        RobotClientConfig(
            robot=SingleArmAgileXRobotConfig(id="robot", arm="right"),
            server_address="localhost:9999",
            policy_type="act",
            pretrained_name_or_path="dummy",
            actions_per_chunk=20,
        )
    )
    try:
        lerobot_features = client.policy_config.lerobot_features
        assert lerobot_features["observation.state"]["shape"] == (7,)
        assert lerobot_features["observation.state"]["names"] == [
            f"right_joint{i}.pos" for i in range(7)
        ]
        assert set(lerobot_features) == {
            "observation.state",
            "observation.images.camera_front",
            "observation.images.camera_right",
        }

        observation = client.robot.get_observation()
        policy_image_features = {
            "observation.images.camera_front": SimpleNamespace(shape=(3, 480, 640)),
            "observation.images.camera_right": SimpleNamespace(shape=(3, 480, 640)),
        }
        prepared = raw_observation_to_observation(
            observation,
            lerobot_features=lerobot_features,
            policy_image_features=policy_image_features,
        )

        assert prepared["observation.state"].shape == (1, 7)
        assert prepared["observation.images.camera_front"].shape == (1, 3, 480, 640)
        assert prepared["observation.images.camera_right"].shape == (1, 3, 480, 640)
    finally:
        client.stop()
