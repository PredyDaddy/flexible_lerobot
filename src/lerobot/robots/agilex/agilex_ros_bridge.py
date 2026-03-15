#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

ACTION_SUFFIX = "pos"
POSITION_SUFFIX = "pos"
VELOCITY_SUFFIX = "vel"
EFFORT_SUFFIX = "effort"
LEFT_PREFIX = "left"
RIGHT_PREFIX = "right"
ARM_PREFIXES = (LEFT_PREFIX, RIGHT_PREFIX)
CAMERA_KEYS = ("camera_front", "camera_left", "camera_right")
SUPPORTED_COLOR_ENCODINGS = {"rgb8", "bgr8"}


def make_joint_feature_names(suffix: str) -> list[str]:
    return [f"{arm}_joint{i}.{suffix}" for arm in ARM_PREFIXES for i in range(7)]


ACTION_FEATURE_NAMES = make_joint_feature_names(ACTION_SUFFIX)
POSITION_FEATURE_NAMES = make_joint_feature_names(POSITION_SUFFIX)
VELOCITY_FEATURE_NAMES = make_joint_feature_names(VELOCITY_SUFFIX)
EFFORT_FEATURE_NAMES = make_joint_feature_names(EFFORT_SUFFIX)


@dataclass(frozen=True)
class JointSample:
    name: tuple[str, ...]
    position: tuple[float, ...]
    velocity: tuple[float, ...]
    effort: tuple[float, ...]


@dataclass(frozen=True)
class ImageTopicConfig:
    topic: str
    feature_key: str


@dataclass(frozen=True)
class BridgeTopics:
    state_left_topic: str
    state_right_topic: str
    command_left_topic: str | None
    command_right_topic: str | None
    image_topics: tuple[ImageTopicConfig, ...]


class AgileXRosBridge:
    _node_initialized = False

    def __init__(
        self,
        *,
        topics: BridgeTopics,
        joint_names: list[str],
        queue_size: int = 1,
    ):
        self.topics = topics
        self.joint_names = tuple(joint_names)
        self.queue_size = queue_size

        self._rospy = None
        self._joint_state_cls = None
        self._image_cls = None
        self._state_subscribers: list[Any] = []
        self._image_subscribers: list[Any] = []
        self._publishers: list[Any] = []
        self._left_command_publisher = None
        self._right_command_publisher = None
        self._connected = False

        self._latest_state: dict[str, JointSample | None] = {
            LEFT_PREFIX: None,
            RIGHT_PREFIX: None,
        }
        self._latest_images: dict[str, np.ndarray | None] = {
            image_config.feature_key: None for image_config in self.topics.image_topics
        }

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, *, node_name: str, needs_publishers: bool) -> None:
        if self._connected:
            return

        import rospy
        from sensor_msgs.msg import Image, JointState

        self._rospy = rospy
        self._joint_state_cls = JointState
        self._image_cls = Image
        self._ensure_node_initialized(node_name)

        self._state_subscribers = [
            rospy.Subscriber(
                self.topics.state_left_topic,
                JointState,
                lambda msg: self._store_joint_sample(LEFT_PREFIX, msg),
                queue_size=self.queue_size,
                tcp_nodelay=True,
            ),
            rospy.Subscriber(
                self.topics.state_right_topic,
                JointState,
                lambda msg: self._store_joint_sample(RIGHT_PREFIX, msg),
                queue_size=self.queue_size,
                tcp_nodelay=True,
            ),
        ]

        self._image_subscribers = [
            rospy.Subscriber(
                image_config.topic,
                Image,
                lambda msg, key=image_config.feature_key: self._store_image(key, msg),
                queue_size=self.queue_size,
                tcp_nodelay=True,
            )
            for image_config in self.topics.image_topics
        ]

        if needs_publishers:
            if self.topics.command_left_topic is None or self.topics.command_right_topic is None:
                raise ValueError("Command publishers requested but command topics are not configured")
            self._left_command_publisher = rospy.Publisher(
                self.topics.command_left_topic,
                JointState,
                queue_size=self.queue_size,
            )
            self._right_command_publisher = rospy.Publisher(
                self.topics.command_right_topic,
                JointState,
                queue_size=self.queue_size,
            )
            self._publishers = [self._left_command_publisher, self._right_command_publisher]

        self._connected = True

    def disconnect(self) -> None:
        for subscriber in self._state_subscribers + self._image_subscribers:
            subscriber.unregister()
        for publisher in self._publishers:
            publisher.unregister()

        self._state_subscribers = []
        self._image_subscribers = []
        self._publishers = []
        self._left_command_publisher = None
        self._right_command_publisher = None
        self._latest_state = {LEFT_PREFIX: None, RIGHT_PREFIX: None}
        self._latest_images = {
            image_config.feature_key: None for image_config in self.topics.image_topics
        }
        self._connected = False

    def wait_for_ready(self, *, timeout_s: float, require_images: bool) -> None:
        if not self._connected:
            raise RuntimeError("Bridge must be connected before waiting for topics")

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._has_required_data(require_images=require_images):
                return
            if self._rospy.is_shutdown():
                break
            self._rospy.sleep(0.01)

        missing = []
        if self._latest_state[LEFT_PREFIX] is None:
            missing.append(self.topics.state_left_topic)
        if self._latest_state[RIGHT_PREFIX] is None:
            missing.append(self.topics.state_right_topic)
        if require_images:
            missing.extend(
                image_config.topic
                for image_config in self.topics.image_topics
                if self._latest_images[image_config.feature_key] is None
            )
        raise TimeoutError(f"Timed out waiting for AgileX topics: {missing}")

    def get_state_features(self) -> dict[str, float]:
        left_sample = self._require_joint_sample(LEFT_PREFIX)
        right_sample = self._require_joint_sample(RIGHT_PREFIX)
        state: dict[str, float] = {}
        # Record only follower joint positions in observation.state to match the reference dataset schema.
        for arm, sample in ((LEFT_PREFIX, left_sample), (RIGHT_PREFIX, right_sample)):
            for idx in range(7):
                state[f"{arm}_joint{idx}.{POSITION_SUFFIX}"] = sample.position[idx]
        return state

    def get_action_features(self) -> dict[str, float]:
        left_sample = self._require_joint_sample(LEFT_PREFIX)
        right_sample = self._require_joint_sample(RIGHT_PREFIX)
        action: dict[str, float] = {}
        for arm, sample in ((LEFT_PREFIX, left_sample), (RIGHT_PREFIX, right_sample)):
            for idx in range(7):
                action[f"{arm}_joint{idx}.{ACTION_SUFFIX}"] = sample.position[idx]
        return action

    def get_images(self) -> dict[str, np.ndarray]:
        images: dict[str, np.ndarray] = {}
        for image_config in self.topics.image_topics:
            image = self._latest_images[image_config.feature_key]
            if image is None:
                raise RuntimeError(f"No image received on topic {image_config.topic}")
            images[image_config.feature_key] = image.copy()
        return images

    def publish_action(self, action: dict[str, float]) -> None:
        if self._left_command_publisher is None or self._right_command_publisher is None:
            raise RuntimeError("Command publishers are not available in this bridge")

        now = self._rospy.Time.now()
        left_message = self._joint_state_cls()
        left_message.header.stamp = now
        left_message.name = list(self.joint_names)
        left_message.position = [float(action[f"{LEFT_PREFIX}_joint{i}.{ACTION_SUFFIX}"]) for i in range(7)]

        right_message = self._joint_state_cls()
        right_message.header.stamp = now
        right_message.name = list(self.joint_names)
        right_message.position = [float(action[f"{RIGHT_PREFIX}_joint{i}.{ACTION_SUFFIX}"]) for i in range(7)]

        self._left_command_publisher.publish(left_message)
        self._right_command_publisher.publish(right_message)

    @classmethod
    def _ensure_node_initialized(cls, node_name: str) -> None:
        import rospy

        if cls._node_initialized:
            return
        if not rospy.core.is_initialized():
            rospy.init_node(node_name, anonymous=True, disable_signals=True)
        cls._node_initialized = True

    def _has_required_data(self, *, require_images: bool) -> bool:
        has_state = all(sample is not None for sample in self._latest_state.values())
        if not has_state:
            return False
        if not require_images:
            return True
        return all(image is not None for image in self._latest_images.values())

    def _store_joint_sample(self, arm: str, msg: Any) -> None:
        self._latest_state[arm] = JointSample(
            name=tuple(msg.name) if msg.name else self.joint_names,
            position=self._coerce_joint_vector(msg.position, "position"),
            velocity=self._coerce_joint_vector(msg.velocity, "velocity"),
            effort=self._coerce_joint_vector(msg.effort, "effort"),
        )

    def _store_image(self, key: str, msg: Any) -> None:
        self._latest_images[key] = self._decode_image(msg)

    def _coerce_joint_vector(self, values: Any, field_name: str) -> tuple[float, ...]:
        if values:
            values_list = [float(value) for value in values]
        else:
            values_list = [0.0] * 7
        if len(values_list) != 7:
            raise ValueError(f"Expected 7 values for JointState.{field_name}, got {len(values_list)}")
        return tuple(values_list)

    def _require_joint_sample(self, arm: str) -> JointSample:
        sample = self._latest_state[arm]
        if sample is None:
            topic = self.topics.state_left_topic if arm == LEFT_PREFIX else self.topics.state_right_topic
            raise RuntimeError(f"No JointState received on topic {topic}")
        return sample

    def _decode_image(self, msg: Any) -> np.ndarray:
        if msg.encoding not in SUPPORTED_COLOR_ENCODINGS:
            raise NotImplementedError(
                f"Unsupported AgileX image encoding '{msg.encoding}'. Supported encodings: "
                f"{sorted(SUPPORTED_COLOR_ENCODINGS)}"
            )

        channels = 3
        expected_bytes = msg.height * msg.step
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        if raw.size < expected_bytes:
            raise ValueError(
                f"Image buffer too small for topic data: expected at least {expected_bytes} bytes, got {raw.size}"
            )

        row_bytes = msg.width * channels
        image = raw[:expected_bytes].reshape(msg.height, msg.step)[:, :row_bytes]
        image = image.reshape(msg.height, msg.width, channels)

        if msg.encoding == "bgr8":
            image = image[:, :, ::-1]

        return np.ascontiguousarray(image)
