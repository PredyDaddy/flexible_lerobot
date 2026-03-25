#!/usr/bin/env python

from __future__ import annotations

import unittest

try:
    import rclpy  # noqa: F401
    import sensor_msgs.msg  # noqa: F401
except ImportError as exc:
    raise unittest.SkipTest(str(exc))

from lerobot.cameras.ros2_topic import ROS2TopicCamera, ROS2TopicCameraConfig


class _TimeoutEvent:
    def __init__(self) -> None:
        self.timeout: float | None = None

    def wait(self, timeout: float) -> bool:
        self.timeout = timeout
        return False


def _make_connected_camera(timeout_ms: int = 5000) -> ROS2TopicCamera:
    camera = ROS2TopicCamera(ROS2TopicCameraConfig(image_topic="/robot1/test_camera", timeout_ms=timeout_ms))
    camera._connected = True
    camera._node = object()
    camera._executor = object()
    camera.new_frame_event = _TimeoutEvent()
    return camera


class ROS2TopicCameraTimeoutTest(unittest.TestCase):
    def test_async_read_uses_configured_timeout_by_default(self) -> None:
        camera = _make_connected_camera(timeout_ms=5000)

        with self.assertRaises(TimeoutError):
            camera.async_read()

        self.assertEqual(camera.new_frame_event.timeout, 5.0)

    def test_async_read_allows_explicit_timeout_override(self) -> None:
        camera = _make_connected_camera(timeout_ms=5000)

        with self.assertRaises(TimeoutError):
            camera.async_read(timeout_ms=250)

        self.assertEqual(camera.new_frame_event.timeout, 0.25)


if __name__ == "__main__":
    unittest.main()
