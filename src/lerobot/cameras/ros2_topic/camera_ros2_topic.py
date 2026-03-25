#!/usr/bin/env python

from __future__ import annotations

import logging
import sys
import time
from threading import Event, Lock, Thread
from typing import Any

import numpy as np
import rclpy
from numpy.typing import NDArray
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..camera import Camera
from ..configs import ColorMode
from .configuration_ros2_topic import ROS2TopicCameraConfig

logger = logging.getLogger(__name__)


def _encoding_to_dtype_and_channels(encoding: str) -> tuple[np.dtype, int]:
    if encoding in ("rgb8", "bgr8"):
        return np.dtype(np.uint8), 3
    if encoding in ("rgba8", "bgra8"):
        return np.dtype(np.uint8), 4
    if encoding == "mono8":
        return np.dtype(np.uint8), 1
    if encoding == "mono16":
        return np.dtype(np.uint16), 1
    if encoding == "8UC1":
        return np.dtype(np.uint8), 1
    if encoding == "16UC1":
        return np.dtype(np.uint16), 1
    if encoding == "32FC1":
        return np.dtype(np.float32), 1
    raise ValueError(f"Unsupported ROS image encoding: {encoding!r}")


def ros_image_to_numpy(msg: Image, *, want_rgb: bool) -> NDArray[Any]:
    dtype, channels = _encoding_to_dtype_and_channels(msg.encoding)
    itemsize = dtype.itemsize

    if msg.step % itemsize != 0:
        raise ValueError(f"Invalid Image.step={msg.step} for dtype itemsize={itemsize}")

    row_stride_elems = msg.step // itemsize
    expected_elems = row_stride_elems * msg.height
    arr = np.frombuffer(msg.data, dtype=dtype, count=expected_elems)
    arr = arr.reshape((msg.height, row_stride_elems))

    pixel_elems = msg.width * channels
    if pixel_elems > row_stride_elems:
        raise ValueError(
            f"Invalid image layout: width*channels={pixel_elems} > step/itemsize={row_stride_elems}"
        )

    arr = arr[:, :pixel_elems]
    if channels == 1:
        arr = arr.reshape((msg.height, msg.width))
    else:
        arr = arr.reshape((msg.height, msg.width, channels))

    if itemsize > 1 and bool(msg.is_bigendian) != (sys.byteorder == "big"):
        arr = arr.byteswap().newbyteorder()

    if channels == 4:
        if msg.encoding == "bgra8" and want_rgb:
            arr = arr[:, :, [2, 1, 0, 3]]
        elif msg.encoding == "rgba8" and not want_rgb:
            arr = arr[:, :, [2, 1, 0, 3]]
        arr = arr[:, :, :3]

    if channels == 3:
        if msg.encoding == "bgr8" and want_rgb:
            arr = arr[:, :, ::-1]
        elif msg.encoding == "rgb8" and not want_rgb:
            arr = arr[:, :, ::-1]

    if channels == 1 and arr.dtype == np.uint8:
        arr = np.repeat(arr[:, :, None], 3, axis=2)

    return np.ascontiguousarray(arr)


class ROS2TopicCamera(Camera):
    def __init__(self, config: ROS2TopicCameraConfig):
        super().__init__(config)
        self.config = config
        self.image_topic = config.image_topic
        self.color_mode = config.color_mode
        self.timeout_ms = config.timeout_ms
        self.qos_depth = config.qos_depth
        self.reliability = config.reliability
        self.durability = config.durability
        self.warmup_s = config.warmup_s

        self._connected = False
        self._owns_rclpy_init = False
        self._node = None
        self._subscription = None
        self._executor: SingleThreadedExecutor | None = None
        self._spin_thread: Thread | None = None

        self.frame_lock: Lock = Lock()
        self.latest_frame: NDArray[Any] | None = None
        self.new_frame_event: Event = Event()
        self._warned_shape_mismatch = False
        self._frame_counter = 0

    def __str__(self) -> str:
        return f"ROS2TopicCamera(topic={self.image_topic})"

    @property
    def is_connected(self) -> bool:
        return self._connected and self._node is not None and self._executor is not None

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        return []

    def _make_qos(self) -> QoSProfile:
        reliability = (
            ReliabilityPolicy.RELIABLE
            if self.reliability == "reliable"
            else ReliabilityPolicy.BEST_EFFORT
        )
        durability = (
            DurabilityPolicy.TRANSIENT_LOCAL
            if self.durability == "transient_local"
            else DurabilityPolicy.VOLATILE
        )
        return QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=self.qos_depth,
            reliability=reliability,
            durability=durability,
        )

    def connect(self, warmup: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")

        try:
            if not rclpy.ok():
                rclpy.init()
                self._owns_rclpy_init = True

            node_name = f"lerobot_ros2_camera_{abs(hash(self.image_topic)) % 10**8}"
            self._node = rclpy.create_node(node_name)
            self._subscription = self._node.create_subscription(
                Image,
                self.image_topic,
                self._image_callback,
                self._make_qos(),
            )
            self._executor = SingleThreadedExecutor()
            self._executor.add_node(self._node)
            self._spin_thread = Thread(target=self._executor.spin, daemon=True, name=node_name)
            self._spin_thread.start()
            self._connected = True

            self.read(timeout_ms=max(self.timeout_ms, 5000))
            if warmup and self.warmup_s > 0:
                time.sleep(self.warmup_s)

            logger.info("%s connected.", self)
        except Exception:
            self._cleanup()
            raise

    def _image_callback(self, msg: Image) -> None:
        try:
            frame = ros_image_to_numpy(msg, want_rgb=self.color_mode == ColorMode.RGB)
        except Exception as exc:
            logger.warning("%s failed to decode frame from %s: %s", self, self.image_topic, exc)
            return

        height, width = frame.shape[:2]
        if self.width is None:
            self.width = width
        if self.height is None:
            self.height = height
        if (self.width, self.height) != (width, height) and not self._warned_shape_mismatch:
            logger.warning(
                "%s configured for %sx%s but topic is publishing %sx%s; recorder will use the incoming frame size.",
                self,
                self.width,
                self.height,
                width,
                height,
            )
            self._warned_shape_mismatch = True
            self.width = width
            self.height = height

        with self.frame_lock:
            self.latest_frame = frame
            self._frame_counter += 1
            self.new_frame_event.set()

    def _convert_output_color_mode(self, frame: NDArray[Any], color_mode: ColorMode | None) -> NDArray[Any]:
        target = self.color_mode if color_mode is None else color_mode
        if frame.ndim != 3 or frame.shape[2] != 3 or target == self.color_mode:
            return frame
        return np.ascontiguousarray(frame[:, :, ::-1])

    def read(self, color_mode: ColorMode | None = None, timeout_ms: int | None = None) -> NDArray[Any]:
        return self.async_read(timeout_ms=float(self.timeout_ms if timeout_ms is None else timeout_ms), color_mode=color_mode)

    def async_read(self, timeout_ms: float | None = None, color_mode: ColorMode | None = None) -> NDArray[Any]:
        timeout_ms = float(self.timeout_ms if timeout_ms is None else timeout_ms)

        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(f"Timed out waiting for frame from {self} after {timeout_ms} ms.")

        with self.frame_lock:
            frame = None if self.latest_frame is None else self.latest_frame.copy()
            self.new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"Internal error: frame event was set but {self} has no cached frame.")

        return self._convert_output_color_mode(frame, color_mode)

    def _cleanup(self) -> None:
        self._connected = False

        if self._executor is not None:
            try:
                self._executor.shutdown(timeout_sec=1.0)
            except Exception:
                pass
            self._executor = None

        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:
                pass
            self._node = None

        if self._spin_thread is not None and self._spin_thread.is_alive():
            self._spin_thread.join(timeout=1.0)
        self._spin_thread = None
        self._subscription = None

        if self._owns_rclpy_init and rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception:
                pass
        self._owns_rclpy_init = False

    def disconnect(self) -> None:
        if not self.is_connected and self._spin_thread is None:
            raise DeviceNotConnectedError(f"{self} not connected.")
        self._cleanup()
        logger.info("%s disconnected.", self)
