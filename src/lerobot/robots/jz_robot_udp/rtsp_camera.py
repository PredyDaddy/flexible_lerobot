#!/usr/bin/env python

from __future__ import annotations

import os
from typing import Any

from .config_jz_robot_udp import RTSPCameraConfig


class RTSPCamera:
    def __init__(self, config: RTSPCameraConfig):
        self.config = config
        self._cap: Any | None = None

    @property
    def is_connected(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def connect(self) -> None:
        import cv2

        if self.is_connected:
            return
        if self.config.transport == "tcp":
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        cap = cv2.VideoCapture(self.config.url)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.config.timeout_ms)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.config.timeout_ms)
        if not cap.isOpened():
            cap.release()
            raise ConnectionError(f"Failed to open RTSP camera {self.config.url}")
        self._cap = cap
        for _ in range(max(0, self.config.warmup_frames)):
            self.read()

    def read(self) -> Any:
        import cv2

        if not self.is_connected:
            raise RuntimeError(f"RTSP camera is not connected: {self.config.url}")
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise TimeoutError(f"Failed to read RTSP frame: {self.config.url}")
        height, width, channels = frame.shape
        if (height, width, channels) != (self.config.height, self.config.width, 3):
            raise RuntimeError(
                f"RTSP camera frame shape {(height, width, channels)} does not match configured "
                f"{(self.config.height, self.config.width, 3)} for {self.config.url}"
            )
        if self.config.color_mode == "rgb":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def async_read(self) -> Any:
        return self.read()

    def disconnect(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
