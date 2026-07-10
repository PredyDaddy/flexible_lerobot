#!/usr/bin/env python

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

from .config_jz_robot_udp import RTSPCameraConfig

TCP_CAPTURE_OPTIONS = "rtsp_transport;tcp"
logger = logging.getLogger(__name__)


def configure_opencv_rtsp_environment(config: RTSPCameraConfig) -> None:
    if config.ffmpeg_capture_options:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = config.ffmpeg_capture_options
        return
    if config.transport == "tcp":
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = TCP_CAPTURE_OPTIONS


class RTSPCamera:
    def __init__(self, config: RTSPCameraConfig):
        self.config = config
        self._cap: Any | None = None
        self._reader_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._frame_ready = threading.Event()
        self._lock = threading.Lock()
        self._new_frame = threading.Condition(self._lock)
        self._latest_frame: Any | None = None
        self._latest_frame_monotonic_s: float | None = None
        self._frames_read = 0
        self._read_failures = 0
        self._last_error: Exception | None = None

    @property
    def is_connected(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def connect(self) -> None:
        import cv2

        if self.is_connected:
            return
        configure_opencv_rtsp_environment(self.config)
        cap = cv2.VideoCapture(self.config.url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.config.timeout_ms)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.config.timeout_ms)
        if not cap.isOpened():
            cap.release()
            raise ConnectionError(f"Failed to open RTSP camera {self.config.url}")
        self._cap = cap
        if self.config.threaded_reader:
            self._start_reader_thread()
            self._wait_for_fresh_frame(timeout_s=self.config.timeout_ms / 1000)
            self.flush(self.config.warmup_frames, timeout_s=self.config.timeout_ms / 1000)
        else:
            for _ in range(max(0, self.config.warmup_frames)):
                self.read()

    @property
    def frame_age_s(self) -> float | None:
        if self._latest_frame_monotonic_s is None:
            return None
        return max(0.0, time.monotonic() - self._latest_frame_monotonic_s)

    @property
    def diagnostics(self) -> dict[str, int | float | None]:
        return {
            "frames_read": self._frames_read,
            "read_failures": self._read_failures,
            "frame_age_s": self.frame_age_s,
        }

    def _decode_frame(self) -> Any:
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

    def _reader_loop(self) -> None:
        retry_sleep_s = self.config.read_retry_sleep_ms / 1000
        while not self._stop_event.is_set():
            try:
                frame = self._decode_frame()
            except Exception as exc:
                with self._lock:
                    self._read_failures += 1
                    self._last_error = exc
                logger.debug("RTSP camera read failed for %s: %s", self.config.url, exc)
                if retry_sleep_s > 0:
                    time.sleep(retry_sleep_s)
                continue

            with self._lock:
                self._latest_frame = frame
                self._latest_frame_monotonic_s = time.monotonic()
                self._frames_read += 1
                self._last_error = None
                self._frame_ready.set()
                self._new_frame.notify_all()

    def _start_reader_thread(self) -> None:
        if self._reader_thread is not None and self._reader_thread.is_alive():
            return
        self._stop_event.clear()
        self._frame_ready.clear()
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            name=f"rtsp_reader_{self.config.url.rsplit('/', maxsplit=1)[-1]}",
            daemon=True,
        )
        self._reader_thread.start()

    def _wait_for_fresh_frame(self, timeout_s: float) -> Any:
        if not self._frame_ready.wait(timeout=timeout_s):
            with self._lock:
                last_error = self._last_error
            message = f"Timed out waiting for the first RTSP frame: {self.config.url}"
            if last_error is not None:
                message = f"{message}; last_error={last_error}"
            raise TimeoutError(message)
        return self.read()

    def flush(self, frames: int, timeout_s: float | None = None) -> None:
        if frames <= 0:
            return
        timeout_s = self.config.timeout_ms / 1000 if timeout_s is None else timeout_s
        deadline_s = time.monotonic() + timeout_s
        last_seen_count = self._frames_read
        consumed = 0
        while consumed < frames:
            remaining_s = deadline_s - time.monotonic()
            if remaining_s <= 0:
                raise TimeoutError(f"Timed out flushing RTSP frames: {self.config.url}")
            with self._lock:
                self._new_frame.wait_for(lambda: self._frames_read > last_seen_count, timeout=remaining_s)
                current_count = self._frames_read
            if current_count > last_seen_count:
                consumed += current_count - last_seen_count
                last_seen_count = current_count

    def read(self) -> Any:
        if not self.config.threaded_reader:
            return self._decode_frame()

        if not self.is_connected:
            raise RuntimeError(f"RTSP camera is not connected: {self.config.url}")
        if not self._frame_ready.wait(timeout=self.config.timeout_ms / 1000):
            raise TimeoutError(f"Timed out waiting for RTSP frame: {self.config.url}")

        with self._lock:
            frame = None if self._latest_frame is None else self._latest_frame.copy()
            frame_age_s = self.frame_age_s
            last_error = self._last_error

        if frame is None:
            message = f"No RTSP frame available: {self.config.url}"
            if last_error is not None:
                message = f"{message}; last_error={last_error}"
            raise TimeoutError(message)
        if frame_age_s is None or frame_age_s > self.config.stale_frame_timeout_ms / 1000:
            raise TimeoutError(
                f"Latest RTSP frame is stale for {self.config.url}: "
                f"age_s={frame_age_s}, timeout_s={self.config.stale_frame_timeout_ms / 1000}"
            )
        return frame

    def async_read(self) -> Any:
        return self.read()

    def disconnect(self) -> None:
        self._stop_event.set()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=self.config.timeout_ms / 1000)
            self._reader_thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._frame_ready.clear()
        with self._lock:
            self._latest_frame = None
            self._latest_frame_monotonic_s = None
            self._last_error = None
