from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any

from .buffers import SampleBuffer
from .config import CameraConfig
from .time_utils import now_wall_time_ns
from .types import TimestampedPayload

LOW_LATENCY_TCP_OPTIONS = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0"


@dataclass(frozen=True)
class FramePayload:
    camera_name: str
    frame_index: int
    pts_ns: int
    image: Any


def configure_opencv_rtsp_environment(camera: CameraConfig) -> None:
    if camera.transport == "tcp":
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = LOW_LATENCY_TCP_OPTIONS


class OpenCvRtspReceiver:
    """Simple RTSP receiver backed by OpenCV.

    It is intentionally isolated from the recorder so it can be replaced with a
    GStreamer appsink implementation later without changing dataset code.
    """

    def __init__(self, camera_name: str, camera: CameraConfig, buffer: SampleBuffer):
        self.camera_name = camera_name
        self.camera = camera
        self.buffer = buffer
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name=f"rtsp-{self.camera_name}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        import cv2

        configure_opencv_rtsp_environment(self.camera)
        cap = cv2.VideoCapture(self.camera.rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        frame_index = 0
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                receive_time_ns = now_wall_time_ns()
                if not ok:
                    continue
                pts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                pts_ns = int(pts_ms * 1_000_000) if pts_ms and pts_ms > 0 else receive_time_ns
                self.buffer.append(
                    TimestampedPayload(
                        source_time_ns=receive_time_ns,
                        receive_time_ns=receive_time_ns,
                        payload=FramePayload(
                            camera_name=self.camera_name,
                            frame_index=frame_index,
                            pts_ns=pts_ns,
                            image=frame,
                        ),
                    )
                )
                frame_index += 1
        finally:
            cap.release()
