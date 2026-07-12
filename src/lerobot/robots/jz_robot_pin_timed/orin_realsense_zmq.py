#!/usr/bin/env python

from __future__ import annotations

import argparse
import base64
import contextlib
import json
import signal
import threading
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import zmq

from lerobot.cameras.configs import ColorMode
from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig


PROTOCOL_VERSION = 1


@dataclass(frozen=True)
class DirectCameraPreset:
    camera_id: str
    serial_number: str
    width: int
    height: int
    fps: int
    port: int


CAMERA_PRESETS = {
    "camera_head": DirectCameraPreset("camera_head", "318122304464", 1280, 720, 30, 5555),
    "camera_left": DirectCameraPreset("camera_left", "230422272306", 640, 480, 30, 5556),
    "camera_right": DirectCameraPreset("camera_right", "230322272819", 640, 480, 30, 5557),
}


def encode_rgb_jpeg(image: np.ndarray, quality: int) -> bytes:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected uint8 RGB image, got shape={image.shape} dtype={image.dtype}")
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(
        ".jpg",
        bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), quality],
    )
    if not ok:
        raise RuntimeError("OpenCV failed to encode JPEG")
    return encoded.tobytes()


def build_message(
    preset: DirectCameraPreset,
    *,
    sequence: int,
    capture_wall_ns: int,
    capture_monotonic_ns: int,
    encode_completed_monotonic_ns: int,
    jpeg: bytes,
    jpeg_quality: int,
) -> str:
    image_b64 = base64.b64encode(jpeg).decode("ascii")
    timing = {
        "sequence": sequence,
        "timestamp_stage": "after_realsense_read_before_jpeg",
        "capture_wall_ns": capture_wall_ns,
        "capture_monotonic_ns": capture_monotonic_ns,
        "encode_completed_monotonic_ns": encode_completed_monotonic_ns,
        "width": preset.width,
        "height": preset.height,
        "channels": 3,
        "pixel_format": "RGB8",
        "encoding": "jpeg",
        "jpeg_quality": jpeg_quality,
        "payload_bytes": len(jpeg),
    }
    return json.dumps(
        {
            "protocol": "jz_realsense_zmq",
            "protocol_version": PROTOCOL_VERSION,
            "timestamps": {preset.camera_id: capture_wall_ns / 1_000_000_000},
            "images": {preset.camera_id: image_b64},
            "camera_timing": {preset.camera_id: timing},
        },
        separators=(",", ":"),
    )


class CameraPublisherWorker:
    def __init__(
        self,
        preset: DirectCameraPreset,
        context: zmq.Context,
        *,
        bind_host: str,
        jpeg_quality: int,
        count: int,
        stop_event: threading.Event,
        camera_factory: Any = RealSenseCamera,
    ) -> None:
        self.preset = preset
        self.context = context
        self.bind_host = bind_host
        self.jpeg_quality = jpeg_quality
        self.count = count
        self.stop_event = stop_event
        self.camera_factory = camera_factory
        self.ready = threading.Event()
        self.thread = threading.Thread(
            target=self._run,
            name=f"jz-realsense-zmq-{preset.camera_id}",
            daemon=False,
        )
        self.exception: BaseException | None = None
        self.frames_sent = 0

    def start(self) -> None:
        self.thread.start()

    def join(self, timeout: float | None = None) -> None:
        self.thread.join(timeout=timeout)

    def _run(self) -> None:
        camera = self.camera_factory(
            RealSenseCameraConfig(
                serial_number_or_name=self.preset.serial_number,
                fps=self.preset.fps,
                width=self.preset.width,
                height=self.preset.height,
                color_mode=ColorMode.RGB,
                use_depth=False,
                warmup_s=1,
            )
        )
        socket = self.context.socket(zmq.PUB)
        socket.setsockopt(zmq.SNDHWM, 2)
        socket.setsockopt(zmq.LINGER, 0)
        socket.bind(f"tcp://{self.bind_host}:{self.preset.port}")
        started_ns = time.monotonic_ns()
        try:
            camera.connect()
            self.ready.set()
            print(
                f"[direct realsense zmq] ready camera={self.preset.camera_id} "
                f"serial={self.preset.serial_number} bind=tcp://{self.bind_host}:{self.preset.port} "
                f"format=RGB8 size={self.preset.width}x{self.preset.height} fps={self.preset.fps}",
                flush=True,
            )
            while not self.stop_event.is_set() and (
                self.count <= 0 or self.frames_sent < self.count
            ):
                frame = camera.read(timeout_ms=500)
                capture_monotonic_ns = time.monotonic_ns()
                capture_wall_ns = time.time_ns()
                expected_shape = (self.preset.height, self.preset.width, 3)
                if frame.dtype != np.uint8 or frame.shape != expected_shape:
                    raise RuntimeError(
                        f"unexpected {self.preset.camera_id} frame shape={frame.shape} "
                        f"dtype={frame.dtype} expected={expected_shape}/uint8"
                    )
                jpeg = encode_rgb_jpeg(frame, self.jpeg_quality)
                encode_completed_monotonic_ns = time.monotonic_ns()
                sequence = self.frames_sent + 1
                message = build_message(
                    self.preset,
                    sequence=sequence,
                    capture_wall_ns=capture_wall_ns,
                    capture_monotonic_ns=capture_monotonic_ns,
                    encode_completed_monotonic_ns=encode_completed_monotonic_ns,
                    jpeg=jpeg,
                    jpeg_quality=self.jpeg_quality,
                )
                with contextlib.suppress(zmq.Again):
                    socket.send_string(message, flags=zmq.NOBLOCK)
                self.frames_sent = sequence
                if sequence == 1 or sequence % self.preset.fps == 0:
                    elapsed_s = (time.monotonic_ns() - started_ns) / 1_000_000_000
                    encode_ms = (encode_completed_monotonic_ns - capture_monotonic_ns) / 1_000_000
                    print(
                        f"[direct realsense zmq] camera={self.preset.camera_id} "
                        f"sequence={sequence} capture_hz={sequence / elapsed_s:.3f} "
                        f"encode_ms={encode_ms:.3f} jpeg_bytes={len(jpeg)}",
                        flush=True,
                    )
        except BaseException as exc:
            self.exception = exc
            self.stop_event.set()
        finally:
            if camera.is_connected:
                camera.disconnect()
            socket.close()


class DirectRealSenseZmqServer:
    def __init__(
        self,
        *,
        bind_host: str = "*",
        jpeg_quality: int = 95,
        count: int = 0,
        presets: tuple[DirectCameraPreset, ...] | None = None,
    ) -> None:
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        self.bind_host = bind_host
        self.jpeg_quality = jpeg_quality
        self.count = count
        self.presets = presets or tuple(CAMERA_PRESETS.values())
        self.stop_event = threading.Event()
        self.context = zmq.Context()
        self.workers = tuple(
            CameraPublisherWorker(
                preset,
                self.context,
                bind_host=bind_host,
                jpeg_quality=jpeg_quality,
                count=count,
                stop_event=self.stop_event,
            )
            for preset in self.presets
        )

    def request_stop(self) -> None:
        self.stop_event.set()

    def run(self) -> int:
        for worker in self.workers:
            worker.start()
        try:
            deadline = time.monotonic() + 20
            while not all(worker.ready.is_set() for worker in self.workers):
                self._raise_if_failed()
                if time.monotonic() >= deadline:
                    raise TimeoutError("timed out opening all RealSense cameras")
                time.sleep(0.05)
            print("[direct realsense zmq] all cameras ready", flush=True)
            while not self.stop_event.is_set():
                self._raise_if_failed()
                if self.count > 0 and all(
                    worker.frames_sent >= self.count for worker in self.workers
                ):
                    break
                time.sleep(0.05)
            self._raise_if_failed()
            return 0
        finally:
            self.stop_event.set()
            for worker in self.workers:
                worker.join(timeout=5)
            self.context.term()

    def _raise_if_failed(self) -> None:
        for worker in self.workers:
            if worker.exception is not None:
                raise RuntimeError(f"camera worker failed: {worker.preset.camera_id}") from worker.exception


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve Orin RealSense RGB frames over ZMQ")
    parser.add_argument("--bind-host", default="*")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--count", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    server = DirectRealSenseZmqServer(
        bind_host=args.bind_host,
        jpeg_quality=args.jpeg_quality,
        count=args.count,
    )
    signal.signal(signal.SIGINT, lambda _signum, _frame: server.request_stop())
    signal.signal(signal.SIGTERM, lambda _signum, _frame: server.request_stop())
    return server.run()


if __name__ == "__main__":
    raise SystemExit(main())
