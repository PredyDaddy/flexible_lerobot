#!/usr/bin/env python

from __future__ import annotations

import argparse
import base64
import json
import multiprocessing
import queue
import signal
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import zmq

from lerobot.cameras.configs import ColorMode
from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig


PROTOCOL_VERSION = 1
ZMQ_SEND_HIGH_WATER_MARK = 64


@dataclass(frozen=True)
class DirectCameraPreset:
    camera_id: str
    serial_number: str
    width: int
    height: int
    fps: int
    port: int


@dataclass(frozen=True)
class CapturedFrame:
    sequence: int
    image: np.ndarray
    capture_wall_ns: int
    capture_monotonic_ns: int
    read_enter_monotonic_ns: int
    read_return_monotonic_ns: int
    capture_interval_ns: int | None
    capture_thread_interval_ns: int | None
    realsense_frame_number: int | None
    realsense_frame_gap: int | None
    realsense_device_timestamp_ms: float | None
    realsense_device_interval_ms: float | None
    realsense_timestamp_domain: str | None


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
    image_b64: str | None = None,
    extra_timing: dict[str, Any] | None = None,
) -> str:
    if image_b64 is None:
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
    if extra_timing:
        timing.update(extra_timing)
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
        trace_every_frame: bool = False,
    ) -> None:
        self.preset = preset
        self.context = context
        self.bind_host = bind_host
        self.jpeg_quality = jpeg_quality
        self.count = count
        self.stop_event = stop_event
        self.camera_factory = camera_factory
        self.trace_every_frame = trace_every_frame
        self.ready = threading.Event()
        self.thread = threading.Thread(
            target=self._run,
            name=f"jz-realsense-zmq-{preset.camera_id}",
            daemon=False,
        )
        self.exception: BaseException | None = None
        self.frames_sent = 0
        self.frames_captured = 0
        self.capture_queue: queue.Queue[CapturedFrame] = queue.Queue(maxsize=8)
        self.capture_intervals_ms: deque[float] = deque(maxlen=300)
        self.read_durations_ms: deque[float] = deque(maxlen=300)
        self.encode_durations_ms: deque[float] = deque(maxlen=300)

    def start(self) -> None:
        self.thread.start()

    def join(self, timeout: float | None = None) -> None:
        self.thread.join(timeout=timeout)

    def _run(self) -> None:
        capture_ready = threading.Event()
        publisher_ready = threading.Event()
        capture_thread = threading.Thread(
            target=self._capture_loop,
            args=(capture_ready,),
            name=f"jz-realsense-capture-{self.preset.camera_id}",
            daemon=False,
        )
        publisher_thread = threading.Thread(
            target=self._publisher_loop,
            args=(publisher_ready,),
            name=f"jz-realsense-publisher-{self.preset.camera_id}",
            daemon=False,
        )
        try:
            capture_thread.start()
            publisher_thread.start()
            deadline = time.monotonic() + 20
            while not (capture_ready.is_set() and publisher_ready.is_set()):
                if self.exception is not None:
                    raise self.exception
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"timed out starting {self.preset.camera_id} workers")
                time.sleep(0.02)
            self.ready.set()
            print(
                f"[direct realsense zmq] ready camera={self.preset.camera_id} "
                f"serial={self.preset.serial_number} bind=tcp://{self.bind_host}:{self.preset.port} "
                f"format=RGB8 size={self.preset.width}x{self.preset.height} fps={self.preset.fps} "
                "workers=capture+publisher queue_size=8",
                flush=True,
            )
            while not self.stop_event.is_set():
                if self.exception is not None:
                    raise self.exception
                if self.count > 0 and self.frames_sent >= self.count:
                    break
                time.sleep(0.02)
        except BaseException as exc:
            self.exception = exc
            self.stop_event.set()
        finally:
            capture_thread.join(timeout=5)
            publisher_thread.join(timeout=5)

    def _make_camera(self) -> Any:
        return self.camera_factory(
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

    def _read_frame(self, camera: Any) -> tuple[np.ndarray, int | None, float | None, str | None]:
        pipeline = getattr(camera, "rs_pipeline", None)
        if pipeline is None or not isinstance(camera, RealSenseCamera):
            return camera.read(timeout_ms=500), None, None, None

        ok, frames = pipeline.try_wait_for_frames(timeout_ms=500)
        if not ok or frames is None:
            raise RuntimeError(f"{self.preset.camera_id} RealSense read failed status={ok}")
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError(f"{self.preset.camera_id} RealSense frameset has no color frame")
        frame_number = int(color_frame.get_frame_number())
        device_timestamp_ms = float(color_frame.get_timestamp())
        timestamp_domain = str(color_frame.get_frame_timestamp_domain())
        image = np.asanyarray(color_frame.get_data())
        return camera._postprocess_image(image), frame_number, device_timestamp_ms, timestamp_domain

    def _capture_loop(self, ready: threading.Event) -> None:
        camera = self._make_camera()
        previous_capture_ns: int | None = None
        previous_loop_enter_ns: int | None = None
        previous_frame_number: int | None = None
        previous_device_timestamp_ms: float | None = None
        try:
            camera.connect()
            ready.set()
            while not self.stop_event.is_set() and (
                self.count <= 0 or self.frames_captured < self.count
            ):
                loop_enter_ns = time.monotonic_ns()
                read_enter_ns = loop_enter_ns
                image, frame_number, device_timestamp_ms, timestamp_domain = self._read_frame(camera)
                read_return_ns = time.monotonic_ns()
                capture_wall_ns = time.time_ns()
                expected_shape = (self.preset.height, self.preset.width, 3)
                if image.dtype != np.uint8 or image.shape != expected_shape:
                    raise RuntimeError(
                        f"unexpected {self.preset.camera_id} frame shape={image.shape} "
                        f"dtype={image.dtype} expected={expected_shape}/uint8"
                    )
                sequence = self.frames_captured + 1
                captured = CapturedFrame(
                    sequence=sequence,
                    image=image,
                    capture_wall_ns=capture_wall_ns,
                    capture_monotonic_ns=read_return_ns,
                    read_enter_monotonic_ns=read_enter_ns,
                    read_return_monotonic_ns=read_return_ns,
                    capture_interval_ns=(
                        None if previous_capture_ns is None else read_return_ns - previous_capture_ns
                    ),
                    capture_thread_interval_ns=(
                        None if previous_loop_enter_ns is None else loop_enter_ns - previous_loop_enter_ns
                    ),
                    realsense_frame_number=frame_number,
                    realsense_frame_gap=(
                        None
                        if previous_frame_number is None or frame_number is None
                        else max(0, frame_number - previous_frame_number - 1)
                    ),
                    realsense_device_timestamp_ms=device_timestamp_ms,
                    realsense_device_interval_ms=(
                        None
                        if previous_device_timestamp_ms is None or device_timestamp_ms is None
                        else device_timestamp_ms - previous_device_timestamp_ms
                    ),
                    realsense_timestamp_domain=timestamp_domain,
                )
                try:
                    self.capture_queue.put(captured, timeout=0.5)
                except queue.Full as exc:
                    raise RuntimeError(f"{self.preset.camera_id} publisher queue remained full") from exc
                self.frames_captured = sequence
                previous_capture_ns = read_return_ns
                previous_loop_enter_ns = loop_enter_ns
                previous_frame_number = frame_number
                previous_device_timestamp_ms = device_timestamp_ms
        except BaseException as exc:
            self.exception = exc
            self.stop_event.set()
        finally:
            if camera.is_connected:
                camera.disconnect()

    @staticmethod
    def _percentile(values: deque[float], percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = min(len(ordered) - 1, round((len(ordered) - 1) * percentile))
        return ordered[index]

    def _publisher_loop(self, ready: threading.Event) -> None:
        socket = self.context.socket(zmq.PUB)
        socket.setsockopt(zmq.SNDHWM, ZMQ_SEND_HIGH_WATER_MARK)
        socket.setsockopt(zmq.LINGER, 0)
        socket.bind(f"tcp://{self.bind_host}:{self.preset.port}")
        started_ns = time.monotonic_ns()
        ready.set()
        try:
            while not self.stop_event.is_set() or not self.capture_queue.empty():
                if self.count > 0 and self.frames_sent >= self.count:
                    break
                try:
                    captured = self.capture_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                dequeue_ns = time.monotonic_ns()
                encode_start_ns = dequeue_ns
                jpeg = encode_rgb_jpeg(captured.image, self.jpeg_quality)
                encode_completed_ns = time.monotonic_ns()
                base64_start_ns = encode_completed_ns
                image_b64 = base64.b64encode(jpeg).decode("ascii")
                base64_completed_ns = time.monotonic_ns()
                json_start_ns = base64_completed_ns
                extra_timing = {
                    "read_enter_monotonic_ns": captured.read_enter_monotonic_ns,
                    "read_return_monotonic_ns": captured.read_return_monotonic_ns,
                    "capture_interval_ns": captured.capture_interval_ns,
                    "capture_thread_interval_ns": captured.capture_thread_interval_ns,
                    "publisher_dequeue_monotonic_ns": dequeue_ns,
                    "encode_started_monotonic_ns": encode_start_ns,
                    "base64_started_monotonic_ns": base64_start_ns,
                    "base64_completed_monotonic_ns": base64_completed_ns,
                    "json_started_monotonic_ns": json_start_ns,
                    "realsense_frame_number": captured.realsense_frame_number,
                    "realsense_frame_gap": captured.realsense_frame_gap,
                    "realsense_device_timestamp_ms": captured.realsense_device_timestamp_ms,
                    "realsense_device_interval_ms": captured.realsense_device_interval_ms,
                    "realsense_timestamp_domain": captured.realsense_timestamp_domain,
                }
                message = build_message(
                    self.preset,
                    sequence=captured.sequence,
                    capture_wall_ns=captured.capture_wall_ns,
                    capture_monotonic_ns=captured.capture_monotonic_ns,
                    encode_completed_monotonic_ns=encode_completed_ns,
                    jpeg=jpeg,
                    jpeg_quality=self.jpeg_quality,
                    image_b64=image_b64,
                    extra_timing=extra_timing,
                )
                json_completed_ns = time.monotonic_ns()
                send_start_ns = json_completed_ns
                sent = True
                try:
                    socket.send_string(message, flags=zmq.NOBLOCK)
                except zmq.Again:
                    sent = False
                send_completed_ns = time.monotonic_ns()
                if not sent:
                    raise RuntimeError(f"{self.preset.camera_id} ZMQ send would block")
                self.frames_sent = captured.sequence
                self._log_stage_trace(
                    captured,
                    dequeue_ns=dequeue_ns,
                    encode_start_ns=encode_start_ns,
                    encode_completed_ns=encode_completed_ns,
                    base64_start_ns=base64_start_ns,
                    base64_completed_ns=base64_completed_ns,
                    json_start_ns=json_start_ns,
                    json_completed_ns=json_completed_ns,
                    send_start_ns=send_start_ns,
                    send_completed_ns=send_completed_ns,
                    jpeg_bytes=len(jpeg),
                    started_ns=started_ns,
                )
        except BaseException as exc:
            self.exception = exc
            self.stop_event.set()
        finally:
            socket.close()

    def _log_stage_trace(
        self,
        captured: CapturedFrame,
        *,
        dequeue_ns: int,
        encode_start_ns: int,
        encode_completed_ns: int,
        base64_start_ns: int,
        base64_completed_ns: int,
        json_start_ns: int,
        json_completed_ns: int,
        send_start_ns: int,
        send_completed_ns: int,
        jpeg_bytes: int,
        started_ns: int,
    ) -> None:
        def ms(end_ns: int, start_ns: int) -> float:
            return (end_ns - start_ns) / 1_000_000

        capture_interval_ms = (
            None if captured.capture_interval_ns is None else captured.capture_interval_ns / 1_000_000
        )
        read_ms = ms(captured.read_return_monotonic_ns, captured.read_enter_monotonic_ns)
        encode_ms = ms(encode_completed_ns, encode_start_ns)
        base64_ms = ms(base64_completed_ns, base64_start_ns)
        json_ms = ms(json_completed_ns, json_start_ns)
        send_ms = ms(send_completed_ns, send_start_ns)
        queue_ms = ms(dequeue_ns, captured.capture_monotonic_ns)
        if capture_interval_ms is not None:
            self.capture_intervals_ms.append(capture_interval_ms)
        self.read_durations_ms.append(read_ms)
        self.encode_durations_ms.append(encode_ms)
        slow = (
            max(capture_interval_ms or 0.0, read_ms, encode_ms, base64_ms, json_ms, send_ms, queue_ms) > 50
            or (captured.realsense_frame_gap or 0) > 0
        )
        if self.trace_every_frame or slow:
            trace = {
                "event": "camera_stage_trace",
                "slow": slow,
                "camera": self.preset.camera_id,
                "sequence": captured.sequence,
                "realsense_frame_number": captured.realsense_frame_number,
                "realsense_frame_gap": captured.realsense_frame_gap,
                "realsense_device_timestamp_ms": captured.realsense_device_timestamp_ms,
                "realsense_device_interval_ms": captured.realsense_device_interval_ms,
                "capture_interval_ms": capture_interval_ms,
                "capture_thread_interval_ms": None
                if captured.capture_thread_interval_ns is None
                else captured.capture_thread_interval_ns / 1_000_000,
                "read_ms": read_ms,
                "queue_ms": queue_ms,
                "encode_ms": encode_ms,
                "base64_ms": base64_ms,
                "json_ms": json_ms,
                "send_ms": send_ms,
                "jpeg_bytes": jpeg_bytes,
            }
            print(f"[direct realsense zmq] stage={json.dumps(trace, separators=(',', ':'))}", flush=True)
        if captured.sequence == 1 or captured.sequence % (self.preset.fps * 10) == 0:
            elapsed_s = (time.monotonic_ns() - started_ns) / 1_000_000_000
            print(
                f"[direct realsense zmq] camera={self.preset.camera_id} sequence={captured.sequence} "
                f"capture_hz={captured.sequence / elapsed_s:.3f} jpeg_bytes={jpeg_bytes} "
                f"capture_interval_ms_p95={self._percentile(self.capture_intervals_ms, 0.95):.3f} "
                f"capture_interval_ms_max={max(self.capture_intervals_ms, default=0.0):.3f} "
                f"read_ms_p95={self._percentile(self.read_durations_ms, 0.95):.3f} "
                f"read_ms_max={max(self.read_durations_ms, default=0.0):.3f} "
                f"encode_ms_p95={self._percentile(self.encode_durations_ms, 0.95):.3f} "
                f"encode_ms_max={max(self.encode_durations_ms, default=0.0):.3f}",
                flush=True,
            )


def _run_camera_process(
    preset: DirectCameraPreset,
    bind_host: str,
    jpeg_quality: int,
    count: int,
    trace_every_frame: bool,
    stop_event: Any,
    ready_queue: Any,
    error_queue: Any,
) -> None:
    context = zmq.Context()
    worker = CameraPublisherWorker(
        preset,
        context,
        bind_host=bind_host,
        jpeg_quality=jpeg_quality,
        count=count,
        stop_event=stop_event,
        trace_every_frame=trace_every_frame,
    )
    ready_reported = False
    try:
        worker.start()
        while worker.thread.is_alive():
            if worker.ready.is_set() and not ready_reported:
                ready_queue.put(preset.camera_id)
                ready_reported = True
            if worker.exception is not None:
                raise worker.exception
            worker.join(timeout=0.05)
        if worker.exception is not None:
            raise worker.exception
        if not ready_reported:
            raise RuntimeError(f"{preset.camera_id} exited before readiness")
    except BaseException as exc:
        error_queue.put((preset.camera_id, f"{type(exc).__name__}: {exc}"))
        stop_event.set()
        raise
    finally:
        if worker.exception is not None:
            stop_event.set()
        worker.join(timeout=5)
        context.term()


class DirectRealSenseZmqServer:
    def __init__(
        self,
        *,
        bind_host: str = "*",
        jpeg_quality: int = 95,
        count: int = 0,
        trace_every_frame: bool = False,
        presets: tuple[DirectCameraPreset, ...] | None = None,
    ) -> None:
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        self.bind_host = bind_host
        self.jpeg_quality = jpeg_quality
        self.count = count
        self.trace_every_frame = trace_every_frame
        self.presets = presets or tuple(CAMERA_PRESETS.values())
        self.mp_context = multiprocessing.get_context("spawn")
        self.stop_event = self.mp_context.Event()
        self.ready_queue = self.mp_context.Queue()
        self.error_queue = self.mp_context.Queue()
        self.processes = tuple(
            self.mp_context.Process(
                target=_run_camera_process,
                args=(
                    preset,
                    bind_host,
                    jpeg_quality,
                    count,
                    trace_every_frame,
                    self.stop_event,
                    self.ready_queue,
                    self.error_queue,
                ),
                name=f"jz-realsense-process-{preset.camera_id}",
                daemon=False,
            )
            for preset in self.presets
        )

    def request_stop(self) -> None:
        self.stop_event.set()

    def run(self) -> int:
        for process in self.processes:
            process.start()
        try:
            deadline = time.monotonic() + 20
            ready_cameras: set[str] = set()
            while len(ready_cameras) < len(self.processes):
                self._raise_if_failed()
                if time.monotonic() >= deadline:
                    raise TimeoutError("timed out opening all RealSense cameras")
                try:
                    ready_cameras.add(self.ready_queue.get(timeout=0.05))
                except queue.Empty:
                    pass
            print("[direct realsense zmq] all cameras ready", flush=True)
            while not self.stop_event.is_set():
                self._raise_if_failed()
                if self.count > 0 and all(not process.is_alive() for process in self.processes):
                    break
                time.sleep(0.05)
            self._raise_if_failed()
            return 0
        finally:
            self.stop_event.set()
            for process in self.processes:
                process.join(timeout=5)
            for process in self.processes:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=2)
            self.ready_queue.close()
            self.error_queue.close()

    def _raise_if_failed(self) -> None:
        try:
            camera_id, error = self.error_queue.get_nowait()
        except queue.Empty:
            camera_id = None
            error = None
        if camera_id is not None:
            raise RuntimeError(f"camera process failed: {camera_id}: {error}")
        for process in self.processes:
            if process.exitcode not in (None, 0):
                raise RuntimeError(
                    f"camera process failed: name={process.name} exitcode={process.exitcode}"
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve Orin RealSense RGB frames over ZMQ")
    parser.add_argument("--bind-host", default="*")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--count", type=int, default=0)
    parser.add_argument("--trace-every-frame", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    server = DirectRealSenseZmqServer(
        bind_host=args.bind_host,
        jpeg_quality=args.jpeg_quality,
        count=args.count,
        trace_every_frame=args.trace_every_frame,
    )
    signal.signal(signal.SIGINT, lambda _signum, _frame: server.request_stop())
    signal.signal(signal.SIGTERM, lambda _signum, _frame: server.request_stop())
    return server.run()


if __name__ == "__main__":
    raise SystemExit(main())
