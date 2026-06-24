from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from .buffers import SampleBuffer
from .config import BridgeCaptureConfig
from .raw_episode import RawEpisodeWriter
from .rtsp_receiver import OpenCvRtspReceiver
from .time_utils import now_monotonic_ns, now_wall_time_ns
from .types import TimestampedPayload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record raw JZRobot episodes from RTSP and ROS vector topics.")
    parser.add_argument("--config", required=True, help="Path to bridge_capture YAML.")
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument(
        "--no-cameras",
        action="store_true",
        help="Record only ROS vectors. Useful for first network/topic validation.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = BridgeCaptureConfig.from_yaml(args.config)
    recorder = RawDatasetRecorder(config=config, episode_id=args.episode_id, task=args.task)
    recorder.record(duration_s=args.duration_s, enable_cameras=not args.no_cameras)


class RawDatasetRecorder:
    def __init__(self, *, config: BridgeCaptureConfig, episode_id: str, task: str):
        self.config = config
        self.episode_id = episode_id
        self.task = task
        self.camera_buffers = {name: SampleBuffer(maxlen=256) for name in config.cameras}
        self.state_buffer = SampleBuffer(maxlen=1024)
        self.action_buffer = SampleBuffer(maxlen=1024)
        self.receivers: list[OpenCvRtspReceiver] = []
        self._cameras_enabled = True

    def record(self, *, duration_s: float, enable_cameras: bool = True) -> Path:
        writer = RawEpisodeWriter(
            root=self.config.recording.output_root,
            episode_id=self.episode_id,
            metadata=self._metadata(),
        )
        writer.start()
        writer.write_event("start", {"task": self.task})
        self._start_ros_subscribers()
        self._cameras_enabled = enable_cameras
        if enable_cameras:
            self._start_camera_receivers()

        period_s = 1.0 / self.config.recording.sample_rate_hz
        sample_count = int(duration_s * self.config.recording.sample_rate_hz)
        try:
            for sample_index in range(sample_count):
                capture_time_ns = now_wall_time_ns()
                writer.write_sample(self._build_sample(sample_index, capture_time_ns, writer.tmp_dir))
                time.sleep(period_s)
        finally:
            for receiver in self.receivers:
                receiver.stop()
            self._stop_ros_subscribers()
            writer.write_event("stop", {"duration_s": duration_s})
        return writer.finish()

    def _metadata(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "robot": self.config.robot.__dict__,
            "sample_rate_hz": self.config.recording.sample_rate_hz,
            "state_names": self.config.state.names,
            "action_names": self.config.action.names,
            "cameras": {name: camera.__dict__ for name, camera in self.config.cameras.items()},
        }

    def _start_camera_receivers(self) -> None:
        for name, camera in self.config.cameras.items():
            receiver = OpenCvRtspReceiver(name, camera, self.camera_buffers[name])
            receiver.start()
            self.receivers.append(receiver)

    def _start_ros_subscribers(self) -> None:
        # Import lazily so unit tests and offline conversion do not require ROS 2.
        try:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.node import Node
            from std_msgs.msg import Float64MultiArray
        except Exception as exc:
            raise ImportError("raw_recorder must run in a ROS 2 Python environment with rclpy installed") from exc

        if not rclpy.ok():
            rclpy.init()

        recorder = self

        class RecorderRosNode(Node):
            def __init__(self) -> None:
                super().__init__("lerobot_raw_dataset_recorder")
                self.create_subscription(Float64MultiArray, recorder.config.state.topic, self._state_cb, 10)
                self.create_subscription(Float64MultiArray, recorder.config.action.topic, self._action_cb, 10)

            def _state_cb(self, message: Any) -> None:
                receive_time_ns = now_wall_time_ns()
                recorder.state_buffer.append(
                    TimestampedPayload(
                        source_time_ns=receive_time_ns,
                        receive_time_ns=receive_time_ns,
                        payload=[float(value) for value in message.data],
                    )
                )

            def _action_cb(self, message: Any) -> None:
                receive_time_ns = now_wall_time_ns()
                recorder.action_buffer.append(
                    TimestampedPayload(
                        source_time_ns=receive_time_ns,
                        receive_time_ns=receive_time_ns,
                        payload=[float(value) for value in message.data],
                    )
                )

        self._ros_node = RecorderRosNode()
        self._ros_executor = SingleThreadedExecutor()
        self._ros_executor.add_node(self._ros_node)

        import threading

        self._ros_thread = threading.Thread(target=self._ros_executor.spin, daemon=True)
        self._ros_thread.start()

    def _stop_ros_subscribers(self) -> None:
        executor = getattr(self, "_ros_executor", None)
        node = getattr(self, "_ros_node", None)
        thread = getattr(self, "_ros_thread", None)
        if executor is not None:
            executor.shutdown()
        if thread is not None:
            thread.join(timeout=2.0)
        if node is not None:
            node.destroy_node()

    def _build_sample(self, sample_index: int, capture_time_ns: int, episode_dir: Path) -> dict[str, Any]:
        camera_max_delta_ns = int(self.config.recording.max_camera_delta_ms * 1_000_000)
        state_max_delta_ns = int(self.config.recording.max_state_delta_ms * 1_000_000)
        action_max_delta_ns = int(self.config.recording.max_action_delta_ms * 1_000_000)

        sample: dict[str, Any] = {
            "sample_index": sample_index,
            "capture_time_ns": capture_time_ns,
            "monotonic_time_ns": now_monotonic_ns(),
            "valid": True,
            "invalid_reason": "",
        }

        invalid_reasons = []
        if self._cameras_enabled:
            for camera_name, buffer in self.camera_buffers.items():
                match = buffer.latest_not_after(capture_time_ns, camera_max_delta_ns)
                if match is None:
                    invalid_reasons.append(f"{camera_name}_missing")
                    sample[f"{camera_name}_frame_index"] = -1
                    sample[f"{camera_name}_delta_ms"] = None
                else:
                    frame = match.payload
                    frame_relpath = self._write_frame(episode_dir, camera_name, sample_index, frame.image)
                    sample[f"{camera_name}_frame_index"] = frame.frame_index
                    sample[f"{camera_name}_frame_path"] = frame_relpath
                    sample[f"{camera_name}_pts_ns"] = frame.pts_ns
                    sample[f"{camera_name}_delta_ms"] = match.delta_ns / 1_000_000

        state_match = self.state_buffer.latest_not_after(capture_time_ns, state_max_delta_ns)
        action_match = self.action_buffer.latest_not_after(capture_time_ns, action_max_delta_ns)
        if state_match is None:
            invalid_reasons.append("state_missing")
            sample["state"] = []
            sample["state_delta_ms"] = None
        else:
            sample["state"] = state_match.payload
            sample["state_delta_ms"] = state_match.delta_ns / 1_000_000
        if action_match is None:
            invalid_reasons.append("action_missing")
            sample["action"] = []
            sample["action_delta_ms"] = None
        else:
            sample["action"] = action_match.payload
            sample["action_delta_ms"] = action_match.delta_ns / 1_000_000

        if invalid_reasons:
            sample["valid"] = False
            sample["invalid_reason"] = ",".join(invalid_reasons)
        return sample

    def _write_frame(self, episode_dir: Path, camera_name: str, sample_index: int, image: Any) -> str:
        import cv2

        camera_dir = episode_dir / "frames" / camera_name
        camera_dir.mkdir(parents=True, exist_ok=True)
        suffix = self.config.recording.image_format.lower().lstrip(".")
        relpath = Path("frames") / camera_name / f"{sample_index:06d}.{suffix}"
        output_path = episode_dir / relpath
        params = []
        if suffix in {"jpg", "jpeg"}:
            params = [cv2.IMWRITE_JPEG_QUALITY, self.config.recording.image_quality]
        cv2.imwrite(str(output_path), image, params)
        return relpath.as_posix()


if __name__ == "__main__":
    main()
