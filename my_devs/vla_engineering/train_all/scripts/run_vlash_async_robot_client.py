#!/usr/bin/env python
from __future__ import annotations

import argparse
import base64
import json
import os
import queue
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path("/data/cqy_workspace/flexible_lerobot")
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())


DEFAULT_TASK = "Put the eraser into the small box"
DEFAULT_ROBOT_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00"
DEFAULT_CALIB_DIR = "/home/cqy/.cache/huggingface/lerobot/calibration/robots/so_follower"
ACTION_NAMES = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)


@dataclass
class InferenceTask:
    seq: int
    payload: dict[str, Any]
    raw_observation: dict[str, Any]
    submitted_at: float
    trigger: str


@dataclass
class ActionChunk:
    seq: int
    actions: np.ndarray
    raw_observation: dict[str, Any]
    submitted_at: float
    received_at: float
    trigger: str
    server_timing: dict[str, Any]

    @property
    def infer_ms(self) -> float:
        return (self.received_at - self.submitted_at) * 1000.0


@dataclass
class InferenceFailure:
    seq: int
    error: str
    traceback_text: str
    submitted_at: float
    received_at: float
    trigger: str


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {value}")


def optional_float(value: str | None) -> float | None:
    if value is None or value.strip().lower() in {"", "none", "null"}:
        return None
    return float(value)


def maybe_path(path_str: str | None) -> Path | None:
    return None if not path_str else Path(path_str).expanduser()


def parse_camera(value: str) -> int | Path:
    return int(value) if value.isdecimal() else Path(value).expanduser()


def encode_array(value: np.ndarray) -> dict[str, Any]:
    value = np.ascontiguousarray(value)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "data_b64": base64.b64encode(value.tobytes()).decode("ascii"),
    }


def _image_to_hwc_uint8(value: Any) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D image, got shape={image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.moveaxis(image, 0, -1)
    if np.issubdtype(image.dtype, np.floating):
        scale = 255.0 if float(np.nanmax(image)) <= 1.5 else 1.0
        image = np.clip(image * scale, 0, 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8, copy=False)
    return np.ascontiguousarray(image)


def _state_to_float32(value: Any) -> np.ndarray:
    state = np.asarray(value, dtype=np.float32)
    if state.shape != (6,):
        raise ValueError(f"Expected SO101 state shape=(6,), got shape={state.shape}")
    return np.ascontiguousarray(state)


def _post_json(url: str, payload: dict[str, Any], timeout_s: float = 60.0) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc


def _get_json(url: str, timeout_s: float = 10.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


class AsyncInferenceWorker:
    def __init__(self, infer_url: str, *, timeout_s: float, queue_size: int = 1) -> None:
        self.infer_url = infer_url
        self.timeout_s = timeout_s
        self.tasks: queue.Queue[InferenceTask | None] = queue.Queue(maxsize=queue_size)
        self.results: queue.Queue[ActionChunk | InferenceFailure] = queue.Queue()
        self.thread = threading.Thread(target=self._run, name="vlash-http-inference", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        try:
            self.tasks.put_nowait(None)
        except queue.Full:
            pass
        self.thread.join(timeout=2.0)

    def submit(self, task: InferenceTask) -> bool:
        try:
            self.tasks.put_nowait(task)
        except queue.Full:
            return False
        return True

    def pop_result(self, timeout_s: float = 0.0) -> ActionChunk | InferenceFailure | None:
        try:
            if timeout_s <= 0:
                return self.results.get_nowait()
            return self.results.get(timeout=timeout_s)
        except queue.Empty:
            return None

    def wait_result(self, timeout_s: float) -> ActionChunk:
        deadline = time.perf_counter() + timeout_s
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError(f"Timed out waiting for async VLASH inference after {timeout_s:.1f}s.")
            result = self.pop_result(timeout_s=min(remaining, 0.25))
            if result is None:
                continue
            if isinstance(result, InferenceFailure):
                raise RuntimeError(
                    f"Async inference request seq={result.seq} failed: {result.error}\n"
                    f"{result.traceback_text}"
                )
            return result

    def _run(self) -> None:
        while True:
            task = self.tasks.get()
            if task is None:
                return
            try:
                result = _post_json(self.infer_url, task.payload, timeout_s=self.timeout_s)
                if not result.get("ok", False):
                    raise RuntimeError(f"Server returned error: {result}")
                actions = np.asarray(result["actions"], dtype=np.float32)
                if actions.ndim != 2 or actions.shape[1] != 6:
                    raise ValueError(f"Expected action chunk shape=(horizon, 6), got shape={actions.shape}")
                if not np.isfinite(actions).all():
                    raise ValueError("Server returned NaN or Inf action values.")
                self.results.put(
                    ActionChunk(
                        seq=task.seq,
                        actions=actions,
                        raw_observation=task.raw_observation,
                        submitted_at=task.submitted_at,
                        received_at=time.perf_counter(),
                        trigger=task.trigger,
                        server_timing=result.get("server_timing", {}),
                    )
                )
            except Exception as exc:
                self.results.put(
                    InferenceFailure(
                        seq=task.seq,
                        error=f"{type(exc).__name__}: {exc}",
                        traceback_text=traceback.format_exc(),
                        submitted_at=task.submitted_at,
                        received_at=time.perf_counter(),
                        trigger=task.trigger,
                    )
                )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SO101 robot with VLASH-style async HTTP prefetch.")
    parser.add_argument("--host", default=os.getenv("VLASH_SERVER_HOST", os.getenv("HOST", "localhost")))
    parser.add_argument("--port", type=int, default=int(os.getenv("VLASH_SERVER_PORT", os.getenv("PORT", "8005"))))
    parser.add_argument("--task", default=os.getenv("DATASET_TASK", DEFAULT_TASK))
    parser.add_argument("--robot-id", default=os.getenv("ROBOT_ID", "hfy_follower"))
    parser.add_argument("--robot-port", default=os.getenv("ROBOT_PORT", DEFAULT_ROBOT_PORT))
    parser.add_argument("--calib-dir", default=os.getenv("CALIB_DIR", DEFAULT_CALIB_DIR))
    parser.add_argument(
        "--max-relative-target",
        type=optional_float,
        default=optional_float(os.getenv("MAX_RELATIVE_TARGET", "10")),
        help="SO follower safety clip per motor target. Use none/null to disable.",
    )
    parser.add_argument("--top-cam", type=parse_camera, default=parse_camera(os.getenv("TOP_CAM", "/dev/video4")))
    parser.add_argument("--wrist-cam", type=parse_camera, default=parse_camera(os.getenv("WRIST_CAM", "/dev/video6")))
    parser.add_argument("--top-cam-fourcc", default=os.getenv("TOP_CAM_FOURCC", "YUYV"))
    parser.add_argument("--wrist-cam-fourcc", default=os.getenv("WRIST_CAM_FOURCC", "MJPG"))
    parser.add_argument("--img-width", type=int, default=int(os.getenv("IMG_WIDTH", "640")))
    parser.add_argument("--img-height", type=int, default=int(os.getenv("IMG_HEIGHT", "480")))
    parser.add_argument("--fps", type=int, default=int(os.getenv("FPS", "30")))
    parser.add_argument("--run-time-s", type=float, default=float(os.getenv("RUN_TIME_S", "0")))
    parser.add_argument("--log-interval", type=int, default=int(os.getenv("LOG_INTERVAL", "10")))
    parser.add_argument("--execute-actions", type=parse_bool, default=parse_bool(os.getenv("EXECUTE_ACTIONS", "false")))
    parser.add_argument(
        "--action-chunk-steps",
        type=int,
        default=int(os.getenv("ACTION_CHUNK_STEPS", "40")),
        help="Preferred switch point in each returned action chunk.",
    )
    parser.add_argument(
        "--prefetch-at-step",
        type=int,
        default=int(os.getenv("PREFETCH_AT_STEP", "20")),
        help="Submit the next /infer request when this many actions of the current chunk have been executed.",
    )
    parser.add_argument(
        "--max-chunk-execute-steps",
        type=int,
        default=int(os.getenv("MAX_CHUNK_EXECUTE_STEPS", "50")),
        help="Maximum number of actions to consume from a chunk when async prefetch is late.",
    )
    parser.add_argument(
        "--allow-tail-fallback",
        type=parse_bool,
        default=parse_bool(os.getenv("ALLOW_TAIL_FALLBACK", "true")),
        help="If next chunk is late at action_chunk_steps, keep consuming the current chunk tail.",
    )
    parser.add_argument("--inference-timeout-s", type=float, default=float(os.getenv("INFERENCE_TIMEOUT_S", "60")))
    parser.add_argument("--motor-io-retries", type=int, default=int(os.getenv("MOTOR_IO_RETRIES", "10")))
    parser.add_argument("--dry-run", type=parse_bool, default=parse_bool(os.getenv("DRY_RUN", "false")))
    return parser


def _build_robot(args: argparse.Namespace):
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots import make_robot_from_config, so_follower  # noqa: F401
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

    cameras = {
        "top": OpenCVCameraConfig(
            index_or_path=args.top_cam,
            width=args.img_width,
            height=args.img_height,
            fps=args.fps,
            fourcc=args.top_cam_fourcc,
        ),
        "wrist": OpenCVCameraConfig(
            index_or_path=args.wrist_cam,
            width=args.img_width,
            height=args.img_height,
            fps=args.fps,
            fourcc=args.wrist_cam_fourcc,
        ),
    }
    config = SOFollowerRobotConfig(
        id=args.robot_id,
        calibration_dir=maybe_path(args.calib_dir),
        port=args.robot_port,
        max_relative_target=args.max_relative_target,
        cameras=cameras,
    )
    return make_robot_from_config(config), config


def _build_dataset_features(robot: Any) -> tuple[dict[str, Any], Any, Any]:
    from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
    from lerobot.datasets.utils import combine_feature_dicts
    from lerobot.processor import make_default_processors

    _, robot_action_processor, robot_observation_processor = make_default_processors()
    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=robot_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True,
        ),
    )
    return dataset_features, robot_action_processor, robot_observation_processor


def _build_request(
    robot: Any,
    robot_observation_processor: Any,
    dataset_features: dict[str, Any],
    task: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from lerobot.datasets.utils import build_dataset_frame
    from lerobot.utils.constants import OBS_STR

    raw_observation = robot.get_observation()
    processed_observation = robot_observation_processor(raw_observation)
    frame = build_dataset_frame(dataset_features, processed_observation, prefix=OBS_STR)
    request = {
        "observation.images.top": encode_array(_image_to_hwc_uint8(frame["observation.images.top"])),
        "observation.images.wrist": encode_array(_image_to_hwc_uint8(frame["observation.images.wrist"])),
        "observation.state": encode_array(_state_to_float32(frame["observation.state"])),
        "prompt": task,
    }
    return request, raw_observation


def _action_to_robot_action(action: np.ndarray, dataset_features: dict[str, Any]) -> dict[str, Any]:
    from lerobot.policies.utils import make_robot_action

    action = np.asarray(action, dtype=np.float32)
    if action.shape != (6,):
        raise ValueError(f"Expected one SO101 action shape=(6,), got shape={action.shape}")
    return make_robot_action(torch.as_tensor(action, dtype=torch.float32).unsqueeze(0), dataset_features)


def _patch_motor_bus_retries(robot: Any, retries: int) -> None:
    retries = max(int(retries), 0)
    if retries <= 0:
        print("[DIAG] Motor bus I/O uses native LeRobot retry behavior.")
        return

    bus = getattr(robot, "bus", None)
    if bus is None:
        print("[DIAG] Robot has no motor bus; skip motor I/O retry patch.")
        return

    original_read = bus.read
    original_sync_read = bus.sync_read
    original_write = bus.write
    original_sync_write = bus.sync_write

    def read_with_min_retries(data_name: str, motor: str, *, normalize: bool = True, num_retry: int = 0) -> Any:
        return original_read(data_name, motor, normalize=normalize, num_retry=max(num_retry, retries))

    def sync_read_with_min_retries(
        data_name: str,
        motors: str | list[str] | None = None,
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> Any:
        return original_sync_read(data_name, motors, normalize=normalize, num_retry=max(num_retry, retries))

    def write_with_min_retries(
        data_name: str,
        motor: str | None,
        value: Any,
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> Any:
        return original_write(data_name, motor, value, normalize=normalize, num_retry=max(num_retry, retries))

    def sync_write_with_min_retries(
        data_name: str,
        values: Any,
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> Any:
        return original_sync_write(data_name, values, normalize=normalize, num_retry=max(num_retry, retries))

    bus.read = read_with_min_retries
    bus.sync_read = sync_read_with_min_retries
    bus.write = write_with_min_retries
    bus.sync_write = sync_write_with_min_retries
    print(f"[DIAG] Motor bus read/write calls will use at least {retries} retries.")


def _raise_if_failure(result: ActionChunk | InferenceFailure | None) -> ActionChunk | None:
    if result is None:
        return None
    if isinstance(result, InferenceFailure):
        raise RuntimeError(
            f"Async inference request seq={result.seq} failed: {result.error}\n"
            f"{result.traceback_text}"
        )
    return result


def main() -> int:
    args = build_parser().parse_args()
    base_url = f"http://{args.host}:{args.port}"
    infer_url = f"{base_url}/infer"
    print(f"[INFO] Repo root: {REPO_ROOT}")
    print(f"[INFO] Server: {base_url}")
    print(f"[INFO] Task: {args.task}")
    print(f"[INFO] Robot id: {args.robot_id}")
    print(f"[INFO] Robot port: {args.robot_port}")
    print(f"[INFO] Calibration dir: {args.calib_dir}")
    print(f"[INFO] Cameras: top={args.top_cam} wrist={args.wrist_cam}")
    print(f"[INFO] execute_actions: {args.execute_actions}")
    print(f"[INFO] action_chunk_steps: {args.action_chunk_steps}")
    print(f"[INFO] prefetch_at_step: {args.prefetch_at_step}")
    print(f"[INFO] max_chunk_execute_steps: {args.max_chunk_execute_steps}")
    print(f"[INFO] allow_tail_fallback: {args.allow_tail_fallback}")
    print(f"[INFO] run_time_s: {args.run_time_s} (<=0 means until Ctrl+C)")

    if args.action_chunk_steps < 1:
        raise ValueError("--action-chunk-steps must be >= 1")
    if args.prefetch_at_step < 0:
        raise ValueError("--prefetch-at-step must be >= 0")
    if args.max_chunk_execute_steps < args.action_chunk_steps:
        raise ValueError("--max-chunk-execute-steps must be >= --action-chunk-steps")

    if args.dry_run:
        print("[INFO] DRY_RUN=true, exiting before server or hardware access.")
        return 0

    worker = AsyncInferenceWorker(infer_url, timeout_s=args.inference_timeout_s)
    robot = None
    try:
        metadata = _get_json(f"{base_url}/metadata")
        print(f"[INFO] Server metadata: {metadata}")

        robot, robot_config = _build_robot(args)
        _patch_motor_bus_retries(robot, args.motor_io_retries)
        dataset_features, robot_action_processor, robot_observation_processor = _build_dataset_features(robot)
        if dataset_features["action"]["names"] != list(ACTION_NAMES):
            raise ValueError(f"Unexpected action names: {dataset_features['action']['names']}")

        print(f"[INFO] Robot config type resolved by current LeRobot registry: {robot_config.type}")
        print(f"[INFO] Action names: {dataset_features['action']['names']}")

        worker.start()
        robot.connect()
        print("[INFO] Robot connected. Starting VLASH-style async runtime.")

        seq = 0
        pending = False
        next_chunk: ActionChunk | None = None
        current_chunk: ActionChunk | None = None
        chunk_index = 0
        chunk_step = 0
        global_step = 0
        miss_count = 0
        async_hit_count = 0
        start_t = time.perf_counter()
        end_t = start_t + args.run_time_s if args.run_time_s > 0 else None

        def submit_request(trigger: str) -> bool:
            nonlocal seq, pending
            request_t = time.perf_counter()
            payload, raw_observation = _build_request(
                robot,
                robot_observation_processor,
                dataset_features,
                args.task,
            )
            seq += 1
            task = InferenceTask(
                seq=seq,
                payload=payload,
                raw_observation=raw_observation,
                submitted_at=request_t,
                trigger=trigger,
            )
            if not worker.submit(task):
                return False
            pending = True
            return True

        print("[INFO] Submitting initial inference request.")
        if not submit_request("initial"):
            raise RuntimeError("Failed to submit initial inference request.")
        current_chunk = worker.wait_result(args.inference_timeout_s)
        pending = False
        print(
            f"[INFO] Initial chunk seq={current_chunk.seq} ready "
            f"infer_ms={current_chunk.infer_ms:.1f} shape={current_chunk.actions.shape}"
        )

        while True:
            if end_t is not None and time.perf_counter() >= end_t:
                print("[INFO] Reached requested run_time_s. Exiting.")
                break

            result = _raise_if_failure(worker.pop_result())
            if result is not None:
                pending = False
                next_chunk = result
                print(
                    f"[INFO] Async chunk ready seq={result.seq} trigger={result.trigger} "
                    f"infer_ms={result.infer_ms:.1f} server_timing={result.server_timing}"
                )

            assert current_chunk is not None
            current_limit = min(args.max_chunk_execute_steps, current_chunk.actions.shape[0])
            switch_step = min(args.action_chunk_steps, current_limit)
            if chunk_step >= current_limit:
                if next_chunk is None and not pending:
                    print("[WARN] No next chunk pending at chunk tail; submitting emergency request.")
                    if not submit_request(f"emergency_after_seq_{current_chunk.seq}"):
                        raise RuntimeError("Failed to submit emergency inference request.")
                if next_chunk is None:
                    wait_start = time.perf_counter()
                    next_chunk = worker.wait_result(args.inference_timeout_s)
                    pending = False
                    miss_count += 1
                    print(f"[WARN] Waited {(time.perf_counter() - wait_start) * 1000:.1f}ms for next chunk.")
                current_chunk = next_chunk
                next_chunk = None
                chunk_index += 1
                chunk_step = 0
                print(f"[INFO] Switched to chunk seq={current_chunk.seq} chunk_index={chunk_index}")
                continue

            if chunk_step >= args.prefetch_at_step and not pending and next_chunk is None:
                trigger = f"seq_{current_chunk.seq}_step_{chunk_step}"
                submitted = submit_request(trigger)
                if submitted and args.log_interval > 0:
                    print(f"[INFO] Submitted async prefetch trigger={trigger}")

            action = current_chunk.actions[chunk_step]
            step_t = time.perf_counter()
            robot_action = _action_to_robot_action(action, dataset_features)
            if args.execute_actions:
                robot_action_to_send = robot_action_processor((robot_action, current_chunk.raw_observation))
                robot.send_action(robot_action_to_send)

            global_step += 1
            chunk_step += 1

            if chunk_step >= switch_step:
                result = _raise_if_failure(worker.pop_result())
                if result is not None:
                    pending = False
                    next_chunk = result

                if next_chunk is not None:
                    async_hit_count += 1
                    current_chunk = next_chunk
                    next_chunk = None
                    chunk_index += 1
                    chunk_step = 0
                    print(f"[INFO] Async switch to chunk seq={current_chunk.seq} chunk_index={chunk_index}")
                elif not args.allow_tail_fallback or chunk_step >= current_limit:
                    if not pending:
                        print("[WARN] Next chunk not pending at switch point; submitting emergency request.")
                        if not submit_request(f"switch_after_seq_{current_chunk.seq}"):
                            raise RuntimeError("Failed to submit switch-point inference request.")
                    wait_start = time.perf_counter()
                    current_chunk = worker.wait_result(args.inference_timeout_s)
                    pending = False
                    next_chunk = None
                    chunk_index += 1
                    chunk_step = 0
                    miss_count += 1
                    print(
                        f"[WARN] Sync wait at switch point: "
                        f"{(time.perf_counter() - wait_start) * 1000:.1f}ms; "
                        f"new seq={current_chunk.seq}"
                    )

            if args.log_interval > 0 and global_step % args.log_interval == 0:
                elapsed = time.perf_counter() - start_t
                action_text = np.array2string(
                    np.array([robot_action[name] for name in ACTION_NAMES], dtype=np.float32),
                    precision=3,
                    suppress_small=True,
                )
                print(
                    f"[INFO] Step {global_step} elapsed={elapsed:.2f}s chunk_index={chunk_index} "
                    f"chunk_step={chunk_step} pending={pending} next_ready={next_chunk is not None} "
                    f"async_hits={async_hit_count} misses={miss_count} action={action_text}"
                )

            dt_s = time.perf_counter() - step_t
            time.sleep(max(1 / args.fps - dt_s, 0.0))

    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt received. Stopping.")
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        return 1
    finally:
        worker.stop()
        if robot is not None and getattr(robot, "is_connected", False):
            try:
                robot.disconnect()
            except Exception:
                pass
        print("[INFO] Async client finished.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
