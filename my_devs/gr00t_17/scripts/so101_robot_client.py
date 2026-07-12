#!/usr/bin/env python
from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import msgpack
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
GR00T17_ROOT = Path(__file__).resolve().parents[1]
CLIENT_DEPS = GR00T17_ROOT / "tools" / "robot_client_deps"
SRC_ROOT = REPO_ROOT / "src"
for import_path in (CLIENT_DEPS, SRC_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

JOINT_KEYS = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)
LANGUAGE_KEY = "annotation.human.task_description"
KNOWN_TASKS = (
    "Put the eraser into the small box",
    "Move the cup back to the upper-right corner",
    "First put the eraser into the small box, then move the cup back to the upper-right corner",
)
ACTUATION_CONFIRMATION = "SO101_GR00T_N17"
DEFAULT_CHECKPOINT = (
    GR00T17_ROOT / "outputs" / "formal" / "so101_n17_b2_e10_20260711" / "train" / "checkpoint-63600"
)
DEFAULT_CALIBRATION_DIR = GR00T17_ROOT / "configs" / "calibration" / "so_follower"
DEFAULT_ROBOT_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00"


def ensure_within(path: Path, root: Path, *, must_exist: bool = False) -> Path:
    resolved = path.expanduser().resolve(strict=must_exist)
    allowed = root.expanduser().resolve(strict=True)
    try:
        resolved.relative_to(allowed)
    except ValueError as exc:
        raise ValueError(f"Path escapes allowed root: {resolved}") from exc
    return resolved


def parse_camera(value: str) -> int | Path:
    return int(value) if value.isdecimal() else Path(value).expanduser()


def encode_msgpack(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        output = io.BytesIO()
        np.save(output, value, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": output.getvalue()}
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported msgpack value: {type(value)!r}")


def decode_msgpack(value: Any) -> Any:
    if isinstance(value, dict) and "__ndarray_class__" in value:
        return np.load(io.BytesIO(value["as_npy"]), allow_pickle=False)
    return value


class LocalPolicyClient:
    def __init__(self, host: str, port: int, timeout_s: float):
        import zmq

        self.zmq = zmq
        self.host = host
        self.port = port
        self.timeout_ms = max(1, int(timeout_s * 1000))
        self.context = zmq.Context()
        self.socket = self._new_socket()

    def _new_socket(self):
        socket = self.context.socket(self.zmq.REQ)
        socket.setsockopt(self.zmq.LINGER, 0)
        socket.setsockopt(self.zmq.RCVTIMEO, self.timeout_ms)
        socket.setsockopt(self.zmq.SNDTIMEO, self.timeout_ms)
        socket.connect(f"tcp://{self.host}:{self.port}")
        return socket

    def _reset_socket(self) -> None:
        self.socket.close(linger=0)
        self.socket = self._new_socket()

    def call(self, endpoint: str, data: dict[str, Any] | None = None) -> Any:
        request = {"endpoint": endpoint, "data": data or {}}
        try:
            self.socket.send(msgpack.packb(request, default=encode_msgpack))
            result = msgpack.unpackb(self.socket.recv(), object_hook=decode_msgpack)
        except self.zmq.error.ZMQError:
            self._reset_socket()
            raise
        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(f"Policy server error: {result['error']}")
        return result

    def ping(self) -> dict[str, Any]:
        result = self.call("ping")
        if result.get("status") != "ok":
            raise RuntimeError(f"Unexpected policy server ping response: {result}")
        return result

    def get_action(self, observation: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict]:
        result = self.call("get_action", {"observation": observation, "options": None})
        if not isinstance(result, list | tuple) or len(result) != 2:
            raise RuntimeError(f"Unexpected policy response type: {type(result)!r}")
        return result[0], result[1]

    def close(self) -> None:
        self.socket.close(linger=0)
        self.context.term()


@dataclass(frozen=True)
class ActionBounds:
    lower: np.ndarray
    upper: np.ndarray
    source: str


def validate_checkpoint(checkpoint: Path) -> dict[str, Any]:
    checkpoint = ensure_within(checkpoint, GR00T17_ROOT, must_exist=True)
    required = (
        "config.json",
        "model.safetensors.index.json",
        "processor_config.json",
        "statistics.json",
        "trainer_state.json",
    )
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Checkpoint is missing inference artifacts: {missing}")
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shards = sorted(set(index["weight_map"].values()))
    for shard in shards:
        path = checkpoint / shard
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Checkpoint shard is missing or empty: {path}")
    trainer_state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    return {
        "path": str(checkpoint),
        "global_step": trainer_state.get("global_step"),
        "model_shards": shards,
        "model_bytes": sum((checkpoint / shard).stat().st_size for shard in shards),
    }


def load_action_bounds(checkpoint: Path, mode: str) -> ActionBounds:
    statistics = json.loads((checkpoint / "statistics.json").read_text(encoding="utf-8"))
    action_stats = statistics["new_embodiment"]["action"]
    if mode == "q01_q99":
        arm_lower = action_stats["single_arm"]["q01"]
        arm_upper = action_stats["single_arm"]["q99"]
        gripper_lower = action_stats["gripper"]["q01"]
        gripper_upper = action_stats["gripper"]["q99"]
    elif mode == "dataset_minmax":
        arm_lower = action_stats["single_arm"]["min"]
        arm_upper = action_stats["single_arm"]["max"]
        gripper_lower = action_stats["gripper"]["min"]
        gripper_upper = action_stats["gripper"]["max"]
    elif mode == "physical":
        arm_lower, arm_upper = [-100.0] * 5, [100.0] * 5
        gripper_lower, gripper_upper = [0.0], [100.0]
    else:
        raise ValueError(f"Unsupported bounds mode: {mode}")

    lower = np.asarray([*arm_lower, *gripper_lower], dtype=np.float32)
    upper = np.asarray([*arm_upper, *gripper_upper], dtype=np.float32)
    physical_lower = np.asarray([-100.0] * 5 + [0.0], dtype=np.float32)
    physical_upper = np.asarray([100.0] * 6, dtype=np.float32)
    return ActionBounds(
        lower=np.maximum(lower, physical_lower),
        upper=np.minimum(upper, physical_upper),
        source=mode,
    )


def validate_image(name: str, image: Any) -> dict[str, Any]:
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Camera {name} did not return a numpy array: {type(image)!r}")
    if image.dtype != np.uint8 or image.shape != (480, 640, 3):
        raise ValueError(f"Camera {name} expected uint8 (480, 640, 3), got {image.dtype} {image.shape}")
    if not image.flags.c_contiguous:
        image = np.ascontiguousarray(image)
    mean = float(image.mean())
    std = float(image.std())
    if mean <= 5 or mean >= 250 or std < 5:
        raise RuntimeError(f"Camera {name} appears blank or saturated: mean={mean:.3f}, std={std:.3f}")
    return {
        "shape": list(image.shape),
        "dtype": str(image.dtype),
        "min": int(image.min()),
        "max": int(image.max()),
        "mean": mean,
        "std": std,
    }


def extract_state(observation: dict[str, Any]) -> np.ndarray:
    state = np.asarray([observation[key] for key in JOINT_KEYS], dtype=np.float32)
    if state.shape != (6,) or not np.isfinite(state).all():
        raise RuntimeError(f"Invalid robot state: shape={state.shape}, values={state}")
    return state


def build_model_observation(observation: dict[str, Any], task: str) -> dict[str, Any]:
    state = extract_state(observation)
    top = np.ascontiguousarray(observation["top"], dtype=np.uint8)
    wrist = np.ascontiguousarray(observation["wrist"], dtype=np.uint8)
    return {
        "video": {
            "top": top[None, None, ...],
            "wrist": wrist[None, None, ...],
        },
        "state": {
            "single_arm": state[:5][None, None, ...],
            "gripper": state[5:][None, None, ...],
        },
        "language": {LANGUAGE_KEY: [[task]]},
    }


def validate_action_chunk(action: dict[str, Any]) -> np.ndarray:
    expected = {"single_arm": 5, "gripper": 1}
    arrays: dict[str, np.ndarray] = {}
    horizon = None
    for key, width in expected.items():
        if key not in action:
            raise KeyError(f"Policy action is missing key: {key}")
        value = np.asarray(action[key], dtype=np.float32)
        if value.ndim != 3 or value.shape[0] != 1 or value.shape[2] != width:
            raise ValueError(f"Action {key} expected (1, T, {width}), got {value.shape}")
        if not np.isfinite(value).all():
            raise RuntimeError(f"Action {key} contains NaN or Inf")
        if horizon is None:
            horizon = value.shape[1]
        elif value.shape[1] != horizon:
            raise RuntimeError("Arm and gripper action horizons do not match")
        arrays[key] = value
    if horizon != 16:
        raise RuntimeError(f"Expected the trained 16-step action horizon, got {horizon}")
    return np.concatenate([arrays["single_arm"][0], arrays["gripper"][0]], axis=-1)


def sanitize_action(
    predicted: np.ndarray,
    reference: np.ndarray,
    bounds: ActionBounds,
    max_command_delta: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    predicted = np.asarray(predicted, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    if predicted.shape != (6,) or reference.shape != (6,):
        raise ValueError(f"Expected two 6D vectors, got {predicted.shape} and {reference.shape}")
    if not np.isfinite(predicted).all() or not np.isfinite(reference).all():
        raise RuntimeError("Predicted action or reference state contains NaN/Inf")
    bounded = np.clip(predicted, bounds.lower, bounds.upper)
    delta_limited = np.clip(bounded, reference - max_command_delta, reference + max_command_delta)
    safe = np.clip(delta_limited, bounds.lower, bounds.upper).astype(np.float32)
    return safe, {
        "hard_bound_clipped": bool(not np.array_equal(predicted, bounded)),
        "delta_clipped": bool(not np.array_equal(bounded, delta_limited)),
        "predicted": predicted.tolist(),
        "safe": safe.tolist(),
        "reference": reference.tolist(),
    }


def action_to_robot_dict(action: np.ndarray) -> dict[str, float]:
    return {key: float(action[index]) for index, key in enumerate(JOINT_KEYS)}


def disconnect_robot_safely(robot: Any) -> list[str]:
    """Best-effort cleanup, including a bus-only partial connection."""
    errors: list[str] = []
    bus = getattr(robot, "bus", None)
    if bus is not None:
        try:
            if bus.is_connected:
                bus.disconnect(disable_torque=True)
        except BaseException as exc:
            errors.append(f"motor bus: {type(exc).__name__}: {exc}")

    for name, camera in getattr(robot, "cameras", {}).items():
        try:
            if camera.is_connected:
                camera.disconnect()
        except BaseException as exc:
            errors.append(f"camera {name}: {type(exc).__name__}: {exc}")
    return errors


def save_snapshot(output_dir: Path, name: str, rgb: np.ndarray) -> str:
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.png"
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Failed to save camera snapshot: {path}")
    return str(path)


def build_robot(args: argparse.Namespace):
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots import make_robot_from_config
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

    calibration_dir = ensure_within(args.calibration_dir, GR00T17_ROOT, must_exist=True)
    calibration_file = calibration_dir / f"{args.robot_id}.json"
    if not calibration_file.is_file():
        raise FileNotFoundError(f"Robot calibration file is missing: {calibration_file}")
    cameras = {
        "top": OpenCVCameraConfig(
            index_or_path=args.top_cam,
            width=640,
            height=480,
            fps=30,
            fourcc=args.top_cam_fourcc,
        ),
        "wrist": OpenCVCameraConfig(
            index_or_path=args.wrist_cam,
            width=640,
            height=480,
            fps=30,
            fourcc=args.wrist_cam_fourcc,
        ),
    }
    config = SOFollowerRobotConfig(
        id=args.robot_id,
        calibration_dir=calibration_dir,
        port=args.robot_port,
        max_relative_target=args.max_relative_target,
        disable_torque_on_disconnect=True,
        cameras=cameras,
    )
    robot = make_robot_from_config(config)
    if tuple(robot.action_features) != JOINT_KEYS:
        raise RuntimeError(
            f"Robot action feature order changed: {tuple(robot.action_features)} != {JOINT_KEYS}"
        )
    if set(robot.observation_features) != {*JOINT_KEYS, "top", "wrist"}:
        raise RuntimeError(f"Unexpected robot observation features: {robot.observation_features}")
    return robot, calibration_file


def prediction_summary(chunk: np.ndarray, state: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(chunk.shape),
        "min_by_joint": chunk.min(axis=0).tolist(),
        "max_by_joint": chunk.max(axis=0).tolist(),
        "first_action": chunk[0].tolist(),
        "first_action_delta_from_state": (chunk[0] - state).tolist(),
    }


def write_report(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path = ensure_within(path, GR00T17_ROOT)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite inference report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[REPORT] {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Safety-gated GR00T N1.7 client for SO101.")
    parser.add_argument("--mode", choices=("ping", "observe", "predict", "actuate"), required=True)
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=5555)
    parser.add_argument("--request-timeout-s", type=float, default=2.0)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--robot-id", default="hfy_follower")
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_CALIBRATION_DIR)
    parser.add_argument("--robot-port", default=DEFAULT_ROBOT_PORT)
    parser.add_argument("--top-cam", type=parse_camera, default=parse_camera("/dev/video4"))
    parser.add_argument("--wrist-cam", type=parse_camera, default=parse_camera("/dev/video6"))
    parser.add_argument("--top-cam-fourcc", default="YUYV")
    parser.add_argument("--wrist-cam-fourcc", default="MJPG")
    parser.add_argument("--camera-warmup-s", type=float, default=2.0)
    parser.add_argument("--task", default=KNOWN_TASKS[0])
    parser.add_argument("--allow-unknown-task", action="store_true")
    parser.add_argument("--prediction-count", type=int, default=2)
    parser.add_argument("--run-time-s", type=float, default=120.0)
    parser.add_argument("--execution-horizon", type=int, default=8)
    parser.add_argument("--control-hz", type=float, default=5.0)
    parser.add_argument(
        "--bounds-mode", choices=("q01_q99", "dataset_minmax", "physical"), default="physical"
    )
    parser.add_argument("--max-command-delta", type=float, default=3.0)
    parser.add_argument("--max-relative-target", type=float, default=3.0)
    parser.add_argument("--max-inference-s", type=float, default=2.0)
    parser.add_argument("--reject-clipped-action", action="store_true")
    parser.add_argument("--enable-actuation", action="store_true")
    parser.add_argument("--confirm-actuation", default="")
    parser.add_argument("--report", type=Path)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not CLIENT_DEPS.is_dir():
        raise FileNotFoundError(
            f"Robot client dependencies are missing: {CLIENT_DEPS}. Run scripts/setup_robot_client_deps.sh"
        )
    if args.server_host not in {"127.0.0.1", "localhost"}:
        raise ValueError("The guarded client only accepts a localhost policy server")
    if not 1 <= args.server_port <= 65535:
        raise ValueError(f"Invalid server port: {args.server_port}")
    if args.task not in KNOWN_TASKS and not args.allow_unknown_task:
        raise ValueError(f"Task was not present in training data: {args.task!r}")
    finite_positive = {
        "request_timeout_s": args.request_timeout_s,
        "run_time_s": args.run_time_s,
        "control_hz": args.control_hz,
        "max_command_delta": args.max_command_delta,
        "max_relative_target": args.max_relative_target,
        "max_inference_s": args.max_inference_s,
    }
    invalid = {
        name: value for name, value in finite_positive.items() if not math.isfinite(value) or value <= 0
    }
    if invalid:
        raise ValueError(f"Numeric safety arguments must be finite and positive: {invalid}")
    if not math.isfinite(args.camera_warmup_s) or not 0 <= args.camera_warmup_s <= 10:
        raise ValueError("camera_warmup_s must be finite and in [0, 10]")
    if not 1 <= args.execution_horizon <= 16:
        raise ValueError("execution_horizon must be in [1, 16]")
    if args.prediction_count < 1:
        raise ValueError("prediction_count must be positive")
    if args.control_hz > 30:
        raise ValueError("control_hz must be in (0, 30]")
    if args.mode == "actuate":
        if args.run_time_s > 600:
            raise ValueError("Guarded actuation run_time_s must not exceed 600 seconds")
        if args.max_inference_s > 10 or args.request_timeout_s > args.max_inference_s:
            raise ValueError("Actuation requires request_timeout_s <= max_inference_s <= 10 seconds")
        if not args.enable_actuation or args.confirm_actuation != ACTUATION_CONFIRMATION:
            raise PermissionError(
                f"Actuation requires --enable-actuation and --confirm-actuation {ACTUATION_CONFIRMATION}"
            )


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    checkpoint = ensure_within(args.checkpoint_path, GR00T17_ROOT, must_exist=True)
    checkpoint_info = validate_checkpoint(checkpoint)
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "timestamp": datetime.now().astimezone().isoformat(),
        "mode": args.mode,
        "checkpoint": checkpoint_info,
        "server": {"host": args.server_host, "port": args.server_port},
        "task": args.task,
        "actuation_enabled": args.mode == "actuate",
    }

    client = None
    if args.mode in {"ping", "predict", "actuate"}:
        client = LocalPolicyClient(args.server_host, args.server_port, args.request_timeout_s)
        try:
            client.ping()
        except BaseException:
            client.close()
            raise
        report["server_ping"] = "passed"
    if args.mode == "ping":
        assert client is not None
        client.close()
        report["status"] = "passed"
        return report

    try:
        bounds = load_action_bounds(checkpoint, args.bounds_mode)
        report["safety"] = {
            "bounds_source": bounds.source,
            "lower": bounds.lower.tolist(),
            "upper": bounds.upper.tolist(),
            "max_command_delta": args.max_command_delta,
            "max_relative_target": args.max_relative_target,
            "execution_horizon": args.execution_horizon,
            "control_hz": args.control_hz,
            "run_time_s": args.run_time_s if args.mode == "actuate" else None,
        }
        robot, calibration_file = build_robot(args)
        report["devices"] = {
            "robot_port": str(Path(args.robot_port).resolve(strict=True)),
            "top_camera": str(args.top_cam),
            "wrist_camera": str(args.wrist_cam),
            "calibration_file": str(calibration_file),
        }
    except BaseException:
        if client is not None:
            client.close()
        raise
    predictions: list[dict[str, Any]] = []
    sent_actions: list[dict[str, Any]] = []
    inference_latencies: list[float] = []
    clipping_events = 0
    preview_clipping_events = 0

    try:
        robot.connect(calibrate=False)
        if not robot.is_calibrated:
            raise RuntimeError("Robot is not calibrated; guarded inference refuses interactive calibration")
        warmup_start = time.perf_counter()
        warmup_observation_count = 0
        while time.perf_counter() - warmup_start < args.camera_warmup_s:
            robot.get_observation()
            warmup_observation_count += 1
            time.sleep(1 / 30)
        report["camera_warmup"] = {
            "requested_s": args.camera_warmup_s,
            "actual_s": time.perf_counter() - warmup_start,
            "observation_count": warmup_observation_count,
        }
        first_observation = robot.get_observation()
        report["initial_state"] = extract_state(first_observation).tolist()
        report["cameras"] = {
            "top": validate_image("top", first_observation["top"]),
            "wrist": validate_image("wrist", first_observation["wrist"]),
        }
        if args.report is not None:
            snapshot_dir = ensure_within(args.report, GR00T17_ROOT).parent / "snapshots"
            report["snapshots"] = {
                "top": save_snapshot(snapshot_dir, "top", first_observation["top"]),
                "wrist": save_snapshot(snapshot_dir, "wrist", first_observation["wrist"]),
            }
        if args.mode == "observe":
            report["status"] = "passed"
            return report

        assert client is not None
        deadline = time.perf_counter() + args.run_time_s if args.mode == "actuate" else None
        prediction_limit = math.inf if args.mode == "actuate" else args.prediction_count
        prediction_index = 0
        while prediction_index < prediction_limit:
            if deadline is not None and time.perf_counter() >= deadline:
                break
            observation = first_observation if prediction_index == 0 else robot.get_observation()
            state = extract_state(observation)
            validate_image("top", observation["top"])
            validate_image("wrist", observation["wrist"])
            model_observation = build_model_observation(observation, args.task)
            inference_start = time.perf_counter()
            action_dict, _ = client.get_action(model_observation)
            inference_s = time.perf_counter() - inference_start
            if inference_s > args.max_inference_s:
                raise TimeoutError(
                    f"Policy inference exceeded safety limit: {inference_s:.3f}s > {args.max_inference_s:.3f}s"
                )
            inference_latencies.append(inference_s)
            chunk = validate_action_chunk(action_dict)
            summary = prediction_summary(chunk, state)
            summary["inference_s"] = inference_s
            _, preview = sanitize_action(chunk[0], state, bounds, args.max_command_delta)
            preview_clipped = preview["hard_bound_clipped"] or preview["delta_clipped"]
            preview_clipping_events += int(preview_clipped)
            summary["first_action_safety_preview"] = preview
            predictions.append(summary)
            print(
                f"[PREDICT] chunk={prediction_index} latency={inference_s:.3f}s "
                f"first={np.array2string(chunk[0], precision=3)}"
            )

            if args.mode == "actuate":
                reference = state
                for action_index in range(args.execution_horizon):
                    if deadline is not None and time.perf_counter() >= deadline:
                        break
                    step_start = time.perf_counter()
                    safe, safety = sanitize_action(
                        chunk[action_index], reference, bounds, args.max_command_delta
                    )
                    clipped = safety["hard_bound_clipped"] or safety["delta_clipped"]
                    clipping_events += int(clipped)
                    if clipped and args.reject_clipped_action:
                        raise RuntimeError(f"Predicted action required clipping: {safety}")
                    actual = robot.send_action(action_to_robot_dict(safe))
                    actual_vector = np.asarray([actual[key] for key in JOINT_KEYS], dtype=np.float32)
                    if not np.isfinite(actual_vector).all():
                        raise RuntimeError(f"Robot returned a non-finite sent action: {actual}")
                    sent_actions.append(
                        {
                            "chunk": prediction_index,
                            "index": action_index,
                            "predicted": safety["predicted"],
                            "safe": safety["safe"],
                            "actual": actual_vector.tolist(),
                            "clipped": clipped,
                        }
                    )
                    reference = actual_vector
                    sleep_s = max(1.0 / args.control_hz - (time.perf_counter() - step_start), 0.0)
                    time.sleep(sleep_s)
            prediction_index += 1

        report["predictions"] = predictions
        report["inference_latency_s"] = {
            "count": len(inference_latencies),
            "min": min(inference_latencies),
            "max": max(inference_latencies),
            "mean": float(np.mean(inference_latencies)),
        }
        report["sent_action_count"] = len(sent_actions)
        report["clipping_events"] = clipping_events
        report["prediction_preview_clipping_events"] = preview_clipping_events
        if args.mode == "actuate" and not sent_actions:
            raise RuntimeError("Actuation run ended without sending any guarded action")
        if sent_actions:
            report["first_sent_action"] = sent_actions[0]
            report["last_sent_action"] = sent_actions[-1]
        report["status"] = "passed"
        return report
    finally:
        cleanup_errors = disconnect_robot_safely(robot)
        if client is not None:
            try:
                client.close()
            except BaseException as exc:
                cleanup_errors.append(f"policy client: {type(exc).__name__}: {exc}")
        if cleanup_errors:
            message = f"Device cleanup errors: {cleanup_errors}"
            print(f"[ERROR] {message}", file=sys.stderr)
            if sys.exc_info()[0] is None:
                raise RuntimeError(message)


def main() -> None:
    args = build_parser().parse_args()
    report_path = args.report
    if report_path is not None:
        checked_report = ensure_within(report_path, GR00T17_ROOT)
        if checked_report.exists():
            raise FileExistsError(f"Refusing to overwrite inference report: {checked_report}")
    try:
        payload = run(args)
    except BaseException as exc:
        failure = {
            "schema_version": 1,
            "status": "failed",
            "timestamp": datetime.now().astimezone().isoformat(),
            "mode": args.mode,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        write_report(report_path, failure)
        raise
    write_report(report_path, payload)
    print(f"[OK] SO101 mode passed: {args.mode}")


if __name__ == "__main__":
    main()
