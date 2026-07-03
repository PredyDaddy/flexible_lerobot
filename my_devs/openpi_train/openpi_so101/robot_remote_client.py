from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from openpi_so101 import paths


REPO_ROOT = paths.PROJECT_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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


def _image_to_hwc_uint8(value) -> np.ndarray:
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


def _state_to_float32(value) -> np.ndarray:
    state = np.asarray(value, dtype=np.float32)
    if state.shape != (6,):
        raise ValueError(f"Expected SO101 state shape=(6,), got shape={state.shape}")
    return np.ascontiguousarray(state)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SO101 robot inference through a remote OpenPI server.")
    parser.add_argument("--host", default=os.getenv("OPENPI_SERVER_HOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.getenv("OPENPI_SERVER_PORT", "8000")))
    parser.add_argument("--api-key", default=os.getenv("OPENPI_API_KEY"))
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
    parser.add_argument(
        "--wrist-cam",
        type=parse_camera,
        default=parse_camera(os.getenv("WRIST_CAM", "/dev/video6")),
    )
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
        default=int(os.getenv("ACTION_CHUNK_STEPS", "30")),
        help="How many actions to execute from each returned 50-step OpenPI action chunk. Use 1 for old behavior.",
    )
    parser.add_argument(
        "--motor-io-retries",
        type=int,
        default=int(os.getenv("MOTOR_IO_RETRIES", "10")),
        help="Minimum retry count for SO follower motor bus read/write calls.",
    )
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


def _build_dataset_features(robot) -> dict:
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


def _build_observation(robot, robot_observation_processor, dataset_features: dict, task: str) -> tuple[dict, dict]:
    from lerobot.datasets.utils import build_dataset_frame
    from lerobot.utils.constants import OBS_STR

    raw_observation = robot.get_observation()
    processed_observation = robot_observation_processor(raw_observation)
    frame = build_dataset_frame(dataset_features, processed_observation, prefix=OBS_STR)
    observation = {
        "observation.images.top": _image_to_hwc_uint8(frame["observation.images.top"]),
        "observation.images.wrist": _image_to_hwc_uint8(frame["observation.images.wrist"]),
        "observation.state": _state_to_float32(frame["observation.state"]),
        "prompt": task,
    }
    return observation, raw_observation


def _first_action_to_robot_action(actions: np.ndarray, dataset_features: dict) -> dict:
    from lerobot.policies.utils import make_robot_action

    if actions.ndim != 2 or actions.shape[1] != 6:
        raise ValueError(f"Expected action chunk shape=(horizon, 6), got shape={actions.shape}")
    return make_robot_action(torch.as_tensor(actions[0], dtype=torch.float32).unsqueeze(0), dataset_features)


def _action_to_robot_action(action: np.ndarray, dataset_features: dict) -> dict:
    from lerobot.policies.utils import make_robot_action

    action = np.asarray(action, dtype=np.float32)
    if action.shape != (6,):
        raise ValueError(f"Expected one SO101 action shape=(6,), got shape={action.shape}")
    return make_robot_action(torch.as_tensor(action, dtype=torch.float32).unsqueeze(0), dataset_features)


def _motor_metadata_value(metadata: Any, name: str, fallback_index: int | None = None) -> Any:
    if hasattr(metadata, name):
        return getattr(metadata, name)
    if fallback_index is not None:
        try:
            return metadata[fallback_index]
        except Exception:
            pass
    return None


def _motor_id(robot: Any, motor: str) -> Any:
    metadata = getattr(robot.bus, "motors", {}).get(motor)
    return _motor_metadata_value(metadata, "id", 0)


def _log_expected_motors(robot: Any) -> None:
    motors = getattr(getattr(robot, "bus", None), "motors", {})
    if not motors:
        print("[DIAG] Robot has no bus.motors metadata.")
        return

    print("[DIAG] Expected motor map:")
    for name, metadata in motors.items():
        motor_id = _motor_metadata_value(metadata, "id", 0)
        model = _motor_metadata_value(metadata, "model", 1)
        norm_mode = _motor_metadata_value(metadata, "norm_mode", 2)
        print(f"[DIAG]   {name}: id={motor_id}, model={model}, norm_mode={norm_mode}")


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

    def read_with_min_retries(
        data_name: str,
        motor: str,
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> Any:
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
        value: int | float | list | tuple,
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> Any:
        return original_write(
            data_name,
            motor,
            value,
            normalize=normalize,
            num_retry=max(num_retry, retries),
        )

    def sync_write_with_min_retries(
        data_name: str,
        values: Any,
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> Any:
        return original_sync_write(
            data_name,
            values,
            normalize=normalize,
            num_retry=max(num_retry, retries),
        )

    bus.read = read_with_min_retries
    bus.sync_read = sync_read_with_min_retries
    bus.write = write_with_min_retries
    bus.sync_write = sync_write_with_min_retries
    print(f"[DIAG] Motor bus read/write calls will use at least {retries} retries.")


def _diagnose_robot_connect_failure(robot: Any, exc: BaseException, retries: int) -> None:
    print(f"[ERROR] robot.connect() failed: {type(exc).__name__}: {exc}")
    print("[ERROR] This happened before the control loop, so no actions were sent.")
    print(f"[DIAG] Current motor_io_retries={retries}.")
    _log_expected_motors(robot)

    bus = getattr(robot, "bus", None)
    if bus is None:
        print("[DIAG] Robot has no bus object; cannot run motor ping diagnostics.")
        return
    if not getattr(bus, "is_connected", False):
        print("[DIAG] Motor bus is not connected; check serial port, USB permission, and power.")
        return

    print("[DIAG] Motor bus is open. Pinging each configured motor:")
    for motor in getattr(bus, "motors", {}):
        motor_id = _motor_id(robot, motor)
        try:
            ping_result = bus.ping(motor, num_retry=max(retries, 5))
            if ping_result is None:
                print(f"[DIAG]   {motor}: id={motor_id} -> NO RESPONSE")
            else:
                print(f"[DIAG]   {motor}: id={motor_id} -> {ping_result}")
        except Exception as ping_exc:
            print(f"[DIAG]   {motor}: id={motor_id} -> ping failed: {type(ping_exc).__name__}: {ping_exc}")

    print("[HINT] The failing write names the motor id. On this SO follower map, id=3 is elbow_flex.")
    print("[HINT] If this is intermittent, rerun with --motor-io-retries 20.")
    print("[HINT] If the same id keeps failing, check that motor's power, cable, id, and bus stability.")


def main() -> int:
    args = build_parser().parse_args()
    print(f"[INFO] Repo root: {REPO_ROOT}")
    print(f"[INFO] Server: ws://{args.host}:{args.port}")
    print(f"[INFO] Task: {args.task}")
    print(f"[INFO] Robot id: {args.robot_id}")
    print(f"[INFO] Robot port: {args.robot_port}")
    print(f"[INFO] Calibration dir: {args.calib_dir}")
    print(f"[INFO] Cameras: top={args.top_cam} wrist={args.wrist_cam}")
    print(f"[INFO] max_relative_target: {args.max_relative_target}")
    print(f"[INFO] execute_actions: {args.execute_actions}")
    print(f"[INFO] action_chunk_steps: {args.action_chunk_steps}")
    print(f"[INFO] motor_io_retries: {args.motor_io_retries}")
    print(f"[INFO] run_time_s: {args.run_time_s} (<=0 means until Ctrl+C)")

    if args.action_chunk_steps < 1:
        raise ValueError("--action-chunk-steps must be >= 1")

    if args.dry_run:
        print("[INFO] DRY_RUN=true, exiting before server or hardware access.")
        return 0

    try:
        try:
            from openpi_client import websocket_client_policy
        except ModuleNotFoundError as exc:
            if exc.name == "msgpack":
                raise ModuleNotFoundError(
                    "The robot environment is missing `msgpack`, which is required by the local "
                    "openpi-client websocket code.\n"
                    "Install it without changing the rest of the robot environment:\n"
                    "  source /home/cqy/miniconda3/etc/profile.d/conda.sh\n"
                    "  conda activate lerobot_flex\n"
                    "  python -m pip install msgpack\n"
                ) from exc
            raise

        robot, robot_config = _build_robot(args)
        _patch_motor_bus_retries(robot, args.motor_io_retries)
        dataset_features, robot_action_processor, robot_observation_processor = _build_dataset_features(robot)
        if dataset_features["action"]["names"] != list(ACTION_NAMES):
            raise ValueError(
                f"Unexpected action names: {dataset_features['action']['names']}. Expected {list(ACTION_NAMES)}"
            )

        print(f"[INFO] Robot config type resolved by current LeRobot registry: {robot_config.type}")
        print(f"[INFO] Robot runtime type: {robot.robot_type}")
        print(f"[INFO] Action names: {dataset_features['action']['names']}")

        client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port, api_key=args.api_key)
        print(f"[INFO] Server metadata: {client.get_server_metadata()}")

        step = 0
        start_t = time.perf_counter()
        end_t = start_t + args.run_time_s if args.run_time_s > 0 else None
        try:
            robot.connect()
        except Exception as exc:
            _diagnose_robot_connect_failure(robot, exc, args.motor_io_retries)
            raise
        print("[INFO] Robot connected. Starting remote inference loop.")

        while True:
            if end_t is not None and time.perf_counter() >= end_t:
                print("[INFO] Reached requested run_time_s. Exiting.")
                break

            infer_t = time.perf_counter()
            observation, raw_observation = _build_observation(
                robot,
                robot_observation_processor,
                dataset_features,
                args.task,
            )
            result = client.infer(observation)
            actions = np.asarray(result["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[1] != 6:
                raise ValueError(f"Expected action chunk shape=(horizon, 6), got shape={actions.shape}")
            infer_ms = (time.perf_counter() - infer_t) * 1000

            steps_to_execute = min(args.action_chunk_steps, actions.shape[0])
            first_action = actions[0].copy()
            for chunk_index, action in enumerate(actions[:steps_to_execute]):
                if end_t is not None and time.perf_counter() >= end_t:
                    break

                step_t = time.perf_counter()
                robot_action = _action_to_robot_action(action, dataset_features)
                if args.execute_actions:
                    robot_action_to_send = robot_action_processor((robot_action, raw_observation))
                    robot.send_action(robot_action_to_send)

                step += 1
                if args.log_interval > 0 and step % args.log_interval == 0:
                    elapsed = time.perf_counter() - start_t
                    action_text = np.array2string(
                        np.array([robot_action[name] for name in ACTION_NAMES], dtype=np.float32),
                        precision=3,
                        suppress_small=True,
                    )
                    print(
                        f"[INFO] Step {step} elapsed={elapsed:.2f}s chunk_index={chunk_index}/"
                        f"{steps_to_execute - 1} infer_ms={infer_ms:.1f} first_action={action_text}"
                    )
                    if "server_timing" in result:
                        print(f"[INFO] server_timing={result['server_timing']}")

                dt_s = time.perf_counter() - step_t
                time.sleep(max(1 / args.fps - dt_s, 0.0))

    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt received. Stopping.")
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        return 1
    finally:
        if "robot" in locals() and getattr(robot, "is_connected", False):
            try:
                robot.disconnect()
            except Exception:
                pass
        print("[INFO] Client finished.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
