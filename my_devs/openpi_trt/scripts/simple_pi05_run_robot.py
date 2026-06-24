#!/usr/bin/env python

"""Simple real-robot runner for the compact PI0.5 hybrid TensorRT runtime."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
OPENPI_TRT_DIR = SCRIPT_DIR.parent
REPO_ROOT = OPENPI_TRT_DIR.parents[1]
for path in (OPENPI_TRT_DIR, REPO_ROOT):
    if path.as_posix() not in sys.path:
        sys.path.insert(0, path.as_posix())

from runtime.simple_pi05_split import (  # noqa: E402
    DEFAULT_PREFIX_ENGINE,
    DEFAULT_DENOISE_FP16_CONSTRAINED_ENGINE,
    SimplePI05SplitTRTRuntime,
    SimplePI05TRTProfile,
)
from runtime.pure_pi05_trt import PurePI05TRTPolicyAdapter, PurePI05TRTProfile, PurePI05TRTRuntime  # noqa: E402
from scripts.pi05_onnx_common import (  # noqa: E402
    DEFAULT_POLICY_PATH,
    DEFAULT_TASK,
    configure_runtime,
    ensure_local_tokenizer_dir,
    load_policy,
)

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: E402
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features  # noqa: E402
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts  # noqa: E402
from lerobot.policies.utils import make_robot_action  # noqa: E402
from lerobot.processor import PolicyAction, PolicyProcessorPipeline, make_default_processors  # noqa: E402
from lerobot.processor.converters import (  # noqa: E402
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig  # noqa: E402
from lerobot.utils.constants import OBS_STR  # noqa: E402
from lerobot.utils.control_utils import predict_action  # noqa: E402
from lerobot.utils.robot_utils import precise_sleep  # noqa: E402
from lerobot.utils.utils import get_safe_torch_device  # noqa: E402


DEFAULT_ROBOT_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00"
DEFAULT_CALIB_DIR = "/home/cqy/.cache/huggingface/lerobot/calibration/robots/so_follower"
DEFAULT_MOTOR_IO_RETRIES = 10
DEFAULT_RUNTIME_ASSETS_DIR = Path("my_devs/openpi_trt/artifacts/pi05_runtime_assets")


def log(message: str) -> None:
    print(message, flush=True)


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {value}")


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    return default if raw is None else parse_bool(raw)


def parse_camera(value: str) -> int | Path:
    return int(value) if value.isdecimal() else Path(value).expanduser()


def maybe_path(path: str | None) -> Path | None:
    return None if not path else Path(path).expanduser()


def optional_float(value: str | None) -> float | None:
    if value is None or value.strip().lower() in {"", "none", "null"}:
        return None
    return float(value)


def action_preview(action_values, limit: int = 6) -> str:
    try:
        values = action_values.detach().flatten().cpu().tolist()
    except AttributeError:
        values = list(action_values)
    preview = ", ".join(f"{float(value):+.4f}" for value in values[:limit])
    return f"[{preview}{', ...' if len(values) > limit else ''}]"


def motor_metadata_value(metadata: Any, field: str, index: int) -> Any:
    if hasattr(metadata, field):
        return getattr(metadata, field)
    if isinstance(metadata, (tuple, list)) and len(metadata) > index:
        return metadata[index]
    return None


def motor_id(robot: Any, motor: str) -> int | None:
    try:
        value = motor_metadata_value(robot.bus.motors[motor], "id", 0)
        return None if value is None else int(value)
    except (AttributeError, KeyError, TypeError, ValueError):
        return None


def log_expected_motors(robot: Any) -> None:
    try:
        motors = robot.bus.motors
    except AttributeError:
        log("[DIAG] Robot has no bus.motors metadata.")
        return

    log("[DIAG] Expected motor map:")
    for name, metadata in motors.items():
        motor_id_value = motor_metadata_value(metadata, "id", 0)
        model = motor_metadata_value(metadata, "model", 1)
        norm_mode = motor_metadata_value(metadata, "norm_mode", 2)
        log(f"[DIAG]   {name}: id={motor_id_value}, model={model}")
        if norm_mode is not None:
            log(f"[DIAG]     norm_mode={norm_mode}")


def patch_motor_bus_retries(robot: Any, retries: int) -> None:
    retries = max(int(retries), 0)
    if retries <= 0:
        log("[DIAG] Motor bus I/O uses native LeRobot retry behavior.")
        return
    try:
        bus = robot.bus
        original_read = bus.read
        original_sync_read = bus.sync_read
        original_write = bus.write
        original_sync_write = bus.sync_write
    except AttributeError:
        log("[DIAG] Robot has no motor bus; skip motor I/O retry patch.")
        return

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
    log(f"[DIAG] Motor bus read/write calls will use at least {retries} retries.")


def diagnose_robot_connect_failure(robot: Any, exc: BaseException) -> None:
    log(f"[ERROR] robot.connect() failed: {type(exc).__name__}: {exc}")
    log("[ERROR] This happened before the control loop, so no actions were sent.")
    log_expected_motors(robot)

    bus = getattr(robot, "bus", None)
    if bus is None:
        log("[DIAG] Robot has no bus object; cannot run motor ping diagnostics.")
        return
    if not getattr(bus, "is_connected", False):
        log("[DIAG] Motor bus is not connected; check serial port, USB permission, and power.")
        return

    log("[DIAG] Motor bus is open. Pinging each configured motor:")
    for motor in getattr(bus, "motors", {}):
        motor_id_value = motor_id(robot, motor)
        try:
            ping_result = bus.ping(motor, num_retry=5)
            if ping_result is None:
                log(f"[DIAG]   {motor}: id={motor_id_value} -> NO RESPONSE")
            else:
                log(f"[DIAG]   {motor}: id={motor_id_value} -> {ping_result}")
        except Exception as ping_exc:  # noqa: BLE001 - diagnostics must never hide the original error.
            log(f"[DIAG]   {motor}: id={motor_id_value} -> ping failed: {type(ping_exc).__name__}: {ping_exc}")

    log("[HINT] The failing write above names the motor id. For id=5 on SO101, check wrist_roll power/cable/id.")
    log("[HINT] If the motor responds intermittently, rerun with --motor-write-retries 10.")


def diagnose_robot_observation_failure(robot: Any, exc: BaseException, retries: int) -> None:
    log(f"[ERROR] robot.get_observation() failed: {type(exc).__name__}: {exc}")
    log("[ERROR] This happened before policy inference and before robot.send_action().")
    log(f"[DIAG] Current motor_io_retries={retries}.")
    log_expected_motors(robot)
    log("[HINT] This is a motor bus read failure on Present_Position, not a TensorRT/model failure.")
    log("[HINT] Try --motor-io-retries 20, check SO101 power, USB serial stability, and all motor cables.")


def disconnect_robot_best_effort(robot: Any) -> None:
    try:
        if getattr(robot, "is_connected", False):
            log("[INFO] Disconnecting robot.")
            robot.disconnect()
            return
    except Exception as exc:  # noqa: BLE001 - cleanup must be best effort.
        log(f"[WARN] robot.disconnect() failed: {type(exc).__name__}: {exc}")

    bus = getattr(robot, "bus", None)
    if bus is not None and getattr(bus, "is_connected", False):
        try:
            log("[INFO] Closing motor bus without extra torque writes.")
            bus.disconnect(disable_torque=False)
        except Exception as exc:  # noqa: BLE001 - cleanup must be best effort.
            log(f"[WARN] bus.disconnect(disable_torque=False) failed: {type(exc).__name__}: {exc}")

    for camera_name, camera in getattr(robot, "cameras", {}).items():
        if getattr(camera, "is_connected", False):
            try:
                log(f"[INFO] Disconnecting camera {camera_name}.")
                camera.disconnect()
            except Exception as exc:  # noqa: BLE001 - cleanup must be best effort.
                log(f"[WARN] camera {camera_name} disconnect failed: {type(exc).__name__}: {exc}")


def load_pre_post_processors(
    policy_path: Path,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=str(policy_path),
        config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=str(policy_path),
        config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return preprocessor, postprocessor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simple SO101 PI0.5 Torch-prefix + TRT-denoise real-robot runner.")
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument(
        "--runtime-assets-dir",
        type=Path,
        default=DEFAULT_RUNTIME_ASSETS_DIR,
        help=(
            "Small config/preprocessor/postprocessor directory for pure_trt. "
            "It must not contain model.safetensors."
        ),
    )
    parser.add_argument(
        "--runtime-backend",
        choices=["hybrid", "pure_trt"],
        default="hybrid",
        help="hybrid loads PyTorch policy weights for prefix. pure_trt loads prefix+denoise TensorRT engines only.",
    )
    parser.add_argument("--profile", choices=["auto", "fp32", "fp16_constrained"], default="auto")
    parser.add_argument(
        "--prefix-engine-path",
        type=Path,
        default=DEFAULT_PREFIX_ENGINE,
        help="Required for --runtime-backend pure_trt. Ignored by hybrid.",
    )
    parser.add_argument("--denoise-engine-path", type=Path, default=DEFAULT_DENOISE_FP16_CONSTRAINED_ENGINE)
    parser.add_argument("--robot-id", default=os.getenv("ROBOT_ID", "hfy_follower"))
    parser.add_argument("--robot-type", default=os.getenv("ROBOT_TYPE", "so101_follower"))
    parser.add_argument("--calib-dir", default=os.getenv("CALIB_DIR", DEFAULT_CALIB_DIR))
    parser.add_argument("--robot-port", default=os.getenv("ROBOT_PORT", DEFAULT_ROBOT_PORT))
    parser.add_argument(
        "--max-relative-target",
        type=optional_float,
        default=optional_float(os.getenv("MAX_RELATIVE_TARGET")),
    )
    parser.add_argument("--top-cam", type=parse_camera, default=parse_camera(os.getenv("TOP_CAM", "/dev/video4")))
    parser.add_argument("--wrist-cam", type=parse_camera, default=parse_camera(os.getenv("WRIST_CAM", "/dev/video6")))
    parser.add_argument("--top-cam-fourcc", default=os.getenv("TOP_CAM_FOURCC", "YUYV"))
    parser.add_argument("--wrist-cam-fourcc", default=os.getenv("WRIST_CAM_FOURCC", "MJPG"))
    parser.add_argument("--img-width", type=int, default=int(os.getenv("IMG_WIDTH", "640")))
    parser.add_argument("--img-height", type=int, default=int(os.getenv("IMG_HEIGHT", "480")))
    parser.add_argument("--fps", type=int, default=int(os.getenv("FPS", "30")))
    parser.add_argument("--task", default=os.getenv("DATASET_TASK", DEFAULT_TASK))
    parser.add_argument("--run-time-s", type=float, default=float(os.getenv("RUN_TIME_S", "0")))
    parser.add_argument("--log-interval", type=int, default=int(os.getenv("LOG_INTERVAL", "1")))
    parser.add_argument(
        "--log-action-preview",
        type=parse_bool,
        nargs="?",
        const=True,
        default=env_bool("LOG_ACTION_PREVIEW", True),
        help="Print the first few action values at each log interval.",
    )
    parser.add_argument(
        "--motor-write-retries",
        type=int,
        default=None,
        help="Deprecated alias for --motor-io-retries.",
    )
    parser.add_argument(
        "--motor-io-retries",
        type=int,
        default=int(os.getenv("MOTOR_IO_RETRIES", os.getenv("MOTOR_WRITE_RETRIES", str(DEFAULT_MOTOR_IO_RETRIES)))),
        help="Minimum retries for motor read/sync_read/write/sync_write calls; 0 matches native LeRobot.",
    )
    parser.add_argument("--dry-run", type=parse_bool, nargs="?", const=True, default=env_bool("DRY_RUN", False))
    parser.add_argument("--check-policy-load", type=parse_bool, nargs="?", const=True, default=False)
    parser.add_argument("--confirm-control", action="store_true")
    return parser


def main() -> None:
    configure_runtime()
    args = build_parser().parse_args()
    policy_path = args.policy_path.expanduser().resolve()
    runtime_assets_dir = args.runtime_assets_dir.expanduser().resolve()
    prefix_engine = args.prefix_engine_path.expanduser() if args.prefix_engine_path is not None else None
    denoise_engine = args.denoise_engine_path.expanduser()
    profile_name = args.profile
    if profile_name == "auto":
        profile_name = "fp16_constrained" if "fp16" in denoise_engine.name else "fp32"
    if profile_name == "fp32" and args.denoise_engine_path == DEFAULT_DENOISE_FP16_CONSTRAINED_ENGINE:
        denoise_engine = Path("my_devs/openpi_trt/artifacts/pi05_so101_denoise_step_b1_fp32.engine")

    log("[SAFETY] Simple runner controls robot only with --confirm-control.")
    log(f"[INFO] runtime_backend={args.runtime_backend}")
    if args.runtime_backend == "pure_trt":
        log(f"[INFO] runtime_assets_dir={runtime_assets_dir}")
    else:
        log(f"[INFO] policy_path={policy_path}")
    log(f"[INFO] profile={profile_name}")
    prefix_backend = "tensorrt" if args.runtime_backend == "pure_trt" else "torch"
    log(f"[INFO] prefix_backend={prefix_backend}")
    if args.runtime_backend == "pure_trt":
        log(f"[INFO] prefix_engine={prefix_engine}")
    else:
        log(f"[INFO] prefix_engine={prefix_engine} (ignored by hybrid)")
    log(f"[INFO] denoise_engine={denoise_engine}")
    log(f"[INFO] robot={args.robot_id} type={args.robot_type} port={args.robot_port}")
    log(f"[INFO] max_relative_target={args.max_relative_target}")
    log(f"[INFO] top_cam={args.top_cam} {args.top_cam_fourcc}, wrist_cam={args.wrist_cam} {args.wrist_cam_fourcc}")
    log(f"[INFO] image={args.img_width}x{args.img_height}@{args.fps}, task={args.task!r}")
    log(f"[INFO] run_time_s={args.run_time_s}, log_interval={args.log_interval}")
    log(f"[INFO] log_action_preview={args.log_action_preview}")
    motor_io_retries = args.motor_io_retries if args.motor_write_retries is None else args.motor_write_retries
    log(f"[INFO] motor_io_retries={motor_io_retries}")
    if args.dry_run:
        log("[INFO] DRY_RUN=true. Exit before loading model or touching hardware.")
        return

    if args.runtime_backend == "pure_trt":
        if not runtime_assets_dir.is_dir():
            raise FileNotFoundError(f"Runtime assets directory does not exist: {runtime_assets_dir}")
        if (runtime_assets_dir / "model.safetensors").exists():
            raise RuntimeError(
                f"Runtime assets directory should not contain model.safetensors: {runtime_assets_dir}"
            )
    elif not policy_path.is_dir():
        raise FileNotFoundError(f"Policy path does not exist: {policy_path}")
    if args.runtime_backend == "pure_trt" and (prefix_engine is None or not prefix_engine.is_file()):
        raise FileNotFoundError(f"Prefix TensorRT engine does not exist: {prefix_engine}")
    if not denoise_engine.is_file():
        raise FileNotFoundError(f"Denoise TensorRT engine does not exist: {denoise_engine}")

    ensure_local_tokenizer_dir()

    if args.runtime_backend == "pure_trt":
        trt_t0 = time.perf_counter()
        log("[INFO] Loading pure TensorRT runtime without PyTorch model weights...")
        runtime = PurePI05TRTRuntime(PurePI05TRTProfile(profile_name, prefix_engine, denoise_engine))
        policy = PurePI05TRTPolicyAdapter(runtime_assets_dir, runtime, device="cuda")
        log(f"[INFO] Pure TensorRT runtime loaded in {time.perf_counter() - trt_t0:.2f}s")
        log(f"[INFO] runtime={runtime.describe()}")
    else:
        load_t0 = time.perf_counter()
        log("[INFO] Loading policy...")
        policy = load_policy(policy_path, device="cuda", model_dtype="float32")
        log(f"[INFO] Policy loaded in {time.perf_counter() - load_t0:.2f}s")

        trt_t0 = time.perf_counter()
        log("[INFO] Loading simple hybrid TensorRT runtime...")
        runtime = SimplePI05SplitTRTRuntime(SimplePI05TRTProfile(profile_name, prefix_engine, denoise_engine))
        runtime.patch_policy(policy)
        log(f"[INFO] TensorRT runtime loaded in {time.perf_counter() - trt_t0:.2f}s")
        log(f"[INFO] runtime={runtime.describe()}")

    proc_t0 = time.perf_counter()
    log("[INFO] Loading processors...")
    processor_dir = runtime_assets_dir if args.runtime_backend == "pure_trt" else policy_path
    preprocessor, postprocessor = load_pre_post_processors(processor_dir)
    log(f"[INFO] Processors loaded in {time.perf_counter() - proc_t0:.2f}s")
    if args.check_policy_load:
        log("[INFO] CHECK_POLICY_LOAD=true. Exit before robot connection.")
        return
    if not args.confirm_control:
        log("[SAFETY] Missing --confirm-control. Exit before robot connection.")
        return
    if args.robot_type not in {"so100_follower", "so101_follower"}:
        raise ValueError(f"Unsupported robot_type={args.robot_type!r}")

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
    robot_cfg = SOFollowerRobotConfig(
        id=args.robot_id,
        calibration_dir=maybe_path(args.calib_dir),
        port=args.robot_port,
        max_relative_target=args.max_relative_target,
        cameras=cameras,
    )

    from lerobot.robots import make_robot_from_config

    log("[INFO] Creating robot object and dataset feature mapping...")
    robot = make_robot_from_config(robot_cfg)
    patch_motor_bus_retries(robot, motor_io_retries)
    log_expected_motors(robot)
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
    action_shape = dataset_features["action"]["shape"] if "action" in dataset_features else "unknown"
    log(f"[INFO] Dataset action shape: {action_shape}")

    step = 0
    start = time.perf_counter()
    end = start + args.run_time_s if args.run_time_s > 0 else None
    try:
        log("[SAFETY] --confirm-control present. Connecting robot now.")
        try:
            connect_t0 = time.perf_counter()
            robot.connect()
            log(f"[INFO] Robot connected in {time.perf_counter() - connect_t0:.2f}s")
        except Exception as exc:
            diagnose_robot_connect_failure(robot, exc)
            raise
        log("[INFO] Robot connected. Entering control loop. Press Ctrl+C to stop.")
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()
        while True:
            if end is not None and time.perf_counter() >= end:
                log("[INFO] run_time_s reached. Stopping.")
                break
            loop_t = time.perf_counter()
            obs_t = time.perf_counter()
            try:
                obs = robot.get_observation()
            except ConnectionError as exc:
                diagnose_robot_observation_failure(robot, exc, motor_io_retries)
                raise
            obs_ms = (time.perf_counter() - obs_t) * 1000
            prep_t = time.perf_counter()
            obs_processed = robot_observation_processor(obs)
            frame = build_dataset_frame(dataset_features, obs_processed, prefix=OBS_STR)
            prep_ms = (time.perf_counter() - prep_t) * 1000
            infer_t = time.perf_counter()
            action_values = predict_action(
                observation=frame,
                policy=policy,
                device=get_safe_torch_device(policy.config.device),
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy.config.use_amp,
                task=args.task,
                robot_type=robot.robot_type,
            )
            infer_ms = (time.perf_counter() - infer_t) * 1000
            send_t = time.perf_counter()
            action_dict = make_robot_action(action_values, dataset_features)
            robot.send_action(robot_action_processor((action_dict, obs)))
            send_ms = (time.perf_counter() - send_t) * 1000
            step += 1
            loop_ms = (time.perf_counter() - loop_t) * 1000
            sleep_s = max(1 / args.fps - loop_ms / 1000, 0.0)
            if step == 1 or (args.log_interval > 0 and step % args.log_interval == 0):
                elapsed = time.perf_counter() - start
                hz = step / elapsed if elapsed > 0 else 0.0
                log(
                    f"[LOOP] step={step} elapsed={elapsed:.2f}s avg_hz={hz:.2f} "
                    f"loop={loop_ms:.1f}ms obs={obs_ms:.1f}ms prep={prep_ms:.1f}ms "
                    f"infer={infer_ms:.1f}ms send={send_ms:.1f}ms sleep={sleep_s * 1000:.1f}ms "
                    f"action={action_preview(action_values) if args.log_action_preview else '[disabled]'}"
                )
            precise_sleep(sleep_s)
    except KeyboardInterrupt:
        log("[INFO] KeyboardInterrupt. Stopping.")
    finally:
        disconnect_robot_best_effort(robot)
        log(f"[INFO] Finished. steps={step}, elapsed={time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    main()
