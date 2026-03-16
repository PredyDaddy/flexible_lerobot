#!/usr/bin/env python

"""Run Agilex ACT observation checks and single-arm inference.

This script mirrors the repository's standard Agilex ACT inference flow, but
targets single-arm checkpoints:
1) read full Agilex live observation
2) slice the selected arm's 7D state and matching side camera
3) optionally run ACT inference
4) optionally publish the selected arm command while holding the other arm at
   its current live pose

Examples:
python my_devs/train/act/agilex/run_act_single_arm_infer.py --dry-run true

python my_devs/train/act/agilex/run_act_single_arm_infer.py \
    --arm right \
    --execution-mode observation_only \
    --run-time-s 10

python my_devs/train/act/agilex/run_act_single_arm_infer.py \
    --arm right \
    --execution-mode policy_inference \
    --control-mode passive_follow \
    --policy-n-action-steps 1 \
    --policy-temporal-ensemble-coeff 0.01 \
    --run-time-s 12

python my_devs/train/act/agilex/run_act_single_arm_infer.py \
    --arm right \
    --execution-mode policy_inference \
    --control-mode command_master \
    --policy-path /home/agilex/cqy/flexible_lerobot/outputs/train/20260314_215531_act_agilex_first_test_right_full/checkpoints/100000/pretrained_model \
    --run-time-s 8

python my_devs/train/act/agilex/run_act_single_arm_infer.py \
    --arm right \
    --execution-mode policy_inference \
    --control-mode command_master \
    --policy-path /home/agilex/cqy/flexible_lerobot/outputs/train/20260314_215531_act_agilex_first_test_right_full/checkpoints/100000/pretrained_model \
    --policy-n-action-steps 1 \
    --policy-temporal-ensemble-coeff 0.01 \
    --run-time-s 8
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from lerobot import policies  # noqa: F401  # Register policy config classes.
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.processor.converters import (
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.processor.device_processor import DeviceProcessorStep
from lerobot.robots.agilex.agilex_ros_bridge import (
    ACTION_FEATURE_NAMES,
    AgileXRosBridge,
    BridgeTopics,
    ImageTopicConfig,
    LEFT_PREFIX,
    POSITION_FEATURE_NAMES,
    RIGHT_PREFIX,
)
from lerobot.utils.control_utils import predict_action
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import auto_select_torch_device, get_safe_torch_device

DEFAULT_POLICY_PATH = (
    "/home/agilex/cqy/flexible_lerobot/outputs/train/"
    "20260314_215531_act_agilex_first_test_right_full/checkpoints/100000/pretrained_model"
)
EXPECTED_STATE_KEY = "observation.state"
EXPECTED_ACTION_KEY = "action"
ARM_CHOICES = (LEFT_PREFIX, RIGHT_PREFIX)
DEFAULT_ARM = RIGHT_PREFIX
DEFAULT_TASK = "Execute the trained Agilex single-arm ACT task"
ARM_TO_STATE_NAMES = {
    LEFT_PREFIX: tuple(f"{LEFT_PREFIX}_joint{i}.pos" for i in range(7)),
    RIGHT_PREFIX: tuple(f"{RIGHT_PREFIX}_joint{i}.pos" for i in range(7)),
}
ARM_TO_ACTION_NAMES = {
    LEFT_PREFIX: tuple(f"{LEFT_PREFIX}_joint{i}.pos" for i in range(7)),
    RIGHT_PREFIX: tuple(f"{RIGHT_PREFIX}_joint{i}.pos" for i in range(7)),
}
ARM_TO_LIVE_CAMERA_KEY = {
    LEFT_PREFIX: "camera_left",
    RIGHT_PREFIX: "camera_right",
}
ARM_TO_POLICY_CAMERA_KEY = {
    LEFT_PREFIX: "observation.images.camera_left",
    RIGHT_PREFIX: "observation.images.camera_right",
}
ROBOT_TYPE = "agilex"
DEFAULT_JOINT_NAMES = [f"joint{i}" for i in range(7)]
INACTIVE_ARM_HOLD_ATOL = 1e-6


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value

    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {value}")


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    return default if raw is None else parse_bool(raw)


def parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"", "none", "null"}:
        return None
    return float(value)


def parse_optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"", "none", "null", "0"}:
        return None
    return int(value)


def resolve_policy_device(requested: str | None, config_device: str | None) -> str:
    if requested is not None and requested.strip().lower() in {"", "none", "null"}:
        requested = None
    if config_device is None or str(config_device).lower() in {"", "none", "null"}:
        config_device = auto_select_torch_device().type
    if requested is None:
        return str(config_device)
    if requested == "auto":
        return auto_select_torch_device().type
    return requested


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Agilex single-arm ACT observation check, shadow inference, and closed-loop inference."
    )
    parser.add_argument("--robot-id", default=os.getenv("ROBOT_ID", "my_agilex"))
    parser.add_argument(
        "--arm",
        choices=ARM_CHOICES,
        default=os.getenv("ARM", DEFAULT_ARM),
        help="Which arm the single-arm checkpoint controls.",
    )
    parser.add_argument(
        "--execution-mode",
        choices=("observation_only", "policy_inference"),
        default=os.getenv("EXECUTION_MODE", "policy_inference"),
        help=(
            "observation_only only checks live state/images; "
            "policy_inference runs ACT and optionally publishes actions."
        ),
    )
    parser.add_argument(
        "--control-mode",
        choices=("passive_follow", "command_master"),
        default=os.getenv("CONTROL_MODE", "passive_follow"),
        help="passive_follow keeps the loop read-only; command_master publishes real robot commands.",
    )
    parser.add_argument("--state-left-topic", default=os.getenv("STATE_LEFT_TOPIC", "/puppet/joint_left"))
    parser.add_argument("--state-right-topic", default=os.getenv("STATE_RIGHT_TOPIC", "/puppet/joint_right"))
    parser.add_argument("--command-left-topic", default=os.getenv("COMMAND_LEFT_TOPIC", "/master/joint_left"))
    parser.add_argument("--command-right-topic", default=os.getenv("COMMAND_RIGHT_TOPIC", "/master/joint_right"))
    parser.add_argument(
        "--front-camera-topic",
        default=os.getenv("FRONT_CAMERA_TOPIC", "/camera_f/color/image_raw"),
    )
    parser.add_argument(
        "--left-camera-topic",
        default=os.getenv("LEFT_CAMERA_TOPIC", "/camera_l/color/image_raw"),
    )
    parser.add_argument(
        "--right-camera-topic",
        default=os.getenv("RIGHT_CAMERA_TOPIC", "/camera_r/color/image_raw"),
    )
    parser.add_argument(
        "--observation-timeout-s",
        type=float,
        default=float(os.getenv("OBSERVATION_TIMEOUT_S", "2.0")),
        help="Timeout for waiting on state and image topics during connect().",
    )
    parser.add_argument("--queue-size", type=int, default=int(os.getenv("QUEUE_SIZE", "1")))
    parser.add_argument("--fps", type=int, default=int(os.getenv("FPS", "30")))

    parser.add_argument("--policy-path", default=os.getenv("POLICY_PATH", DEFAULT_POLICY_PATH))
    parser.add_argument(
        "--policy-device",
        default=os.getenv("POLICY_DEVICE_OVERRIDE", os.getenv("POLICY_DEVICE")),
        help="Override checkpoint device with one of cpu/cuda/mps/xpu/auto.",
    )
    parser.add_argument(
        "--policy-n-action-steps",
        type=int,
        default=parse_optional_int(os.getenv("POLICY_N_ACTION_STEPS")),
        help="Optional ACT deployment override. Must be in [1, chunk_size].",
    )
    parser.add_argument(
        "--policy-temporal-ensemble-coeff",
        type=parse_optional_float,
        default=parse_optional_float(os.getenv("POLICY_TEMPORAL_ENSEMBLE_COEFF")),
        help="Optional ACT deployment override. If set, requires policy-n-action-steps=1.",
    )
    parser.add_argument(
        "--task",
        default=os.getenv("DATASET_TASK", DEFAULT_TASK),
        help="Language instruction passed to policy inference.",
    )
    parser.add_argument(
        "--run-time-s",
        type=float,
        default=float(os.getenv("RUN_TIME_S", "0")),
        help="Total runtime in seconds. <=0 means run until Ctrl+C.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=int(os.getenv("LOG_INTERVAL", "30")),
        help="Print status every N loop iterations. Set 0 to disable periodic logs.",
    )
    parser.add_argument(
        "--dry-run",
        type=parse_bool,
        nargs="?",
        const=True,
        default=env_bool("DRY_RUN", False),
        help="Print resolved config and exit without connecting robot or loading model weights.",
    )
    return parser


def ensure_checkpoint_assets(policy_path: Path, *, require_weights: bool) -> None:
    required_files = [
        policy_path / "config.json",
        policy_path / "policy_preprocessor.json",
        policy_path / "policy_postprocessor.json",
    ]
    if require_weights:
        required_files.append(policy_path / "model.safetensors")

    missing_files = [path for path in required_files if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(f"Missing checkpoint assets: {missing_files}")


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


def sync_processor_device(pipeline: PolicyProcessorPipeline[Any, Any], device: str) -> None:
    for step in pipeline.steps:
        if isinstance(step, DeviceProcessorStep):
            step.device = device
            step.__post_init__()


def apply_act_runtime_overrides(
    policy_cfg: PreTrainedConfig,
    policy_n_action_steps: int | None,
    policy_temporal_ensemble_coeff: float | None,
) -> None:
    if policy_cfg.type != "act":
        raise ValueError(f"Expected ACT policy, got {policy_cfg.type!r}")

    chunk_size = int(policy_cfg.chunk_size)

    if policy_n_action_steps is not None:
        if not 1 <= policy_n_action_steps <= chunk_size:
            raise ValueError(
                f"--policy-n-action-steps must be within [1, {chunk_size}], got {policy_n_action_steps}"
            )
        policy_cfg.n_action_steps = policy_n_action_steps

    if policy_temporal_ensemble_coeff is not None:
        if policy_cfg.n_action_steps != 1:
            raise ValueError(
                "ACT temporal ensembling requires n_action_steps == 1. "
                f"Current value: {policy_cfg.n_action_steps}"
            )
        policy_cfg.temporal_ensemble_coeff = policy_temporal_ensemble_coeff


def feature_shape(feature: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in getattr(feature, "shape", ()))


def validate_single_arm_checkpoint_schema(policy_cfg: PreTrainedConfig, arm: str) -> tuple[int, int]:
    expected_visual_keys = {
        "observation.images.camera_front": (3, 480, 640),
        ARM_TO_POLICY_CAMERA_KEY[arm]: (3, 480, 640),
    }
    expected_input_shapes = {EXPECTED_STATE_KEY: (7,), **expected_visual_keys}
    expected_output_shapes = {EXPECTED_ACTION_KEY: (7,)}

    actual_input_keys = set(policy_cfg.input_features)
    actual_output_keys = set(policy_cfg.output_features)
    missing_inputs = sorted(set(expected_input_shapes) - actual_input_keys)
    extra_inputs = sorted(actual_input_keys - set(expected_input_shapes))
    missing_outputs = sorted(set(expected_output_shapes) - actual_output_keys)
    extra_outputs = sorted(actual_output_keys - set(expected_output_shapes))
    if missing_inputs or extra_inputs or missing_outputs or extra_outputs:
        raise ValueError(
            "Checkpoint schema does not match the requested Agilex single-arm deployment contract. "
            f"arm={arm}, missing_inputs={missing_inputs}, extra_inputs={extra_inputs}, "
            f"missing_outputs={missing_outputs}, extra_outputs={extra_outputs}"
        )

    for key, expected_shape in expected_input_shapes.items():
        actual_shape = feature_shape(policy_cfg.input_features[key])
        if actual_shape != expected_shape:
            raise ValueError(f"Unexpected shape for {key}: expected {expected_shape}, got {actual_shape}")

    for key, expected_shape in expected_output_shapes.items():
        actual_shape = feature_shape(policy_cfg.output_features[key])
        if actual_shape != expected_shape:
            raise ValueError(f"Unexpected shape for {key}: expected {expected_shape}, got {actual_shape}")

    _, image_height, image_width = expected_visual_keys["observation.images.camera_front"]
    return image_height, image_width


def build_bridge(args: argparse.Namespace) -> AgileXRosBridge:
    image_topics = (
        ImageTopicConfig(args.front_camera_topic, "camera_front"),
        ImageTopicConfig(
            args.left_camera_topic if args.arm == LEFT_PREFIX else args.right_camera_topic,
            ARM_TO_LIVE_CAMERA_KEY[args.arm],
        ),
    )
    return AgileXRosBridge(
        topics=BridgeTopics(
            state_left_topic=args.state_left_topic,
            state_right_topic=args.state_right_topic,
            command_left_topic=args.command_left_topic if args.control_mode == "command_master" else None,
            command_right_topic=args.command_right_topic if args.control_mode == "command_master" else None,
            image_topics=image_topics,
        ),
        joint_names=DEFAULT_JOINT_NAMES,
        queue_size=args.queue_size,
    )


def validate_execution_mode(args: argparse.Namespace) -> None:
    if args.execution_mode == "observation_only" and args.control_mode == "command_master":
        raise ValueError(
            "--execution-mode=observation_only is read-only. Use --control-mode=passive_follow for that mode."
        )


def validate_live_observation(observation: dict[str, Any], arm: str, image_height: int, image_width: int) -> None:
    required_camera_keys = ("camera_front", ARM_TO_LIVE_CAMERA_KEY[arm])
    required_keys = [*POSITION_FEATURE_NAMES, *required_camera_keys]
    missing_keys = [key for key in required_keys if key not in observation]
    if missing_keys:
        raise KeyError(f"Missing Agilex observation keys: {missing_keys}")

    for key in POSITION_FEATURE_NAMES:
        value = observation[key]
        if not np.isscalar(value):
            raise TypeError(f"Expected scalar joint value for {key}, got {type(value)}")

    expected_image_shape = (image_height, image_width, 3)
    for key in required_camera_keys:
        value = observation[key]
        if not isinstance(value, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray image for {key}, got {type(value)}")
        if value.shape != expected_image_shape:
            raise ValueError(
                f"Unexpected live image shape for {key}: "
                f"expected {expected_image_shape}, got {value.shape}"
            )


def build_single_arm_observation_payload(observation: dict[str, Any], arm: str) -> dict[str, np.ndarray]:
    state_vector = np.asarray(
        [float(observation[key]) for key in ARM_TO_STATE_NAMES[arm]],
        dtype=np.float32,
    )
    return {
        EXPECTED_STATE_KEY: state_vector,
        "observation.images.camera_front": observation["camera_front"],
        ARM_TO_POLICY_CAMERA_KEY[arm]: observation[ARM_TO_LIVE_CAMERA_KEY[arm]],
    }


def _to_single_arm_action_vector(arm_action: PolicyAction | np.ndarray | list[float]) -> np.ndarray:
    if hasattr(arm_action, "detach"):
        tensor = arm_action.detach()
        if hasattr(tensor, "squeeze"):
            tensor = tensor.squeeze(0)
        if hasattr(tensor, "to"):
            tensor = tensor.to("cpu")
        action_vector = np.asarray(tensor, dtype=np.float32)
    else:
        action_vector = np.asarray(arm_action, dtype=np.float32)

    if action_vector.shape != (7,):
        raise ValueError(f"Expected single-arm action shape (7,), got {action_vector.shape}")
    return action_vector


def merge_single_arm_action_with_hold_current(
    observation: dict[str, Any],
    arm_action: PolicyAction | np.ndarray | list[float],
    arm: str,
) -> dict[str, float]:
    action_vector = _to_single_arm_action_vector(arm_action)

    full_action = {key: float(observation[key]) for key in ACTION_FEATURE_NAMES}
    for idx, key in enumerate(ARM_TO_ACTION_NAMES[arm]):
        full_action[key] = float(action_vector[idx])
    return full_action


def validate_hold_current_merge(action: dict[str, float], observation: dict[str, Any], arm: str) -> None:
    missing_keys = [key for key in ACTION_FEATURE_NAMES if key not in action]
    if missing_keys:
        raise ValueError(f"Merged robot action is missing keys: {missing_keys}")

    inactive_arm = RIGHT_PREFIX if arm == LEFT_PREFIX else LEFT_PREFIX
    mismatched_keys = [
        key
        for key in ARM_TO_ACTION_NAMES[inactive_arm]
        if not np.isclose(float(action[key]), float(observation[key]), atol=INACTIVE_ARM_HOLD_ATOL)
    ]
    if mismatched_keys:
        raise ValueError(
            "Inactive arm command deviates from the current observation hold pose. "
            f"arm={inactive_arm}, keys={mismatched_keys}"
        )


def summarize_live_observation(observation: dict[str, Any]) -> str:
    state_vector = np.asarray([float(observation[key]) for key in POSITION_FEATURE_NAMES], dtype=np.float32)
    image_summary = ", ".join(
        f"{key}={observation[key].shape}/{observation[key].dtype}"
        for key in ("camera_front", "camera_left", "camera_right")
        if key in observation
    )
    return (
        f"state14[min={state_vector.min():.4f}, max={state_vector.max():.4f}, mean={state_vector.mean():.4f}] | "
        f"images[{image_summary}]"
    )


def summarize_policy_observation(policy_observation: dict[str, np.ndarray], arm: str) -> str:
    state_vector = np.asarray(policy_observation[EXPECTED_STATE_KEY], dtype=np.float32)
    side_policy_key = ARM_TO_POLICY_CAMERA_KEY[arm]
    return (
        f"{arm}_state7[min={state_vector.min():.4f}, max={state_vector.max():.4f}, mean={state_vector.mean():.4f}] | "
        f"images[camera_front={policy_observation['observation.images.camera_front'].shape}/"
        f"{policy_observation['observation.images.camera_front'].dtype}, "
        f"{side_policy_key}={policy_observation[side_policy_key].shape}/{policy_observation[side_policy_key].dtype}]"
    )


def summarize_selected_arm_action(action: dict[str, float], arm: str) -> str:
    action_vector = np.asarray([float(action[key]) for key in ARM_TO_ACTION_NAMES[arm]], dtype=np.float32)
    return (
        f"{arm}_action7[min={action_vector.min():.4f}, max={action_vector.max():.4f}, "
        f"mean={action_vector.mean():.4f}, std={action_vector.std():.4f}]"
    )


def get_live_observation(bridge: AgileXRosBridge) -> dict[str, Any]:
    observation = bridge.get_state_features()
    observation.update(bridge.get_images())
    return observation


def print_runtime_summary(
    args: argparse.Namespace,
    policy_path: Path,
    policy_cfg: PreTrainedConfig,
    image_height: int,
    image_width: int,
) -> None:
    side_policy_key = ARM_TO_POLICY_CAMERA_KEY[args.arm]
    print(f"[INFO] Robot id: {args.robot_id}")
    print(f"[INFO] Arm: {args.arm}")
    print(f"[INFO] Execution mode: {args.execution_mode}")
    print(f"[INFO] Control mode: {args.control_mode}")
    print(f"[INFO] Policy path: {policy_path}")
    print(f"[INFO] Policy type: {policy_cfg.type}")
    print(f"[INFO] Policy device: {policy_cfg.device}")
    print(f"[INFO] Task: {args.task}")
    print(f"[INFO] FPS: {args.fps}")
    print(f"[INFO] run_time_s: {args.run_time_s} (<=0 means until Ctrl+C)")
    print(
        "[INFO] ACT runtime config: "
        f"chunk_size={policy_cfg.chunk_size}, "
        f"n_action_steps={policy_cfg.n_action_steps}, "
        f"temporal_ensemble_coeff={policy_cfg.temporal_ensemble_coeff}"
    )
    print("[INFO] Input features:")
    print(f"[INFO]   {EXPECTED_STATE_KEY}: {feature_shape(policy_cfg.input_features[EXPECTED_STATE_KEY])}")
    print("[INFO]   observation.images.camera_front: "
          f"{feature_shape(policy_cfg.input_features['observation.images.camera_front'])}")
    print(f"[INFO]   {side_policy_key}: {feature_shape(policy_cfg.input_features[side_policy_key])}")
    print(
        f"[INFO] Output features: {EXPECTED_ACTION_KEY}: "
        f"{feature_shape(policy_cfg.output_features[EXPECTED_ACTION_KEY])}"
    )
    print(f"[INFO] Expected live image shape (HWC): ({image_height}, {image_width}, 3)")
    print(f"[INFO] State topics: left={args.state_left_topic} right={args.state_right_topic}")
    print(f"[INFO] Command topics: left={args.command_left_topic} right={args.command_right_topic}")
    print(
        "[INFO] Camera topics: "
        f"front={args.front_camera_topic} left={args.left_camera_topic} right={args.right_camera_topic}"
    )
    print(f"[INFO] hold_inactive_arm_mode: current_observation_pose")
    print(f"[INFO] observation_timeout_s: {args.observation_timeout_s}")
    print(f"[INFO] queue_size: {args.queue_size}")


def main() -> None:
    register_third_party_plugins()
    args = build_parser().parse_args()
    validate_execution_mode(args)

    policy_path = Path(args.policy_path).expanduser()
    if not policy_path.is_dir():
        raise FileNotFoundError(f"Policy path does not exist: {policy_path}")

    ensure_checkpoint_assets(
        policy_path,
        require_weights=args.execution_mode == "policy_inference" and not args.dry_run,
    )

    policy_cfg = PreTrainedConfig.from_pretrained(str(policy_path))
    image_height, image_width = validate_single_arm_checkpoint_schema(policy_cfg, args.arm)
    policy_cfg.pretrained_path = policy_path
    policy_cfg.device = resolve_policy_device(args.policy_device, policy_cfg.device)
    apply_act_runtime_overrides(
        policy_cfg=policy_cfg,
        policy_n_action_steps=args.policy_n_action_steps,
        policy_temporal_ensemble_coeff=args.policy_temporal_ensemble_coeff,
    )

    print_runtime_summary(args, policy_path, policy_cfg, image_height, image_width)
    if args.dry_run:
        print("[INFO] DRY_RUN=true, exit without execution.")
        return

    bridge = build_bridge(args)

    policy = None
    preprocessor = None
    postprocessor = None
    if args.execution_mode == "policy_inference":
        policy_class = get_policy_class(policy_cfg.type)
        policy = policy_class.from_pretrained(str(policy_path), config=policy_cfg, strict=False)
        policy.config.device = policy_cfg.device
        policy.to(policy_cfg.device)

        preprocessor, postprocessor = load_pre_post_processors(policy_path)
        sync_processor_device(preprocessor, policy_cfg.device)
        sync_processor_device(postprocessor, "cpu")

    step = 0
    start_t = time.perf_counter()
    end_t = start_t + args.run_time_s if args.run_time_s > 0 else None
    first_observation_logged = False
    first_policy_observation_logged = False
    publish_enabled = args.execution_mode == "policy_inference" and args.control_mode == "command_master"

    try:
        bridge.connect(
            node_name="lerobot_agilex_single_arm_infer",
            needs_publishers=args.control_mode == "command_master",
        )
        bridge.wait_for_ready(timeout_s=args.observation_timeout_s, require_images=True)
        print(f"[INFO] Agilex connected. publish_enabled={publish_enabled}")

        if policy is not None and preprocessor is not None and postprocessor is not None:
            policy.reset()
            preprocessor.reset()
            postprocessor.reset()

        while True:
            if end_t is not None and time.perf_counter() >= end_t:
                print("[INFO] Reached requested run_time_s. Exiting loop.")
                break

            loop_t = time.perf_counter()
            observation = get_live_observation(bridge)
            validate_live_observation(observation, args.arm, image_height=image_height, image_width=image_width)

            step += 1
            if not first_observation_logged:
                print(f"[INFO] First live observation summary: {summarize_live_observation(observation)}")
                first_observation_logged = True

            if args.execution_mode == "observation_only":
                policy_observation = build_single_arm_observation_payload(observation, args.arm)
                if not first_policy_observation_logged:
                    print(
                        f"[INFO] First policy observation summary: "
                        f"{summarize_policy_observation(policy_observation, args.arm)}"
                    )
                    first_policy_observation_logged = True

                if args.log_interval > 0 and step % args.log_interval == 0:
                    elapsed = time.perf_counter() - start_t
                    print(
                        f"[INFO] Step {step} | elapsed={elapsed:.2f}s | "
                        f"{summarize_policy_observation(policy_observation, args.arm)}"
                    )
            else:
                assert policy is not None
                assert preprocessor is not None
                assert postprocessor is not None

                policy_observation = build_single_arm_observation_payload(observation, args.arm)
                if not first_policy_observation_logged:
                    print(
                        f"[INFO] First policy observation summary: "
                        f"{summarize_policy_observation(policy_observation, args.arm)}"
                    )
                    first_policy_observation_logged = True

                action_values = predict_action(
                    observation=policy_observation,
                    policy=policy,
                    device=get_safe_torch_device(policy.config.device),
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    use_amp=policy.config.use_amp,
                    task=args.task,
                    robot_type=ROBOT_TYPE,
                )
                robot_action_to_send = merge_single_arm_action_with_hold_current(
                    observation=observation,
                    arm_action=action_values,
                    arm=args.arm,
                )
                validate_hold_current_merge(robot_action_to_send, observation, args.arm)
                sent_action = robot_action_to_send
                if publish_enabled:
                    bridge.publish_action(sent_action)

                if args.log_interval > 0 and step % args.log_interval == 0:
                    elapsed = time.perf_counter() - start_t
                    mode_label = "closed_loop" if publish_enabled else "shadow_infer"
                    print(
                        f"[INFO] Step {step} | elapsed={elapsed:.2f}s | mode={mode_label} | "
                        f"{summarize_selected_arm_action(sent_action, args.arm)}"
                    )

            dt_s = time.perf_counter() - loop_t
            precise_sleep(max(1 / args.fps - dt_s, 0.0))
    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt received. Stopping inference.")
    finally:
        if bridge.is_connected:
            bridge.disconnect()
        print("[INFO] Agilex single-arm inference finished.")


if __name__ == "__main__":
    main()
