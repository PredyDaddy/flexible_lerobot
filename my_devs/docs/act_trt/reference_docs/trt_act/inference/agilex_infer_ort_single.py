#!/usr/bin/env python3
"""Agilex 单臂 ACT（ONNXRuntime）推理脚本

与 `agilex_infer_single_cc_vertical.py` 的机器人控制/节拍/队列语义保持一致，
仅将“策略前向”替换为 ONNXRuntime（`trt_act/inference/ort_policy.py`）。

python trt_act/inference/agilex_infer_ort_single.py \
    --checkpoint outputs/act_agilex_left_box/checkpoints/030000/pretrained_model \
    --arm left \
    --fps 30 \
    --binary-gripper \
    --onnx trt_act/export_models/act_single.onnx \
    --export-metadata trt_act/export_models/export_metadata.json \
    --ort-provider cpu

"""

from __future__ import annotations

import argparse
import contextlib
import logging
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from lerobot.cameras.ros_camera import RosCameraConfig
from lerobot.policies.utils import make_robot_action
from lerobot.robots.agilex import AgileXConfig, AgileXRobot
from lerobot.robots.agilex.config_agilex import JOINT_NAMES
from lerobot.utils.robot_utils import precise_sleep

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from trt_act.inference.ort_policy import ActOrtPolicy  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


LEFT_INIT_POS = [
    0.007710248231887817,
    1.9562225341796875,
    -1.9327428340911865,
    -0.01397264376282692,
    1.1765978336334229,
    0.2510017156600952,
    0.0,
]

RIGHT_INIT_POS = [
    -0.09158100187778473,
    1.8357367515563965,
    -1.7473654747009277,
    0.006960155908018351,
    1.2210625410079956,
    0.1726432740688324,
    7.000000186963007e-05,
]


def get_arm_config(arm: str) -> dict[str, Any]:
    if arm == "left":
        return {
            "cameras": ("camera_left", "camera_front"),
            "state_keys": [f"left_{j}.pos" for j in JOINT_NAMES],
            "prefix": "left_",
        }
    if arm == "right":
        return {
            "cameras": ("camera_right", "camera_front"),
            "state_keys": [f"right_{j}.pos" for j in JOINT_NAMES],
            "prefix": "right_",
        }
    raise ValueError(f"Invalid arm: {arm}, must be 'left' or 'right'")


@dataclass
class StopSignal:
    stopped: bool = False

    def __call__(self, _signum: int, _frame: object | None = None) -> None:
        if not self.stopped:
            logger.info("收到停止信号，正在退出...")
        self.stopped = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agilex 单臂 ACT（ONNXRuntime）推理脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="策略 checkpoint 路径 (包含 config.json 和 stats safetensors)",
    )
    parser.add_argument("--arm", type=str, required=True, choices=["left", "right"], help="选择控制的臂")
    parser.add_argument("--fps", type=int, default=30, help="推理频率")
    parser.add_argument("--duration", type=float, default=None, help="推理持续时间(秒)，不指定则无限循环")
    parser.add_argument("--mock", action="store_true", help="使用 mock 模式（不连接真实硬件）")
    parser.add_argument("--binary-gripper", action="store_true", help="启用夹爪二值化（增加抓取力度）")
    parser.add_argument(
        "--gripper-threshold",
        type=float,
        default=0.7,
        help="夹爪二值化阈值 (0-1)，低于此值设为0，高于设为最大值",
    )
    parser.add_argument(
        "--onnx",
        type=str,
        default=None,
        help="ONNX 模型路径（默认 <ckpt>/act_single.onnx）",
    )
    parser.add_argument(
        "--export-metadata",
        type=str,
        default=None,
        help="导出元数据路径（默认从 <ckpt>/export_metadata.json 读取）",
    )
    parser.add_argument(
        "--use-export-metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="优先使用 export_metadata.json 读取相机顺序与 shape",
    )
    parser.add_argument(
        "--ort-provider",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="ONNXRuntime execution provider（使用 cuda 需要安装 onnxruntime-gpu）",
    )

    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps 必须为正整数")
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration 必须为正数")
    return args


def has_valid_images(obs: dict[str, Any], camera_names: tuple[str, ...], *, allow_blank: bool) -> bool:
    image_keys = [k for k in camera_names if k in obs]
    if not image_keys:
        return False
    if allow_blank:
        return True

    for key in image_keys:
        img = obs.get(key)
        if img is not None and np.any(img):
            return True
    return False


def format_observation(obs: dict[str, Any], arm_config: dict[str, Any]) -> dict[str, Any]:
    formatted: dict[str, Any] = {}

    for cam in arm_config["cameras"]:
        if cam in obs:
            formatted[f"observation.images.{cam}"] = obs[cam]

    formatted["observation.state"] = np.fromiter(
        (float(obs.get(k, 0.0)) for k in arm_config["state_keys"]),
        dtype=np.float32,
        count=len(arm_config["state_keys"]),
    )
    return formatted


def binarize_gripper(action: dict[str, Any], arm: str, threshold: float) -> dict[str, Any]:
    gripper_max = 0.085
    threshold_value = gripper_max * threshold
    key = f"{arm}_gripper.pos"

    if key in action:
        if action[key] < threshold_value:
            action[key] = 0.0
        else:
            action[key] = gripper_max
    return action


def make_robot(mock: bool, arm: str) -> AgileXRobot:
    camera_configs = {
        "camera_front": RosCameraConfig(
            topic_name="/camera_f/color/image_raw",
            width=640,
            height=480,
            fps=30,
            mock=mock,
        ),
    }

    if arm == "left":
        camera_configs["camera_left"] = RosCameraConfig(
            topic_name="/camera_l/color/image_raw",
            width=640,
            height=480,
            fps=30,
            mock=mock,
        )
    else:
        camera_configs["camera_right"] = RosCameraConfig(
            topic_name="/camera_r/color/image_raw",
            width=640,
            height=480,
            fps=30,
            mock=mock,
        )

    robot_config = AgileXConfig(id="agilex_infer_ort_single", mock=mock, cameras=camera_configs)
    return AgileXRobot(robot_config)


@contextlib.contextmanager
def connected_robot(robot: AgileXRobot) -> Iterator[AgileXRobot]:
    robot.connect()
    try:
        yield robot
    finally:
        logger.info("断开机器人连接...")
        try:
            robot.disconnect()
        except Exception as e:
            logger.warning("断开连接时出错: %s", e)


def _clip_delta(current: np.ndarray, target: np.ndarray, max_delta: float) -> np.ndarray:
    delta = target - current
    delta = np.clip(delta, -max_delta, max_delta)
    return current + delta


def send_single_arm_action(robot: AgileXRobot, action: dict[str, Any], arm: str) -> dict[str, Any]:
    left_state, right_state = robot.ros_bridge.get_puppet_state()

    if arm == "left":
        left_target = np.array([action[f"left_{jn}.pos"] for jn in JOINT_NAMES], dtype=np.float32)
        right_target = np.array(right_state.position, dtype=np.float32)
    else:
        left_target = np.array(left_state.position, dtype=np.float32)
        right_target = np.array([action[f"right_{jn}.pos"] for jn in JOINT_NAMES], dtype=np.float32)

    if robot.config.max_relative_target > 0:
        left_target = _clip_delta(left_state.position, left_target, robot.config.max_relative_target)
        right_target = _clip_delta(right_state.position, right_target, robot.config.max_relative_target)

    robot.ros_bridge.send_joint_commands(left_target, right_target)

    result = {}
    for i, jn in enumerate(JOINT_NAMES):
        result[f"left_{jn}.pos"] = float(left_target[i])
        result[f"right_{jn}.pos"] = float(right_target[i])
    return result


def move_to_init_position(robot: AgileXRobot, arm: str, fps: int = 30) -> None:
    target = np.array(LEFT_INIT_POS if arm == "left" else RIGHT_INIT_POS, dtype=np.float32)

    logger.info("移动到初始位置... arm=%s", arm)
    period_s = 1.0 / fps
    for _ in range(600):
        left_state, right_state = robot.ros_bridge.get_puppet_state()
        left_current = np.array(left_state.position, dtype=np.float32)
        right_current = np.array(right_state.position, dtype=np.float32)

        if arm == "left":
            left_target = target
            right_target = right_current
        else:
            left_target = left_current
            right_target = target

        if robot.config.max_relative_target > 0:
            left_target = _clip_delta(left_state.position, left_target, robot.config.max_relative_target)
            right_target = _clip_delta(right_state.position, right_target, robot.config.max_relative_target)

        robot.ros_bridge.send_joint_commands(left_target, right_target)

        left_err = np.max(np.abs(left_current - left_target))
        right_err = np.max(np.abs(right_current - right_target))
        if left_err < 0.02 and right_err < 0.02:
            logger.info("已到达初始位置")
            break

        precise_sleep(period_s)

    logger.info("初始位置移动完成")


def run_inference_loop(
    *,
    robot: AgileXRobot,
    policy: ActOrtPolicy,
    fps: int,
    duration_s: float | None,
    stop: StopSignal,
    allow_blank_images: bool,
    arm: str,
    arm_config: dict[str, Any],
    binary_gripper: bool,
    gripper_threshold: float,
) -> tuple[int, float]:
    logger.info("开始推理循环 (FPS=%d, arm=%s)", fps, arm)

    policy.reset()

    ds_features = {"action": {"names": arm_config["state_keys"]}}

    step_count = 0
    start_t = time.perf_counter()
    period_s = 1.0 / fps
    deadline_t = time.monotonic() + duration_s if duration_s is not None else None

    while not stop.stopped:
        loop_start_t = time.perf_counter()

        if deadline_t is not None and time.monotonic() >= deadline_t:
            logger.info("达到设定时间 %.3fs，停止推理", duration_s)
            break

        obs = robot.get_observation()
        if not has_valid_images(obs, arm_config["cameras"], allow_blank=allow_blank_images):
            logger.warning("当前帧观测无有效图像，跳过")
        else:
            formatted_obs = format_observation(obs, arm_config)
            action_vec = policy.select_action(formatted_obs)
            action_tensor = torch.from_numpy(action_vec[None, :])

            robot_action = make_robot_action(action_tensor, ds_features)
            if binary_gripper:
                robot_action = binarize_gripper(robot_action, arm, gripper_threshold)
            send_single_arm_action(robot, robot_action, arm)

            step_count += 1
            if step_count % 100 == 0:
                elapsed = time.perf_counter() - start_t
                fps_actual = step_count / elapsed if elapsed > 0 else 0.0
                logger.info("步数: %d, 实际 FPS: %.1f", step_count, fps_actual)

        precise_sleep(period_s - (time.perf_counter() - loop_start_t))

    return step_count, time.perf_counter() - start_t


def main() -> int:
    args = parse_args()

    stop = StopSignal()
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_dir():
        logger.error("Checkpoint 路径不存在: %s", checkpoint_path)
        return 1

    arm_config = get_arm_config(args.arm)
    logger.info("单臂模式: %s", args.arm)
    logger.info("使用相机: %s", arm_config["cameras"])
    logger.info("控制关节: %s", arm_config["state_keys"])

    onnx_path = Path(args.onnx) if args.onnx is not None else None
    export_metadata_path = Path(args.export_metadata) if args.export_metadata is not None else None

    providers = ["CPUExecutionProvider"]
    if args.ort_provider == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    try:
        policy = ActOrtPolicy(
            checkpoint=checkpoint_path,
            onnx_path=onnx_path,
            use_export_metadata=bool(args.use_export_metadata),
            export_metadata_path=export_metadata_path,
            providers=providers,
        )
    except Exception as e:
        logger.error("ORT policy 加载失败: %s", e)
        return 1

    robot = make_robot(args.mock, args.arm)
    try:
        with connected_robot(robot):
            move_to_init_position(robot, args.arm, fps=args.fps)
            logger.info("等待 1 秒后开始推理...")
            time.sleep(1.0)

            step_count, total_time = run_inference_loop(
                robot=robot,
                policy=policy,
                fps=args.fps,
                duration_s=args.duration,
                stop=stop,
                allow_blank_images=args.mock,
                arm=args.arm,
                arm_config=arm_config,
                binary_gripper=args.binary_gripper,
                gripper_threshold=args.gripper_threshold,
            )
    except Exception as e:
        logger.error("推理过程出错: %s", e)
        raise

    avg_fps = step_count / total_time if total_time > 0 else 0.0
    logger.info("推理结束，共 %d 步，耗时 %.1fs，平均 FPS: %.1f", step_count, total_time, avg_fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
