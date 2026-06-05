#!/usr/bin/env python

"""Camera-only PI0.5 PyTorch vs TensorRT FP16 smoke comparison.

This script intentionally does not connect to, command, or instantiate the
robot. It only reads top/wrist camera frames, runs policy inference offline, and
compares pure PyTorch output against a TensorRT-backed suffix embedding engine.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
OPENPI_TRT_DIR = SCRIPT_DIR.parent
if OPENPI_TRT_DIR.as_posix() not in sys.path:
    sys.path.insert(0, OPENPI_TRT_DIR.as_posix())

from runtime.pi05_trt_suffix import patch_embed_suffix_with_trt  # noqa: E402
from scripts.pi05_onnx_common import (  # noqa: E402
    DEFAULT_POLICY_PATH,
    DEFAULT_TASK,
    configure_runtime,
    ensure_local_tokenizer_dir,
    load_policy,
    load_pre_post_processors,
    tensor_stats,
    write_json,
)

from lerobot.policies.utils import prepare_observation_for_inference  # noqa: E402
from lerobot.utils.utils import get_safe_torch_device  # noqa: E402


DEFAULT_ENGINE_PATH = Path("my_devs/openpi_trt/artifacts/pi05_so101_suffix_embedding_b1_fp16.engine")


def parse_camera(value: str) -> int | str:
    return int(value) if value.isdecimal() else value


def parse_state_values(raw: str) -> np.ndarray:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if len(values) != 6:
        raise argparse.ArgumentTypeError(f"--state-values expects 6 comma-separated floats, got {len(values)}")
    return np.asarray(values, dtype=np.float32)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke compare PI0.5 PyTorch vs TensorRT FP16 using cameras only.")
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--engine-path", type=Path, default=DEFAULT_ENGINE_PATH)
    parser.add_argument("--top-cam", type=parse_camera, default="/dev/video4")
    parser.add_argument("--wrist-cam", type=parse_camera, default="/dev/video6")
    parser.add_argument("--top-cam-fourcc", default="YUYV")
    parser.add_argument("--wrist-cam-fourcc", default="MJPG")
    parser.add_argument("--img-width", type=int, default=640)
    parser.add_argument("--img-height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--state-values", type=parse_state_values, default=np.zeros(6, dtype=np.float32))
    parser.add_argument("--robot-port", default=None, help="Accepted for CLI compatibility; never opened.")
    parser.add_argument("--robot-type", default="so101_follower")
    parser.add_argument("--noise-seed", type=int, default=2026)
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--check-policy-load", action="store_true")
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="JSON report path. Defaults to my_devs/openpi_trt/artifacts/camera_smoke_torch_trt_<timestamp>.json",
    )
    return parser


def open_camera(index_or_path: int | str, width: int, height: int, fps: int, fourcc: str, name: str):
    cap = cv2.VideoCapture(index_or_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {name} camera: {index_or_path}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    return cap


def read_camera_frame(cap, name: str, warmup_frames: int) -> np.ndarray:
    frame = None
    for _ in range(max(warmup_frames, 0) + 1):
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read frame from {name} camera")
        time.sleep(0.01)
    # OpenCV returns BGR HWC uint8; LeRobot image prep expects RGB-like HWC.
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def prepare_batch(policy, policy_path: Path, top_rgb: np.ndarray, wrist_rgb: np.ndarray, state: np.ndarray, task: str, robot_type: str):
    preprocessor, _ = load_pre_post_processors(policy_path)
    preprocessor.reset()

    observation = {
        "observation.state": state.astype(np.float32),
        "observation.images.top": top_rgb,
        "observation.images.wrist": wrist_rgb,
    }
    device = get_safe_torch_device(policy.config.device)
    observation = prepare_observation_for_inference(observation, device, task=task, robot_type=robot_type)
    return preprocessor(observation)


def make_noise(policy, batch: dict[str, torch.Tensor], seed: int) -> torch.Tensor:
    tokens = batch["observation.language.tokens"]
    generator = torch.Generator(device=tokens.device)
    generator.manual_seed(seed)
    return torch.randn(
        tokens.shape[0],
        policy.config.chunk_size,
        policy.config.max_action_dim,
        dtype=torch.float32,
        device=tokens.device,
        generator=generator,
    )


@torch.no_grad()
def run_action_chunk(policy, batch: dict[str, torch.Tensor], noise: torch.Tensor) -> torch.Tensor:
    images, img_masks = policy._preprocess_images(batch)
    tokens = batch["observation.language.tokens"]
    masks = batch["observation.language.attention_mask"]
    actions = policy.model.sample_actions(images, img_masks, tokens, masks, noise=noise)
    original_action_dim = policy.config.output_features["action"].shape[0]
    return actions[:, :, :original_action_dim]


def save_frame(path: Path, frame_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))


def main() -> None:
    configure_runtime()
    args = build_parser().parse_args()

    policy_path = args.policy_path.expanduser().resolve()
    engine_path = args.engine_path.expanduser()
    if not policy_path.is_dir():
        raise FileNotFoundError(f"Policy path not found: {policy_path}")
    if not engine_path.is_file():
        raise FileNotFoundError(f"TensorRT FP16 engine not found: {engine_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TensorRT smoke comparison.")

    ensure_local_tokenizer_dir()

    print("[SAFETY] Camera-only smoke test. This script never opens robot serial and never sends actions.")
    if args.robot_port:
        print(f"[SAFETY] Ignoring --robot-port={args.robot_port}; it is accepted only for CLI compatibility.")

    print(f"[INFO] Loading policy: {policy_path}")
    policy = load_policy(policy_path, device="cuda")
    if args.check_policy_load:
        print("[INFO] CHECK_POLICY_LOAD=true, loaded policy and exiting before camera access.")
        return

    top_cap = None
    wrist_cap = None
    try:
        top_cap = open_camera(args.top_cam, args.img_width, args.img_height, args.fps, args.top_cam_fourcc, "top")
        wrist_cap = open_camera(args.wrist_cam, args.img_width, args.img_height, args.fps, args.wrist_cam_fourcc, "wrist")
        top_rgb = read_camera_frame(top_cap, "top", args.warmup_frames)
        wrist_rgb = read_camera_frame(wrist_cap, "wrist", args.warmup_frames)
    finally:
        if top_cap is not None:
            top_cap.release()
        if wrist_cap is not None:
            wrist_cap.release()

    batch = prepare_batch(policy, policy_path, top_rgb, wrist_rgb, args.state_values, args.task, args.robot_type)
    noise = make_noise(policy, batch, args.noise_seed)

    print("[INFO] Running PyTorch sample_actions...")
    policy.reset()
    torch_actions = run_action_chunk(policy, batch, noise)

    print(f"[INFO] Patching embed_suffix with TensorRT FP16 engine: {engine_path}")
    engine = patch_embed_suffix_with_trt(policy, engine_path)

    print("[INFO] Running TensorRT-backed sample_actions...")
    policy.reset()
    trt_actions = run_action_chunk(policy, batch, noise)

    torch_np = torch_actions.detach().cpu().numpy()
    trt_np = trt_actions.detach().cpu().numpy()
    stats = tensor_stats(torch_np, trt_np)
    passed = bool(np.allclose(torch_np, trt_np, rtol=args.rtol, atol=args.atol))

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    report = args.report
    if report is None:
        report = Path(f"my_devs/openpi_trt/artifacts/camera_smoke_torch_trt_fp16_{timestamp}.json")

    frame_paths = {}
    if args.save_frames:
        frame_dir = report.parent / f"{report.stem}_frames"
        top_path = frame_dir / "top_rgb.png"
        wrist_path = frame_dir / "wrist_rgb.png"
        save_frame(top_path, top_rgb)
        save_frame(wrist_path, wrist_rgb)
        frame_paths = {"top_rgb": str(top_path), "wrist_rgb": str(wrist_path)}

    data = {
        "mode": "camera_smoke_sample_actions_with_trt_suffix_embedding",
        "safety": {
            "robot_connected": False,
            "robot_action_sent": False,
            "robot_port_ignored": args.robot_port,
        },
        "passed_allclose": passed,
        "rtol": args.rtol,
        "atol": args.atol,
        "policy_path": str(policy_path),
        "engine_path": str(engine_path),
        "task": args.task,
        "state_values": args.state_values.astype(float).tolist(),
        "top_cam": str(args.top_cam),
        "wrist_cam": str(args.wrist_cam),
        "top_frame_shape": list(top_rgb.shape),
        "wrist_frame_shape": list(wrist_rgb.shape),
        "engine": engine.describe(),
        "action_chunk_stats": stats,
        "first_action_torch": torch_np[0, 0].astype(float).tolist(),
        "first_action_trt": trt_np[0, 0].astype(float).tolist(),
        "frame_paths": frame_paths,
        "noise_seed": args.noise_seed,
    }
    write_json(report, data)

    print("[INFO] Camera smoke Torch vs TRT FP16 statistics:")
    print(f"[INFO]   action chunk torch={stats['reference_shape']} trt={stats['candidate_shape']}")
    print(f"[INFO]   mean_abs_diff={stats['mean_abs_diff']:.8f}")
    print(f"[INFO]   max_abs_diff={stats['max_abs_diff']:.8f}")
    print(f"[INFO]   cosine_similarity={stats['cosine_similarity']:.8f}")
    print(f"[INFO]   allclose(rtol={args.rtol}, atol={args.atol})={passed}")
    print(f"[INFO]   report={report}")

    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
