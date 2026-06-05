#!/usr/bin/env python

"""Camera-only PI0.5 PyTorch vs split TensorRT smoke comparison."""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
OPENPI_TRT_DIR = SCRIPT_DIR.parent
if OPENPI_TRT_DIR.as_posix() not in sys.path:
    sys.path.insert(0, OPENPI_TRT_DIR.as_posix())

from runtime.pi05_trt_split import PI05TensorRTSplitRuntime  # noqa: E402
from scripts.pi05_onnx_common import (  # noqa: E402
    DEFAULT_POLICY_PATH,
    DEFAULT_TASK,
    configure_runtime,
    ensure_local_tokenizer_dir,
    load_policy,
    tensor_stats,
    write_json,
)
from scripts.smoke_pi05_camera_torch_trt import (  # noqa: E402
    make_noise,
    open_camera,
    parse_camera,
    parse_state_values,
    prepare_batch,
    read_camera_frame,
    run_action_chunk,
    save_frame,
)


DEFAULT_PREFIX_ENGINE_PATH = Path("my_devs/openpi_trt/artifacts/pi05_so101_prefix_cache_b1_fp32.engine")
DEFAULT_DENOISE_ENGINE_PATH = Path("my_devs/openpi_trt/artifacts/pi05_so101_denoise_step_b1_fp32.engine")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke compare PI0.5 PyTorch vs split TensorRT using cameras only.")
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--prefix-engine-path", type=Path, default=DEFAULT_PREFIX_ENGINE_PATH)
    parser.add_argument("--denoise-engine-path", type=Path, default=DEFAULT_DENOISE_ENGINE_PATH)
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
    parser.add_argument("--atol", type=float, default=1e-1)
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--check-policy-load", action="store_true")
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="JSON report path. Defaults to my_devs/openpi_trt/artifacts/camera_smoke_split_trt_<timestamp>.json",
    )
    return parser


@torch.no_grad()
def run_split_trt_action_chunk(policy, batch: dict[str, torch.Tensor], noise: torch.Tensor, runtime):
    images, img_masks = policy._preprocess_images(batch)
    tokens = batch["observation.language.tokens"]
    masks = batch["observation.language.attention_mask"]
    actions = runtime(policy, images, img_masks, tokens, masks, noise)
    original_action_dim = policy.config.output_features["action"].shape[0]
    return actions[:, :, :original_action_dim]


def main() -> None:
    configure_runtime()
    args = build_parser().parse_args()

    policy_path = args.policy_path.expanduser().resolve()
    prefix_engine_path = args.prefix_engine_path.expanduser()
    denoise_engine_path = args.denoise_engine_path.expanduser()
    if not policy_path.is_dir():
        raise FileNotFoundError(f"Policy path not found: {policy_path}")
    if not prefix_engine_path.is_file():
        raise FileNotFoundError(f"Prefix TensorRT engine not found: {prefix_engine_path}")
    if not denoise_engine_path.is_file():
        raise FileNotFoundError(f"Denoise TensorRT engine not found: {denoise_engine_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TensorRT smoke comparison.")

    ensure_local_tokenizer_dir()

    print("[SAFETY] Camera-only smoke test. This script never opens robot serial and never sends actions.")
    if args.robot_port:
        print(f"[SAFETY] Ignoring --robot-port={args.robot_port}; it is accepted only for CLI compatibility.")

    print(f"[INFO] Loading policy: {policy_path}")
    policy = load_policy(policy_path, device="cuda", model_dtype="float32")
    if args.check_policy_load:
        print("[INFO] CHECK_POLICY_LOAD=true, loaded policy and exiting before camera access.")
        return

    top_cap = None
    wrist_cap = None
    try:
        top_cap = open_camera(args.top_cam, args.img_width, args.img_height, args.fps, args.top_cam_fourcc, "top")
        wrist_cap = open_camera(
            args.wrist_cam, args.img_width, args.img_height, args.fps, args.wrist_cam_fourcc, "wrist"
        )
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

    print("[INFO] Loading split TensorRT runtime...")
    runtime = PI05TensorRTSplitRuntime(prefix_engine_path, denoise_engine_path)

    print("[INFO] Running split TensorRT sample_actions...")
    policy.reset()
    trt_actions = run_split_trt_action_chunk(policy, batch, noise, runtime)

    torch_np = torch_actions.detach().cpu().numpy()
    trt_np = trt_actions.detach().cpu().numpy()
    stats = tensor_stats(torch_np, trt_np)
    passed = bool(np.allclose(torch_np, trt_np, rtol=args.rtol, atol=args.atol))

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    report = args.report
    if report is None:
        report = Path(f"my_devs/openpi_trt/artifacts/camera_smoke_split_trt_{timestamp}.json")

    frame_paths = {}
    if args.save_frames:
        frame_dir = report.parent / f"{report.stem}_frames"
        top_path = frame_dir / "top_rgb.png"
        wrist_path = frame_dir / "wrist_rgb.png"
        save_frame(top_path, top_rgb)
        save_frame(wrist_path, wrist_rgb)
        frame_paths = {"top_rgb": str(top_path), "wrist_rgb": str(wrist_path)}

    data = {
        "mode": "camera_smoke_sample_actions_with_split_trt_prefix_cache_denoise_step",
        "safety": {
            "robot_connected": False,
            "robot_action_sent": False,
            "robot_port_ignored": args.robot_port,
        },
        "passed_allclose": passed,
        "rtol": args.rtol,
        "atol": args.atol,
        "policy_path": str(policy_path),
        "prefix_engine_path": str(prefix_engine_path),
        "denoise_engine_path": str(denoise_engine_path),
        "task": args.task,
        "state_values": args.state_values.astype(float).tolist(),
        "top_cam": str(args.top_cam),
        "wrist_cam": str(args.wrist_cam),
        "top_frame_shape": list(top_rgb.shape),
        "wrist_frame_shape": list(wrist_rgb.shape),
        "runtime": runtime.describe(),
        "action_chunk_stats": stats,
        "first_action_torch": torch_np[0, 0].astype(float).tolist(),
        "first_action_trt": trt_np[0, 0].astype(float).tolist(),
        "frame_paths": frame_paths,
        "noise_seed": args.noise_seed,
    }
    write_json(report, data)

    print("[INFO] Camera smoke Torch vs split TensorRT statistics:")
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
