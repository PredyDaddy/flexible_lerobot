#!/usr/bin/env python

"""Benchmark camera-based PI0.5 PyTorch vs TensorRT suffix-engine inference.

This is a read-only benchmark: it opens cameras, captures one frame from each
camera, and runs repeated policy forward passes. It never connects to the robot
serial port and never sends actions.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

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
    tensor_stats,
    write_json,
)
from scripts.smoke_pi05_camera_torch_trt import (  # noqa: E402
    DEFAULT_ENGINE_PATH,
    make_noise,
    open_camera,
    parse_camera,
    parse_state_values,
    prepare_batch,
    read_camera_frame,
    run_action_chunk,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark PI0.5 PyTorch vs TensorRT suffix FP16 path.")
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
    parser.add_argument("--state-values", type=parse_state_values, default="0,0,0,0,0,0")
    parser.add_argument("--robot-port", default=None, help="Accepted for CLI compatibility; never opened.")
    parser.add_argument("--robot-type", default="so101_follower")
    parser.add_argument("--noise-seed", type=int, default=2026)
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("my_devs/openpi_trt/artifacts/benchmark_camera_torch_trt_suffix_fp16.json"),
    )
    return parser


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def summarize_ms(values: list[float]) -> dict:
    return {
        "mean_ms": float(statistics.mean(values)),
        "median_ms": float(statistics.median(values)),
        "min_ms": float(min(values)),
        "max_ms": float(max(values)),
        "std_ms": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "runs_ms": values,
    }


@torch.no_grad()
def time_call(fn, warmup_runs: int, runs: int) -> tuple[object, list[float]]:
    result = None
    for _ in range(warmup_runs):
        result = fn()
    synchronize()

    times = []
    for _ in range(runs):
        synchronize()
        start = time.perf_counter()
        result = fn()
        synchronize()
        times.append((time.perf_counter() - start) * 1000)
    return result, times


def main() -> None:
    configure_runtime()
    args = build_parser().parse_args()

    policy_path = args.policy_path.expanduser().resolve()
    engine_path = args.engine_path.expanduser()
    if not policy_path.is_dir():
        raise FileNotFoundError(f"Policy path not found: {policy_path}")
    if not engine_path.is_file():
        raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")
    if args.runs <= 0:
        raise ValueError("--runs must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    ensure_local_tokenizer_dir()
    print("[SAFETY] Benchmark only reads cameras and runs model forward passes. It never controls the robot.")
    if args.robot_port:
        print(f"[SAFETY] Ignoring --robot-port={args.robot_port}; accepted only for CLI compatibility.")

    print(f"[INFO] Loading policy: {policy_path}")
    policy = load_policy(policy_path, device="cuda")

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
    timestep = torch.ones(noise.shape[0], dtype=torch.float32, device=noise.device)

    print(f"[INFO] Benchmarking PyTorch full sample_actions: warmup={args.warmup_runs}, runs={args.runs}")
    policy.reset()
    torch_actions, torch_full_times = time_call(
        lambda: run_action_chunk(policy, batch, noise),
        warmup_runs=args.warmup_runs,
        runs=args.runs,
    )

    print(f"[INFO] Benchmarking PyTorch embed_suffix only: warmup={args.warmup_runs}, runs={args.runs}")
    torch_suffix_result, torch_suffix_times = time_call(
        lambda: policy.model.embed_suffix(noise, timestep),
        warmup_runs=args.warmup_runs,
        runs=args.runs,
    )

    print(f"[INFO] Patching embed_suffix with TensorRT FP16 engine: {engine_path}")
    engine = patch_embed_suffix_with_trt(policy, engine_path)

    print(f"[INFO] Benchmarking TensorRT-backed full sample_actions: warmup={args.warmup_runs}, runs={args.runs}")
    policy.reset()
    trt_actions, trt_full_times = time_call(
        lambda: run_action_chunk(policy, batch, noise),
        warmup_runs=args.warmup_runs,
        runs=args.runs,
    )

    print(f"[INFO] Benchmarking TensorRT embed_suffix only: warmup={args.warmup_runs}, runs={args.runs}")
    trt_suffix_result, trt_suffix_times = time_call(
        lambda: policy.model.embed_suffix(noise, timestep),
        warmup_runs=args.warmup_runs,
        runs=args.runs,
    )

    torch_np = torch_actions.detach().cpu().numpy()
    trt_np = trt_actions.detach().cpu().numpy()
    action_stats = tensor_stats(torch_np, trt_np)

    torch_full = summarize_ms(torch_full_times)
    trt_full = summarize_ms(trt_full_times)
    torch_suffix = summarize_ms(torch_suffix_times)
    trt_suffix = summarize_ms(trt_suffix_times)

    report = {
        "mode": "camera_benchmark_sample_actions_with_trt_suffix_embedding",
        "safety": {
            "robot_connected": False,
            "robot_action_sent": False,
            "robot_port_ignored": args.robot_port,
        },
        "policy_path": str(policy_path),
        "engine_path": str(engine_path),
        "task": args.task,
        "state_values": args.state_values.astype(float).tolist(),
        "top_cam": str(args.top_cam),
        "wrist_cam": str(args.wrist_cam),
        "top_frame_shape": list(top_rgb.shape),
        "wrist_frame_shape": list(wrist_rgb.shape),
        "warmup_runs": args.warmup_runs,
        "runs": args.runs,
        "torch_full_sample_actions": torch_full,
        "trt_suffix_full_sample_actions": trt_full,
        "torch_embed_suffix_only": torch_suffix,
        "trt_embed_suffix_only": trt_suffix,
        "full_sample_actions_speedup": torch_full["mean_ms"] / trt_full["mean_ms"],
        "embed_suffix_speedup": torch_suffix["mean_ms"] / trt_suffix["mean_ms"],
        "action_chunk_stats": action_stats,
        "engine": engine.describe(),
        "noise_seed": args.noise_seed,
    }
    write_json(args.report, report)

    print("[RESULT] Full sample_actions:")
    print(f"[RESULT]   PyTorch mean: {torch_full['mean_ms']:.2f} ms")
    print(f"[RESULT]   TRT-suffix mean: {trt_full['mean_ms']:.2f} ms")
    print(f"[RESULT]   speedup: {report['full_sample_actions_speedup']:.4f}x")
    print("[RESULT] embed_suffix only:")
    print(f"[RESULT]   PyTorch mean: {torch_suffix['mean_ms']:.4f} ms")
    print(f"[RESULT]   TRT mean: {trt_suffix['mean_ms']:.4f} ms")
    print(f"[RESULT]   speedup: {report['embed_suffix_speedup']:.4f}x")
    print("[RESULT] Action consistency:")
    print(f"[RESULT]   max_abs_diff: {action_stats['max_abs_diff']:.8f}")
    print(f"[RESULT]   cosine_similarity: {action_stats['cosine_similarity']:.8f}")
    print(f"[RESULT] Report written: {args.report}")


if __name__ == "__main__":
    main()

