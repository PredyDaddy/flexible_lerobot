#!/usr/bin/env python

"""CLI entrypoint for the GR00T TensorRT async robot client."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "lerobot").is_dir():
            return candidate
    raise RuntimeError(f"Failed to locate repo root from: {start}")


REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent)
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from my_devs.groot_trt_async_server.robot_client import (
    DEFAULT_ENGINE_DIR,
    DEFAULT_POLICY_PATH,
    DEFAULT_SERVER_RESOURCE_PROFILE,
    build_so_follower_client_config,
    default_calibration_dir,
    run_async_robot_client,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the GR00T async robot client against a remote policy server.")
    parser.add_argument("--robot-id", default="my_so101")
    parser.add_argument("--robot-type", default="so101_follower", choices=["so100_follower", "so101_follower"])
    parser.add_argument(
        "--calib-dir",
        default=None,
        help="Calibration directory. Defaults to ~/.cache/huggingface/lerobot/calibration/robots/<robot-type>.",
    )
    parser.add_argument("--robot-port", default="/dev/ttyACM0")
    parser.add_argument("--top-cam-index", type=int, default=4)
    parser.add_argument("--wrist-cam-index", type=int, default=6)
    parser.add_argument("--img-width", type=int, default=640)
    parser.add_argument("--img-height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--task", default="Put the block in the bin")

    parser.add_argument("--server-address", default="127.0.0.1:8080")
    parser.add_argument(
        "--resource-profile",
        default=DEFAULT_SERVER_RESOURCE_PROFILE,
        help="Logical server-managed resource profile to request. The server resolves local paths.",
    )
    parser.add_argument(
        "--policy-path",
        default=DEFAULT_POLICY_PATH,
        help="Deprecated client-side path. Remote loading now resolves server-managed resources.",
    )
    parser.add_argument(
        "--actions-per-chunk",
        type=int,
        default=50,
        help="How many actions to request per chunk. GR00T checkpoints here were trained with chunk_size=50.",
    )
    parser.add_argument("--policy-device", default="cuda")
    parser.add_argument("--client-device", default="cpu")
    parser.add_argument(
        "--chunk-size-threshold",
        type=float,
        default=0.7,
        help="Refresh the queue once remaining queued actions fall below this fraction of the chunk.",
    )
    parser.add_argument(
        "--aggregate-fn-name",
        default="latest_only",
        choices=["weighted_average", "latest_only", "average", "conservative"],
    )
    parser.add_argument("--debug-visualize-queue-size", action="store_true")

    parser.add_argument("--backend", default="tensorrt", choices=["tensorrt", "pytorch"])
    parser.add_argument(
        "--session-id",
        default=None,
        help="Optional sticky session_id to reuse across client restarts.",
    )
    parser.add_argument(
        "--session-mode",
        default="claim",
        choices=["claim", "takeover"],
        help="How Ready() should acquire the sticky session on startup.",
    )
    parser.add_argument(
        "--engine-dir",
        default=DEFAULT_ENGINE_DIR,
        help="Deprecated client-side path. Remote loading now resolves server-managed resources.",
    )
    parser.add_argument(
        "--tensorrt-py-dir",
        default=None,
        help="Deprecated client-side path. Remote loading now resolves server-managed resources.",
    )
    parser.add_argument("--vit-dtype", default="fp16", choices=["fp16", "fp8"])
    parser.add_argument("--llm-dtype", default="fp16", choices=["fp16", "fp8", "nvfp4", "nvfp4_full"])
    parser.add_argument("--dit-dtype", default="fp16", choices=["fp16", "fp8"])
    parser.add_argument("--num-denoising-steps", type=int, default=None)

    parser.add_argument("--run-time-s", type=float, default=0.0, help="<=0 means run until Ctrl+C.")
    parser.add_argument(
        "--mock-actions",
        action="store_true",
        help="Run the full async client loop but do not send actions to the robot.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose client-side timing logs.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    calib_dir = args.calib_dir or default_calibration_dir(args.robot_type)
    cfg = build_so_follower_client_config(
        robot_id=args.robot_id,
        robot_type=args.robot_type,
        calib_dir=calib_dir,
        robot_port=args.robot_port,
        top_cam_index=args.top_cam_index,
        wrist_cam_index=args.wrist_cam_index,
        img_width=args.img_width,
        img_height=args.img_height,
        fps=args.fps,
        task=args.task,
        server_address=args.server_address,
        policy_path=args.policy_path,
        resource_profile=args.resource_profile,
        actions_per_chunk=args.actions_per_chunk,
        policy_device=args.policy_device,
        client_device=args.client_device,
        chunk_size_threshold=args.chunk_size_threshold,
        aggregate_fn_name=args.aggregate_fn_name,
        debug_visualize_queue_size=args.debug_visualize_queue_size,
        backend=args.backend,
        session_id=args.session_id,
        session_mode=args.session_mode,
        engine_dir=args.engine_dir,
        tensorrt_py_dir=args.tensorrt_py_dir,
        vit_dtype=args.vit_dtype,
        llm_dtype=args.llm_dtype,
        dit_dtype=args.dit_dtype,
        num_denoising_steps=args.num_denoising_steps,
    )
    run_async_robot_client(
        cfg,
        mock_actions=args.mock_actions,
        run_time_s=args.run_time_s,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
