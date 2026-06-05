#!/usr/bin/env python

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
    load_policy,
    make_export_inputs,
    make_policy_batch,
    tensor_stats,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify PI0.5 Torch sample_actions vs split TensorRT runtime.")
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--prefix-engine-path", type=Path, required=True)
    parser.add_argument("--denoise-engine-path", type=Path, required=True)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument(
        "--model-dtype",
        choices=["checkpoint", "bfloat16", "float32"],
        default="float32",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--noise-seed", type=int, default=2026)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--atol", type=float, default=1e-1)
    parser.add_argument("--report", type=Path, default=None)
    return parser


def main() -> None:
    configure_runtime()
    args = build_parser().parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for split TensorRT sample_actions verification.")

    policy_path = args.policy_path.expanduser().resolve()
    model_dtype = None if args.model_dtype == "checkpoint" else args.model_dtype
    print(f"[INFO] Loading policy: {policy_path}")
    policy = load_policy(policy_path, device="cuda", model_dtype=model_dtype)
    batch = make_policy_batch(policy_path, task=args.task, seed=args.seed)
    inputs = make_export_inputs(policy, batch, seed=args.noise_seed)
    image_0, image_1, img_mask_0, img_mask_1, tokens, masks, noise = inputs
    images = [image_0, image_1]
    img_masks = [img_mask_0, img_mask_1]

    print("[INFO] Running Torch sample_actions baseline...")
    with torch.no_grad():
        torch_actions = policy.model.sample_actions(
            images,
            img_masks,
            tokens,
            masks,
            noise=noise,
            num_steps=args.num_steps,
        )

    print("[INFO] Loading split TensorRT runtime...")
    runtime = PI05TensorRTSplitRuntime(
        args.prefix_engine_path.expanduser(),
        args.denoise_engine_path.expanduser(),
    )
    print("[INFO] Running split TensorRT sample_actions...")
    trt_actions = runtime(
        policy,
        images,
        img_masks,
        tokens,
        masks,
        noise,
        num_steps=args.num_steps,
    )

    torch_np = torch_actions.detach().cpu().numpy()
    trt_np = trt_actions.detach().cpu().numpy()
    stats = tensor_stats(torch_np, trt_np)
    passed = bool(np.allclose(torch_np, trt_np, rtol=args.rtol, atol=args.atol))
    report = {
        "mode": "sample_actions_split_trt_prefix_cache_denoise_step",
        "passed_allclose": passed,
        "rtol": args.rtol,
        "atol": args.atol,
        "policy_path": str(policy_path),
        "prefix_engine_path": str(args.prefix_engine_path),
        "denoise_engine_path": str(args.denoise_engine_path),
        "seed": args.seed,
        "noise_seed": args.noise_seed,
        "num_steps": args.num_steps or policy.config.num_inference_steps,
        "action_stats": stats,
        "first_action_torch": torch_np[0, 0, : policy.config.max_action_dim].astype(float).tolist(),
        "first_action_trt": trt_np[0, 0, : policy.config.max_action_dim].astype(float).tolist(),
        "runtime": runtime.describe(),
    }

    output = args.report
    if output is None:
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        output = Path(f"my_devs/openpi_trt/artifacts/verify_split_trt_sample_actions_{timestamp}.json")
    write_json(output, report)

    print("[INFO] Torch vs split TensorRT sample_actions statistics:")
    print(f"[INFO]   actions: torch={stats['reference_shape']} trt={stats['candidate_shape']}")
    print(f"[INFO]     mean_abs_diff={stats['mean_abs_diff']:.8f}")
    print(f"[INFO]     max_abs_diff={stats['max_abs_diff']:.8f}")
    print(f"[INFO]     cosine_similarity={stats['cosine_similarity']:.8f}")
    print(f"[INFO]   allclose(rtol={args.rtol}, atol={args.atol})={passed}")
    print(f"[INFO] Report written: {output}")

    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
