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

from runtime.trt_engine import TorchTensorRTEngine  # noqa: E402
from scripts.pi05_onnx_common import (  # noqa: E402
    DEFAULT_POLICY_PATH,
    DEFAULT_TASK,
    configure_runtime,
    denoise_step_input_names,
    load_policy,
    make_denoise_step_inputs,
    make_prefix_cache_inputs,
    make_prefix_embedding_inputs,
    make_policy_batch,
    make_suffix_embedding_inputs,
    prefix_cache_tensor_names,
    run_torch_denoise_step,
    run_torch_prefix_cache,
    run_torch_prefix_embedding,
    run_torch_suffix_embedding,
    tensor_stats,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify LeRobot PI0.5 Torch vs TensorRT numerics.")
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--engine-path", type=Path, required=True)
    parser.add_argument("--precision", choices=["fp32", "fp16"], required=True)
    parser.add_argument(
        "--mode",
        choices=["suffix_embedding", "prefix_embedding", "prefix_cache", "denoise_step"],
        default="suffix_embedding",
    )
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument(
        "--model-dtype",
        choices=["checkpoint", "bfloat16", "float32"],
        default="checkpoint",
        help="Override PI0.5 config dtype for verification.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--noise-seed", type=int, default=2026)
    parser.add_argument("--rtol", type=float, default=None)
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument("--report", type=Path, default=None)
    return parser


def main() -> None:
    configure_runtime()
    args = build_parser().parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TensorRT verification.")

    default_atol = 2e-3 if args.precision == "fp32" else 2e-2
    default_rtol = 1e-4 if args.precision == "fp32" else 2e-2
    atol = default_atol if args.atol is None else args.atol
    rtol = default_rtol if args.rtol is None else args.rtol

    policy_path = args.policy_path.expanduser().resolve()
    engine_path = args.engine_path.expanduser()
    if not engine_path.is_file():
        raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")

    print(f"[INFO] Loading policy: {policy_path}")
    model_dtype = None if args.model_dtype == "checkpoint" else args.model_dtype
    policy = load_policy(policy_path, device="cuda", model_dtype=model_dtype)
    batch = make_policy_batch(policy_path, task=args.task, seed=args.seed)

    print("[INFO] Running Torch baseline...")
    if args.mode == "suffix_embedding":
        inputs = make_suffix_embedding_inputs(policy, batch, seed=args.noise_seed)
        output_names = ["suffix_embs", "adarms_cond"]
        torch_outputs = run_torch_suffix_embedding(policy, inputs)
        trt_inputs = {
            "noisy_actions": inputs[0],
            "timestep": inputs[1],
        }
    elif args.mode == "prefix_embedding":
        inputs = make_prefix_embedding_inputs(policy, batch)
        output_names = ["prefix_embs", "prefix_pad_masks", "prefix_att_masks"]
        torch_outputs = run_torch_prefix_embedding(policy, inputs)
        trt_inputs = {
            "image_0": inputs[0],
            "image_1": inputs[1],
            "img_mask_0": inputs[2],
            "img_mask_1": inputs[3],
            "tokens": inputs[4],
            "masks": inputs[5],
        }
    elif args.mode == "prefix_cache":
        inputs = make_prefix_cache_inputs(policy, batch)
        output_names = prefix_cache_tensor_names()
        torch_outputs = run_torch_prefix_cache(policy, inputs)
        trt_inputs = {
            "image_0": inputs[0],
            "image_1": inputs[1],
            "img_mask_0": inputs[2],
            "img_mask_1": inputs[3],
            "tokens": inputs[4],
            "masks": inputs[5],
        }
    else:
        inputs = make_denoise_step_inputs(policy, batch, seed=args.noise_seed)
        output_names = ["v_t"]
        torch_outputs = (run_torch_denoise_step(policy, inputs),)
        trt_inputs = {
            name: tensor
            for name, tensor in zip(denoise_step_input_names(), inputs, strict=True)
        }

    print(f"[INFO] Loading TensorRT engine: {engine_path}")
    engine = TorchTensorRTEngine(engine_path)

    for name in engine.input_names:
        expected_dtype = engine.tensor_dtypes[name]
        if trt_inputs[name].dtype != expected_dtype:
            trt_inputs[name] = trt_inputs[name].to(expected_dtype)

    print("[INFO] Running TensorRT...")
    trt_outputs = engine(**trt_inputs)

    torch_arrays = {name: tensor.detach().cpu().numpy() for name, tensor in zip(output_names, torch_outputs, strict=True)}
    trt_arrays = {name: trt_outputs[name].detach().cpu().numpy() for name in output_names}

    output_stats = {
        name: tensor_stats(torch_arrays[name], trt_arrays[name])
        for name in output_names
    }
    passed_items = []
    for name in output_names:
        if torch_arrays[name].dtype == np.bool_ or trt_arrays[name].dtype == np.bool_:
            passed_items.append(bool(np.array_equal(torch_arrays[name], trt_arrays[name])))
        else:
            passed_items.append(bool(np.allclose(torch_arrays[name], trt_arrays[name], rtol=rtol, atol=atol)))
    passed = all(passed_items)

    numeric_stats = [item for item in output_stats.values() if "max_abs_diff" in item]

    stats = {
        "mode": args.mode,
        "precision": args.precision,
        "engine": engine.describe(),
        "outputs": output_stats,
        "mean_abs_diff_worst_output": max((item["mean_abs_diff"] for item in numeric_stats), default=0.0),
        "max_abs_diff_worst_output": max((item["max_abs_diff"] for item in numeric_stats), default=0.0),
        "cosine_similarity_worst_output": min((item["cosine_similarity"] for item in numeric_stats), default=1.0),
        "passed_allclose": passed,
        "rtol": rtol,
        "atol": atol,
        "policy_path": str(policy_path),
        "engine_path": str(engine_path),
        "seed": args.seed,
        "noise_seed": args.noise_seed,
    }

    report = args.report
    if report is None:
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        report = Path(f"my_devs/openpi_trt/artifacts/verify_torch_trt_{args.precision}_{timestamp}.json")
    write_json(report, stats)

    print("[INFO] Torch vs TensorRT statistics:")
    for name, item in output_stats.items():
        print(f"[INFO]   {name}: torch={item['reference_shape']} trt={item['candidate_shape']}")
        if "max_abs_diff" in item:
            print(f"[INFO]     mean_abs_diff={item['mean_abs_diff']:.8f}")
            print(f"[INFO]     max_abs_diff={item['max_abs_diff']:.8f}")
            print(f"[INFO]     cosine_similarity={item['cosine_similarity']:.8f}")
        else:
            print(f"[INFO]     equal={item['equal']} mismatch_count={item['mismatch_count']}")
    print(f"[INFO]   allclose(rtol={rtol}, atol={atol})={passed}")
    print(f"[INFO] Report written: {report}")

    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
