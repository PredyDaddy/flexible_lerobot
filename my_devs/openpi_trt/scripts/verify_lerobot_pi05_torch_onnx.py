#!/usr/bin/env python

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import torch

from pi05_onnx_common import (
    DEFAULT_POLICY_PATH,
    DEFAULT_TASK,
    configure_runtime,
    denoise_step_input_names,
    load_policy,
    make_denoise_step_inputs,
    make_export_inputs,
    make_prefix_cache_inputs,
    make_policy_batch,
    make_prefix_embedding_inputs,
    make_suffix_embedding_inputs,
    prefix_cache_tensor_names,
    run_torch_denoise_step,
    run_torch_prefix_cache,
    run_torch_prefix_embedding,
    run_torch_wrapper,
    run_torch_suffix_embedding,
    tensor_stats,
    write_json,
)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.float()
    return tensor.detach().cpu().numpy()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify LeRobot PI0.5 Torch vs ONNX numerics.")
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument(
        "--onnx-path",
        type=Path,
        default=Path("my_devs/openpi_trt/artifacts/pi05_so101_sample_actions_b1.onnx"),
    )
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--model-dtype",
        choices=["checkpoint", "bfloat16", "float32"],
        default="checkpoint",
        help="Override PI0.5 config dtype for verification.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--noise-seed", type=int, default=2026)
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument(
        "--mode",
        choices=["sample_actions", "suffix_embedding", "prefix_embedding", "prefix_cache", "denoise_step"],
        default="sample_actions",
        help="ONNX boundary to verify.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="JSON report path. Defaults to my_devs/openpi_trt/artifacts/verify_torch_onnx_<timestamp>.json",
    )
    return parser


def run_onnx(onnx_path: Path, inputs: tuple[torch.Tensor, ...], mode: str):
    import onnxruntime as ort

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    available = ort.get_available_providers()
    providers = [provider for provider in providers if provider in available]
    session_options = ort.SessionOptions()
    if mode in {"prefix_embedding", "prefix_cache", "denoise_step"}:
        # ORT may fuse BF16 prefix graphs into SkipLayerNormalization, which is
        # invalid for BF16 in the available runtime. Keeping transformer split
        # graphs as-is also makes numeric debugging easier.
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(onnx_path), sess_options=session_options, providers=providers)
    if mode == "sample_actions":
        input_names = ["image_0", "image_1", "img_mask_0", "img_mask_1", "tokens", "masks", "noise"]
        output_names = ["actions"]
    elif mode == "suffix_embedding":
        input_names = ["noisy_actions", "timestep"]
        output_names = ["suffix_embs", "adarms_cond"]
    elif mode == "prefix_embedding":
        input_names = ["image_0", "image_1", "img_mask_0", "img_mask_1", "tokens", "masks"]
        output_names = ["prefix_embs", "prefix_pad_masks", "prefix_att_masks"]
    elif mode == "prefix_cache":
        input_names = ["image_0", "image_1", "img_mask_0", "img_mask_1", "tokens", "masks"]
        output_names = prefix_cache_tensor_names()
    else:
        input_names = denoise_step_input_names()
        output_names = ["v_t"]
    feed = {name: tensor.detach().cpu().numpy() for name, tensor in zip(input_names, inputs, strict=True)}
    return session.run(output_names, feed)


def main() -> None:
    configure_runtime()
    args = build_parser().parse_args()

    policy_path = args.policy_path.expanduser().resolve()
    onnx_path = args.onnx_path.expanduser()
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this verification command but torch.cuda.is_available() is false.")

    print(f"[INFO] Loading policy: {policy_path}")
    model_dtype = None if args.model_dtype == "checkpoint" else args.model_dtype
    policy = load_policy(policy_path, device=args.device, model_dtype=model_dtype)
    batch = make_policy_batch(policy_path, task=args.task, seed=args.seed)
    if args.mode == "sample_actions":
        inputs = make_export_inputs(policy, batch, seed=args.noise_seed)
    elif args.mode == "suffix_embedding":
        inputs = make_suffix_embedding_inputs(policy, batch, seed=args.noise_seed)
    elif args.mode == "prefix_embedding":
        inputs = make_prefix_embedding_inputs(policy, batch)
    elif args.mode == "prefix_cache":
        inputs = make_prefix_cache_inputs(policy, batch)
    else:
        inputs = make_denoise_step_inputs(policy, batch, seed=args.noise_seed)

    print("[INFO] Running Torch baseline...")
    with torch.no_grad():
        if args.mode == "sample_actions":
            torch_outputs = [run_torch_wrapper(policy, inputs)]
            output_names = ["actions"]
        elif args.mode == "suffix_embedding":
            torch_outputs = list(run_torch_suffix_embedding(policy, inputs))
            output_names = ["suffix_embs", "adarms_cond"]
        elif args.mode == "prefix_embedding":
            torch_outputs = list(run_torch_prefix_embedding(policy, inputs))
            output_names = ["prefix_embs", "prefix_pad_masks", "prefix_att_masks"]
        elif args.mode == "prefix_cache":
            torch_outputs = list(run_torch_prefix_cache(policy, inputs))
            output_names = prefix_cache_tensor_names()
        else:
            torch_outputs = [run_torch_denoise_step(policy, inputs)]
            output_names = ["v_t"]
    torch_arrays = [tensor_to_numpy(out) for out in torch_outputs]

    print("[INFO] Running ONNX Runtime...")
    onnx_arrays = run_onnx(onnx_path, inputs, args.mode)

    output_stats = {
        name: tensor_stats(torch_np, onnx_np)
        for name, torch_np, onnx_np in zip(output_names, torch_arrays, onnx_arrays, strict=True)
    }
    passed_items = []
    for torch_np, onnx_np in zip(torch_arrays, onnx_arrays, strict=True):
        if torch_np.dtype == np.bool_ or onnx_np.dtype == np.bool_:
            passed_items.append(bool(np.array_equal(torch_np, onnx_np)))
        else:
            passed_items.append(bool(np.allclose(torch_np, onnx_np, rtol=args.rtol, atol=args.atol)))
    passed = all(passed_items)
    numeric_stats = [item for item in output_stats.values() if "max_abs_diff" in item]
    max_abs_diff = max((item["max_abs_diff"] for item in numeric_stats), default=0.0)
    mean_abs_diff = max((item["mean_abs_diff"] for item in numeric_stats), default=0.0)
    min_cosine = min((item["cosine_similarity"] for item in numeric_stats), default=1.0)
    stats = {
        "mode": args.mode,
        "outputs": output_stats,
        "mean_abs_diff_worst_output": mean_abs_diff,
        "max_abs_diff_worst_output": max_abs_diff,
        "cosine_similarity_worst_output": min_cosine,
    }
    stats.update(
        {
            "passed_allclose": passed,
            "rtol": args.rtol,
            "atol": args.atol,
            "policy_path": str(policy_path),
            "onnx_path": str(onnx_path),
            "seed": args.seed,
            "noise_seed": args.noise_seed,
        }
    )

    report = args.report
    if report is None:
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        report = Path(f"my_devs/openpi_trt/artifacts/verify_torch_onnx_{timestamp}.json")
    write_json(report, stats)

    print("[INFO] Torch vs ONNX statistics:")
    for name, item in output_stats.items():
        print(f"[INFO]   {name}: torch={item['reference_shape']} onnx={item['candidate_shape']}")
        if "max_abs_diff" in item:
            print(f"[INFO]     mean_abs_diff={item['mean_abs_diff']:.8f}")
            print(f"[INFO]     max_abs_diff={item['max_abs_diff']:.8f}")
            print(f"[INFO]     cosine_similarity={item['cosine_similarity']:.8f}")
        else:
            print(f"[INFO]     equal={item['equal']} mismatch_count={item['mismatch_count']}")
    print(f"[INFO]   allclose(rtol={args.rtol}, atol={args.atol})={passed}")
    print(f"[INFO] Report written: {report}")

    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
