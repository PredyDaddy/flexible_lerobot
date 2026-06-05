#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from pi05_onnx_common import (
    DEFAULT_POLICY_PATH,
    DEFAULT_TASK,
    PI05PrefixEmbeddingONNXWrapper,
    PI05PrefixCacheONNXWrapper,
    PI05SampleActionsONNXWrapper,
    PI05DenoiseStepONNXWrapper,
    PI05SuffixEmbeddingONNXWrapper,
    configure_runtime,
    denoise_step_input_names,
    load_policy,
    make_denoise_step_inputs,
    make_export_inputs,
    make_policy_batch,
    make_prefix_cache_inputs,
    make_prefix_embedding_inputs,
    make_suffix_embedding_inputs,
    patch_transformers_for_onnx_export,
    prefix_cache_tensor_names,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export LeRobot PI0.5 sample_actions to ONNX.")
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("my_devs/openpi_trt/artifacts/pi05_so101_sample_actions_b1.onnx"),
    )
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--noise-seed", type=int, default=2026)
    parser.add_argument("--opset", type=int, default=19)
    parser.add_argument("--dynamo", action="store_true", help="Use the torch.onnx dynamo exporter.")
    parser.add_argument(
        "--model-dtype",
        choices=["checkpoint", "bfloat16", "float32"],
        default="checkpoint",
        help="Override PI0.5 config dtype after loading config. Use float32 for ONNX runtimes that lack BF16 coverage.",
    )
    parser.add_argument(
        "--mode",
        choices=["sample_actions", "suffix_embedding", "prefix_embedding", "prefix_cache", "denoise_step"],
        default="sample_actions",
        help="ONNX boundary to export.",
    )
    return parser


def main() -> None:
    configure_runtime()
    patch_transformers_for_onnx_export()
    args = build_parser().parse_args()

    policy_path = args.policy_path.expanduser().resolve()
    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this export command but torch.cuda.is_available() is false.")

    print(f"[INFO] Loading policy: {policy_path}")
    model_dtype = None if args.model_dtype == "checkpoint" else args.model_dtype
    policy = load_policy(policy_path, device=args.device, model_dtype=model_dtype)
    batch = make_policy_batch(policy_path, task=args.task, seed=args.seed)
    if args.mode == "sample_actions":
        inputs = make_export_inputs(policy, batch, seed=args.noise_seed)
        wrapper = PI05SampleActionsONNXWrapper(policy).eval()
        input_names = ["image_0", "image_1", "img_mask_0", "img_mask_1", "tokens", "masks", "noise"]
        output_names = ["actions"]
    else:
        if args.mode == "suffix_embedding":
            inputs = make_suffix_embedding_inputs(policy, batch, seed=args.noise_seed)
            wrapper = PI05SuffixEmbeddingONNXWrapper(policy).eval()
            input_names = ["noisy_actions", "timestep"]
            output_names = ["suffix_embs", "adarms_cond"]
        elif args.mode == "prefix_embedding":
            inputs = make_prefix_embedding_inputs(policy, batch)
            wrapper = PI05PrefixEmbeddingONNXWrapper(policy).eval()
            input_names = ["image_0", "image_1", "img_mask_0", "img_mask_1", "tokens", "masks"]
            output_names = ["prefix_embs", "prefix_pad_masks", "prefix_att_masks"]
        elif args.mode == "prefix_cache":
            inputs = make_prefix_cache_inputs(policy, batch)
            wrapper = PI05PrefixCacheONNXWrapper(policy).eval()
            input_names = ["image_0", "image_1", "img_mask_0", "img_mask_1", "tokens", "masks"]
            output_names = prefix_cache_tensor_names()
        else:
            inputs = make_denoise_step_inputs(policy, batch, seed=args.noise_seed)
            wrapper = PI05DenoiseStepONNXWrapper(policy).eval()
            input_names = denoise_step_input_names()
            output_names = ["v_t"]

    print("[INFO] Export input shapes:")
    for name, tensor in zip(input_names, inputs, strict=True):
        print(f"[INFO]   {name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device}")

    print(f"[INFO] Exporting ONNX to: {output}")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            inputs,
            str(output),
            input_names=input_names,
            output_names=output_names,
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=args.dynamo,
        )
    print(f"[INFO] ONNX export completed: {output}")


if __name__ == "__main__":
    main()
