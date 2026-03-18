#!/usr/bin/env python

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def _find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "lerobot").is_dir():
            return candidate
    raise RuntimeError(f"Failed to locate repo root from: {start}")


REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent)
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from my_devs.pi_trt.common import (
    DEFAULT_POLICY_PATH,
    DenoiseStepOnnxWrapper,
    PrefixCacheOnnxWrapper,
    VisionEncoderOnnxWrapper,
    build_prefix_from_image_embeddings,
    load_policy,
    load_preprocessor,
    prepare_batch,
    prepare_policy_for_fp16,
    preprocess_observation,
    resolve_policy_dir,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export PI0.5 FP16 ONNX subgraphs.")
    parser.add_argument("--policy-path", default=DEFAULT_POLICY_PATH)
    parser.add_argument("--onnx-out-dir", required=True)
    parser.add_argument("--opset", type=int, default=19)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--task", default="Put the block in the bin")
    return parser


@torch.inference_mode()
def main() -> None:
    args = build_parser().parse_args()
    policy_path = resolve_policy_dir(args.policy_path)
    output_dir = Path(args.onnx_out_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    _, policy = load_policy(policy_path, device="cuda")
    prepare_policy_for_fp16(policy)
    preprocessor = load_preprocessor(policy_path)
    batch = preprocess_observation(policy, preprocessor, seed=args.seed, task=args.task)
    prepared = prepare_batch(policy, batch)

    image_embeddings = policy.model.paligemma_with_expert.embed_image(prepared.image_tensor)
    prefix_embs, prefix_pad_masks, _prefix_att_masks, prefix_attention_mask_4d, prefix_position_ids = (
        build_prefix_from_image_embeddings(
            policy,
            image_embeddings,
            prepared.image_masks,
            prepared.tokens,
            prepared.masks,
        )
    )
    kv_cache = PrefixCacheOnnxWrapper(policy)(prefix_embs, prefix_attention_mask_4d, prefix_position_ids)
    x_t = torch.randn(
        (1, int(policy.config.chunk_size), int(policy.config.max_action_dim)),
        dtype=torch.float16,
        device="cuda",
    )
    timestep = torch.full((1,), 1.0, dtype=torch.float32, device="cuda")

    vision_path = output_dir / "vision_encoder_fp16.onnx"
    prefix_path = output_dir / "prefix_cache_fp16.onnx"
    denoise_path = output_dir / "denoise_step_fp16.onnx"

    torch.onnx.export(
        VisionEncoderOnnxWrapper(policy),
        (prepared.image_tensor,),
        str(vision_path),
        export_params=True,
        do_constant_folding=True,
        opset_version=args.opset,
        input_names=["image_tensor"],
        output_names=["image_embeddings"],
    )

    torch.onnx.export(
        PrefixCacheOnnxWrapper(policy),
        (prefix_embs, prefix_attention_mask_4d, prefix_position_ids),
        str(prefix_path),
        export_params=True,
        do_constant_folding=True,
        opset_version=args.opset,
        input_names=["prefix_embs", "prefix_attention_mask_4d", "prefix_position_ids"],
        output_names=["kv_cache"],
    )

    torch.onnx.export(
        DenoiseStepOnnxWrapper(policy),
        (x_t, timestep, prefix_pad_masks, kv_cache),
        str(denoise_path),
        export_params=True,
        do_constant_folding=True,
        opset_version=args.opset,
        input_names=["x_t", "timestep", "prefix_pad_masks", "kv_cache"],
        output_names=["velocity"],
    )

    print(f"[OK] Exported PI0.5 ONNX to: {output_dir}")
    print(f"[OK] Vision ONNX: {vision_path}")
    print(f"[OK] Prefix cache ONNX: {prefix_path}")
    print(f"[OK] Denoise step ONNX: {denoise_path}")


if __name__ == "__main__":
    main()
