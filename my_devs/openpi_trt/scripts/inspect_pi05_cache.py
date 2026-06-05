#!/usr/bin/env python

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
OPENPI_TRT_DIR = SCRIPT_DIR.parent
if OPENPI_TRT_DIR.as_posix() not in sys.path:
    sys.path.insert(0, OPENPI_TRT_DIR.as_posix())

from scripts.pi05_onnx_common import (  # noqa: E402
    DEFAULT_POLICY_PATH,
    DEFAULT_TASK,
    configure_runtime,
    load_policy,
    make_prefix_embedding_inputs,
    make_policy_batch,
    write_json,
)


def describe_tensor(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "type": type(tensor).__name__,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "requires_grad": bool(tensor.requires_grad),
        "contiguous": bool(tensor.is_contiguous()),
    }


def describe_object(obj: Any, max_depth: int = 5) -> Any:
    if isinstance(obj, torch.Tensor):
        return describe_tensor(obj)
    if max_depth <= 0:
        return {"type": type(obj).__name__, "repr": repr(obj)[:300]}
    if obj is None:
        return None
    if isinstance(obj, (list, tuple)):
        return {
            "type": type(obj).__name__,
            "length": len(obj),
            "items": [describe_object(item, max_depth=max_depth - 1) for item in obj],
        }
    if isinstance(obj, dict):
        return {
            "type": type(obj).__name__,
            "keys": list(obj.keys()),
            "items": {str(key): describe_object(value, max_depth=max_depth - 1) for key, value in obj.items()},
        }
    attrs = {}
    for name in ("key_cache", "value_cache", "seen_tokens", "_seen_tokens"):
        if hasattr(obj, name):
            attrs[name] = describe_object(getattr(obj, name), max_depth=max_depth - 1)
    if attrs:
        return {"type": type(obj).__name__, "attrs": attrs}
    return {"type": type(obj).__name__, "repr": repr(obj)[:300]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect LeRobot PI0.5 prefix cache and denoise tensors.")
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--model-dtype",
        choices=["checkpoint", "bfloat16", "float32"],
        default="checkpoint",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--noise-seed", type=int, default=2026)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("my_devs/openpi_trt/artifacts/inspect_pi05_cache.json"),
    )
    return parser


def main() -> None:
    configure_runtime()
    args = build_parser().parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this inspection command but torch.cuda.is_available() is false.")

    policy_path = args.policy_path.expanduser().resolve()
    model_dtype = None if args.model_dtype == "checkpoint" else args.model_dtype
    print(f"[INFO] Loading policy: {policy_path}")
    policy = load_policy(policy_path, device=args.device, model_dtype=model_dtype)
    model = policy.model
    batch = make_policy_batch(policy_path, task=args.task, seed=args.seed)

    inputs = make_prefix_embedding_inputs(policy, batch)
    image_0, image_1, img_mask_0, img_mask_1, tokens, masks = inputs
    images = [image_0, image_1]
    img_masks = [img_mask_0, img_mask_1]

    generator = torch.Generator(device=tokens.device)
    generator.manual_seed(args.noise_seed)
    x_t = torch.randn(
        tokens.shape[0],
        policy.config.chunk_size,
        policy.config.max_action_dim,
        dtype=torch.float32,
        device=tokens.device,
        generator=generator,
    )
    timestep = torch.full((tokens.shape[0],), 1.0, dtype=torch.float32, device=tokens.device)

    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

    with torch.no_grad():
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, tokens, masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)
        model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
        _, past_key_values = model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = model.embed_suffix(x_t, timestep)
        v_t = model.denoise_step(
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            x_t=x_t,
            timestep=timestep,
        )

    report = {
        "policy_path": str(policy_path),
        "model_dtype": str(next(model.parameters()).dtype),
        "config": {
            "chunk_size": policy.config.chunk_size,
            "max_action_dim": policy.config.max_action_dim,
            "num_inference_steps": policy.config.num_inference_steps,
            "tokenizer_max_length": policy.config.tokenizer_max_length,
        },
        "inputs": {
            "image_0": describe_tensor(image_0),
            "image_1": describe_tensor(image_1),
            "img_mask_0": describe_tensor(img_mask_0),
            "img_mask_1": describe_tensor(img_mask_1),
            "tokens": describe_tensor(tokens),
            "masks": describe_tensor(masks),
            "x_t": describe_tensor(x_t),
            "timestep": describe_tensor(timestep),
        },
        "prefix": {
            "prefix_embs": describe_tensor(prefix_embs),
            "prefix_pad_masks": describe_tensor(prefix_pad_masks),
            "prefix_att_masks": describe_tensor(prefix_att_masks),
            "prefix_att_2d_masks": describe_tensor(prefix_att_2d_masks),
            "prefix_position_ids": describe_tensor(prefix_position_ids),
            "prefix_att_2d_masks_4d": describe_tensor(prefix_att_2d_masks_4d),
        },
        "past_key_values": describe_object(past_key_values),
        "suffix": {
            "suffix_embs": describe_tensor(suffix_embs),
            "suffix_pad_masks": describe_tensor(suffix_pad_masks),
            "suffix_att_masks": describe_tensor(suffix_att_masks),
            "adarms_cond": describe_tensor(adarms_cond),
        },
        "denoise": {
            "v_t": describe_tensor(v_t),
        },
    }
    write_json(args.report.expanduser(), report)
    print(f"[INFO] Report written: {args.report}")


if __name__ == "__main__":
    main()
