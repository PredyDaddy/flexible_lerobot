#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gr00t.model.gr00t_n1d7.setup  # noqa: F401
import torch
from safetensors import safe_open
from transformers import AutoModel

SAMPLE_WEIGHT_KEYS = (
    "backbone.model.model.visual.patch_embed.proj.weight",
    "backbone.model.model.language_model.layers.0.self_attn.q_proj.weight",
    "backbone.model.model.language_model.embed_tokens.weight",
    "action_head.action_decoder.layer1.W",
)


def ensure_within(path: Path, root: Path, *, must_exist: bool = False) -> Path:
    resolved = path.expanduser().resolve(strict=must_exist)
    allowed = root.expanduser().resolve(strict=True)
    try:
        resolved.relative_to(allowed)
    except ValueError as exc:
        raise ValueError(f"Path escapes allowed root: {resolved}") from exc
    return resolved


def tensor_from_checkpoint(model_dir: Path, weight_map: dict[str, str], key: str) -> torch.Tensor:
    shard_path = model_dir / weight_map[key]
    with safe_open(shard_path, framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def main() -> None:
    parser = argparse.ArgumentParser(description="Strictly validate local N1.7 model loading.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--backbone-assets", type=Path, required=True)
    parser.add_argument("--allowed-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    allowed_root = args.allowed_root.expanduser().resolve(strict=True)
    model_dir = ensure_within(args.model_dir, allowed_root, must_exist=True)
    backbone_assets = ensure_within(args.backbone_assets, allowed_root, must_exist=True)
    report_path = ensure_within(args.report, allowed_root)
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite model validation report: {report_path}")

    start = time.monotonic()
    model, loading_info = AutoModel.from_pretrained(
        model_dir,
        model_name=str(backbone_assets),
        backbone_init_from_config=True,
        tune_llm=False,
        tune_visual=False,
        tune_projector=True,
        tune_diffusion_model=True,
        tune_vlln=True,
        state_dropout_prob=0.2,
        backbone_trainable_params_fp32=True,
        load_bf16=False,
        transformers_loading_kwargs={
            "trust_remote_code": True,
            "local_files_only": True,
        },
        local_files_only=True,
        trust_remote_code=True,
        output_loading_info=True,
    )
    elapsed_seconds = time.monotonic() - start

    loading_errors = {
        name: loading_info.get(name, [])
        for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    }
    nonempty_loading_errors = {name: value for name, value in loading_errors.items() if value}
    if nonempty_loading_errors:
        raise RuntimeError(f"Strict checkpoint loading failed: {nonempty_loading_errors}")
    attention_implementation = model.backbone.model.config._attn_implementation
    if attention_implementation != "flash_attention_2":
        raise RuntimeError(f"Model did not select FlashAttention 2: {attention_implementation}")

    index = json.loads((model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    model_state = model.state_dict()
    tensor_checks = []
    for key in SAMPLE_WEIGHT_KEYS:
        if key not in weight_map:
            raise KeyError(f"Sample tensor is absent from checkpoint index: {key}")
        if key not in model_state:
            raise KeyError(f"Sample tensor is absent from loaded model state: {key}")
        expected = tensor_from_checkpoint(model_dir, weight_map, key)
        actual = model_state[key].detach().cpu()
        exact = torch.equal(actual, expected)
        tensor_checks.append(
            {
                "key": key,
                "shard": weight_map[key],
                "shape": list(actual.shape),
                "dtype": str(actual.dtype),
                "exactly_equal": exact,
            }
        )
        if not exact:
            raise RuntimeError(f"Loaded tensor differs from safetensors checkpoint: {key}")

    dtype_counts: dict[str, int] = {}
    for parameter in model.parameters():
        dtype_counts[str(parameter.dtype)] = dtype_counts.get(str(parameter.dtype), 0) + parameter.numel()

    report = {
        "schema_version": 1,
        "status": "passed",
        "model_dir": str(model_dir),
        "backbone_assets": str(backbone_assets),
        "backbone_init_from_config": model.config.backbone_init_from_config,
        "strict_loading": loading_errors,
        "tensor_checks": tensor_checks,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "dtype_parameter_counts": dtype_counts,
        "language_model_layers": len(model.backbone.model.model.language_model.layers),
        "attention_implementation": attention_implementation,
        "model_name": model.config.model_name,
        "elapsed_seconds": elapsed_seconds,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
