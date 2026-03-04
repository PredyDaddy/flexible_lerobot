#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from lerobot import policies  # noqa: F401  # Register policy config classes.
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class


DEFAULT_POLICY_PATH = (
    "/data/cqy_workspace/flexible_lerobot/outputs/train/"
    "groot_grasp_block_in_bin1_repro_20260302_223413/bs32_20260302_223447/"
    "checkpoints/last/pretrained_model"
)


def _extract_hidden_state(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    if isinstance(output, (tuple, list)) and output:
        first = output[0]
        if isinstance(first, torch.Tensor):
            return first
    raise TypeError(f"Unsupported vision output type: {type(output)!r}")


class VisionModelForOnnx(torch.nn.Module):
    def __init__(self, vision_model: torch.nn.Module) -> None:
        super().__init__()
        self.vision_model = vision_model

    def forward(self, pixel_values: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        hidden = _extract_hidden_state(self.vision_model(pixel_values))
        keep_input = position_ids.to(dtype=hidden.dtype).sum(dim=-1, keepdim=True).unsqueeze(-1) * 0.0
        return hidden + keep_input


class LanguageModelForOnnx(torch.nn.Module):
    def __init__(
        self,
        language_model: torch.nn.Module,
        eagle_linear: torch.nn.Module,
        select_layer: int,
    ) -> None:
        super().__init__()
        self.language_model = language_model
        self.eagle_linear = eagle_linear
        self.select_layer = int(select_layer)

    @staticmethod
    def _build_causal_mask(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        batch_size, seq_len = attention_mask.shape
        causal = torch.ones((1, 1, seq_len, seq_len), dtype=torch.bool, device=attention_mask.device).tril()
        padding = attention_mask[:, None, None, :].to(torch.bool)
        full = causal & padding
        keep = torch.zeros((), dtype=dtype, device=attention_mask.device)
        block = torch.full((), torch.finfo(dtype).min, dtype=dtype, device=attention_mask.device)
        return torch.where(full, keep, block).expand(batch_size, 1, seq_len, seq_len)

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        causal_mask = self._build_causal_mask(attention_mask, inputs_embeds.dtype)
        mask_mapping = {"full_attention": causal_mask, "sliding_attention": causal_mask}
        outputs = self.language_model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=mask_mapping,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden = outputs.hidden_states[self.select_layer]
        hidden = self.eagle_linear(hidden)
        return hidden


def _num_patches(backbone: torch.nn.Module) -> int:
    vision_model = backbone.eagle_model.vision_model
    if hasattr(vision_model, "vision_model") and hasattr(vision_model.vision_model, "embeddings"):
        return int(vision_model.vision_model.embeddings.num_patches)
    if hasattr(vision_model, "embeddings"):
        return int(vision_model.embeddings.num_patches)
    raise AttributeError("Failed to locate vision embeddings to infer num_patches")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export LeRobot GROOT backbone modules to ONNX.")
    parser.add_argument(
        "--policy-path",
        default=DEFAULT_POLICY_PATH,
        help="Path to LeRobot GROOT checkpoint folder (the `pretrained_model/` directory).",
    )
    parser.add_argument(
        "--onnx-out-dir",
        required=True,
        help="Output directory. Will create `eagle2/` subfolder inside it.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=296,
        help="Dummy sequence length for LLM ONNX tracing.",
    )
    parser.add_argument(
        "--video-views",
        type=int,
        default=1,
        help="Dummy number of camera views for ViT ONNX tracing.",
    )
    parser.add_argument(
        "--vit-dtype",
        default="fp16",
        choices=["fp16", "fp8"],
        help="Filename suffix for ViT ONNX export (vit_<dtype>.onnx).",
    )
    parser.add_argument(
        "--llm-dtype",
        default="fp16",
        choices=["fp16", "fp8", "nvfp4", "nvfp4_full"],
        help="Filename suffix for LLM ONNX export (llm_<dtype>.onnx).",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=19,
        help="ONNX opset version.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda"],
        help="Export device. TensorRT export is intended to run on CUDA.",
    )
    return parser


@torch.inference_mode()
def main() -> None:
    args = build_parser().parse_args()

    policy_path = Path(args.policy_path).expanduser()
    if not policy_path.is_dir():
        raise FileNotFoundError(f"--policy-path does not exist: {policy_path}")

    out_dir = Path(args.onnx_out_dir).expanduser()
    eagle2_dir = out_dir / "eagle2"
    eagle2_dir.mkdir(parents=True, exist_ok=True)

    cfg = PreTrainedConfig.from_pretrained(str(policy_path))
    cfg.pretrained_path = policy_path
    cfg.device = args.device

    policy_cls = get_policy_class(cfg.type)
    policy = policy_cls.from_pretrained(str(policy_path), config=cfg, strict=False)
    policy.eval()

    backbone = policy._groot_model.backbone
    backbone.eval()
    if hasattr(backbone.eagle_model.vision_model, "config"):
        backbone.eagle_model.vision_model.config._attn_implementation = "eager"
    if hasattr(backbone.eagle_model.language_model, "config"):
        backbone.eagle_model.language_model.config._attn_implementation = "eager"

    num_patches = _num_patches(backbone)
    hidden_size = int(backbone.eagle_model.language_model.config.hidden_size)

    vit_wrapper = VisionModelForOnnx(backbone.eagle_model.vision_model).to(device=args.device, dtype=torch.float16)
    pixel_values = torch.randn(
        (args.video_views, 3, 224, 224),
        dtype=torch.float16,
        device=args.device,
    )
    position_ids = torch.arange(num_patches, dtype=torch.int64, device=args.device).expand((args.video_views, -1))

    vit_path = eagle2_dir / f"vit_{args.vit_dtype}.onnx"
    torch.onnx.export(
        vit_wrapper,
        (pixel_values, position_ids),
        str(vit_path),
        export_params=True,
        do_constant_folding=True,
        opset_version=args.opset,
        input_names=["pixel_values", "position_ids"],
        output_names=["vit_embeds"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "position_ids": {0: "batch_size"},
            "vit_embeds": {0: "batch_size"},
        },
    )

    llm_wrapper = LanguageModelForOnnx(
        backbone.eagle_model.language_model,
        backbone.eagle_linear,
        backbone.select_layer,
    ).to(device=args.device, dtype=torch.float16)
    llm_inputs_embeds = torch.randn(
        (1, args.seq_len, hidden_size),
        dtype=torch.float16,
        device=args.device,
    )
    llm_attention_mask = torch.ones((1, args.seq_len), dtype=torch.int64, device=args.device)

    llm_path = eagle2_dir / f"llm_{args.llm_dtype}.onnx"
    torch.onnx.export(
        llm_wrapper,
        (llm_inputs_embeds, llm_attention_mask),
        str(llm_path),
        export_params=True,
        do_constant_folding=True,
        opset_version=args.opset,
        input_names=["inputs_embeds", "attention_mask"],
        output_names=["embeddings"],
        dynamic_axes={
            "inputs_embeds": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "embeddings": {0: "batch_size", 1: "sequence_length"},
        },
    )

    print(f"[OK] Exported backbone ONNX to: {eagle2_dir}")
    print(f"[OK] ViT ONNX: {vit_path}")
    print(f"[OK] LLM ONNX: {llm_path}")


if __name__ == "__main__":
    main()
