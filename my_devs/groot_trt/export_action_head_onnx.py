#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from lerobot import policies  # noqa: F401  # Register policy config classes.
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class


class VLLNVLSelfAttention(torch.nn.Module):
    def __init__(self, vlln: torch.nn.Module, vl_self_attention: torch.nn.Module) -> None:
        super().__init__()
        self.vlln = vlln
        self.vl_self_attention = vl_self_attention

    def forward(self, backbone_features: torch.Tensor) -> torch.Tensor:
        x = self.vlln(backbone_features)
        x = self.vl_self_attention(x)
        return x


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export LeRobot GROOT action head modules to ONNX.")
    parser.add_argument(
        "--policy-path",
        required=True,
        help="Path to LeRobot GROOT checkpoint folder (the `pretrained_model/` directory).",
    )
    parser.add_argument(
        "--onnx-out-dir",
        required=True,
        help="Output directory. Will create `action_head/` subfolder inside it.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=296,
        help="Dummy backbone sequence length used for ONNX tracing (dynamic axis is enabled).",
    )
    parser.add_argument(
        "--state-horizon",
        type=int,
        default=1,
        help="State horizon used by the action head (normally 1 for GROOT).",
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
    action_head_dir = out_dir / "action_head"
    action_head_dir.mkdir(parents=True, exist_ok=True)

    cfg = PreTrainedConfig.from_pretrained(str(policy_path))
    cfg.pretrained_path = policy_path
    cfg.device = args.device

    policy_cls = get_policy_class(cfg.type)
    policy = policy_cls.from_pretrained(str(policy_path), config=cfg, strict=False)
    policy.eval()

    groot_model = policy._groot_model
    action_head = groot_model.action_head.to(dtype=torch.float16)
    action_head.eval()

    # ----- vlln + vl_self_attention -----
    process_backbone_model = VLLNVLSelfAttention(action_head.vlln, action_head.vl_self_attention).to(
        dtype=torch.float16
    )
    backbone_features = torch.randn(
        (1, args.seq_len, int(action_head.config.backbone_embedding_dim)),
        dtype=torch.float16,
        device=args.device,
    )
    torch.onnx.export(
        process_backbone_model,
        (backbone_features,),
        str(action_head_dir / "vlln_vl_self_attention.onnx"),
        export_params=True,
        do_constant_folding=True,
        opset_version=args.opset,
        input_names=["backbone_features"],
        output_names=["output"],
        dynamic_axes={
            "backbone_features": {0: "batch_size", 1: "sequence_length"},
            "output": {0: "batch_size", 1: "sequence_length"},
        },
    )

    # ----- state_encoder -----
    state_encoder = action_head.state_encoder.to(dtype=torch.float16)
    state_tensor = torch.randn(
        (1, args.state_horizon, int(action_head.config.max_state_dim)),
        dtype=torch.float16,
        device=args.device,
    )
    embodiment_id = torch.ones((1,), dtype=torch.int64, device=args.device)
    torch.onnx.export(
        state_encoder,
        (state_tensor, embodiment_id),
        str(action_head_dir / "state_encoder.onnx"),
        export_params=True,
        do_constant_folding=True,
        opset_version=args.opset,
        input_names=["state", "embodiment_id"],
        output_names=["output"],
        dynamic_axes={
            "state": {0: "batch_size"},
            "embodiment_id": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    # ----- action_encoder -----
    action_encoder = action_head.action_encoder.to(dtype=torch.float16)
    actions_tensor = torch.randn(
        (1, int(action_head.config.action_horizon), int(action_head.config.action_dim)),
        dtype=torch.float16,
        device=args.device,
    )
    timesteps_tensor = torch.ones((1,), dtype=torch.int64, device=args.device)
    torch.onnx.export(
        action_encoder,
        (actions_tensor, timesteps_tensor, embodiment_id),
        str(action_head_dir / "action_encoder.onnx"),
        export_params=True,
        do_constant_folding=True,
        opset_version=args.opset,
        input_names=["actions", "timesteps_tensor", "embodiment_id"],
        output_names=["output"],
        dynamic_axes={
            "actions": {0: "batch_size"},
            "timesteps_tensor": {0: "batch_size"},
            "embodiment_id": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    # ----- DiT -----
    dit = action_head.model.to(dtype=torch.float16)
    sa_seq_len = args.state_horizon + int(action_head.config.action_horizon) + int(
        action_head.config.num_target_vision_tokens
    )
    sa_embs = torch.randn(
        (1, sa_seq_len, int(action_head.config.input_embedding_dim)),
        dtype=torch.float16,
        device=args.device,
    )
    vl_embs = torch.randn(
        (1, args.seq_len, int(action_head.config.backbone_embedding_dim)),
        dtype=torch.float16,
        device=args.device,
    )
    torch.onnx.export(
        dit,
        (sa_embs, vl_embs, timesteps_tensor),
        str(action_head_dir / "DiT_fp16.onnx"),
        export_params=True,
        do_constant_folding=True,
        opset_version=args.opset,
        input_names=["sa_embs", "vl_embs", "timesteps_tensor"],
        output_names=["output"],
        dynamic_axes={
            "sa_embs": {0: "batch_size"},
            "vl_embs": {0: "batch_size", 1: "sequence_length"},
            "timesteps_tensor": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    # ----- action_decoder -----
    action_decoder = action_head.action_decoder.to(dtype=torch.float16)
    model_output = torch.randn(
        (1, sa_seq_len, int(action_head.config.hidden_size)),
        dtype=torch.float16,
        device=args.device,
    )
    torch.onnx.export(
        action_decoder,
        (model_output, embodiment_id),
        str(action_head_dir / "action_decoder.onnx"),
        export_params=True,
        do_constant_folding=True,
        opset_version=args.opset,
        input_names=["model_output", "embodiment_id"],
        output_names=["output"],
        dynamic_axes={
            "model_output": {0: "batch_size"},
            "embodiment_id": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    print(f"[OK] Exported action head ONNX to: {action_head_dir}")
    print("[OK] Next: build engines with my_devs/groot_trt/build_engine.sh")


if __name__ == "__main__":
    main()
