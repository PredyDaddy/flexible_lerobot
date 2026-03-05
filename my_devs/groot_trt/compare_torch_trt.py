#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot import policies  # noqa: F401  # Register policy config classes.
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class

try:
    # When `my_devs/` is importable as a package.
    from my_devs.groot_trt.trt_utils import TrtSession, import_tensorrt  # type: ignore
except Exception:  # pragma: no cover
    from trt_utils import TrtSession, import_tensorrt  # type: ignore


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
    raise TypeError(f"Unsupported model output type: {type(output)!r}")


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


def _np(tensor: torch.Tensor) -> np.ndarray:
    torch.cuda.synchronize()
    detached = tensor.detach()
    if detached.dtype == torch.bfloat16:
        detached = detached.to(torch.float32)
    return detached.cpu().contiguous().numpy()


def _metrics(ref: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    ref64 = ref.astype(np.float64).reshape(-1)
    pred64 = pred.astype(np.float64).reshape(-1)
    diff = np.abs(ref64 - pred64)
    denom = float(np.linalg.norm(ref64) * np.linalg.norm(pred64))
    cosine = float(np.dot(ref64, pred64) / denom) if denom > 0 else float("nan")
    rmse = float(np.sqrt(np.mean((ref64 - pred64) ** 2)))
    return {
        "shape_ref": list(ref.shape),
        "shape_pred": list(pred.shape),
        "max_abs": float(diff.max(initial=0.0)),
        "mean_abs": float(diff.mean() if diff.size else 0.0),
        "rmse": rmse,
        "cosine": cosine,
    }


def _postprocess_vit(backbone: torch.nn.Module, vit_embeds: torch.Tensor) -> torch.Tensor:
    vit_embeds = vit_embeds.view(1, -1, vit_embeds.shape[-1])
    if getattr(backbone.eagle_model, "use_pixel_shuffle", False):
        token_count = int(vit_embeds.shape[1])
        side = int(math.isqrt(token_count))
        if side * side != token_count:
            raise ValueError(
                f"Pixel-shuffle expects square token layout, got token_count={token_count}. "
                "Try `--video-views 1` for consistency compare."
            )
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], side, side, -1)
        pixel_shuffle = getattr(backbone, "pixel_shuffle", None) or getattr(backbone.eagle_model, "pixel_shuffle", None)
        downsample_ratio = getattr(backbone, "downsample_ratio", None) or getattr(
            backbone.eagle_model, "downsample_ratio", None
        )
        if pixel_shuffle is None or downsample_ratio is None:
            raise RuntimeError("Eagle pixel shuffle is enabled but pixel_shuffle/downsample_ratio is missing.")
        vit_embeds = pixel_shuffle(vit_embeds, scale_factor=downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
    vit_embeds = backbone.eagle_model.mlp1(vit_embeds)
    return vit_embeds


def _build_llm_inputs(
    backbone: torch.nn.Module,
    vit_embeds: torch.Tensor,
    seq_len: int,
    generator: torch.Generator,
    input_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    embedding_layer = backbone.eagle_model.language_model.get_input_embeddings()
    vocab_size = int(embedding_layer.num_embeddings)
    image_token_index = int(backbone.eagle_model.image_token_index)

    if input_ids is None:
        input_ids = torch.randint(
            low=0,
            high=vocab_size,
            size=(1, seq_len),
            generator=generator,
            device=vit_embeds.device,
            dtype=torch.int64,
        )
    else:
        input_ids = input_ids.to(device=vit_embeds.device, dtype=torch.int64)
        if tuple(input_ids.shape) != (1, seq_len):
            raise ValueError(f"Expected input_ids shape (1, {seq_len}), got {tuple(input_ids.shape)}")

    n_vis_tokens = int(vit_embeds.shape[1])
    if n_vis_tokens > seq_len:
        raise ValueError(f"Need seq_len >= visual_tokens. seq_len={seq_len} visual_tokens={n_vis_tokens}")

    input_ids[:, :n_vis_tokens] = image_token_index
    input_embeds = embedding_layer(input_ids).to(torch.float16)
    batch, tokens, channels = input_embeds.shape
    input_embeds_flat = input_embeds.reshape(batch * tokens, channels)
    input_ids_flat = input_ids.reshape(batch * tokens)
    selected = input_ids_flat == image_token_index
    input_embeds_flat[selected] = vit_embeds.reshape(-1, channels)[: int(selected.sum())]
    input_embeds = input_embeds_flat.reshape(batch, tokens, channels)
    attention_mask = torch.ones((1, seq_len), dtype=torch.int64, device=vit_embeds.device)
    return input_ids, input_embeds, attention_mask


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare PyTorch modules and TensorRT engines for GROOT.")
    parser.add_argument(
        "--policy-path",
        default=DEFAULT_POLICY_PATH,
        help="Path to LeRobot GROOT checkpoint folder (the `pretrained_model/` directory).",
    )
    parser.add_argument("--engine-dir", required=True, help="Directory containing built TensorRT engines.")
    parser.add_argument("--seq-len", type=int, default=296, help="Sequence length used in synthetic compare inputs.")
    parser.add_argument("--video-views", type=int, default=1, help="Number of synthetic camera views for ViT compare.")
    parser.add_argument("--seed", type=int, default=20260303, help="Random seed for deterministic synthetic inputs.")
    parser.add_argument(
        "--num-denoising-steps",
        type=int,
        default=None,
        help="Override denoising steps for pipeline compare. Defaults to checkpoint config.",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda"], help="Device for PyTorch reference.")
    parser.add_argument(
        "--tensorrt-py-dir",
        default=None,
        help="Optional repo-local TensorRT install dir (folder containing `tensorrt/` and `tensorrt_libs/`).",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional output JSON path. Default: <engine-dir>/compare_metrics_trt.json",
    )
    return parser


@torch.inference_mode()
def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    policy_path = Path(args.policy_path).expanduser()
    if not policy_path.is_dir():
        raise FileNotFoundError(f"--policy-path does not exist: {policy_path}")

    engine_dir = Path(args.engine_dir).expanduser()
    if not engine_dir.is_dir():
        raise FileNotFoundError(f"--engine-dir does not exist: {engine_dir}")

    cfg = PreTrainedConfig.from_pretrained(str(policy_path))
    cfg.pretrained_path = policy_path
    cfg.device = args.device
    policy_cls = get_policy_class(cfg.type)
    policy = policy_cls.from_pretrained(str(policy_path), config=cfg, strict=False)
    policy.eval()
    policy.to(args.device)

    backbone = policy._groot_model.backbone
    action_head = policy._groot_model.action_head
    backbone.eval()
    action_head.eval()
    backbone.eagle_model.vision_model.to(dtype=torch.float16)
    backbone.eagle_model.mlp1.to(dtype=torch.float16)
    backbone.eagle_model.language_model.to(dtype=torch.float16)
    if hasattr(backbone.eagle_model.vision_model, "config"):
        backbone.eagle_model.vision_model.config._attn_implementation = "eager"
    if hasattr(backbone.eagle_model.language_model, "config"):
        backbone.eagle_model.language_model.config._attn_implementation = "eager"
    backbone.eagle_linear.to(dtype=torch.float16)
    action_head.to(dtype=torch.float16)

    trt = import_tensorrt(args.tensorrt_py_dir)

    required_engines = {
        "vit": engine_dir / "vit_fp16.engine",
        "llm": engine_dir / "llm_fp16.engine",
        "vlln": engine_dir / "vlln_vl_self_attention.engine",
        "state_encoder": engine_dir / "state_encoder.engine",
        "action_encoder": engine_dir / "action_encoder.engine",
        "dit": engine_dir / "DiT_fp16.engine",
        "action_decoder": engine_dir / "action_decoder.engine",
    }
    missing = [path for path in required_engines.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required engine files:\n" + "\n".join(f"  - {p}" for p in missing))

    sessions = {name: TrtSession.load(path, trt_py_dir=args.tensorrt_py_dir) for name, path in required_engines.items()}

    report: dict[str, Any] = {
        "policy_path": str(policy_path),
        "engine_dir": str(engine_dir),
        "tensorrt_version": trt.__version__,
        "seed": args.seed,
        "seq_len": args.seq_len,
        "video_views": args.video_views,
        "results": {},
    }

    generator = torch.Generator(device=args.device)
    generator.manual_seed(args.seed)

    llm_wrapper = LanguageModelForOnnx(
        backbone.eagle_model.language_model,
        backbone.eagle_linear,
        backbone.select_layer,
    ).to(device=args.device, dtype=torch.float16)

    # ----- ViT -----
    if hasattr(backbone.eagle_model.vision_model, "vision_model"):
        num_patches = int(backbone.eagle_model.vision_model.vision_model.embeddings.num_patches)
    else:
        num_patches = int(backbone.eagle_model.vision_model.embeddings.num_patches)
    pixel_values = torch.randn(
        (args.video_views, 3, 224, 224),
        dtype=torch.float16,
        device=args.device,
        generator=generator,
    )
    position_ids = torch.arange(num_patches, device=args.device, dtype=torch.int64).expand((args.video_views, -1))

    vit_torch = _extract_hidden_state(backbone.eagle_model.vision_model(pixel_values))
    vit_trt = sessions["vit"].run({"pixel_values": pixel_values, "position_ids": position_ids})["vit_embeds"]
    report["results"]["vit"] = _metrics(_np(vit_torch), _np(vit_trt))

    # ----- ViT postprocess + LLM (from ViT) -----
    vit_torch_post = _postprocess_vit(backbone, vit_torch.to(torch.float16))
    vit_trt_post = _postprocess_vit(backbone, vit_trt.to(torch.float16))
    report["results"]["vit_postprocess"] = _metrics(_np(vit_torch_post), _np(vit_trt_post))

    input_ids, inputs_embeds_torch, attention_mask = _build_llm_inputs(backbone, vit_torch_post, args.seq_len, generator)
    _, inputs_embeds_trt, _ = _build_llm_inputs(
        backbone, vit_trt_post, args.seq_len, generator, input_ids=input_ids
    )

    llm_torch = llm_wrapper(inputs_embeds_torch, attention_mask)
    llm_trt = sessions["llm"].run({"inputs_embeds": inputs_embeds_trt, "attention_mask": attention_mask})["embeddings"]
    report["results"]["llm_from_vit_pipeline"] = _metrics(_np(llm_torch), _np(llm_trt))

    # ----- LLM direct -----
    hidden_size = int(backbone.eagle_model.language_model.config.hidden_size)
    llm_inputs_embeds = torch.randn(
        (1, args.seq_len, hidden_size),
        dtype=torch.float16,
        device=args.device,
        generator=generator,
    )
    llm_attention_mask = torch.ones((1, args.seq_len), dtype=torch.int64, device=args.device)
    llm_torch = llm_wrapper(llm_inputs_embeds, llm_attention_mask)
    llm_trt = sessions["llm"].run({"inputs_embeds": llm_inputs_embeds, "attention_mask": llm_attention_mask})[
        "embeddings"
    ]
    report["results"]["llm_direct"] = _metrics(_np(llm_torch), _np(llm_trt))

    # ----- Action head (module-level) -----
    backbone_dim = int(action_head.config.backbone_embedding_dim)
    input_dim = int(action_head.config.input_embedding_dim)
    hidden_dim = int(action_head.config.hidden_size)
    state_dim = int(action_head.config.max_state_dim)
    action_dim = int(action_head.config.action_dim)
    action_horizon = int(action_head.config.action_horizon)
    sa_seq_len = 1 + action_horizon + int(action_head.config.num_target_vision_tokens)

    backbone_features = torch.randn(
        (1, args.seq_len, backbone_dim),
        dtype=torch.float16,
        device=args.device,
        generator=generator,
    )
    state = torch.randn((1, 1, state_dim), dtype=torch.float16, device=args.device, generator=generator)
    embodiment_id = torch.ones((1,), dtype=torch.int64, device=args.device)
    actions = torch.randn(
        (1, action_horizon, action_dim),
        dtype=torch.float16,
        device=args.device,
        generator=generator,
    )
    timesteps = torch.randint(
        low=0,
        high=max(1, int(action_head.num_timestep_buckets)),
        size=(1,),
        device=args.device,
        generator=generator,
        dtype=torch.int64,
    )
    sa_embs = torch.randn(
        (1, sa_seq_len, input_dim),
        dtype=torch.float16,
        device=args.device,
        generator=generator,
    )
    model_output = torch.randn(
        (1, sa_seq_len, hidden_dim),
        dtype=torch.float16,
        device=args.device,
        generator=generator,
    )

    vlln_torch = action_head.vl_self_attention(action_head.vlln(backbone_features))
    vlln_trt = sessions["vlln"].run({"backbone_features": backbone_features})["output"]
    report["results"]["action_vlln_vl_self_attention"] = _metrics(_np(vlln_torch), _np(vlln_trt))

    state_encoder_torch = action_head.state_encoder(state, embodiment_id)
    state_encoder_trt = sessions["state_encoder"].run({"state": state, "embodiment_id": embodiment_id})["output"]
    report["results"]["action_state_encoder"] = _metrics(_np(state_encoder_torch), _np(state_encoder_trt))

    action_encoder_torch = action_head.action_encoder(actions, timesteps, embodiment_id)
    action_encoder_trt = sessions["action_encoder"].run(
        {"actions": actions, "timesteps_tensor": timesteps, "embodiment_id": embodiment_id}
    )["output"]
    report["results"]["action_action_encoder"] = _metrics(_np(action_encoder_torch), _np(action_encoder_trt))

    dit_torch = action_head.model(sa_embs, backbone_features, timesteps)
    dit_trt = sessions["dit"].run({"sa_embs": sa_embs, "vl_embs": backbone_features, "timesteps_tensor": timesteps})[
        "output"
    ]
    report["results"]["action_dit"] = _metrics(_np(dit_torch), _np(dit_trt))

    decoder_torch = action_head.action_decoder(model_output, embodiment_id)
    decoder_trt = sessions["action_decoder"].run({"model_output": model_output, "embodiment_id": embodiment_id})[
        "output"
    ]
    report["results"]["action_decoder"] = _metrics(_np(decoder_torch), _np(decoder_trt))

    # ----- Action denoising pipeline (end-to-end) -----
    num_steps = (
        int(args.num_denoising_steps)
        if args.num_denoising_steps is not None
        else int(action_head.num_inference_timesteps)
    )
    dt = 1.0 / float(num_steps)

    init_actions = torch.randn(
        (1, action_horizon, action_dim),
        dtype=torch.float16,
        device=args.device,
        generator=generator,
    )

    # PyTorch reference.
    vl_embs_torch = action_head.vl_self_attention(action_head.vlln(backbone_features))
    state_features_torch = action_head.state_encoder(state, embodiment_id)
    actions_torch = init_actions.clone()
    for step in range(num_steps):
        t_cont = step / float(num_steps)
        t_discretized = int(t_cont * int(action_head.num_timestep_buckets))
        step_t = torch.full((1,), t_discretized, dtype=torch.int64, device=args.device)
        action_features_torch = action_head.action_encoder(actions_torch, step_t, embodiment_id)
        if action_head.config.add_pos_embed:
            pos_ids = torch.arange(action_features_torch.shape[1], dtype=torch.long, device=args.device)
            pos_embs = action_head.position_embedding(pos_ids).unsqueeze(0).to(torch.float16)
            action_features_torch = action_features_torch + pos_embs
        future_tokens_torch = action_head.future_tokens.weight.unsqueeze(0).expand(1, -1, -1)
        sa_embs_torch = torch.cat((state_features_torch, future_tokens_torch, action_features_torch), dim=1).to(
            torch.float16
        )
        model_output_torch = action_head.model(sa_embs_torch, vl_embs_torch, step_t)
        pred_torch = action_head.action_decoder(model_output_torch, embodiment_id)
        pred_velocity_torch = pred_torch[:, -action_horizon:]
        actions_torch = actions_torch + dt * pred_velocity_torch

    # TensorRT pipeline (module-by-module, with the same glue code in torch).
    vl_embs_trt = sessions["vlln"].run({"backbone_features": backbone_features})["output"].to(torch.float16)
    state_features_trt = sessions["state_encoder"].run({"state": state, "embodiment_id": embodiment_id})["output"].to(
        torch.float16
    )
    future_tokens_trt = action_head.future_tokens.weight.unsqueeze(0).expand(1, -1, -1).to(torch.float16)
    if action_head.config.add_pos_embed:
        pos_ids = torch.arange(action_horizon, dtype=torch.long, device=args.device)
        pos_embs_trt = action_head.position_embedding(pos_ids).unsqueeze(0).to(torch.float16)
    else:
        pos_embs_trt = None

    actions_trt = init_actions.clone()
    for step in range(num_steps):
        t_cont = step / float(num_steps)
        t_discretized = int(t_cont * int(action_head.num_timestep_buckets))
        step_t = torch.full((1,), t_discretized, dtype=torch.int64, device=args.device)
        action_features_trt = sessions["action_encoder"].run(
            {"actions": actions_trt, "timesteps_tensor": step_t, "embodiment_id": embodiment_id}
        )["output"].to(torch.float16)
        if pos_embs_trt is not None:
            action_features_trt = action_features_trt + pos_embs_trt
        sa_embs_trt = torch.cat((state_features_trt, future_tokens_trt, action_features_trt), dim=1).to(torch.float16)
        model_output_trt = sessions["dit"].run(
            {"sa_embs": sa_embs_trt, "vl_embs": vl_embs_trt, "timesteps_tensor": step_t}
        )["output"].to(torch.float16)
        pred_trt = sessions["action_decoder"].run({"model_output": model_output_trt, "embodiment_id": embodiment_id})[
            "output"
        ].to(torch.float16)
        pred_velocity_trt = pred_trt[:, -action_horizon:, :]
        actions_trt = actions_trt + dt * pred_velocity_trt

    report["results"]["action_denoising_pipeline"] = _metrics(_np(actions_torch), _np(actions_trt))

    for key, value in report["results"].items():
        print(
            f"[COMPARE] {key}: "
            f"cos={value['cosine']:.8f}, rmse={value['rmse']:.8f}, "
            f"mean_abs={value['mean_abs']:.8f}, max_abs={value['max_abs']:.8f}"
        )

    json_out = Path(args.json_out).expanduser() if args.json_out else engine_dir / "compare_metrics_trt.json"
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2))
    print(f"[OK] Comparison report saved to: {json_out}")


if __name__ == "__main__":
    main()
