from __future__ import annotations

import math
import sys
from functools import partial
from pathlib import Path
from typing import Any

import torch
from transformers.feature_extraction_utils import BatchFeature

DEPLOYMENT_DIR = (
    Path(__file__).resolve().parents[1] / "workspace" / "Isaac-GR00T-n1.7" / "scripts" / "deployment"
)
if str(DEPLOYMENT_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOYMENT_DIR))


def _initialize_rtc_actions(
    action_head: Any,
    action_input: BatchFeature,
    options: dict[str, Any] | None,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(action_head, "init_actions"):
        actions = action_head.init_actions.expand((batch_size, -1, -1)).to(dtype=dtype).clone()
    else:
        actions = torch.randn(
            (batch_size, action_head.config.action_horizon, action_head.action_dim),
            dtype=dtype,
            device=device,
        )
    velocity_strength = torch.ones_like(actions)
    if "action" not in action_input:
        return actions, velocity_strength
    if options is None:
        raise ValueError("RTC action input requires TensorRT RTC options")

    required = ("action_horizon", "rtc_overlap_steps", "rtc_frozen_steps", "rtc_ramp_rate")
    missing = [key for key in required if key not in options]
    if missing:
        raise ValueError(f"TensorRT RTC options are missing: {missing}")
    input_horizon = int(options["action_horizon"])
    overlap = int(options["rtc_overlap_steps"])
    frozen = int(options["rtc_frozen_steps"])
    ramp_rate = float(options["rtc_ramp_rate"])
    if not 1 <= input_horizon <= action_input["action"].shape[1]:
        raise ValueError("Invalid TensorRT RTC action_horizon")
    if not 0 <= frozen <= overlap <= input_horizon:
        raise ValueError("TensorRT RTC requires 0 <= frozen <= overlap <= action_horizon")
    if not math.isfinite(ramp_rate) or ramp_rate <= 0:
        raise ValueError("TensorRT RTC ramp rate must be finite and positive")

    previous = action_input["action"].to(device=device, dtype=dtype)
    actions[:, :overlap, :] = previous[:, input_horizon - overlap : input_horizon, :]
    velocity_strength[:, :frozen, :] = 0
    intermediate = overlap - frozen
    if intermediate:
        t = torch.linspace(0.0, 1.0, intermediate + 2, device=device)
        ramp = 1 - torch.exp(-ramp_rate * t)
        ramp = ramp / ramp[-1].clamp_min(1e-8)
        velocity_strength[:, frozen:overlap, :] = ramp[1:-1][None, :, None].to(dtype)
    return actions, velocity_strength


def rtc_action_head_tensorrt_forward(
    action_head: Any,
    backbone_output: BatchFeature,
    action_input: BatchFeature,
    options: dict[str, Any] | None = None,
) -> BatchFeature:
    """Reference full-pipeline TensorRT forward with N1.7 RTC preserved."""
    backbone_features = action_head.vlln(backbone_output.backbone_features)
    if getattr(action_head, "vl_sa_engine", None) is not None:
        backbone_features = backbone_features.to(torch.bfloat16)
        action_head.vl_sa_engine.set_runtime_tensor_shape("hidden_states", backbone_features.shape)
        backbone_features = action_head.vl_sa_engine(backbone_features)["output"]
    else:
        backbone_features = action_head.vl_self_attention(backbone_features)
    vl_embs = backbone_features.to(torch.bfloat16)

    embodiment_id = action_input.embodiment_id.to(torch.int64)
    state = action_input.state.to(torch.bfloat16)
    if state.ndim == 3 and state.shape[1] > 1:
        state = state.view(state.shape[0], 1, -1)
    if state.ndim != 3 or state.shape[1] != 1:
        raise ValueError(f"Unexpected TensorRT state shape: {tuple(state.shape)}")

    action_head.state_encoder_engine.set_runtime_tensor_shape("state", state.shape)
    action_head.state_encoder_engine.set_runtime_tensor_shape("embodiment_id", embodiment_id.shape)
    state_features = action_head.state_encoder_engine(state, embodiment_id)["output"]

    actions, velocity_strength = _initialize_rtc_actions(
        action_head,
        action_input,
        options,
        batch_size=vl_embs.shape[0],
        device=vl_embs.device,
        dtype=torch.bfloat16,
    )
    dt = 1.0 / action_head.num_inference_timesteps
    for step in range(action_head.num_inference_timesteps):
        timestep = int(step / float(action_head.num_inference_timesteps) * action_head.num_timestep_buckets)
        timesteps = torch.full((vl_embs.shape[0],), timestep, device=vl_embs.device, dtype=torch.int64)
        action_head.action_encoder_engine.set_runtime_tensor_shape("actions", actions.shape)
        action_head.action_encoder_engine.set_runtime_tensor_shape("timesteps", timesteps.shape)
        action_head.action_encoder_engine.set_runtime_tensor_shape("embodiment_id", embodiment_id.shape)
        action_features = action_head.action_encoder_engine(actions, timesteps, embodiment_id)["output"]
        if action_head.config.add_pos_embed:
            positions = torch.arange(action_features.shape[1], device=vl_embs.device)
            action_features = action_features + action_head.position_embedding(positions).unsqueeze(0).to(
                torch.bfloat16
            )

        state_action = torch.cat((state_features, action_features), dim=1).to(torch.bfloat16)
        action_head.dit_engine.set_runtime_tensor_shape("sa_embs", state_action.shape)
        action_head.dit_engine.set_runtime_tensor_shape("vl_embs", vl_embs.shape)
        action_head.dit_engine.set_runtime_tensor_shape("timestep", timesteps.shape)
        dit_kwargs: dict[str, torch.Tensor] = {}
        for name in ("image_mask", "backbone_attention_mask"):
            value = getattr(backbone_output, name, None)
            if value is not None:
                action_head.dit_engine.set_runtime_tensor_shape(name, value.shape)
                dit_kwargs[name] = value
        model_output = action_head.dit_engine(state_action, vl_embs, timesteps, **dit_kwargs)["output"]

        action_head.action_decoder_engine.set_runtime_tensor_shape("model_output", model_output.shape)
        action_head.action_decoder_engine.set_runtime_tensor_shape("embodiment_id", embodiment_id.shape)
        prediction = action_head.action_decoder_engine(model_output, embodiment_id)["output"]
        velocity = prediction[:, -action_head.action_horizon :]
        actions = actions + dt * velocity * velocity_strength

    return BatchFeature(data={"action_pred": actions})


def setup_rtc_tensorrt_engines(policy: Any, engine_dir: Path, mode: str) -> None:
    from trt_model_forward import setup_tensorrt_engines

    if mode not in {"n17_full_pipeline", "action_head"}:
        raise ValueError("RTC TensorRT backend supports n17_full_pipeline or action_head")
    setup_tensorrt_engines(policy, str(engine_dir), mode=mode)
    action_head = policy.model.action_head
    action_head.get_action = partial(rtc_action_head_tensorrt_forward, action_head)
