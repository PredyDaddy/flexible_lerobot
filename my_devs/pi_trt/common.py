#!/usr/bin/env python

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers.cache_utils import DynamicCache


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "lerobot").is_dir():
            return candidate
    raise RuntimeError(f"Failed to locate repo root from: {start}")


REPO_ROOT = find_repo_root(Path(__file__).resolve().parent)
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

os.chdir(REPO_ROOT)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from lerobot import policies  # noqa: F401
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class
from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.converters import batch_to_transition, transition_to_batch
from lerobot.utils.constants import OBS_IMAGES, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE


DEFAULT_POLICY_PATH = (
    "/data/cqy_workspace/flexible_lerobot/outputs/train/"
    "pi05_grasp_block_in_bin1_repro_20260303_170727/bs8_20260303_170727/"
    "checkpoints/last/pretrained_model"
)


def ensure_local_tokenizer_dir(repo_root: Path = REPO_ROOT) -> None:
    local_tokenizer_dir = repo_root / "google" / "paligemma-3b-pt-224"
    if not local_tokenizer_dir.is_dir():
        raise FileNotFoundError(
            "Missing local tokenizer directory for offline PI0.5 processing.\n"
            f"Expected: {local_tokenizer_dir}"
        )


def resolve_policy_dir(policy_path: str | Path) -> Path:
    path = Path(policy_path).expanduser()
    candidates = [
        path,
        path / "pretrained_model",
        path / "checkpoints" / "last" / "pretrained_model",
    ]
    for candidate in candidates:
        if (candidate / "config.json").is_file():
            return candidate
    searched = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "Policy directory must contain config.json.\n"
        f"You passed: {path}\n"
        f"Searched:\n{searched}"
    )


def load_policy(policy_path: str | Path, device: str = "cuda") -> tuple[PreTrainedConfig, torch.nn.Module]:
    policy_dir = resolve_policy_dir(policy_path)
    cfg = PreTrainedConfig.from_pretrained(str(policy_dir))
    cfg.pretrained_path = policy_dir
    cfg.device = device
    policy_cls = get_policy_class(cfg.type)
    policy = policy_cls.from_pretrained(str(policy_dir), config=cfg, strict=False)
    policy.eval()
    policy.to(device)
    return cfg, policy


def prepare_policy_for_fp16(policy: torch.nn.Module) -> None:
    model = policy.model
    model.paligemma_with_expert.to(device=next(policy.parameters()).device, dtype=torch.float16)
    model.action_in_proj.to(device=next(policy.parameters()).device, dtype=torch.float16)
    model.action_out_proj.to(device=next(policy.parameters()).device, dtype=torch.float16)
    model.time_mlp_in.to(device=next(policy.parameters()).device, dtype=torch.float16)
    model.time_mlp_out.to(device=next(policy.parameters()).device, dtype=torch.float16)
    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
    model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001


def load_preprocessor(policy_path: str | Path) -> PolicyProcessorPipeline[dict[str, Any], dict[str, Any]]:
    policy_dir = resolve_policy_dir(policy_path)
    ensure_local_tokenizer_dir(REPO_ROOT)
    return PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=str(policy_dir),
        config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )


def create_sinusoidal_pos_embedding_fp32(
    time: Tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    device: str | torch.device,
) -> Tensor:
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("time must have shape (batch_size,)")
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=torch.float32, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = (1.0 / period) * (2.0 * math.pi)
    sin_input = scaling_factor[None, :] * time[:, None].to(torch.float32)
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def make_synthetic_observation(policy: torch.nn.Module, seed: int = 7, task: str = "Put the block in the bin") -> dict[str, Any]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    state_dim = int(policy.config.input_features[OBS_STATE].shape[0])
    observation: dict[str, Any] = {
        OBS_STATE: torch.linspace(-0.35, 0.35, state_dim, dtype=torch.float32),
        "task": task,
    }
    for image_key in policy.config.image_features:
        image_shape = tuple(int(v) for v in policy.config.input_features[image_key].shape)
        observation[image_key] = torch.rand(image_shape, dtype=torch.float32, generator=generator)
    return observation


def preprocess_observation(
    policy: torch.nn.Module,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    seed: int = 7,
    task: str = "Put the block in the bin",
) -> dict[str, Any]:
    _ = policy
    observation = make_synthetic_observation(policy, seed=seed, task=task)
    return preprocessor(observation)


@dataclass
class PreparedBatch:
    batch: dict[str, Any]
    images: list[Tensor]
    image_masks: list[Tensor]
    image_tensor: Tensor
    tokens: Tensor
    masks: Tensor


def prepare_batch(policy: torch.nn.Module, batch: dict[str, Any]) -> PreparedBatch:
    images, image_masks = policy._preprocess_images(batch)
    if len(images) == 0:
        raise ValueError("PI0.5 requires at least one image")
    batch_size = int(images[0].shape[0])
    if batch_size != 1:
        raise ValueError(f"This first FP16 ONNX baseline only supports batch_size=1, got {batch_size}")
    image_tensor = torch.cat(images, dim=0).to(dtype=torch.float16).contiguous()
    tokens = batch[OBS_LANGUAGE_TOKENS].to(device=image_tensor.device, dtype=torch.int64).contiguous()
    masks = batch[OBS_LANGUAGE_ATTENTION_MASK].to(device=image_tensor.device, dtype=torch.bool).contiguous()
    return PreparedBatch(
        batch=batch,
        images=images,
        image_masks=image_masks,
        image_tensor=image_tensor,
        tokens=tokens,
        masks=masks,
    )


def compute_language_embeddings(policy: torch.nn.Module, tokens: Tensor) -> Tensor:
    lang_embs = policy.model.paligemma_with_expert.embed_language_tokens(tokens)
    return (lang_embs * math.sqrt(lang_embs.shape[-1])).to(dtype=torch.float16)


def build_prefix_from_image_embeddings(
    policy: torch.nn.Module,
    image_embeddings: Tensor,
    image_masks: list[Tensor],
    tokens: Tensor,
    masks: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    num_cameras = len(image_masks)
    if image_embeddings.shape[0] != num_cameras:
        raise ValueError(
            f"Image embedding batch mismatch. Expected {num_cameras}, got {image_embeddings.shape[0]}"
        )

    embs: list[Tensor] = []
    pad_masks: list[Tensor] = []
    attention_mask_pattern: list[int] = []
    for camera_index in range(num_cameras):
        camera_embs = image_embeddings[camera_index : camera_index + 1]
        camera_mask = image_masks[camera_index].to(device=image_embeddings.device, dtype=torch.bool)
        token_count = int(camera_embs.shape[1])
        embs.append(camera_embs)
        pad_masks.append(camera_mask[:, None].expand(camera_mask.shape[0], token_count))
        attention_mask_pattern.extend([0] * token_count)

    language_embeddings = compute_language_embeddings(policy, tokens)
    embs.append(language_embeddings)
    pad_masks.append(masks.to(dtype=torch.bool))
    attention_mask_pattern.extend([0] * int(language_embeddings.shape[1]))

    prefix_embs = torch.cat(embs, dim=1).to(dtype=torch.float16).contiguous()
    prefix_pad_masks = torch.cat(pad_masks, dim=1).to(dtype=torch.bool).contiguous()
    prefix_att_masks = torch.tensor(
        attention_mask_pattern,
        dtype=torch.int64,
        device=prefix_embs.device,
    )[None, :].expand(prefix_embs.shape[0], -1)
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_attention_mask_4d = policy.model._prepare_attention_masks_4d(prefix_att_2d_masks).to(torch.float32)
    prefix_position_ids = (torch.cumsum(prefix_pad_masks.to(dtype=torch.int64), dim=1) - 1).to(dtype=torch.int64)
    return (
        prefix_embs,
        prefix_pad_masks,
        prefix_att_masks,
        prefix_attention_mask_4d,
        prefix_position_ids,
    )


def compute_prefix_cache(
    policy: torch.nn.Module,
    prefix_embs: Tensor,
    prefix_attention_mask_4d: Tensor,
    prefix_position_ids: Tensor,
) -> Tensor:
    outputs = policy.model.paligemma_with_expert.paligemma.language_model.forward(
        inputs_embeds=prefix_embs,
        attention_mask=prefix_attention_mask_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        use_cache=True,
    )
    keys = torch.stack([layer_cache[0].to(torch.float16) for layer_cache in outputs.past_key_values], dim=0)
    values = torch.stack([layer_cache[1].to(torch.float16) for layer_cache in outputs.past_key_values], dim=0)
    return torch.stack((keys, values), dim=1).contiguous()


def embed_suffix_fp16(policy: torch.nn.Module, x_t: Tensor, timestep: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    model = policy.model
    hidden_dtype = model.action_in_proj.weight.dtype
    device = x_t.device
    time_embeddings = create_sinusoidal_pos_embedding_fp32(
        timestep.to(dtype=torch.float32),
        model.action_in_proj.out_features,
        min_period=model.config.min_period,
        max_period=model.config.max_period,
        device=device,
    ).to(dtype=torch.float32)
    action_embeddings = model.action_in_proj(x_t.to(dtype=hidden_dtype))
    time_hidden = model.time_mlp_in(time_embeddings.to(dtype=hidden_dtype))
    time_hidden = F.silu(time_hidden)
    time_hidden = model.time_mlp_out(time_hidden)
    time_hidden = F.silu(time_hidden)
    batch_size, chunk_size = action_embeddings.shape[:2]
    suffix_pad_masks = torch.ones(batch_size, chunk_size, dtype=torch.bool, device=device)
    suffix_att_masks = torch.tensor(
        [1] + ([0] * (chunk_size - 1)),
        dtype=torch.int64,
        device=device,
    )[None, :].expand(batch_size, -1)
    return action_embeddings.contiguous(), suffix_pad_masks, suffix_att_masks, time_hidden.contiguous()


def compute_denoise_step(
    policy: torch.nn.Module,
    x_t: Tensor,
    timestep: Tensor,
    prefix_pad_masks: Tensor,
    kv_cache: Tensor,
) -> Tensor:
    suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = embed_suffix_fp16(policy, x_t, timestep)
    suffix_length = int(suffix_pad_masks.shape[1])
    prefix_length = int(prefix_pad_masks.shape[1])
    batch_size = int(prefix_pad_masks.shape[0])
    prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_length, prefix_length)
    suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
    full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
    prefix_offsets = torch.sum(prefix_pad_masks.to(dtype=torch.int64), dim=-1)[:, None]
    position_ids = (
        prefix_offsets + torch.cumsum(suffix_pad_masks.to(dtype=torch.int64), dim=1) - 1
    ).to(dtype=torch.int64)
    full_att_2d_masks_4d = policy.model._prepare_attention_masks_4d(full_att_2d_masks).to(torch.float32)
    keys = kv_cache[:, 0]
    values = kv_cache[:, 1]
    num_layers = int(keys.shape[0])
    past_key_values = DynamicCache.from_legacy_cache(
        tuple((keys[layer_index], values[layer_index]) for layer_index in range(num_layers))
    )
    outputs = policy.model.paligemma_with_expert.gemma_expert.model.forward(
        inputs_embeds=suffix_embs,
        attention_mask=full_att_2d_masks_4d,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=False,
        adarms_cond=adarms_cond,
    )
    suffix_out = outputs.last_hidden_state[:, -policy.config.chunk_size :].to(policy.model.action_out_proj.weight.dtype)
    return policy.model.action_out_proj(suffix_out).to(torch.float16).contiguous()


def compute_action_chunk(
    policy: torch.nn.Module,
    prepared_batch: PreparedBatch,
    noise: Tensor | None = None,
    num_steps: int | None = None,
) -> Tensor:
    image_embeddings = policy.model.paligemma_with_expert.embed_image(prepared_batch.image_tensor)
    prefix_embs, prefix_pad_masks, _prefix_att_masks, prefix_attention_mask_4d, prefix_position_ids = (
        build_prefix_from_image_embeddings(
            policy,
            image_embeddings,
            prepared_batch.image_masks,
            prepared_batch.tokens,
            prepared_batch.masks,
        )
    )
    kv_cache = compute_prefix_cache(policy, prefix_embs, prefix_attention_mask_4d, prefix_position_ids)

    if num_steps is None:
        num_steps = int(policy.config.num_inference_steps)
    if noise is None:
        noise = torch.randn(
            (1, int(policy.config.chunk_size), int(policy.config.max_action_dim)),
            device=prepared_batch.image_tensor.device,
            dtype=torch.float16,
        )
    x_t = noise.to(torch.float16).contiguous()
    dt = -1.0 / float(num_steps)
    for step in range(num_steps):
        time_value = 1.0 + step * dt
        time_tensor = torch.full((1,), time_value, dtype=torch.float32, device=x_t.device)
        velocity = compute_denoise_step(policy, x_t, time_tensor, prefix_pad_masks, kv_cache)
        x_t = (x_t + dt * velocity).to(torch.float16)
    return x_t


class VisionEncoderOnnxWrapper(nn.Module):
    def __init__(self, policy: torch.nn.Module):
        super().__init__()
        self.policy = policy

    def forward(self, image_tensor: Tensor) -> Tensor:
        return self.policy.model.paligemma_with_expert.embed_image(image_tensor.to(torch.float16)).to(torch.float16)


class PrefixCacheOnnxWrapper(nn.Module):
    def __init__(self, policy: torch.nn.Module):
        super().__init__()
        self.policy = policy

    def forward(self, prefix_embs: Tensor, prefix_attention_mask_4d: Tensor, prefix_position_ids: Tensor) -> Tensor:
        return compute_prefix_cache(
            self.policy,
            prefix_embs.to(torch.float16),
            prefix_attention_mask_4d.to(torch.float32),
            prefix_position_ids.to(torch.int64),
        )


class DenoiseStepOnnxWrapper(nn.Module):
    def __init__(self, policy: torch.nn.Module):
        super().__init__()
        self.policy = policy

    def forward(self, x_t: Tensor, timestep: Tensor, prefix_pad_masks: Tensor, kv_cache: Tensor) -> Tensor:
        return compute_denoise_step(
            self.policy,
            x_t.to(torch.float16),
            timestep.to(torch.float32),
            prefix_pad_masks.to(torch.bool),
            kv_cache.to(torch.float16),
        )


def numpy_metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    reference64 = reference.astype(np.float64)
    prediction64 = prediction.astype(np.float64)
    numerator = float(np.dot(reference64.reshape(-1), prediction64.reshape(-1)))
    denominator = float(np.linalg.norm(reference64.reshape(-1)) * np.linalg.norm(prediction64.reshape(-1)))
    cosine = numerator / denominator if denominator > 0 else 1.0
    rmse = float(np.sqrt(np.mean((reference64 - prediction64) ** 2)))
    mean_abs = float(np.mean(np.abs(reference64 - prediction64)))
    max_abs = float(np.max(np.abs(reference64 - prediction64)))
    return {
        "cosine": cosine,
        "rmse": rmse,
        "mean_abs": mean_abs,
        "max_abs": max_abs,
    }


def tensor_to_numpy(tensor: Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def summarize_metrics(results: dict[str, dict[str, float]]) -> dict[str, Any]:
    lowest_cosine_key = min(results, key=lambda key: results[key]["cosine"])
    return {
        "lowest_cosine_key": lowest_cosine_key,
        "lowest_cosine": results[lowest_cosine_key]["cosine"],
        "results": results,
    }
