#!/usr/bin/env python

"""Shared helpers for LeRobot PI0.5 ONNX export and verification."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def resolve_repo_root(start: Path | None = None) -> Path:
    """Resolve the flexible_lerobot repository root without importing repo code first."""
    current = (start or Path(__file__)).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "lerobot").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate repository root from: {current}")


REPO_ROOT = resolve_repo_root()
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from lerobot import policies  # noqa: E402,F401  # Register policy config classes.
from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.datasets.pipeline_features import (  # noqa: E402
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts  # noqa: E402
from lerobot.policies.factory import get_policy_class  # noqa: E402
from lerobot.processor import PolicyAction, PolicyProcessorPipeline  # noqa: E402
from lerobot.processor.converters import (  # noqa: E402
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks  # noqa: E402
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig  # noqa: E402
from lerobot.utils.constants import OBS_STR  # noqa: E402
from transformers.cache_utils import DynamicCache  # noqa: E402


DEFAULT_POLICY_PATH = (
    REPO_ROOT
    / "outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model"
)
DEFAULT_TASK = "First put the eraser into the small box, then move the cup back to the upper-right corner"
PI05_PREFIX_CACHE_LAYERS = 18


def prefix_cache_tensor_names(num_layers: int = PI05_PREFIX_CACHE_LAYERS) -> list[str]:
    names = ["prefix_pad_masks"]
    for layer_idx in range(num_layers):
        names.append(f"past_key_values.{layer_idx}.key")
        names.append(f"past_key_values.{layer_idx}.value")
    return names


def denoise_step_input_names(num_layers: int = PI05_PREFIX_CACHE_LAYERS) -> list[str]:
    names = prefix_cache_tensor_names(num_layers)
    names.extend(["x_t", "timestep"])
    return names


def flatten_past_key_values(past_key_values) -> tuple[torch.Tensor, ...]:
    if not hasattr(past_key_values, "key_cache") or not hasattr(past_key_values, "value_cache"):
        raise TypeError(f"Expected a DynamicCache-like object, got {type(past_key_values).__name__}")
    if len(past_key_values.key_cache) != len(past_key_values.value_cache):
        raise ValueError(
            "past_key_values key/value cache length mismatch: "
            f"{len(past_key_values.key_cache)} vs {len(past_key_values.value_cache)}"
        )
    flat: list[torch.Tensor] = []
    for key, value in zip(past_key_values.key_cache, past_key_values.value_cache, strict=True):
        flat.append(key.contiguous())
        flat.append(value.contiguous())
    return tuple(flat)


def unflatten_past_key_values(flat_cache: tuple[torch.Tensor, ...]) -> DynamicCache:
    if len(flat_cache) % 2 != 0:
        raise ValueError(f"Expected key/value tensor pairs, got {len(flat_cache)} tensors")
    pairs = []
    for i in range(0, len(flat_cache), 2):
        pairs.append((flat_cache[i].contiguous(), flat_cache[i + 1].contiguous()))
    cache = DynamicCache(pairs)
    if pairs:
        cache._seen_tokens = int(pairs[0][0].shape[-2])  # noqa: SLF001
    return cache


def make_att_2d_masks_for_onnx(pad_masks: torch.Tensor, att_masks: torch.Tensor) -> torch.Tensor:
    cumsum = torch.cumsum(att_masks.to(dtype=torch.int64), dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] & pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class PI05SampleActionsONNXWrapper(torch.nn.Module):
    """Flatten LeRobot PI0.5 list inputs into an ONNX-friendly module signature."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(
        self,
        image_0: torch.Tensor,
        image_1: torch.Tensor,
        img_mask_0: torch.Tensor,
        img_mask_1: torch.Tensor,
        tokens: torch.Tensor,
        masks: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        images = [image_0, image_1]
        img_masks = [img_mask_0, img_mask_1]
        return self.policy.model.sample_actions(images, img_masks, tokens, masks, noise=noise)


class PI05SuffixEmbeddingONNXWrapper(torch.nn.Module):
    """Export the PI0.5 action/time suffix embedding subgraph.

    This is the first split-graph boundary used when full `sample_actions`
    export is blocked by transformer/RoPE exporter limitations.
    """

    def __init__(self, policy):
        super().__init__()
        self.model = policy.model

    def forward(self, noisy_actions: torch.Tensor, timestep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        suffix_embs, _, _, adarms_cond = self.model.embed_suffix(noisy_actions, timestep)
        return suffix_embs, adarms_cond


class PI05PrefixEmbeddingONNXWrapper(torch.nn.Module):
    """Export the PI0.5 image/language prefix embedding subgraph."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(
        self,
        image_0: torch.Tensor,
        image_1: torch.Tensor,
        img_mask_0: torch.Tensor,
        img_mask_1: torch.Tensor,
        tokens: torch.Tensor,
        masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        images = [image_0, image_1]
        img_masks = [img_mask_0, img_mask_1]
        return self.policy.model.embed_prefix(images, img_masks, tokens, masks)


class PI05PrefixCacheONNXWrapper(torch.nn.Module):
    """Export prefix embedding plus PaliGemma prefix KV cache creation."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        self.model = policy.model

    def forward(
        self,
        image_0: torch.Tensor,
        image_1: torch.Tensor,
        img_mask_0: torch.Tensor,
        img_mask_1: torch.Tensor,
        tokens: torch.Tensor,
        masks: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        images = [image_0, image_1]
        img_masks = [img_mask_0, img_mask_1]
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
            images, img_masks, tokens, masks
        )
        prefix_att_2d_masks = make_att_2d_masks_for_onnx(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks.to(dtype=torch.int64), dim=1) - 1
        prefix_att_2d_masks_4d = self.model._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
        _, past_key_values = self.model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        return (prefix_pad_masks.contiguous(), *flatten_past_key_values(past_key_values))


class PI05DenoiseStepONNXWrapper(torch.nn.Module):
    """Export one denoise step using flattened prefix KV cache tensors."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        self.model = policy.model

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        prefix_pad_masks = inputs[0]
        flat_cache = tuple(inputs[1:-2])
        x_t = inputs[-2]
        timestep = inputs[-1]
        past_key_values = unflatten_past_key_values(flat_cache)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.model.embed_suffix(x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks_for_onnx(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks.to(dtype=torch.int64), dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks.to(dtype=torch.int64), dim=1) - 1

        full_att_2d_masks_4d = self.model._prepare_attention_masks_4d(full_att_2d_masks)
        self.model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"

        outputs_embeds, _ = self.model.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.policy.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.model.action_out_proj(suffix_out)


def load_pre_post_processors(policy_path: Path):
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=str(policy_path),
        config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=str(policy_path),
        config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return preprocessor, postprocessor


def ensure_local_tokenizer_dir(repo_root: Path = REPO_ROOT) -> None:
    local_tok = repo_root / "google" / "paligemma-3b-pt-224"
    if not local_tok.is_dir():
        raise FileNotFoundError(
            "Missing local tokenizer directory for offline PI0.5 inference.\n"
            f"Expected: {local_tok}"
        )


def load_policy(policy_path: Path, device: str | None = None, model_dtype: str | None = None):
    config = PreTrainedConfig.from_pretrained(str(policy_path))
    if device is not None:
        config.device = device
    if model_dtype is not None:
        if model_dtype not in {"bfloat16", "float32"}:
            raise ValueError(f"Unsupported PI0.5 dtype override: {model_dtype}")
        config.dtype = model_dtype
    config.pretrained_path = policy_path

    policy_class = get_policy_class(config.type)
    policy = policy_class.from_pretrained(str(policy_path), config=config, strict=False)
    policy.to(config.device)
    policy.eval()

    # Export/inference should not use training-time gradient checkpoint code paths.
    if hasattr(policy, "model") and hasattr(policy.model, "gradient_checkpointing_disable"):
        policy.model.gradient_checkpointing_disable()

    return policy


def make_so101_dataset_features():
    """Create the same dataset feature mapping used by the SO101 real inference script."""
    from lerobot.processor import make_default_processors

    _, robot_action_processor, robot_observation_processor = make_default_processors()
    robot_cfg = SOFollowerRobotConfig(
        id="openpi_trt_dummy_follower",
        port="/dev/null",
        cameras={},
    )
    from lerobot.robots.so_follower.so_follower import SOFollower

    robot = SOFollower(robot_cfg)
    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=robot_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True,
        ),
    )
    return dataset_features


def make_synthetic_observation(task: str = DEFAULT_TASK, seed: int = 1234) -> dict[str, Any]:
    """Build a deterministic SO101-like observation frame before policy preprocessing.

    The real robot script reaches these keys after robot processors and
    `build_dataset_frame(...)`. For offline export verification we can create the
    same policy-facing keys directly and let the saved policy preprocessor handle
    batching, normalization, state-token prompt construction, tokenization, and
    device transfer.
    """
    rng = np.random.default_rng(seed)
    return {
        "observation.state": np.zeros(6, dtype=np.float32),
        "observation.images.top": rng.random((3, 480, 640), dtype=np.float32),
        "observation.images.wrist": rng.random((3, 480, 640), dtype=np.float32),
        "task": [task],
    }


def make_policy_batch(policy_path: Path, task: str = DEFAULT_TASK, seed: int = 1234):
    """Create a deterministic policy-facing batch for model-backend verification.

    This intentionally starts at the same boundary used by
    `PI05Policy.predict_action_chunk(...)`: image tensors plus tokenized language
    fields. The full saved preprocessor is still used in real inference, but
    Torch/ONNX verification should isolate the model backend.
    """
    from transformers import AutoTokenizer

    ensure_local_tokenizer_dir()

    rng = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")

    # Mirror Pi05PrepareStateTokenizerProcessorStep with a zero normalized state.
    discretized_states = np.digitize(np.zeros(32, dtype=np.float32), bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
    state_str = " ".join(map(str, discretized_states))
    full_prompt = f"Task: {task.strip().replace('_', ' ').replace(chr(10), ' ')}, State: {state_str};\nAction: "
    tokenized = tokenizer(
        [full_prompt],
        max_length=200,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    return {
        "observation.images.top": torch.from_numpy(rng.random((1, 3, 480, 640), dtype=np.float32)).to(device),
        "observation.images.wrist": torch.from_numpy(rng.random((1, 3, 480, 640), dtype=np.float32)).to(device),
        "observation.language.tokens": tokenized["input_ids"].to(device),
        "observation.language.attention_mask": tokenized["attention_mask"].to(device).bool(),
    }


def make_export_inputs(policy, batch: dict[str, torch.Tensor], seed: int = 2026):
    """Convert a policy batch into flattened ONNX wrapper inputs plus golden noise."""
    images, img_masks = policy._preprocess_images(batch)
    tokens = batch["observation.language.tokens"]
    masks = batch["observation.language.attention_mask"]

    if len(images) != 2 or len(img_masks) != 2:
        raise ValueError(f"Expected exactly 2 image inputs for SO101, got images={len(images)} masks={len(img_masks)}")

    generator = torch.Generator(device=tokens.device)
    generator.manual_seed(seed)
    noise = torch.randn(
        tokens.shape[0],
        policy.config.chunk_size,
        policy.config.max_action_dim,
        dtype=torch.float32,
        device=tokens.device,
        generator=generator,
    )

    inputs = (
        images[0].contiguous(),
        images[1].contiguous(),
        img_masks[0].contiguous(),
        img_masks[1].contiguous(),
        tokens.contiguous(),
        masks.contiguous(),
        noise.contiguous(),
    )
    return inputs


def make_suffix_embedding_inputs(policy, batch: dict[str, torch.Tensor], seed: int = 2026):
    tokens = batch["observation.language.tokens"]
    generator = torch.Generator(device=tokens.device)
    generator.manual_seed(seed)
    noisy_actions = torch.randn(
        tokens.shape[0],
        policy.config.chunk_size,
        policy.config.max_action_dim,
        dtype=torch.float32,
        device=tokens.device,
        generator=generator,
    )
    timestep = torch.full((tokens.shape[0],), 1.0, dtype=torch.float32, device=tokens.device)
    return noisy_actions.contiguous(), timestep.contiguous()


def make_prefix_embedding_inputs(policy, batch: dict[str, torch.Tensor]):
    images, img_masks = policy._preprocess_images(batch)
    tokens = batch["observation.language.tokens"]
    masks = batch["observation.language.attention_mask"]

    if len(images) != 2 or len(img_masks) != 2:
        raise ValueError(f"Expected exactly 2 image inputs for SO101, got images={len(images)} masks={len(img_masks)}")

    return (
        images[0].contiguous(),
        images[1].contiguous(),
        img_masks[0].contiguous(),
        img_masks[1].contiguous(),
        tokens.contiguous(),
        masks.contiguous(),
    )


def make_prefix_cache_inputs(policy, batch: dict[str, torch.Tensor]):
    return make_prefix_embedding_inputs(policy, batch)


@torch.no_grad()
def make_denoise_step_inputs(policy, batch: dict[str, torch.Tensor], seed: int = 2026):
    prefix_inputs = make_prefix_cache_inputs(policy, batch)
    prefix_outputs = run_torch_prefix_cache(policy, prefix_inputs)
    tokens = batch["observation.language.tokens"]
    generator = torch.Generator(device=tokens.device)
    generator.manual_seed(seed)
    x_t = torch.randn(
        tokens.shape[0],
        policy.config.chunk_size,
        policy.config.max_action_dim,
        dtype=torch.float32,
        device=tokens.device,
        generator=generator,
    )
    timestep = torch.full((tokens.shape[0],), 1.0, dtype=torch.float32, device=tokens.device)
    return (*prefix_outputs, x_t.contiguous(), timestep.contiguous())


@torch.no_grad()
def run_torch_wrapper(policy, inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    wrapper = PI05SampleActionsONNXWrapper(policy).eval()
    return wrapper(*inputs)


@torch.no_grad()
def run_torch_suffix_embedding(policy, inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    wrapper = PI05SuffixEmbeddingONNXWrapper(policy).eval()
    return wrapper(*inputs)


@torch.no_grad()
def run_torch_prefix_embedding(policy, inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    wrapper = PI05PrefixEmbeddingONNXWrapper(policy).eval()
    return wrapper(*inputs)


@torch.no_grad()
def run_torch_prefix_cache(policy, inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    wrapper = PI05PrefixCacheONNXWrapper(policy).eval()
    return wrapper(*inputs)


@torch.no_grad()
def run_torch_denoise_step(policy, inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    wrapper = PI05DenoiseStepONNXWrapper(policy).eval()
    return wrapper(*inputs)


def tensor_stats(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    if reference.dtype == np.bool_ or candidate.dtype == np.bool_:
        equal = reference == candidate
        return {
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
            "equal": bool(np.array_equal(reference, candidate)),
            "mismatch_count": int(np.size(equal) - np.count_nonzero(equal)),
            "mismatch_fraction": float(1.0 - (np.count_nonzero(equal) / np.size(equal))),
            "reference_sample": reference.reshape(-1)[:10].astype(bool).tolist(),
            "candidate_sample": candidate.reshape(-1)[:10].astype(bool).tolist(),
        }

    diff = np.abs(reference - candidate)
    ref_flat = reference.reshape(-1).astype(np.float64)
    cand_flat = candidate.reshape(-1).astype(np.float64)
    cosine = float(np.dot(ref_flat, cand_flat) / ((np.linalg.norm(ref_flat) * np.linalg.norm(cand_flat)) + 1e-12))
    return {
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "mean_abs_diff": float(diff.mean()),
        "max_abs_diff": float(diff.max()),
        "median_abs_diff": float(np.median(diff)),
        "cosine_similarity": cosine,
        "reference_sample": reference.reshape(-1)[:10].astype(float).tolist(),
        "candidate_sample": candidate.reshape(-1)[:10].astype(float).tolist(),
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def configure_runtime() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.chdir(REPO_ROOT)


def patch_transformers_for_onnx_export() -> None:
    """Apply local exporter-only compatibility patches.

    The dynamo ONNX exporter calls `repr(model)` before graph capture. In this
    repository's PI0.5/Gemma combination, some GemmaRMSNorm modules can be
    configured without a `weight` attribute while transformers' `extra_repr`
    still assumes it exists. Patching only the repr keeps model math untouched.
    """
    try:
        from transformers.models.gemma import modeling_gemma
    except Exception:
        return

    rms_norm = getattr(modeling_gemma, "GemmaRMSNorm", None)
    if rms_norm is None or getattr(rms_norm, "_openpi_trt_repr_patched", False):
        return

    def extra_repr(self):
        shape = tuple(self.weight.shape) if hasattr(self, "weight") else "weightless"
        return f"{shape}, eps={self.eps}"

    rms_norm.extra_repr = extra_repr
    rms_norm._openpi_trt_repr_patched = True
