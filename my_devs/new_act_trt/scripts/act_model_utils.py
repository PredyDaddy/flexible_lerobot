from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ is None or __package__ == "":
    from act_trt_paths import DEFAULT_CHECKPOINT, DEPLOY_ROOT
else:
    from .act_trt_paths import DEFAULT_CHECKPOINT, DEPLOY_ROOT

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.modeling_act import ACTPolicy, ACTTemporalEnsembler
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.processor.converters import (
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE


@dataclass(frozen=True)
class ActModelSpec:
    checkpoint: str
    visual_keys: list[str]
    state_dim: int
    image_height: int
    image_width: int
    action_dim: int
    chunk_size: int
    n_action_steps: int

    @property
    def obs_state_shape(self) -> tuple[int, int]:
        return (1, self.state_dim)

    @property
    def image_shape(self) -> tuple[int, int, int, int]:
        return (1, 3, self.image_height, self.image_width)

    @property
    def action_shape(self) -> tuple[int, int, int]:
        return (1, self.chunk_size, self.action_dim)


def resolve_checkpoint(path: str | Path | None) -> Path:
    checkpoint = Path(path or DEFAULT_CHECKPOINT).expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint}")
    return checkpoint


def resolve_run_name(checkpoint: str | Path) -> str:
    checkpoint_path = resolve_checkpoint(checkpoint)
    try:
        return checkpoint_path.parents[2].name
    except IndexError as exc:
        raise ValueError(f"Unexpected checkpoint layout: {checkpoint_path}") from exc


def resolve_checkpoint_step(checkpoint: str | Path) -> str:
    checkpoint_path = resolve_checkpoint(checkpoint)
    return checkpoint_path.parent.name


def resolve_output_dir(checkpoint: str | Path, output_dir: str | Path | None = None) -> Path:
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve()

    checkpoint_path = resolve_checkpoint(checkpoint)
    return (DEPLOY_ROOT / resolve_run_name(checkpoint_path) / resolve_checkpoint_step(checkpoint_path)).resolve()


def resolve_default_engine_path(checkpoint: str | Path, precision: str = "fp32") -> Path:
    checkpoint_path = resolve_checkpoint(checkpoint)
    candidates = [
        DEPLOY_ROOT / resolve_run_name(checkpoint_path) / resolve_checkpoint_step(checkpoint_path) / f"act_single_{precision}.plan",
        DEPLOY_ROOT / resolve_run_name(checkpoint_path) / f"act_single_{precision}.plan",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def resolve_default_metadata_path(checkpoint: str | Path) -> Path:
    checkpoint_path = resolve_checkpoint(checkpoint)
    candidates = [
        DEPLOY_ROOT / resolve_run_name(checkpoint_path) / resolve_checkpoint_step(checkpoint_path) / "export_metadata.json",
        DEPLOY_ROOT / resolve_run_name(checkpoint_path) / "export_metadata.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def get_visual_feature_keys_in_order(config_dict: dict[str, Any]) -> list[str]:
    input_features = config_dict.get("input_features", {})
    return [key for key, value in input_features.items() if isinstance(value, dict) and value.get("type") == "VISUAL"]


def load_model_spec(checkpoint: str | Path) -> ActModelSpec:
    checkpoint_path = resolve_checkpoint(checkpoint)
    config_dict = load_json(checkpoint_path / "config.json")
    visual_keys = get_visual_feature_keys_in_order(config_dict)
    if len(visual_keys) != 2:
        raise ValueError(f"Expected exactly 2 visual inputs, got {visual_keys}")

    state_shape = config_dict["input_features"]["observation.state"]["shape"]
    image_shape = config_dict["input_features"][visual_keys[0]]["shape"]
    action_shape = config_dict["output_features"]["action"]["shape"]

    if int(image_shape[0]) != 3:
        raise ValueError(f"Expected 3 image channels, got {image_shape[0]}")

    return ActModelSpec(
        checkpoint=str(checkpoint_path),
        visual_keys=visual_keys,
        state_dim=int(state_shape[0]),
        image_height=int(image_shape[1]),
        image_width=int(image_shape[2]),
        action_dim=int(action_shape[0]),
        chunk_size=int(config_dict["chunk_size"]),
        n_action_steps=int(config_dict["n_action_steps"]),
    )


def load_act_policy(checkpoint: str | Path, device: str = "cpu") -> ACTPolicy:
    checkpoint_path = resolve_checkpoint(checkpoint)
    config = PreTrainedConfig.from_pretrained(str(checkpoint_path))
    config.device = device
    if hasattr(config, "pretrained_backbone_weights"):
        config.pretrained_backbone_weights = None
    policy = ACTPolicy.from_pretrained(str(checkpoint_path), config=config)
    policy.eval()
    return policy


def clone_policy_config(checkpoint: str | Path, *, device: str | None = None) -> PreTrainedConfig:
    checkpoint_path = resolve_checkpoint(checkpoint)
    config = copy.deepcopy(PreTrainedConfig.from_pretrained(str(checkpoint_path)))
    if device is not None:
        config.device = device
    return config


def apply_act_runtime_overrides(
    policy_cfg: PreTrainedConfig,
    policy_n_action_steps: int | None,
    policy_temporal_ensemble_coeff: float | None,
) -> None:
    if policy_cfg.type != "act":
        raise ValueError(f"Expected ACT policy, got {policy_cfg.type!r}")

    chunk_size = int(policy_cfg.chunk_size)

    if policy_n_action_steps is not None:
        if not 1 <= policy_n_action_steps <= chunk_size:
            raise ValueError(
                f"--policy-n-action-steps must be within [1, {chunk_size}], got {policy_n_action_steps}"
            )
        policy_cfg.n_action_steps = policy_n_action_steps

    if policy_temporal_ensemble_coeff is not None:
        if policy_cfg.n_action_steps != 1:
            raise ValueError(
                "ACT temporal ensembling requires n_action_steps == 1. "
                f"Current value: {policy_cfg.n_action_steps}"
            )
        policy_cfg.temporal_ensemble_coeff = policy_temporal_ensemble_coeff


def apply_act_policy_runtime_overrides(
    policy: ACTPolicy,
    policy_n_action_steps: int | None,
    policy_temporal_ensemble_coeff: float | None,
) -> None:
    apply_act_runtime_overrides(policy.config, policy_n_action_steps, policy_temporal_ensemble_coeff)

    if policy.config.temporal_ensemble_coeff is not None:
        policy.temporal_ensembler = ACTTemporalEnsembler(
            policy.config.temporal_ensemble_coeff,
            policy.config.chunk_size,
        )
    elif hasattr(policy, "temporal_ensembler"):
        delattr(policy, "temporal_ensembler")

    policy.reset()


def load_pre_post_processors(
    checkpoint: str | Path,
    *,
    policy_device: str | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    checkpoint_path = resolve_checkpoint(checkpoint)
    preprocessor_kwargs: dict[str, Any] = {
        "pretrained_model_name_or_path": str(checkpoint_path),
        "config_filename": "policy_preprocessor.json",
        "to_transition": batch_to_transition,
        "to_output": transition_to_batch,
    }
    if policy_device is not None:
        preprocessor_kwargs["overrides"] = {"device_processor": {"device": policy_device}}

    preprocessor = PolicyProcessorPipeline.from_pretrained(**preprocessor_kwargs)
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=str(checkpoint_path),
        config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return preprocessor, postprocessor


@torch.no_grad()
def torch_core_forward(
    policy: ACTPolicy,
    obs_state_norm: torch.Tensor,
    img0_norm: torch.Tensor,
    img1_norm: torch.Tensor,
) -> torch.Tensor:
    batch = {
        OBS_STATE: obs_state_norm,
        OBS_IMAGES: [img0_norm, img1_norm],
    }
    actions, _ = policy.model(batch)
    return actions


@torch.no_grad()
def make_case_inputs(spec: ActModelSpec, seed: int | None = None, case: str = "random") -> tuple[torch.Tensor, ...]:
    if case not in {"random", "zeros", "ones", "linspace"}:
        raise ValueError(f"Unsupported case: {case}")

    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(int(seed))

    if case == "random":
        obs_state = torch.randn(spec.obs_state_shape, dtype=torch.float32, generator=generator)
        img0 = torch.randn(spec.image_shape, dtype=torch.float32, generator=generator)
        img1 = torch.randn(spec.image_shape, dtype=torch.float32, generator=generator)
    elif case == "zeros":
        obs_state = torch.zeros(spec.obs_state_shape, dtype=torch.float32)
        img0 = torch.zeros(spec.image_shape, dtype=torch.float32)
        img1 = torch.zeros(spec.image_shape, dtype=torch.float32)
    elif case == "ones":
        obs_state = torch.ones(spec.obs_state_shape, dtype=torch.float32)
        img0 = torch.ones(spec.image_shape, dtype=torch.float32)
        img1 = torch.ones(spec.image_shape, dtype=torch.float32)
    else:
        obs_state = torch.linspace(-1.0, 1.0, steps=int(np.prod(spec.obs_state_shape)), dtype=torch.float32).reshape(
            spec.obs_state_shape
        )
        img0 = torch.linspace(-2.0, 2.0, steps=int(np.prod(spec.image_shape)), dtype=torch.float32).reshape(
            spec.image_shape
        )
        img1 = torch.linspace(2.0, -2.0, steps=int(np.prod(spec.image_shape)), dtype=torch.float32).reshape(
            spec.image_shape
        )

    return obs_state.contiguous(), img0.contiguous(), img1.contiguous()


def make_case_feed_dict(spec: ActModelSpec, seed: int | None = None, case: str = "random") -> dict[str, np.ndarray]:
    obs_state, img0, img1 = make_case_inputs(spec=spec, seed=seed, case=case)
    return {
        "obs_state_norm": obs_state.numpy(),
        "img0_norm": img0.numpy(),
        "img1_norm": img1.numpy(),
    }


def make_mock_observation(
    spec: ActModelSpec,
    seed: int | None = None,
    case: str = "random",
) -> dict[str, np.ndarray]:
    if case not in {"random", "zeros", "ones", "linspace"}:
        raise ValueError(f"Unsupported case: {case}")

    rng = np.random.default_rng(seed)
    state_key = "observation.state"
    img0_key, img1_key = spec.visual_keys

    if case == "random":
        obs_state = rng.standard_normal((spec.state_dim,), dtype=np.float32)
        img0 = rng.integers(0, 256, size=(spec.image_height, spec.image_width, 3), dtype=np.uint8)
        img1 = rng.integers(0, 256, size=(spec.image_height, spec.image_width, 3), dtype=np.uint8)
    elif case == "zeros":
        obs_state = np.zeros((spec.state_dim,), dtype=np.float32)
        img0 = np.zeros((spec.image_height, spec.image_width, 3), dtype=np.uint8)
        img1 = np.zeros((spec.image_height, spec.image_width, 3), dtype=np.uint8)
    elif case == "ones":
        obs_state = np.ones((spec.state_dim,), dtype=np.float32)
        img0 = np.full((spec.image_height, spec.image_width, 3), 255, dtype=np.uint8)
        img1 = np.full((spec.image_height, spec.image_width, 3), 255, dtype=np.uint8)
    else:
        obs_state = np.linspace(-1.0, 1.0, num=spec.state_dim, dtype=np.float32)
        grid_x = np.linspace(0, 255, num=spec.image_width, dtype=np.float32)
        grid_y = np.linspace(255, 0, num=spec.image_height, dtype=np.float32)
        channel_r = np.tile(grid_x[None, :], (spec.image_height, 1))
        channel_g = np.tile(grid_y[:, None], (1, spec.image_width))
        channel_b = np.full((spec.image_height, spec.image_width), 127, dtype=np.float32)
        img0 = np.stack([channel_r, channel_g, channel_b], axis=-1).astype(np.uint8)
        img1 = np.stack([channel_g, channel_r, 255 - channel_b], axis=-1).astype(np.uint8)

    return {
        state_key: np.ascontiguousarray(obs_state),
        img0_key: np.ascontiguousarray(img0),
        img1_key: np.ascontiguousarray(img1),
    }


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return np.asarray(tensor.detach().cpu().float().numpy(), dtype=np.float32)


def compare_arrays(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, float]:
    lhs_f = np.asarray(lhs, dtype=np.float32)
    rhs_f = np.asarray(rhs, dtype=np.float32)
    diff = np.abs(lhs_f - rhs_f)
    denom = np.maximum(np.abs(lhs_f), 1e-8)
    rel = diff / denom
    flat_lhs = lhs_f.reshape(-1)
    flat_rhs = rhs_f.reshape(-1)
    cosine = float(np.dot(flat_lhs, flat_rhs) / (np.linalg.norm(flat_lhs) * np.linalg.norm(flat_rhs) + 1e-12))
    return {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "max_rel_diff": float(rel.max()),
        "cosine_similarity": cosine,
    }


def merge_summary(metrics_list: list[dict[str, float]]) -> dict[str, float]:
    return {
        "max_abs_diff": max(item["max_abs_diff"] for item in metrics_list),
        "mean_abs_diff": max(item["mean_abs_diff"] for item in metrics_list),
        "max_rel_diff": max(item["max_rel_diff"] for item in metrics_list),
        "min_cosine_similarity": min(item["cosine_similarity"] for item in metrics_list),
    }


def passes_thresholds(
    summary: dict[str, float],
    *,
    threshold_max_abs_diff: float,
    threshold_max_rel_diff: float | None = None,
    threshold_min_cosine_similarity: float | None = None,
) -> bool:
    passed = summary["max_abs_diff"] <= threshold_max_abs_diff
    if threshold_max_rel_diff is not None:
        passed = passed and summary["max_rel_diff"] <= threshold_max_rel_diff
    if threshold_min_cosine_similarity is not None:
        passed = passed and summary["min_cosine_similarity"] >= threshold_min_cosine_similarity
    return passed


def build_case_catalog(random_cases: int) -> list[dict[str, Any]]:
    catalog: list[dict[str, Any]] = []
    for seed in range(int(random_cases)):
        catalog.append({"name": f"random_seed_{seed}", "case": "random", "seed": seed})
    catalog.extend(
        [
            {"name": "zeros", "case": "zeros", "seed": None},
            {"name": "ones", "case": "ones", "seed": None},
            {"name": "linspace", "case": "linspace", "seed": None},
        ]
    )
    return catalog


def spec_to_dict(spec: ActModelSpec) -> dict[str, Any]:
    return asdict(spec)
