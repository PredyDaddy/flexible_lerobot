#!/usr/bin/env python

"""Compact PI0.5 split TensorRT runtime.

This file is the new minimal runtime path. It intentionally keeps only the
production split boundary:

prefix_cache engine + denoise_step engine + Python denoise loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from runtime.protocol import (
    DENOISE_OUTPUT_NAMES,
    PREFIX_CACHE_INPUT_NAMES,
    denoise_step_input_names,
    prefix_cache_tensor_names,
    validate_names,
)
from runtime.trt_engine import TorchTensorRTEngine


DEFAULT_PREFIX_ENGINE = Path("my_devs/openpi_trt/artifacts/pi05_so101_prefix_cache_b1_fp32.engine")
DEFAULT_DENOISE_FP32_ENGINE = Path("my_devs/openpi_trt/artifacts/pi05_so101_denoise_step_b1_fp32.engine")
DEFAULT_DENOISE_FP16_CONSTRAINED_ENGINE = Path(
    "my_devs/openpi_trt/artifacts/pi05_so101_denoise_step_b1_fp16_constrained.engine"
)


@dataclass(frozen=True)
class SimplePI05TRTProfile:
    """A deployable split TensorRT profile."""

    name: str
    prefix_engine_path: Path
    denoise_engine_path: Path
    num_layers: int = 18

    @classmethod
    def fp32(cls) -> "SimplePI05TRTProfile":
        return cls(
            name="fp32",
            prefix_engine_path=DEFAULT_PREFIX_ENGINE,
            denoise_engine_path=DEFAULT_DENOISE_FP32_ENGINE,
        )

    @classmethod
    def fp16_constrained(cls) -> "SimplePI05TRTProfile":
        return cls(
            name="fp16_constrained",
            prefix_engine_path=DEFAULT_PREFIX_ENGINE,
            denoise_engine_path=DEFAULT_DENOISE_FP16_CONSTRAINED_ENGINE,
        )

    @classmethod
    def from_name(cls, name: str) -> "SimplePI05TRTProfile":
        if name == "fp32":
            return cls.fp32()
        if name == "fp16_constrained":
            return cls.fp16_constrained()
        raise ValueError(f"Unknown PI0.5 TensorRT profile: {name}")

    def resolved(self) -> "SimplePI05TRTProfile":
        return SimplePI05TRTProfile(
            name=self.name,
            prefix_engine_path=self.prefix_engine_path.expanduser(),
            denoise_engine_path=self.denoise_engine_path.expanduser(),
            num_layers=self.num_layers,
        )

    def validate_files(self) -> None:
        if not self.prefix_engine_path.is_file():
            raise FileNotFoundError(f"Missing prefix_cache engine: {self.prefix_engine_path}")
        if not self.denoise_engine_path.is_file():
            raise FileNotFoundError(f"Missing denoise_step engine: {self.denoise_engine_path}")


class SimplePI05SplitTRTRuntime:
    """Minimal TensorRT implementation of PI0.5 sample_actions."""

    def __init__(self, profile: SimplePI05TRTProfile):
        self.profile = profile.resolved()
        self.profile.validate_files()
        self.prefix_engine = TorchTensorRTEngine(self.profile.prefix_engine_path)
        self.denoise_engine = TorchTensorRTEngine(self.profile.denoise_engine_path)
        self.cache_names = prefix_cache_tensor_names(self.profile.num_layers)
        self.denoise_input_names = denoise_step_input_names(self.profile.num_layers)
        self._validate_engine_io()

    def _validate_engine_io(self) -> None:
        validate_names(self.prefix_engine.input_names, list(PREFIX_CACHE_INPUT_NAMES), label="prefix_cache inputs")
        validate_names(self.prefix_engine.output_names, self.cache_names, label="prefix_cache outputs")
        validate_names(self.denoise_engine.input_names, self.denoise_input_names, label="denoise_step inputs")
        validate_names(self.denoise_engine.output_names, list(DENOISE_OUTPUT_NAMES), label="denoise_step outputs")

    @staticmethod
    def _cast_for_engine(engine: TorchTensorRTEngine, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        casted = {}
        for name in engine.input_names:
            tensor = inputs[name].contiguous()
            dtype = engine.tensor_dtypes[name]
            if tensor.dtype != dtype:
                tensor = tensor.to(dtype)
            casted[name] = tensor
        return casted

    @torch.no_grad()
    def sample_actions(
        self,
        policy,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        tokens: torch.Tensor,
        masks: torch.Tensor,
        noise: torch.Tensor,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        if len(images) != 2 or len(img_masks) != 2:
            raise ValueError(f"Expected two SO101 images/masks, got {len(images)} images and {len(img_masks)} masks")
        if num_steps is None:
            num_steps = policy.config.num_inference_steps

        prefix_inputs = {
            "image_0": images[0],
            "image_1": images[1],
            "img_mask_0": img_masks[0],
            "img_mask_1": img_masks[1],
            "tokens": tokens,
            "masks": masks,
        }
        prefix_outputs = self.prefix_engine(**self._cast_for_engine(self.prefix_engine, prefix_inputs))
        cache_inputs = {name: prefix_outputs[name].contiguous() for name in self.cache_names}

        dt = -1.0 / num_steps
        x_t = noise.contiguous()
        batch_size = tokens.shape[0]
        for step in range(num_steps):
            timestep = torch.full(
                (batch_size,),
                1.0 + step * dt,
                dtype=torch.float32,
                device=tokens.device,
            )
            denoise_inputs = dict(cache_inputs)
            denoise_inputs["x_t"] = x_t
            denoise_inputs["timestep"] = timestep
            denoise_outputs = self.denoise_engine(**self._cast_for_engine(self.denoise_engine, denoise_inputs))
            x_t = x_t + dt * denoise_outputs["v_t"]
        return x_t

    def patch_policy(self, policy):
        """Patch policy.model.sample_actions and return the original method."""
        original_sample_actions = policy.model.sample_actions

        @torch.no_grad()
        def trt_sample_actions(images, img_masks, tokens, masks, noise=None, num_steps=None, **kwargs):
            del kwargs
            if noise is None:
                action_shape = (tokens.shape[0], policy.config.chunk_size, policy.config.max_action_dim)
                noise_tensor = policy.model.sample_noise(action_shape, tokens.device)
            else:
                noise_tensor = noise
            return self.sample_actions(policy, images, img_masks, tokens, masks, noise_tensor, num_steps=num_steps)

        policy.model._original_sample_actions_for_simple_openpi_trt = original_sample_actions
        policy.model.sample_actions = trt_sample_actions
        return original_sample_actions

    def describe(self) -> dict:
        return {
            "profile": self.profile.name,
            "prefix_engine_path": str(self.profile.prefix_engine_path),
            "denoise_engine_path": str(self.profile.denoise_engine_path),
            "prefix_inputs": self.prefix_engine.input_names,
            "prefix_outputs": self.prefix_engine.output_names,
            "denoise_inputs": self.denoise_engine.input_names,
            "denoise_outputs": self.denoise_engine.output_names,
        }
