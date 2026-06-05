#!/usr/bin/env python

"""Run LeRobot PI0.5 sample_actions with prefix-cache and denoise TensorRT engines."""

from __future__ import annotations

from pathlib import Path

import torch

from runtime.trt_engine import TorchTensorRTEngine


def prefix_cache_tensor_names(num_layers: int = 18) -> list[str]:
    names = ["prefix_pad_masks"]
    for layer_idx in range(num_layers):
        names.append(f"past_key_values.{layer_idx}.key")
        names.append(f"past_key_values.{layer_idx}.value")
    return names


def denoise_step_input_names(num_layers: int = 18) -> list[str]:
    names = prefix_cache_tensor_names(num_layers)
    names.extend(["x_t", "timestep"])
    return names


class PI05TensorRTSplitRuntime:
    """TensorRT implementation of PI0.5 prefix cache + Python denoise loop."""

    def __init__(self, prefix_engine_path: str | Path, denoise_engine_path: str | Path, num_layers: int = 18):
        self.prefix_engine = TorchTensorRTEngine(prefix_engine_path)
        self.denoise_engine = TorchTensorRTEngine(denoise_engine_path)
        self.num_layers = num_layers
        self.cache_names = prefix_cache_tensor_names(num_layers)
        self.denoise_input_names = denoise_step_input_names(num_layers)

    def _cast_inputs_for_engine(
        self, engine: TorchTensorRTEngine, inputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        casted = {}
        for name in engine.input_names:
            tensor = inputs[name].contiguous()
            expected_dtype = engine.tensor_dtypes[name]
            if tensor.dtype != expected_dtype:
                tensor = tensor.to(expected_dtype)
            casted[name] = tensor
        return casted

    @torch.no_grad()
    def __call__(
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
            "image_0": images[0].contiguous(),
            "image_1": images[1].contiguous(),
            "img_mask_0": img_masks[0].contiguous(),
            "img_mask_1": img_masks[1].contiguous(),
            "tokens": tokens.contiguous(),
            "masks": masks.contiguous(),
        }
        prefix_outputs = self.prefix_engine(**self._cast_inputs_for_engine(self.prefix_engine, prefix_inputs))
        cache_inputs = {name: prefix_outputs[name].contiguous() for name in self.cache_names}

        dt = -1.0 / num_steps
        x_t = noise.contiguous()
        bsize = tokens.shape[0]
        for step in range(num_steps):
            time = 1.0 + step * dt
            timestep = torch.full((bsize,), time, dtype=torch.float32, device=tokens.device)
            denoise_inputs = dict(cache_inputs)
            denoise_inputs["x_t"] = x_t.contiguous()
            denoise_inputs["timestep"] = timestep.contiguous()
            denoise_outputs = self.denoise_engine(
                **self._cast_inputs_for_engine(self.denoise_engine, denoise_inputs)
            )
            v_t = denoise_outputs["v_t"]
            x_t = x_t + dt * v_t
        return x_t

    def describe(self) -> dict:
        return {
            "prefix_engine": self.prefix_engine.describe(),
            "denoise_engine": self.denoise_engine.describe(),
            "num_layers": self.num_layers,
        }


def patch_sample_actions_with_split_trt(
    policy,
    prefix_engine_path: str | Path,
    denoise_engine_path: str | Path,
) -> PI05TensorRTSplitRuntime:
    """Replace `policy.model.sample_actions(...)` with split TensorRT inference."""
    runtime = PI05TensorRTSplitRuntime(prefix_engine_path, denoise_engine_path)
    original_sample_actions = policy.model.sample_actions

    @torch.no_grad()
    def trt_sample_actions(
        images,
        img_masks,
        tokens,
        masks,
        noise=None,
        num_steps=None,
        **kwargs,
    ):
        del kwargs
        if noise is None:
            actions_shape = (
                tokens.shape[0],
                policy.config.chunk_size,
                policy.config.max_action_dim,
            )
            noise_tensor = policy.model.sample_noise(actions_shape, tokens.device)
        else:
            noise_tensor = noise
        return runtime(
            policy,
            images,
            img_masks,
            tokens,
            masks,
            noise_tensor,
            num_steps=num_steps,
        )

    policy.model._original_sample_actions_for_openpi_split_trt = original_sample_actions
    policy.model.sample_actions = trt_sample_actions
    return runtime
