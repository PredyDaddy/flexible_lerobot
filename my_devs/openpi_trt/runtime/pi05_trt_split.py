#!/usr/bin/env python

"""Compatibility wrapper for the compact PI0.5 split TensorRT runtime.

`my_devs/vla_engineering` imports this module directly, so keep this stable
while the real implementation lives in `runtime.simple_pi05_split`.
"""

from __future__ import annotations

from pathlib import Path

import torch

from runtime.config import PI05SplitTRTConfig
from runtime.simple_pi05_split import SimplePI05SplitTRTRuntime, SimplePI05TRTProfile


class PI05TensorRTSplitRuntime(SimplePI05SplitTRTRuntime):
    """Backward-compatible name for the production split TensorRT runtime."""

    def __init__(
        self,
        prefix_engine_path: str | Path | PI05SplitTRTConfig,
        denoise_engine_path: str | Path | None = None,
        num_layers: int = 18,
        *,
        validate_io: bool = True,
    ):
        del validate_io
        if isinstance(prefix_engine_path, PI05SplitTRTConfig):
            if denoise_engine_path is not None:
                raise ValueError("denoise_engine_path must be None when passing PI05SplitTRTConfig")
            config = prefix_engine_path
            profile_name = "fp16_constrained" if "fp16" in config.denoise_engine_path.name else "fp32"
            profile = SimplePI05TRTProfile(
                name=profile_name,
                prefix_engine_path=config.prefix_engine_path,
                denoise_engine_path=config.denoise_engine_path,
                num_layers=config.num_layers,
            )
            self.config = config
        else:
            if denoise_engine_path is None:
                raise ValueError("denoise_engine_path is required when not passing PI05SplitTRTConfig")
            profile_name = "fp16_constrained" if "fp16" in Path(denoise_engine_path).name else "fp32"
            profile = SimplePI05TRTProfile(
                name=profile_name,
                prefix_engine_path=Path(prefix_engine_path),
                denoise_engine_path=Path(denoise_engine_path),
                num_layers=num_layers,
            )
            self.config = PI05SplitTRTConfig.from_paths(
                prefix_engine_path,
                denoise_engine_path,
                num_layers=num_layers,
            )
        super().__init__(profile)

    @torch.no_grad()
    def __call__(
        self,
        policy,
        images,
        img_masks,
        tokens,
        masks,
        noise,
        num_steps: int | None = None,
    ):
        return self.sample_actions(policy, images, img_masks, tokens, masks, noise, num_steps=num_steps)

    def describe(self) -> dict:
        description = super().describe()
        return {
            "config": self.config.to_dict(),
            "profile": description["profile"],
            "prefix_engine": self.prefix_engine.describe(),
            "denoise_engine": self.denoise_engine.describe(),
            "num_layers": self.profile.num_layers,
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
    def trt_sample_actions(images, img_masks, tokens, masks, noise=None, num_steps=None, **kwargs):
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
        return runtime(policy, images, img_masks, tokens, masks, noise_tensor, num_steps=num_steps)

    policy.model._original_sample_actions_for_openpi_split_trt = original_sample_actions
    policy.model.sample_actions = trt_sample_actions
    return runtime
