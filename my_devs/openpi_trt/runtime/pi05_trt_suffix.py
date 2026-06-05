#!/usr/bin/env python

"""Patch LeRobot PI0.5 suffix embedding with a TensorRT engine."""

from __future__ import annotations

from pathlib import Path

import torch

from runtime.trt_engine import TorchTensorRTEngine


def patch_embed_suffix_with_trt(policy, engine_path: str | Path) -> TorchTensorRTEngine:
    """Replace `policy.model.embed_suffix(...)` with a TensorRT-backed implementation.

    The engine is expected to implement the split-graph boundary:

    ```text
    noisy_actions + timestep -> suffix_embs + adarms_cond
    ```

    The masks are deterministic for PI0.5 and are reconstructed in Python to
    match `PI05Pytorch.embed_suffix(...)`.
    """
    engine = TorchTensorRTEngine(engine_path)
    original_embed_suffix = policy.model.embed_suffix

    def trt_embed_suffix(noisy_actions: torch.Tensor, timestep: torch.Tensor):
        inputs = {
            "noisy_actions": noisy_actions.contiguous(),
            "timestep": timestep.contiguous(),
        }
        for name in engine.input_names:
            expected_dtype = engine.tensor_dtypes[name]
            if inputs[name].dtype != expected_dtype:
                inputs[name] = inputs[name].to(expected_dtype)
        outputs = engine(**inputs)
        suffix_embs = outputs["suffix_embs"]
        adarms_cond = outputs["adarms_cond"]

        bsize = noisy_actions.shape[0]
        chunk_size = policy.config.chunk_size
        suffix_pad_masks = torch.ones(bsize, chunk_size, dtype=torch.bool, device=noisy_actions.device)
        suffix_att_masks = torch.tensor(
            [1] + ([0] * (chunk_size - 1)),
            dtype=suffix_embs.dtype,
            device=noisy_actions.device,
        )
        suffix_att_masks = suffix_att_masks[None, :].expand(bsize, chunk_size)
        return suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond

    policy.model._original_embed_suffix_for_openpi_trt = original_embed_suffix
    policy.model.embed_suffix = trt_embed_suffix
    return engine

