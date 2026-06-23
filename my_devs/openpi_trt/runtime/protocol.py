#!/usr/bin/env python

"""Stable TensorRT tensor-name protocol for LeRobot PI0.5 split engines."""

from __future__ import annotations

PI05_PREFIX_CACHE_LAYERS = 18

PREFIX_CACHE_INPUT_NAMES = (
    "image_0",
    "image_1",
    "img_mask_0",
    "img_mask_1",
    "tokens",
    "masks",
)

DENOISE_EXTRA_INPUT_NAMES = ("x_t", "timestep")
DENOISE_OUTPUT_NAMES = ("v_t",)


def prefix_cache_tensor_names(num_layers: int = PI05_PREFIX_CACHE_LAYERS) -> list[str]:
    """Return the flattened prefix-cache output names expected by split TRT."""
    names = ["prefix_pad_masks"]
    for layer_idx in range(num_layers):
        names.append(f"past_key_values.{layer_idx}.key")
        names.append(f"past_key_values.{layer_idx}.value")
    return names


def denoise_step_input_names(num_layers: int = PI05_PREFIX_CACHE_LAYERS) -> list[str]:
    """Return denoise_step engine inputs: flattened prefix cache plus x_t/timestep."""
    return [*prefix_cache_tensor_names(num_layers), *DENOISE_EXTRA_INPUT_NAMES]


def validate_names(actual: list[str], expected: list[str], *, label: str) -> None:
    """Validate TensorRT I/O names with a message that is useful during deployment."""
    missing = [name for name in expected if name not in actual]
    unexpected = [name for name in actual if name not in expected]
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unexpected:
            details.append(f"unexpected={unexpected}")
        raise ValueError(f"{label} TensorRT I/O mismatch: " + ", ".join(details))
