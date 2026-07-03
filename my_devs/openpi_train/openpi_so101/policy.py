from __future__ import annotations

import dataclasses

import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.moveaxis(image, 0, -1)
    return image


@dataclasses.dataclass(frozen=True)
class SO101Inputs(transforms.DataTransformFn):
    """Map SO101 top/wrist observations into OpenPI's canonical image keys."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        inputs = {
            "state": np.asarray(data["observation/state"], dtype=np.float32),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class SO101Outputs(transforms.DataTransformFn):
    """Return the first six action dimensions used by the SO101 follower."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :6], dtype=np.float32)}


SO101_REPACK_TRANSFORMS = transforms.Group(
    inputs=[
        transforms.RepackTransform(
            {
                "observation/image": "observation.images.top",
                "observation/wrist_image": "observation.images.wrist",
                "observation/state": "observation.state",
                "actions": "action",
                "prompt": "prompt",
            }
        )
    ]
)


SO101_INFERENCE_REPACK_TRANSFORMS = transforms.Group(
    inputs=[
        transforms.RepackTransform(
            {
                "observation/image": "observation.images.top",
                "observation/wrist_image": "observation.images.wrist",
                "observation/state": "observation.state",
                "prompt": "prompt",
            }
        )
    ]
)
