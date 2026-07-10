from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
import time
from typing import Any

import numpy as np


class BaseModelAdapter(ABC):
    @classmethod
    @abstractmethod
    def from_config(cls, model_cfg: Any) -> "BaseModelAdapter":
        raise NotImplementedError

    @abstractmethod
    def infer_actions(self, request: dict) -> list[list[float]]:
        raise NotImplementedError


def _first_available(request: dict, keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in request and request[key] is not None:
            return request[key]
    return None


def _extract_state(request: dict, state_dim: int) -> np.ndarray:
    value = _first_available(
        request,
        (
            "state",
            "observation.state",
            "action",
            "actions",
        ),
    )
    if value is None:
        raise ValueError("Request payload must contain state, observation.state, action, or actions.")

    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[0]
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D SO101 state or [T,D] state history, got shape={arr.shape}")
    if arr.shape[0] < state_dim:
        raise ValueError(f"Expected state_dim >= {state_dim}, got shape={arr.shape}")
    return np.ascontiguousarray(arr[:state_dim], dtype=np.float32)


def _as_hwc_uint8(image: Any) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D image, got shape={arr.shape}")
    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.moveaxis(arr, 0, -1)
    if np.issubdtype(arr.dtype, np.floating):
        scale = 255.0 if float(np.nanmax(arr)) <= 1.5 else 1.0
        arr = np.clip(arr * scale, 0, 255).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8, copy=False)
    return np.ascontiguousarray(arr)


def _decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    import cv2

    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode image bytes.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _extract_image(request: dict, canonical_key: str, aliases: tuple[str, ...]) -> np.ndarray:
    for key in (canonical_key, *aliases):
        if key in request:
            return _as_hwc_uint8(request[key])

    images = request.get("images") or {}
    for key in (canonical_key, *aliases):
        if key not in images:
            continue
        value = images[key]
        if isinstance(value, (bytes, bytearray, memoryview)):
            return _decode_image_bytes(bytes(value))
        return _as_hwc_uint8(value)

    raise ValueError(f"Request payload missing image {canonical_key!r}; aliases={aliases!r}")


def _normalize_action_array(actions: Any, action_dim: int) -> list[list[float]]:
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected action chunk shape=(horizon, dim), got shape={arr.shape}")
    if arr.shape[1] < action_dim:
        raise ValueError(f"Expected action_dim >= {action_dim}, got shape={arr.shape}")
    return np.ascontiguousarray(arr[:, :action_dim], dtype=np.float32).tolist()


@dataclass
class MockSO101Adapter(BaseModelAdapter):
    action_horizon: int
    valid_action_num: int
    state_dim: int
    action_dim: int

    @classmethod
    def from_config(cls, model_cfg: Any) -> "MockSO101Adapter":
        return cls(
            action_horizon=int(model_cfg.action_horizon),
            valid_action_num=int(model_cfg.valid_action_num),
            state_dim=int(model_cfg.state_dim),
            action_dim=int(model_cfg.action_dim),
        )

    def infer_actions(self, request: dict) -> list[list[float]]:
        state = _extract_state(request, self.state_dim)
        horizon = max(1, min(self.valid_action_num, self.action_horizon))
        action = np.zeros((horizon, self.action_dim), dtype=np.float32)
        action[:, : min(self.action_dim, state.shape[0])] = state[: min(self.action_dim, state.shape[0])]
        return action.tolist()


@dataclass
class OpenPISO101Adapter(BaseModelAdapter):
    checkpoint_dir: str
    default_prompt: str
    dataset_format: str
    asset_id: str | None
    pytorch_device: str | None
    valid_action_num: int
    state_dim: int
    action_dim: int

    @classmethod
    def from_config(cls, model_cfg: Any) -> "OpenPISO101Adapter":
        adapter = cls(
            checkpoint_dir=str(model_cfg.checkpoint_dir),
            default_prompt=str(model_cfg.default_prompt),
            dataset_format=str(model_cfg.dataset_format),
            asset_id=model_cfg.asset_id,
            pytorch_device=model_cfg.pytorch_device,
            valid_action_num=int(model_cfg.valid_action_num),
            state_dim=int(model_cfg.state_dim),
            action_dim=int(model_cfg.action_dim),
        )
        adapter._load_policy()
        return adapter

    def _load_policy(self) -> None:
        if not self.checkpoint_dir:
            raise ValueError("model.checkpoint_dir is required for adapter=openpi_so101")

        from pathlib import Path

        checkpoint_path = Path(self.checkpoint_dir).expanduser()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"SO101 checkpoint does not exist: {checkpoint_path}")

        from openpi_so101 import config as so101_config
        from openpi_so101 import policy as so101_policy
        from openpi_so101 import runtime

        runtime.bootstrap()

        from openpi.policies import policy_config

        asset_id = self.asset_id
        if asset_id is None and self.dataset_format == "v21":
            asset_id = "desk_cleanup_v1/eraser_cup_multi_task_v21_full"

        config = so101_config.make_config(
            exp_name="realtimevla_backend_serve",
            asset_id=asset_id,
        )
        so101_config.register_config(config)
        start = time.perf_counter()
        self._policy = policy_config.create_trained_policy(
            config,
            str(checkpoint_path),
            repack_transforms=so101_policy.SO101_INFERENCE_REPACK_TRANSFORMS,
            default_prompt=self.default_prompt,
            pytorch_device=self.pytorch_device,
        )
        self._load_time_s = time.perf_counter() - start
        print(
            "[OpenPISO101Adapter] loaded checkpoint "
            f"{checkpoint_path} in {self._load_time_s:.2f}s "
            f"asset_id={asset_id!r} pytorch_device={self.pytorch_device!r}"
        )

    def _build_policy_observation(self, request: dict) -> dict:
        prompt = request.get("prompt") or self.default_prompt
        return {
            "observation.images.top": _extract_image(
                request,
                "observation.images.top",
                ("top", "high", "base_0_rgb", "observation/image"),
            ),
            "observation.images.wrist": _extract_image(
                request,
                "observation.images.wrist",
                ("wrist", "left_hand", "left_wrist_0_rgb", "observation/wrist_image"),
            ),
            "observation.state": _extract_state(request, self.state_dim),
            "prompt": prompt,
        }

    def infer_actions(self, request: dict) -> list[list[float]]:
        observation = self._build_policy_observation(request)
        result = self._policy.infer(observation)
        actions = result.get("actions") if isinstance(result, dict) else result
        action_list = _normalize_action_array(actions, self.action_dim)
        if self.valid_action_num > 0:
            action_list = action_list[: self.valid_action_num]
        return action_list

