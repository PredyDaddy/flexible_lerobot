#!/usr/bin/env python3
"""ONNXRuntime 推理封装：ACT 单臂（normalized in → normalized out）

本模块只负责：
- 加载 ONNXRuntime Session
- 读取 checkpoint 的 mean/std stats（safetensors）
- 复刻 `ACTPolicy.select_action` 的 action queue 语义（chunk → 单步 action）
- 提供 numpy 版本的 select_action：raw obs -> real action（已反归一化）

注意：
- ONNX 输入/输出均为 mean/std 归一化空间；本模块会在内部完成 normalize/unnormalize。
- 相机顺序必须与导出时一致；默认优先读取导出元数据（export_metadata.json）。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from safetensors.numpy import load_file

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MeanStd:
    mean: np.ndarray
    std: np.ndarray

    def normalize(self, x: np.ndarray, *, eps: float) -> np.ndarray:
        return (x - self.mean) / (self.std + eps)

    def unnormalize(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid json format: expected dict at {path}")
    return data


def _get_visual_feature_keys_in_order(config_dict: dict[str, Any]) -> list[str]:
    input_features = config_dict.get("input_features", {})
    if not isinstance(input_features, dict):
        raise ValueError("Invalid config.json: input_features is not a dict.")
    return [k for k, v in input_features.items() if isinstance(v, dict) and v.get("type") == "VISUAL"]


def _find_step_cfg(steps: list[dict[str, Any]], *, registry_name: str) -> dict[str, Any]:
    for step in steps:
        if step.get("registry_name") == registry_name:
            cfg = step.get("config", {})
            if not isinstance(cfg, dict):
                raise ValueError(f"Invalid processor step config for '{registry_name}': expected dict.")
            return cfg
    raise KeyError(f"Processor step '{registry_name}' not found.")


def _load_mean_std(stats: dict[str, np.ndarray], *, feature_key: str) -> MeanStd:
    mean_key = f"{feature_key}.mean"
    std_key = f"{feature_key}.std"
    if mean_key not in stats or std_key not in stats:
        available = ", ".join(sorted(list(stats.keys())[:20]))
        raise KeyError(
            f"Missing mean/std for '{feature_key}'. Expected keys: '{mean_key}', '{std_key}'. "
            f"Available (first 20): {available}"
        )

    mean = np.asarray(stats[mean_key], dtype=np.float32)
    std = np.asarray(stats[std_key], dtype=np.float32)
    return MeanStd(mean=mean, std=std)


class ActOrtPolicy:
    """ACT 单臂 ORT 推理器（带队列语义）。

    输入：formatted observation（raw，未归一化）
      - observation.state: (state_dim,) float32
      - observation.images.<camera>: (H,W,3) uint8

    输出：single-step action（已反归一化）
      - (action_dim,) float32
    """

    def __init__(
        self,
        *,
        checkpoint: Path,
        onnx_path: Path | None = None,
        use_export_metadata: bool = True,
        export_metadata_path: Path | None = None,
        providers: list[str] | None = None,
    ) -> None:
        checkpoint = Path(checkpoint)
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint}")

        self.checkpoint = checkpoint

        self.export_metadata_path = Path(export_metadata_path) if export_metadata_path is not None else None
        self._export_metadata: dict[str, Any] | None = None
        if use_export_metadata:
            meta_path = (
                self.export_metadata_path
                if self.export_metadata_path is not None
                else checkpoint / "export_metadata.json"
            )
            if meta_path.is_file():
                self._export_metadata = _read_json(meta_path)

        self.onnx_path = Path(onnx_path) if onnx_path is not None else checkpoint / "act_single.onnx"
        if not self.onnx_path.is_file():
            raise FileNotFoundError(
                f"ONNX file not found: {self.onnx_path}. "
                "Run `python trt_act/export/export_single.py --checkpoint <ckpt>` first."
            )

        config_path = checkpoint / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"config.json not found under checkpoint: {config_path}")
        self._config = _read_json(config_path)

        temporal_ensemble_coeff = self._config.get("temporal_ensemble_coeff", None)
        if temporal_ensemble_coeff is not None:
            raise NotImplementedError(
                "temporal_ensemble_coeff is not supported in ORT runner yet; "
                "please export/use a checkpoint with temporal_ensemble_coeff=None."
            )

        self.chunk_size = int(self._config.get("chunk_size", 0))
        self.n_action_steps = int(self._config.get("n_action_steps", self.chunk_size))

        action_dim = int(self._config.get("output_features", {}).get("action", {}).get("shape", [0])[0])
        if action_dim <= 0:
            raise ValueError(f"Invalid action_dim parsed from config.json: {action_dim}")
        self.action_dim = action_dim

        self.camera_order_visual_keys = self._resolve_camera_order_visual_keys()
        if len(self.camera_order_visual_keys) != 2:
            raise ValueError(
                "This runner expects 2 cameras. " f"Got camera_order_visual_keys={self.camera_order_visual_keys}"
            )

        self._state_key = "observation.state"
        if self._state_key not in self._config.get("input_features", {}):
            raise ValueError("Expected 'observation.state' in config.json input_features.")
        self.state_dim = int(self._config["input_features"][self._state_key]["shape"][0])

        self._validate_visual_shapes()
        self._validate_export_metadata_shapes()

        self.eps = float(self._resolve_eps())

        pre_stats_path = checkpoint / "policy_preprocessor_step_3_normalizer_processor.safetensors"
        post_stats_path = checkpoint / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
        if not pre_stats_path.is_file():
            raise FileNotFoundError(f"Preprocessor stats not found: {pre_stats_path}")
        if not post_stats_path.is_file():
            raise FileNotFoundError(f"Postprocessor stats not found: {post_stats_path}")

        pre_stats = load_file(str(pre_stats_path))
        post_stats = load_file(str(post_stats_path))

        self._state_stats = _load_mean_std(pre_stats, feature_key=self._state_key)
        self._image_stats: dict[str, MeanStd] = {
            k: _load_mean_std(pre_stats, feature_key=k) for k in self.camera_order_visual_keys
        }
        self._action_stats = _load_mean_std(post_stats, feature_key="action")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        available_providers = ort.get_available_providers()
        if providers is None:
            providers = ["CPUExecutionProvider"]
        for p in providers:
            if p not in available_providers:
                extra_hint = ""
                if p == "CUDAExecutionProvider":
                    extra_hint = " (hint: install onnxruntime-gpu and ensure CUDA/cuDNN are available)"
                raise ValueError(
                    f"Requested provider '{p}' not available{extra_hint}. Available: {available_providers}"
                )

        self.session = ort.InferenceSession(str(self.onnx_path), sess_options=sess_options, providers=providers)
        input_names = [i.name for i in self.session.get_inputs()]
        output_names = [o.name for o in self.session.get_outputs()]

        required_inputs = {"obs_state_norm", "img0_norm", "img1_norm"}
        required_outputs = {"actions_norm"}
        if not required_inputs.issubset(set(input_names)):
            raise ValueError(f"ONNX inputs mismatch. Required={required_inputs}, got={input_names}")
        if not required_outputs.issubset(set(output_names)):
            raise ValueError(f"ONNX outputs mismatch. Required={required_outputs}, got={output_names}")

        self.reset()

        logger.info("Loaded ORT session: %s", self.onnx_path)
        logger.info("Providers: %s", providers)
        logger.info("Camera order: %s", self.camera_order_visual_keys)
        logger.info(
            "Shapes: state_dim=%d, image_hw=(%d,%d), chunk_size=%d, n_action_steps=%d, action_dim=%d",
            self.state_dim,
            self.image_h,
            self.image_w,
            self.chunk_size,
            self.n_action_steps,
            self.action_dim,
        )

    def reset(self) -> None:
        self._actions_norm_queue: np.ndarray | None = None
        self._queue_idx: int = 0

    def _resolve_camera_order_visual_keys(self) -> list[str]:
        if self._export_metadata is not None:
            keys = self._export_metadata.get("camera_order_visual_keys", None)
            if isinstance(keys, list) and all(isinstance(k, str) for k in keys):
                return list(keys)
        return _get_visual_feature_keys_in_order(self._config)

    def _resolve_eps(self) -> float:
        preprocessor_path = self.checkpoint / "policy_preprocessor.json"
        if not preprocessor_path.is_file():
            return 1e-8

        preprocessor = _read_json(preprocessor_path)
        steps = preprocessor.get("steps", [])
        if not isinstance(steps, list):
            return 1e-8

        try:
            cfg = _find_step_cfg(steps, registry_name="normalizer_processor")
        except Exception:
            return 1e-8

        eps = cfg.get("eps", 1e-8)
        return float(eps)

    def _validate_visual_shapes(self) -> None:
        input_features = self._config.get("input_features", {})
        c0, h0, w0 = input_features[self.camera_order_visual_keys[0]]["shape"]
        c1, h1, w1 = input_features[self.camera_order_visual_keys[1]]["shape"]

        if int(c0) != 3 or int(c1) != 3:
            raise ValueError(
                f"Expected 3-channel images, got C=({c0},{c1}) "
                f"for ({self.camera_order_visual_keys[0]},{self.camera_order_visual_keys[1]})"
            )
        if (int(h0), int(w0)) != (int(h1), int(w1)):
            raise ValueError(
                "Expected both cameras to have the same (H,W), got "
                f"{self.camera_order_visual_keys[0]}=({h0},{w0}) and {self.camera_order_visual_keys[1]}=({h1},{w1})"
            )

        self.image_h, self.image_w = int(h0), int(w0)

    def _validate_export_metadata_shapes(self) -> None:
        if self._export_metadata is None:
            return

        shapes = self._export_metadata.get("shapes", None)
        if not isinstance(shapes, dict):
            return

        expected = {
            "obs_state_norm": [1, self.state_dim],
            "img0_norm": [1, 3, self.image_h, self.image_w],
            "img1_norm": [1, 3, self.image_h, self.image_w],
            "actions_norm": [1, self.chunk_size, self.action_dim],
        }
        for key, exp in expected.items():
            got = shapes.get(key, None)
            if got is None:
                continue
            if list(got) != exp:
                raise ValueError(
                    f"export_metadata.json shapes mismatch for '{key}': expected {exp}, got {got}"
                )

    def _preprocess_observation(self, observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self._state_key not in observation:
            raise KeyError(f"Missing required key in observation: '{self._state_key}'")

        state = np.asarray(observation[self._state_key], dtype=np.float32)
        if state.shape != (self.state_dim,):
            raise ValueError(f"Invalid state shape: expected ({self.state_dim},), got {state.shape}")

        obs_state = state[None, :]
        obs_state_norm = self._state_stats.normalize(obs_state, eps=self.eps).astype(np.float32, copy=False)
        obs_state_norm = np.ascontiguousarray(obs_state_norm)

        imgs_norm: list[np.ndarray] = []
        for key in self.camera_order_visual_keys:
            if key not in observation:
                raise KeyError(f"Missing required key in observation: '{key}'")

            img = np.asarray(observation[key])
            if img.dtype != np.uint8:
                raise TypeError(f"Expected image dtype=uint8 for '{key}', got {img.dtype}")
            if img.shape != (self.image_h, self.image_w, 3):
                raise ValueError(
                    f"Invalid image shape for '{key}': expected ({self.image_h},{self.image_w},3), got {img.shape}"
                )

            img_f32 = img.astype(np.float32) / 255.0
            img_chw = np.ascontiguousarray(img_f32.transpose(2, 0, 1))
            img_bchw = img_chw[None, :, :, :]
            img_norm = self._image_stats[key].normalize(img_bchw, eps=self.eps).astype(np.float32, copy=False)
            imgs_norm.append(np.ascontiguousarray(img_norm))

        return {
            "obs_state_norm": obs_state_norm,
            "img0_norm": imgs_norm[0],
            "img1_norm": imgs_norm[1],
        }

    def _infer_actions_norm_chunk(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        ort_inputs = self._preprocess_observation(observation)
        outputs = self.session.run(["actions_norm"], ort_inputs)
        actions_norm = np.asarray(outputs[0], dtype=np.float32)
        if actions_norm.ndim != 3 or actions_norm.shape[0] != 1:
            raise ValueError(f"Unexpected actions_norm shape: {actions_norm.shape} (expected (1,S,A))")
        return actions_norm

    def select_action(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        """复刻 `ACTPolicy.select_action` 语义：每次返回 1 步 action。

        Returns:
            action: (action_dim,) float32，已反归一化
        """
        if self._actions_norm_queue is None or self._queue_idx >= self._actions_norm_queue.shape[0]:
            actions_norm_chunk = self._infer_actions_norm_chunk(observation)
            actions_norm_steps = actions_norm_chunk[0, : self.n_action_steps, :]
            if actions_norm_steps.shape != (self.n_action_steps, self.action_dim):
                raise ValueError(
                    f"Unexpected sliced actions_norm shape: {actions_norm_steps.shape} "
                    f"(expected ({self.n_action_steps},{self.action_dim}))"
                )

            self._actions_norm_queue = np.ascontiguousarray(actions_norm_steps)
            self._queue_idx = 0

        action_norm = self._actions_norm_queue[self._queue_idx]
        self._queue_idx += 1

        action = self._action_stats.unnormalize(action_norm).astype(np.float32, copy=False)
        return np.asarray(action, dtype=np.float32)
