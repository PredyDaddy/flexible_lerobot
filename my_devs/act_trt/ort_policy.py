from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from torch import Tensor

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

from my_devs.act_trt.common import load_json, load_model_spec, resolve_checkpoint


def _clone_onnx_output_to_torch(array: np.ndarray, *, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(array, dtype=np.float32)).to(device=device, dtype=torch.float32)


class ActOrtPolicyAdapter:
    """ACT ONNXRuntime adapter operating on normalized tensors.

    This mirrors the current repository's inference boundary:
    `prepare_observation_for_inference -> preprocessor -> policy.select_action -> postprocessor`

    Inputs to `select_action` / `predict_action_chunk` are expected to be normalized torch tensors using the
    current checkpoint's feature names, not raw numpy observations.
    """

    def __init__(
        self,
        *,
        checkpoint: str | Path,
        onnx_path: str | Path,
        config: PreTrainedConfig | None = None,
        export_metadata_path: str | Path | None = None,
        providers: list[str] | None = None,
    ) -> None:
        self.checkpoint = resolve_checkpoint(checkpoint)
        self.onnx_path = Path(onnx_path).expanduser().resolve()
        if not self.onnx_path.is_file():
            raise FileNotFoundError(f"ONNX file not found: {self.onnx_path}")

        loaded_config = config if config is not None else PreTrainedConfig.from_pretrained(str(self.checkpoint))
        if loaded_config.type != "act":
            raise ValueError(f"Expected ACT config, got {loaded_config.type!r}")
        self.config: ACTConfig = loaded_config  # type: ignore[assignment]
        self.spec = load_model_spec(self.checkpoint)

        metadata_path = (
            Path(export_metadata_path).expanduser().resolve()
            if export_metadata_path is not None
            else self.onnx_path.parent / "export_metadata.json"
        )
        self.export_metadata = load_json(metadata_path) if metadata_path.is_file() else None

        self.visual_keys = self._resolve_visual_keys()
        self._validate_export_metadata_shapes()

        available_providers = ort.get_available_providers()
        if providers is None:
            providers = ["CPUExecutionProvider"]
        for provider in providers:
            if provider not in available_providers:
                raise ValueError(f"Requested provider {provider!r} not available. Available: {available_providers}")

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(self.onnx_path), sess_options=session_options, providers=providers)

        input_names = {item.name for item in self.session.get_inputs()}
        output_names = {item.name for item in self.session.get_outputs()}
        expected_inputs = {"obs_state_norm", "img0_norm", "img1_norm"}
        expected_outputs = {"actions_norm"}
        if not expected_inputs.issubset(input_names):
            raise ValueError(f"ONNX inputs mismatch. Required={expected_inputs}, got={sorted(input_names)}")
        if not expected_outputs.issubset(output_names):
            raise ValueError(f"ONNX outputs mismatch. Required={expected_outputs}, got={sorted(output_names)}")

        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(
                self.config.temporal_ensemble_coeff,
                self.config.chunk_size,
            )
        self.reset()

    def _resolve_visual_keys(self) -> list[str]:
        config_keys = list(getattr(self.config, "image_features", []) or [])
        if config_keys:
            return config_keys
        if self.export_metadata is not None:
            metadata_keys = self.export_metadata.get("camera_order_visual_keys", None)
            if isinstance(metadata_keys, list) and all(isinstance(item, str) for item in metadata_keys):
                return list(metadata_keys)
        return list(self.spec.visual_keys)

    def _validate_export_metadata_shapes(self) -> None:
        if self.export_metadata is None:
            return
        shapes = self.export_metadata.get("shapes", None)
        if not isinstance(shapes, dict):
            return
        expected = {
            "obs_state_norm": [1, self.spec.state_dim],
            "img0_norm": [1, 3, self.spec.image_height, self.spec.image_width],
            "img1_norm": [1, 3, self.spec.image_height, self.spec.image_width],
            "actions_norm": [1, self.config.chunk_size, self.spec.action_dim],
        }
        for key, expected_shape in expected.items():
            got = shapes.get(key, None)
            if got is None:
                continue
            if list(got) != expected_shape:
                raise ValueError(f"export_metadata shape mismatch for {key!r}: expected {expected_shape}, got {got}")

    def reset(self) -> None:
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue: deque[Tensor] = deque([], maxlen=self.config.n_action_steps)

    def _get_images(self, batch: dict[str, Tensor]) -> list[Tensor]:
        if OBS_IMAGES in batch:
            images = batch[OBS_IMAGES]
            if not isinstance(images, list) or len(images) != len(self.visual_keys):
                raise ValueError(f"Expected {len(self.visual_keys)} tensors in batch[{OBS_IMAGES!r}]")
            return images
        return [batch[key] for key in self.visual_keys]

    def _build_feed_dict(self, batch: dict[str, Tensor]) -> tuple[dict[str, np.ndarray], torch.device]:
        if OBS_STATE not in batch:
            raise KeyError(f"Missing required key: {OBS_STATE}")

        obs_state = batch[OBS_STATE]
        if not isinstance(obs_state, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor for {OBS_STATE}, got {type(obs_state)}")
        if tuple(obs_state.shape) != self.spec.obs_state_shape:
            raise ValueError(f"Unexpected {OBS_STATE} shape: expected {self.spec.obs_state_shape}, got {tuple(obs_state.shape)}")

        images = self._get_images(batch)
        for index, image in enumerate(images):
            if tuple(image.shape) != self.spec.image_shape:
                raise ValueError(
                    f"Unexpected image shape for {self.visual_keys[index]}: "
                    f"expected {self.spec.image_shape}, got {tuple(image.shape)}"
                )

        device = obs_state.device
        feed_dict = {
            "obs_state_norm": np.ascontiguousarray(obs_state.detach().to("cpu", dtype=torch.float32).numpy()),
            "img0_norm": np.ascontiguousarray(images[0].detach().to("cpu", dtype=torch.float32).numpy()),
            "img1_norm": np.ascontiguousarray(images[1].detach().to("cpu", dtype=torch.float32).numpy()),
        }
        return feed_dict, device

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        feed_dict, device = self._build_feed_dict(batch)
        outputs = self.session.run(["actions_norm"], feed_dict)
        actions_norm = np.asarray(outputs[0], dtype=np.float32)
        if tuple(actions_norm.shape) != self.spec.action_shape:
            raise ValueError(f"Unexpected actions_norm shape: expected {self.spec.action_shape}, got {actions_norm.shape}")
        return _clone_onnx_output_to_torch(actions_norm, device=device)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        if self.config.temporal_ensemble_coeff is not None:
            actions = self.predict_action_chunk(batch)
            return self.temporal_ensembler.update(actions)

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()
