from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from torch import Tensor

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

from my_devs.act_trt.common import load_json, load_model_spec, resolve_checkpoint
from my_devs.act_trt.trt_runtime import TensorRTRunner, trt_dtype_to_torch_dtype


def _clone_trt_output_to_torch(tensor: Tensor, *, device: torch.device) -> Tensor:
    return tensor.detach().to(device=device, dtype=torch.float32).contiguous()


class ActTrtPolicyAdapter:
    """ACT TensorRT adapter operating on normalized tensors.

    This mirrors the current repository's inference boundary:
    `prepare_observation_for_inference -> preprocessor -> policy.select_action -> postprocessor`

    Inputs to `select_action` / `predict_action_chunk` are expected to be normalized torch tensors using the
    current checkpoint's feature names, not raw numpy observations.
    """

    def __init__(
        self,
        *,
        checkpoint: str | Path,
        engine_path: str | Path,
        config: PreTrainedConfig | None = None,
        export_metadata_path: str | Path | None = None,
        device: str = "cuda:0",
    ) -> None:
        self.checkpoint = resolve_checkpoint(checkpoint)
        self.engine_path = Path(engine_path).expanduser().resolve()
        if not self.engine_path.is_file():
            raise FileNotFoundError(f"TensorRT engine not found: {self.engine_path}")

        loaded_config = config if config is not None else PreTrainedConfig.from_pretrained(str(self.checkpoint))
        if loaded_config.type != "act":
            raise ValueError(f"Expected ACT config, got {loaded_config.type!r}")
        self.config: ACTConfig = loaded_config  # type: ignore[assignment]
        self._validate_runtime_config()
        self.spec = load_model_spec(self.checkpoint)

        metadata_path = (
            Path(export_metadata_path).expanduser().resolve()
            if export_metadata_path is not None
            else self.engine_path.parent / "export_metadata.json"
        )
        self.export_metadata = load_json(metadata_path) if metadata_path.is_file() else None

        self.visual_keys = self._resolve_visual_keys()
        if len(self.visual_keys) != 2:
            raise ValueError(f"Expected exactly 2 visual feature keys, got {self.visual_keys}")
        self._validate_export_metadata_shapes()

        self.runner = TensorRTRunner(engine_path=self.engine_path, device=device)
        self._validate_engine_io()

        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(
                self.config.temporal_ensemble_coeff,
                self.config.chunk_size,
            )
        self.reset()

    def _validate_runtime_config(self) -> None:
        if self.config.temporal_ensemble_coeff is not None and self.config.n_action_steps != 1:
            raise NotImplementedError(
                "ACT temporal ensembling requires n_action_steps == 1 in ActTrtPolicyAdapter. "
                f"Got n_action_steps={self.config.n_action_steps}."
            )

    def _get_config_visual_keys(self) -> list[str]:
        image_features = getattr(self.config, "image_features", None)
        if image_features is None:
            return []
        if isinstance(image_features, Mapping):
            return list(image_features.keys())
        if isinstance(image_features, Sequence) and not isinstance(image_features, (str, bytes)):
            if not all(isinstance(item, str) for item in image_features):
                raise TypeError(
                    "ACT TRT config.image_features must be an ordered collection of string feature keys. "
                    f"Got {image_features!r}"
                )
            return list(image_features)
        raise TypeError(
            "ACT TRT config.image_features must preserve visual feature order via a mapping or sequence. "
            f"Got {type(image_features)!r}"
        )

    def _get_metadata_visual_keys(self) -> list[str]:
        if self.export_metadata is None:
            return []

        metadata_value = self.export_metadata.get("camera_order_visual_keys", None)
        if metadata_value is None:
            return []
        if not isinstance(metadata_value, list) or not all(isinstance(item, str) for item in metadata_value):
            raise TypeError(
                "ACT TRT export_metadata.camera_order_visual_keys must be a list of strings when present. "
                f"Got {metadata_value!r}"
            )
        return list(metadata_value)

    def _resolve_visual_keys(self) -> list[str]:
        spec_keys = list(self.spec.visual_keys)
        config_keys = self._get_config_visual_keys()
        metadata_keys = self._get_metadata_visual_keys()

        if config_keys and metadata_keys and config_keys != metadata_keys:
            raise ValueError(
                "ACT TRT visual key mismatch: config.image_features and "
                "export_metadata.camera_order_visual_keys must match exactly. "
                f"config={config_keys}, metadata={metadata_keys}"
            )

        if config_keys:
            if config_keys != spec_keys:
                raise ValueError(
                    "ACT TRT visual key mismatch: config.image_features must match the checkpoint visual feature "
                    f"order {spec_keys}. Got {config_keys}"
                )
            return config_keys
        if metadata_keys:
            if metadata_keys != spec_keys:
                raise ValueError(
                    "ACT TRT visual key mismatch: export_metadata.camera_order_visual_keys must match the "
                    f"checkpoint visual feature order {spec_keys}. Got {metadata_keys}"
                )
            return metadata_keys
        return list(self.spec.visual_keys)

    def _validate_export_metadata_shapes(self) -> None:
        if self.export_metadata is None:
            return
        shapes = self.export_metadata.get("shapes", None)
        if not isinstance(shapes, dict):
            return
        expected = self._expected_shapes()
        for key, expected_shape in expected.items():
            got = shapes.get(key, None)
            if got is None:
                continue
            if list(got) != expected_shape:
                raise ValueError(
                    f"export_metadata shape mismatch for {key!r}: expected {expected_shape}, got {got}"
                )

    def _expected_shapes(self) -> dict[str, list[int]]:
        return {
            "obs_state_norm": [1, self.spec.state_dim],
            "img0_norm": [1, 3, self.spec.image_height, self.spec.image_width],
            "img1_norm": [1, 3, self.spec.image_height, self.spec.image_width],
            "actions_norm": [1, self.config.chunk_size, self.spec.action_dim],
        }

    def _shape_matches_expected(self, actual: list[int], expected: list[int]) -> bool:
        if len(actual) != len(expected):
            return False
        return all(
            actual_dim < 0 or actual_dim == expected_dim
            for actual_dim, expected_dim in zip(actual, expected)
        )

    def _validate_engine_io(self) -> None:
        expected_inputs = {"obs_state_norm", "img0_norm", "img1_norm"}
        expected_outputs = {"actions_norm"}

        input_names = set(self.runner.input_names)
        output_names = set(self.runner.output_names)
        if not expected_inputs.issubset(input_names):
            raise ValueError(
                f"TensorRT inputs mismatch. Required={expected_inputs}, got={sorted(input_names)}"
            )
        if not expected_outputs.issubset(output_names):
            raise ValueError(
                f"TensorRT outputs mismatch. Required={expected_outputs}, got={sorted(output_names)}"
            )

        tensor_meta = {meta.name: meta for meta in self.runner.describe()}
        for name, expected_shape in self._expected_shapes().items():
            meta = tensor_meta.get(name)
            if meta is None:
                continue
            if not self._shape_matches_expected(list(meta.shape), expected_shape):
                raise ValueError(
                    f"TensorRT shape mismatch for {name!r}: "
                    f"expected compatible with {expected_shape}, got {meta.shape}"
                )

        for name in expected_inputs | expected_outputs:
            dtype = trt_dtype_to_torch_dtype(self.runner.engine.get_tensor_dtype(name))
            if dtype not in {torch.float16, torch.float32}:
                raise TypeError(f"Expected floating TensorRT dtype for {name!r}, got {dtype}")

    def reset(self) -> None:
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue: deque[Tensor] = deque([], maxlen=self.config.n_action_steps)

    def _get_images(self, batch: dict[str, Tensor]) -> list[Tensor]:
        missing_keys = [key for key in self.visual_keys if key not in batch]
        if missing_keys:
            if OBS_IMAGES in batch:
                raise KeyError(
                    "ActTrtPolicyAdapter expects keyed image tensors matching the resolved visual feature "
                    f"order {self.visual_keys}; batch[{OBS_IMAGES!r}] is not accepted as the image source of "
                    f"truth. Missing keys: {missing_keys}"
                )
            raise KeyError(f"Missing required visual feature keys: {missing_keys}")
        return [batch[key] for key in self.visual_keys]

    def _validate_obs_images_compatibility(
        self,
        batch: dict[str, Tensor],
        *,
        keyed_images: list[Tensor],
        expected_device: torch.device,
    ) -> None:
        if OBS_IMAGES not in batch:
            return

        legacy_images = batch[OBS_IMAGES]
        if not isinstance(legacy_images, Sequence) or isinstance(legacy_images, (str, bytes)):
            raise TypeError(
                f"Expected batch[{OBS_IMAGES!r}] to be a sequence of image tensors when provided, "
                f"got {type(legacy_images)}"
            )
        if len(legacy_images) != len(keyed_images):
            raise ValueError(
                f"Expected {len(keyed_images)} tensors in batch[{OBS_IMAGES!r}] to match visual keys "
                f"{self.visual_keys}, got {len(legacy_images)}"
            )

        for index, (key, keyed_image, legacy_image) in enumerate(zip(self.visual_keys, keyed_images, legacy_images)):
            self._validate_batch_tensor(
                f"{OBS_IMAGES}[{index}]",
                legacy_image,
                expected_shape=self.spec.image_shape,
                expected_device=expected_device,
            )
            if not torch.equal(legacy_image, keyed_image):
                raise ValueError(
                    f"batch[{OBS_IMAGES!r}] must match keyed visual tensors exactly in resolved order "
                    f"{self.visual_keys}. Mismatch at index {index} for key {key!r}."
                )

    def _validate_batch_tensor(
        self,
        name: str,
        tensor: Tensor,
        *,
        expected_shape: tuple[int, ...],
        expected_device: torch.device | None,
    ) -> None:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor for {name}, got {type(tensor)}")
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"Unexpected {name} shape: expected {expected_shape}, got {tuple(tensor.shape)}"
            )
        if not tensor.is_floating_point():
            raise TypeError(f"Expected floating tensor for {name}, got {tensor.dtype}")
        if expected_device is not None and tensor.device != expected_device:
            raise ValueError(
                f"Expected {name} to be on device {expected_device}, got {tensor.device}. "
                "All normalized inputs must share the same device."
            )

    def _build_feed_dict(self, batch: dict[str, Tensor]) -> tuple[dict[str, Tensor], torch.device]:
        if OBS_STATE not in batch:
            raise KeyError(f"Missing required key: {OBS_STATE}")

        obs_state = batch[OBS_STATE]
        self._validate_batch_tensor(
            OBS_STATE,
            obs_state,
            expected_shape=self.spec.obs_state_shape,
            expected_device=None,
        )

        images = self._get_images(batch)
        device = obs_state.device
        for index, image in enumerate(images):
            self._validate_batch_tensor(
                self.visual_keys[index],
                image,
                expected_shape=self.spec.image_shape,
                expected_device=device,
            )
        self._validate_obs_images_compatibility(batch, keyed_images=images, expected_device=device)

        feed_dict = {
            "obs_state_norm": obs_state.detach().contiguous(),
            "img0_norm": images[0].detach().contiguous(),
            "img1_norm": images[1].detach().contiguous(),
        }
        return feed_dict, device

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        feed_dict, device = self._build_feed_dict(batch)
        outputs = self.runner.infer(feed_dict)
        if "actions_norm" not in outputs:
            raise KeyError(
                f"TensorRT output missing 'actions_norm'. Available outputs: {sorted(outputs)}"
            )

        actions_norm = outputs["actions_norm"]
        if tuple(actions_norm.shape) != self.spec.action_shape:
            raise ValueError(
                f"Unexpected actions_norm shape: expected {self.spec.action_shape}, "
                f"got {tuple(actions_norm.shape)}"
            )
        return _clone_trt_output_to_torch(actions_norm, device=device)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        if self.config.temporal_ensemble_coeff is not None:
            actions = self.predict_action_chunk(batch)
            return self.temporal_ensembler.update(actions)

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()


__all__ = ["ActTrtPolicyAdapter"]
