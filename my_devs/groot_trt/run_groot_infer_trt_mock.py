#!/usr/bin/env python

"""Safe mock compare: PyTorch vs TensorRT engines for GR00T (LeRobot Groot policy).

This script is designed to reduce robot-risk before "real" deployment:

1) Record observations (no action is sent) from:
   - a real robot (`--source robot`), OR
   - deterministic random mock frames (`--source random`), OR
   - a previously recorded frames directory (`--source frames_dir`)
2) Run policy inference on the SAME frames with:
   - PyTorch (baseline)
   - TensorRT engines (via TensorRT Python API, using pre-built .engine files)
3) Compare action outputs and write a JSON report.

Notes:
- This script never sends actions to the robot.
- It uses the checkpoint-saved pre/post processors to match runtime behavior.
- TensorRT inference uses engines built by `my_devs/groot_trt/build_engine.py`.

Example (random frames, quick sanity check):

  conda run -n lerobot_flex env TENSORRT_PY_DIR=/data/cqy_workspace/third_party/tensorrt_10_13_0_35 \\
    python my_devs/groot_trt/run_groot_infer_trt_mock.py \\
      --policy-path /path/to/pretrained_model \\
      --engine-dir outputs/trt/<run_id>/gr00t_engine_api_trt1013 \\
      --source random --num-steps 8

Example (record from robot, then compare):

  conda run -n lerobot_flex env TENSORRT_PY_DIR=/data/cqy_workspace/third_party/tensorrt_10_13_0_35 \\
    python my_devs/groot_trt/run_groot_infer_trt_mock.py \\
      --policy-path /path/to/pretrained_model \\
      --engine-dir outputs/trt/<run_id>/gr00t_engine_api_trt1013 \\
      --source robot --run-time-s 10 --fps 10
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

# When invoked as `python my_devs/groot_trt/run_groot_infer_trt_mock.py`,
# sys.path[0] becomes the script directory, so we add repo root explicitly.
def _find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "lerobot").is_dir():
            return candidate
    raise RuntimeError(f"Failed to locate repo root from: {start}")


REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent)
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from lerobot import policies  # noqa: F401  # Register policy config classes.
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.policies.factory import get_policy_class
from lerobot.processor import PolicyAction, PolicyProcessorPipeline, make_default_processors
from lerobot.processor.converters import (
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import predict_action
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import get_safe_torch_device

from my_devs.groot_trt.trt_utils import TrtSession


DEFAULT_POLICY_PATH = (
    "/data/cqy_workspace/flexible_lerobot/outputs/train/"
    "groot_grasp_block_in_bin1_repro_20260302_223413/bs32_20260302_223447/"
    "checkpoints/last/pretrained_model"
)


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value

    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {value}")


def maybe_path(path_str: str | None) -> Path | None:
    return None if not path_str else Path(path_str).expanduser()


def _now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _np(t: torch.Tensor) -> np.ndarray:
    # Ensure the tensor is ready before pulling to CPU.
    if t.is_cuda:
        torch.cuda.synchronize()
    return t.detach().cpu().contiguous().numpy()


def _metrics(ref: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    ref64 = ref.astype(np.float64).reshape(-1)
    pred64 = pred.astype(np.float64).reshape(-1)
    diff = np.abs(ref64 - pred64)
    denom = float(np.linalg.norm(ref64) * np.linalg.norm(pred64))
    cosine = float(np.dot(ref64, pred64) / denom) if denom > 0 else float("nan")
    rmse = float(np.sqrt(np.mean((ref64 - pred64) ** 2)))
    return {
        "shape_ref": list(ref.shape),
        "shape_pred": list(pred.shape),
        "max_abs": float(diff.max(initial=0.0)),
        "mean_abs": float(diff.mean() if diff.size else 0.0),
        "rmse": rmse,
        "cosine": cosine,
    }


def load_pre_post_processors(
    policy_path: Path,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Load saved policy processors directly from checkpoint directory."""
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=str(policy_path),
        config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=str(policy_path),
        config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return preprocessor, postprocessor


def _num_patches(backbone: torch.nn.Module) -> int:
    vision_model = backbone.eagle_model.vision_model
    if hasattr(vision_model, "vision_model") and hasattr(vision_model.vision_model, "embeddings"):
        return int(vision_model.vision_model.embeddings.num_patches)
    if hasattr(vision_model, "embeddings"):
        return int(vision_model.embeddings.num_patches)
    raise AttributeError("Cannot determine num_patches from backbone vision model.")


def _postprocess_vit(backbone: torch.nn.Module, vit_embeds: torch.Tensor) -> torch.Tensor:
    # Keep identical behavior to `my_devs/groot_trt/compare_torch_onnx.py` and friends.
    vit_embeds = vit_embeds.view(1, -1, vit_embeds.shape[-1])
    if getattr(backbone.eagle_model, "use_pixel_shuffle", False):
        token_count = int(vit_embeds.shape[1])
        side = int(int(token_count) ** 0.5)
        if side * side != token_count:
            raise ValueError(
                f"Pixel-shuffle expects square token layout, got token_count={token_count}. "
                "Try --fps=1 (1-view) for debugging."
            )
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], side, side, -1)
        pixel_shuffle = getattr(backbone, "pixel_shuffle", None) or getattr(backbone.eagle_model, "pixel_shuffle", None)
        downsample_ratio = getattr(backbone, "downsample_ratio", None) or getattr(
            backbone.eagle_model, "downsample_ratio", None
        )
        if pixel_shuffle is None or downsample_ratio is None:
            raise RuntimeError("Eagle pixel shuffle is enabled but pixel_shuffle/downsample_ratio is missing.")
        vit_embeds = pixel_shuffle(vit_embeds, scale_factor=downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
    vit_embeds = backbone.eagle_model.mlp1(vit_embeds)
    return vit_embeds


def _build_inputs_embeds_from_vit(
    backbone: torch.nn.Module,
    input_ids: torch.Tensor,
    vit_embeds: torch.Tensor,
) -> torch.Tensor:
    """Rebuild `inputs_embeds` by replacing image-token embeddings with ViT token embeddings."""
    embedding_layer = backbone.eagle_model.language_model.get_input_embeddings()
    image_token_index = int(backbone.eagle_model.image_token_index)

    input_ids = input_ids.to(device=vit_embeds.device, dtype=torch.int64)
    inputs_embeds = embedding_layer(input_ids).to(torch.float16)

    batch, tokens, channels = inputs_embeds.shape
    inputs_embeds_flat = inputs_embeds.reshape(batch * tokens, channels)
    input_ids_flat = input_ids.reshape(batch * tokens)
    selected = input_ids_flat == image_token_index

    vit_flat = vit_embeds.reshape(-1, channels)
    n = int(selected.sum().item())
    if n == 0:
        # If this happens, the prompt template probably changed.
        return inputs_embeds
    if n > vit_flat.shape[0]:
        raise ValueError(f"Need {n} ViT tokens, but got {vit_flat.shape[0]}.")

    inputs_embeds_flat[selected] = vit_flat[:n]
    return inputs_embeds_flat.reshape(batch, tokens, channels)


@dataclass
class TrtEnginePaths:
    vit: Path
    llm: Path
    vlln: Path
    state_encoder: Path
    action_encoder: Path
    dit: Path
    action_decoder: Path


class TrtGrootPolicyAdapter:
    """A minimal policy-like adapter that runs GR00T with TensorRT engines.

    It reuses the checkpoint-loaded policy for config and for the small "glue" modules that
    were not exported to ONNX (e.g. mlp1, embedding tables, future tokens, pos embedding).
    """

    def __init__(
        self,
        torch_policy: torch.nn.Module,
        engine_dir: Path,
        *,
        vit_dtype: str = "fp16",
        llm_dtype: str = "fp16",
        dit_dtype: str = "fp16",
        tensorrt_py_dir: str | None = None,
        num_denoising_steps: int | None = None,
    ) -> None:
        self.torch_policy = torch_policy
        self.config = getattr(torch_policy, "config", None)
        self.device = next(torch_policy.parameters()).device

        self.groot_model = torch_policy._groot_model
        self.backbone = self.groot_model.backbone
        self.action_head = self.groot_model.action_head

        # We feed engines with fp16 tensors; keep the glue in fp16 as well.
        self.backbone.eagle_model.mlp1.to(device=self.device, dtype=torch.float16)
        self.backbone.eagle_model.language_model.get_input_embeddings().to(device=self.device, dtype=torch.float16)
        self.action_head.future_tokens.to(device=self.device, dtype=torch.float16)
        if getattr(self.action_head.config, "add_pos_embed", False):
            self.action_head.position_embedding.to(device=self.device, dtype=torch.float16)

        self.num_patches = _num_patches(self.backbone)
        self.original_action_dim = int(self.torch_policy.config.output_features[ACTION].shape[0])

        paths = TrtEnginePaths(
            vit=(engine_dir / f"vit_{vit_dtype}.engine"),
            llm=(engine_dir / f"llm_{llm_dtype}.engine"),
            vlln=(engine_dir / "vlln_vl_self_attention.engine"),
            state_encoder=(engine_dir / "state_encoder.engine"),
            action_encoder=(engine_dir / "action_encoder.engine"),
            dit=(engine_dir / f"DiT_{dit_dtype}.engine"),
            action_decoder=(engine_dir / "action_decoder.engine"),
        )
        missing = [p for p in paths.__dict__.values() if not Path(p).is_file()]
        if missing:
            raise FileNotFoundError("Missing required TensorRT engine files:\n" + "\n".join(f"  - {p}" for p in missing))

        self.sess_vit = TrtSession.load(paths.vit, trt_py_dir=tensorrt_py_dir)
        self.sess_llm = TrtSession.load(paths.llm, trt_py_dir=tensorrt_py_dir)
        self.sess_vlln = TrtSession.load(paths.vlln, trt_py_dir=tensorrt_py_dir)
        self.sess_state_encoder = TrtSession.load(paths.state_encoder, trt_py_dir=tensorrt_py_dir)
        self.sess_action_encoder = TrtSession.load(paths.action_encoder, trt_py_dir=tensorrt_py_dir)
        self.sess_dit = TrtSession.load(paths.dit, trt_py_dir=tensorrt_py_dir)
        self.sess_action_decoder = TrtSession.load(paths.action_decoder, trt_py_dir=tensorrt_py_dir)

        self._action_queue: deque[torch.Tensor] = deque()
        self._num_denoising_steps = num_denoising_steps

    def reset(self) -> None:
        self._action_queue.clear()

    @torch.inference_mode()
    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if len(self._action_queue) == 0:
            actions = self._predict_action_chunk(batch)  # (B, T, D)
            self._action_queue.extend(actions.transpose(0, 1))  # (T, B, D) -> list of (B, D)
        return self._action_queue.popleft()

    def _predict_action_chunk(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        # 1) Backbone: (eagle_pixel_values, eagle_input_ids, eagle_attention_mask) -> backbone_features
        if "eagle_pixel_values" not in batch or "eagle_input_ids" not in batch or "eagle_attention_mask" not in batch:
            raise KeyError("Missing eagle_* keys in batch. Is the preprocessor pipeline correct?")
        pixel_values = batch["eagle_pixel_values"].to(device=self.device, dtype=torch.float16).contiguous()
        input_ids = batch["eagle_input_ids"].to(device=self.device, dtype=torch.int64).contiguous()
        attention_mask = batch["eagle_attention_mask"].to(device=self.device, dtype=torch.int64).contiguous()

        # ViT: position_ids is a dummy input for export signature; values do not affect output.
        position_ids = (
            torch.arange(self.num_patches, dtype=torch.int64, device=self.device)
            .unsqueeze(0)
            .expand(pixel_values.shape[0], -1)
            .contiguous()
        )
        vit_embeds = self.sess_vit.run({"pixel_values": pixel_values, "position_ids": position_ids})["vit_embeds"]
        vit_embeds = _postprocess_vit(self.backbone, vit_embeds.to(torch.float16))

        inputs_embeds = _build_inputs_embeds_from_vit(self.backbone, input_ids, vit_embeds)
        llm_out = self.sess_llm.run({"inputs_embeds": inputs_embeds, "attention_mask": attention_mask})["embeddings"]
        backbone_features = llm_out.to(torch.float16)

        # 2) Action head: use TRT engines for the exported submodules.
        action_head = self.action_head
        embodiment_id = batch["embodiment_id"].to(device=self.device, dtype=torch.int64).contiguous()
        state = batch["state"].to(device=self.device, dtype=torch.float16).contiguous()

        vl_embs = self.sess_vlln.run({"backbone_features": backbone_features})["output"].to(torch.float16)
        state_features = self.sess_state_encoder.run({"state": state, "embodiment_id": embodiment_id})["output"].to(
            torch.float16
        )

        batch_size = int(vl_embs.shape[0])
        action_horizon = int(action_head.config.action_horizon)
        action_dim = int(action_head.config.action_dim)
        num_steps = int(self._num_denoising_steps) if self._num_denoising_steps is not None else int(
            action_head.num_inference_timesteps
        )
        dt = 1.0 / float(num_steps)

        actions = torch.randn(
            size=(batch_size, action_horizon, action_dim),
            dtype=torch.float16,
            device=self.device,
        )

        future_tokens = action_head.future_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1).to(torch.float16)
        if getattr(action_head.config, "add_pos_embed", False):
            pos_ids = torch.arange(action_horizon, dtype=torch.long, device=self.device)
            pos_embs = action_head.position_embedding(pos_ids).unsqueeze(0).to(torch.float16)
        else:
            pos_embs = None

        for step in range(num_steps):
            t_cont = step / float(num_steps)
            t_discretized = int(t_cont * int(action_head.num_timestep_buckets))
            timesteps_tensor = torch.full((batch_size,), t_discretized, dtype=torch.int64, device=self.device)

            action_features = self.sess_action_encoder.run(
                {
                    "actions": actions,
                    "timesteps_tensor": timesteps_tensor,
                    "embodiment_id": embodiment_id,
                }
            )["output"].to(torch.float16)
            if pos_embs is not None:
                action_features = action_features + pos_embs

            sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1).to(torch.float16)
            model_output = self.sess_dit.run(
                {
                    "sa_embs": sa_embs,
                    "vl_embs": vl_embs,
                    "timesteps_tensor": timesteps_tensor,
                }
            )["output"].to(torch.float16)
            pred = self.sess_action_decoder.run({"model_output": model_output, "embodiment_id": embodiment_id})[
                "output"
            ].to(torch.float16)
            pred_velocity = pred[:, -action_horizon:, :]
            actions = actions + dt * pred_velocity

        actions = actions[:, :, : self.original_action_dim]
        return actions


class SeededPolicyWrapper:
    """Wrap a policy-like object (must have `select_action`, optionally `_action_queue`) to make sampling reproducible."""

    def __init__(self, policy: Any, *, seed: int) -> None:
        self.policy = policy
        self.seed = int(seed)
        self._chunk_idx = 0

    def reset(self) -> None:
        self._chunk_idx = 0
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def select_action(self, batch: dict[str, Any]) -> torch.Tensor:
        queue = getattr(self.policy, "_action_queue", None)
        need_seed = queue is not None and len(queue) == 0
        if need_seed:
            chunk_seed = self.seed + self._chunk_idx
            torch.manual_seed(chunk_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(chunk_seed)
            self._chunk_idx += 1
        return self.policy.select_action(batch)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mock compare: GROOT PyTorch vs TensorRT engine outputs.")
    parser.add_argument("--policy-path", default=os.getenv("POLICY_PATH", DEFAULT_POLICY_PATH))
    parser.add_argument("--task", default=os.getenv("DATASET_TASK", "Put the block in the bin"))
    parser.add_argument("--robot-type", default=os.getenv("ROBOT_TYPE", "so101_follower"))

    parser.add_argument("--engine-dir", required=True, help="Directory containing TensorRT .engine files.")
    parser.add_argument(
        "--tensorrt-py-dir",
        default=os.getenv("TENSORRT_PY_DIR"),
        help="Optional TensorRT pip --target dir (contains `tensorrt/` and `tensorrt_libs/`).",
    )
    parser.add_argument("--vit-dtype", default=os.getenv("TRT_VIT_DTYPE", "fp16"), choices=["fp16", "fp8"])
    parser.add_argument("--llm-dtype", default=os.getenv("TRT_LLM_DTYPE", "fp16"), choices=["fp16", "fp8", "nvfp4", "nvfp4_full"])
    parser.add_argument("--dit-dtype", default=os.getenv("TRT_DIT_DTYPE", "fp16"), choices=["fp16", "fp8"])

    parser.add_argument(
        "--source",
        default=os.getenv("MOCK_SOURCE", "random"),
        choices=["robot", "random", "frames_dir"],
        help="Where to get observations. 'robot' records frames without acting; 'random' generates deterministic frames.",
    )
    parser.add_argument(
        "--frames-dir",
        default=os.getenv("FRAMES_DIR"),
        help="When --source=frames_dir, load frames from this directory (pickle files).",
    )
    parser.add_argument("--out-dir", default=os.getenv("OUT_DIR"), help="Output dir. Default: outputs/trt/mock_compare_<ts>/")
    parser.add_argument("--seed", type=int, default=int(os.getenv("SEED", "20260305")))
    parser.add_argument(
        "--num-steps",
        type=int,
        default=int(os.getenv("NUM_STEPS", "32")),
        help="Number of frames to record/generate. Used by source=robot/random.",
    )
    parser.add_argument("--run-time-s", type=float, default=float(os.getenv("RUN_TIME_S", "0")))
    parser.add_argument("--fps", type=int, default=int(os.getenv("FPS", "10")))
    parser.add_argument("--img-width", type=int, default=int(os.getenv("IMG_WIDTH", "640")))
    parser.add_argument("--img-height", type=int, default=int(os.getenv("IMG_HEIGHT", "480")))

    parser.add_argument("--robot-id", default=os.getenv("ROBOT_ID", "my_so101"))
    parser.add_argument(
        "--calib-dir",
        default=os.getenv(
            "CALIB_DIR", "/home/cqy/.cache/huggingface/lerobot/calibration/robots/so101_follower"
        ),
    )
    parser.add_argument("--robot-port", default=os.getenv("ROBOT_PORT", "/dev/ttyACM0"))
    parser.add_argument("--top-cam-index", type=int, default=int(os.getenv("TOP_CAM_INDEX", "4")))
    parser.add_argument("--wrist-cam-index", type=int, default=int(os.getenv("WRIST_CAM_INDEX", "6")))

    parser.add_argument(
        "--num-denoising-steps",
        type=int,
        default=None,
        help="Override denoising steps for faster mock compare. Default: checkpoint config.",
    )
    parser.add_argument(
        "--dry-run",
        type=parse_bool,
        nargs="?",
        const=True,
        default=False,
        help="Print resolved config and exit without running.",
    )
    return parser


def _resolve_out_dir(args: argparse.Namespace) -> Path:
    if args.out_dir:
        return Path(args.out_dir).expanduser()
    return Path("outputs/trt") / f"mock_compare_{_now_ts()}"


def _save_frame(frame_path: Path, frame: dict[str, np.ndarray]) -> None:
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    with frame_path.open("wb") as f:
        pickle.dump(frame, f, protocol=4)


def _load_frame(frame_path: Path) -> dict[str, np.ndarray]:
    with frame_path.open("rb") as f:
        return pickle.load(f)


def record_frames_random(
    frames_dir: Path,
    *,
    num_steps: int,
    img_height: int,
    img_width: int,
    seed: int,
) -> list[Path]:
    rng = np.random.default_rng(int(seed))
    paths: list[Path] = []
    for i in range(num_steps):
        frame = {
            "observation.images.top": rng.integers(0, 256, size=(img_height, img_width, 3), dtype=np.uint8),
            "observation.images.wrist": rng.integers(0, 256, size=(img_height, img_width, 3), dtype=np.uint8),
            "observation.state": rng.standard_normal(size=(64,), dtype=np.float32),
        }
        p = frames_dir / f"{i:06d}.pkl"
        _save_frame(p, frame)
        paths.append(p)
    return paths


def record_frames_robot(
    frames_dir: Path,
    *,
    policy_path: Path,
    robot_cfg: SOFollowerRobotConfig,
    fps: int,
    run_time_s: float,
    num_steps: int,
) -> list[Path]:
    # Build robot.
    from lerobot.robots import make_robot_from_config

    robot = make_robot_from_config(robot_cfg)

    # Mirror the feature building in `run_groot_infer.py` for robust key mapping.
    _, robot_action_processor, robot_observation_processor = make_default_processors()
    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=robot_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True,
        ),
    )

    frames_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    start_t = time.perf_counter()
    end_t = start_t + run_time_s if run_time_s > 0 else None

    try:
        robot.connect()
        step = 0
        while True:
            if end_t is not None and time.perf_counter() >= end_t:
                break
            if run_time_s <= 0 and step >= num_steps:
                break
            if run_time_s > 0 and step >= num_steps:
                # If user provided both, stop at whichever comes first.
                break

            loop_t = time.perf_counter()
            obs = robot.get_observation()
            obs_processed = robot_observation_processor(obs)
            observation_frame = build_dataset_frame(dataset_features, obs_processed, prefix=OBS_STR)
            p = frames_dir / f"{step:06d}.pkl"
            _save_frame(p, observation_frame)
            paths.append(p)
            step += 1

            dt_s = time.perf_counter() - loop_t
            precise_sleep(max(1 / float(fps) - dt_s, 0.0))
    finally:
        if robot.is_connected:
            robot.disconnect()

    return paths


def run_inference_on_frames(
    *,
    policy: Any,
    policy_path: Path,
    frame_paths: list[Path],
    task: str,
    robot_type: str,
    seed: int,
    json_out: Path,
    use_amp: bool,
) -> np.ndarray:
    preprocessor, postprocessor = load_pre_post_processors(policy_path)
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    seeded_policy = SeededPolicyWrapper(policy, seed=seed)
    seeded_policy.reset()

    device = get_safe_torch_device(getattr(policy, "config", None).device if hasattr(policy, "config") else "cuda")
    actions: list[np.ndarray] = []
    start = time.perf_counter()
    for i, p in enumerate(frame_paths):
        obs_frame = _load_frame(p)
        action_t = predict_action(
            observation=obs_frame,
            policy=seeded_policy,  # type: ignore[arg-type]
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=use_amp,
            task=task,
            robot_type=robot_type,
        )
        a_np = _np(action_t).astype(np.float32)
        # Normalize shapes to (D,) for easy stacking.
        if a_np.ndim == 2 and a_np.shape[0] == 1:
            a_np = a_np[0]
        actions.append(a_np)
        if (i + 1) % 10 == 0:
            elapsed = time.perf_counter() - start
            print(f"[INFO] Inference step {i+1}/{len(frame_paths)} elapsed={elapsed:.2f}s")

    actions_arr = np.stack(actions, axis=0)
    total_s = time.perf_counter() - start
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(
        json.dumps(
            {
                "num_frames": len(frame_paths),
                "actions_shape": list(actions_arr.shape),
                "total_seconds": total_s,
                "fps_effective": float(len(frame_paths) / total_s) if total_s > 0 else float("nan"),
            },
            indent=2,
        )
    )
    return actions_arr


def main() -> None:
    register_third_party_plugins()
    args = build_parser().parse_args()

    policy_path = Path(args.policy_path).expanduser()
    if not policy_path.is_dir():
        raise FileNotFoundError(f"--policy-path does not exist: {policy_path}")

    engine_dir = Path(args.engine_dir).expanduser()
    if not engine_dir.is_dir():
        raise FileNotFoundError(f"--engine-dir does not exist: {engine_dir}")

    out_dir = _resolve_out_dir(args)
    frames_dir = out_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] policy_path: {policy_path}")
    print(f"[INFO] engine_dir: {engine_dir}")
    print(f"[INFO] source: {args.source}")
    print(f"[INFO] out_dir: {out_dir}")
    print(f"[INFO] seed: {args.seed}")
    print(f"[INFO] task: {args.task}")
    print(f"[INFO] robot_type: {args.robot_type}")
    print(f"[INFO] num_steps: {args.num_steps}")
    print(f"[INFO] run_time_s: {args.run_time_s}")
    print(f"[INFO] fps: {args.fps}")
    print(f"[INFO] tensorrt_py_dir: {args.tensorrt_py_dir}")
    print(f"[INFO] dtypes: vit={args.vit_dtype} llm={args.llm_dtype} dit={args.dit_dtype}")
    if args.num_denoising_steps is not None:
        print(f"[INFO] num_denoising_steps override: {args.num_denoising_steps}")

    if args.dry_run:
        print("[INFO] DRY_RUN=true, exiting.")
        return

    # 1) Prepare frames.
    if args.source == "frames_dir":
        if not args.frames_dir:
            raise ValueError("--frames-dir is required when --source=frames_dir")
        frames_dir = Path(args.frames_dir).expanduser()
        frame_paths = sorted(frames_dir.glob("*.pkl"))
        if not frame_paths:
            raise FileNotFoundError(f"No *.pkl frames found in: {frames_dir}")
    elif args.source == "random":
        frame_paths = record_frames_random(
            frames_dir,
            num_steps=int(args.num_steps),
            img_height=int(args.img_height),
            img_width=int(args.img_width),
            seed=int(args.seed),
        )
    else:
        cameras = {
            "top": OpenCVCameraConfig(
                index_or_path=args.top_cam_index,
                width=args.img_width,
                height=args.img_height,
                fps=args.fps,
            ),
            "wrist": OpenCVCameraConfig(
                index_or_path=args.wrist_cam_index,
                width=args.img_width,
                height=args.img_height,
                fps=args.fps,
            ),
        }
        robot_cfg = SOFollowerRobotConfig(
            id=args.robot_id,
            calibration_dir=maybe_path(args.calib_dir),
            port=args.robot_port,
            cameras=cameras,
        )
        frame_paths = record_frames_robot(
            frames_dir,
            policy_path=policy_path,
            robot_cfg=robot_cfg,
            fps=int(args.fps),
            run_time_s=float(args.run_time_s),
            num_steps=int(args.num_steps),
        )

    print(f"[OK] Prepared {len(frame_paths)} frame(s). frames_dir={frames_dir}")

    # 2) Baseline: PyTorch inference.
    torch.cuda.empty_cache()
    policy_cfg = PreTrainedConfig.from_pretrained(str(policy_path))
    policy_cfg.pretrained_path = policy_path
    policy_class = get_policy_class(policy_cfg.type)
    torch_policy = policy_class.from_pretrained(str(policy_path), strict=False)
    torch_policy.to(policy_cfg.device)
    torch_policy.eval()
    use_amp = bool(torch_policy.config.use_amp)

    print(f"[INFO] Running PyTorch baseline inference (use_amp={use_amp})...")
    actions_torch = run_inference_on_frames(
        policy=torch_policy,
        policy_path=policy_path,
        frame_paths=frame_paths,
        task=args.task,
        robot_type=args.robot_type,
        seed=int(args.seed),
        json_out=out_dir / "torch_infer_meta.json",
        use_amp=use_amp,
    )
    torch_actions_path = out_dir / "actions_torch.npy"
    np.save(torch_actions_path, actions_torch)
    print(f"[OK] Saved PyTorch actions: {torch_actions_path} shape={actions_torch.shape}")

    # Free VRAM before TRT run.
    del torch_policy
    gc.collect()
    torch.cuda.empty_cache()

    # 3) TensorRT inference via engines.
    trt_policy = policy_class.from_pretrained(str(policy_path), strict=False)
    trt_policy.to(policy_cfg.device)
    trt_policy.eval()

    trt_adapter = TrtGrootPolicyAdapter(
        trt_policy,
        engine_dir,
        vit_dtype=args.vit_dtype,
        llm_dtype=args.llm_dtype,
        dit_dtype=args.dit_dtype,
        tensorrt_py_dir=args.tensorrt_py_dir,
        num_denoising_steps=args.num_denoising_steps,
    )
    print(f"[INFO] Running TensorRT engine inference (use_amp={use_amp})...")
    actions_trt = run_inference_on_frames(
        policy=trt_adapter,
        policy_path=policy_path,
        frame_paths=frame_paths,
        task=args.task,
        robot_type=args.robot_type,
        seed=int(args.seed),
        json_out=out_dir / "trt_infer_meta.json",
        use_amp=use_amp,
    )
    trt_actions_path = out_dir / "actions_trt.npy"
    np.save(trt_actions_path, actions_trt)
    print(f"[OK] Saved TensorRT actions: {trt_actions_path} shape={actions_trt.shape}")

    # 4) Compare.
    report: dict[str, Any] = {
        "policy_path": policy_path.as_posix(),
        "engine_dir": engine_dir.as_posix(),
        "source": args.source,
        "frames_dir": frames_dir.as_posix(),
        "num_frames": int(actions_torch.shape[0]),
        "seed": int(args.seed),
        "task": args.task,
        "robot_type": args.robot_type,
        "use_amp": use_amp,
        "actions_torch_path": torch_actions_path.as_posix(),
        "actions_trt_path": trt_actions_path.as_posix(),
        "overall": _metrics(actions_torch, actions_trt),
        "per_step": [],
    }

    worst_cos = (None, float("inf"))
    worst_max_abs = (None, -1.0)
    for i in range(actions_torch.shape[0]):
        m = _metrics(actions_torch[i], actions_trt[i])
        report["per_step"].append({"step": i, **m})
        if m["cosine"] < worst_cos[1]:
            worst_cos = (i, m["cosine"])
        if m["max_abs"] > worst_max_abs[1]:
            worst_max_abs = (i, m["max_abs"])

    report["summary"] = {
        "worst_cosine_step": int(worst_cos[0]) if worst_cos[0] is not None else None,
        "worst_cosine": float(worst_cos[1]),
        "worst_max_abs_step": int(worst_max_abs[0]) if worst_max_abs[0] is not None else None,
        "worst_max_abs": float(worst_max_abs[1]),
    }

    report_path = out_dir / "compare_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[OK] Compare report saved: {report_path}")
    print(
        "[RESULT] overall: "
        f"cos={report['overall']['cosine']:.8f} rmse={report['overall']['rmse']:.8f} "
        f"mean_abs={report['overall']['mean_abs']:.8f} max_abs={report['overall']['max_abs']:.8f}"
    )
    print(
        "[RESULT] worst: "
        f"cos(step={report['summary']['worst_cosine_step']})={report['summary']['worst_cosine']:.8f} | "
        f"max_abs(step={report['summary']['worst_max_abs_step']})={report['summary']['worst_max_abs']:.8f}"
    )


if __name__ == "__main__":
    main()
