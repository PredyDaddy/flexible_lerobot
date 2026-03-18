#!/usr/bin/env python

"""Run GR00T robot inference using TensorRT engines (no Torch backbone/actionhead compute).

This is the "real on-robot" script that mirrors `run_groot_infer.py`, but uses the TensorRT engines
you built under `outputs/.../gr00t_engine_api_trt1013/`.

Safety:
- Supports `--mock true` to run the full loop WITHOUT sending actions to the robot.
- Start with a short `--run-time-s` and low `--fps`, and keep your E-stop ready.

Quick start on this machine (mock, no action send):

  export TENSORRT_PY_DIR=/data/cqy_workspace/third_party/tensorrt_10_13_0_35

  python3 my_devs/groot_trt/run_groot_infer_trt.py \\
    --engine-dir /data/cqy_workspace/flexible_lerobot/outputs/trt/consistency_rerun_20260305_102931/gr00t_engine_api_trt1013 \\
    --robot-port /dev/ttyACM0 \\
    --top-cam-index 4 \\
    --wrist-cam-index 6 \\
    --task "Put the block in the bin" \\
    --run-time-s 10 \\
    --fps 5 \\
    --mock true

Example (real action send):

  export TENSORRT_PY_DIR=/data/cqy_workspace/third_party/tensorrt_10_13_0_35

  python3 my_devs/groot_trt/run_groot_infer_trt.py \
    --engine-dir /data/cqy_workspace/flexible_lerobot/outputs/trt/consistency_rerun_20260305_102931/gr00t_engine_api_trt1013 \
    --robot-port /dev/ttyACM0 \
    --top-cam-index 4 \
    --wrist-cam-index 6 \
    --task "Put the block in the bin" \
    --run-time-s 120 \
    --fps 30 \
    --mock false
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import torch

# Ensure repo root is importable when invoked as a script.
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
from lerobot.policies.utils import make_robot_action
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

DEFAULT_ENGINE_DIR_EXAMPLE = (
    "/data/cqy_workspace/flexible_lerobot/outputs/trt/"
    "consistency_rerun_20260305_102931/gr00t_engine_api_trt1013"
)


def resolve_policy_dir(policy_path: Path) -> Path:
    """Accept a few common user inputs and resolve to the actual pretrained_model dir.

    We need a directory containing `config.json` for `PreTrainedConfig.from_pretrained(...)`.
    """

    candidates = [
        policy_path,
        policy_path / "pretrained_model",
        policy_path / "checkpoints" / "last" / "pretrained_model",
    ]
    for cand in candidates:
        if (cand / "config.json").is_file():
            if cand != policy_path:
                print(
                    "[WARN] --policy-path does not look like a `pretrained_model/` directory "
                    f"(missing config.json). Using detected path: {cand}"
                )
            return cand

    searched = "\n".join(f"  - {c}" for c in candidates)
    raise FileNotFoundError(
        "Policy directory must contain `config.json` (the lerobot `pretrained_model/` folder).\n"
        f"You passed: {policy_path}\n"
        "Searched:\n"
        f"{searched}\n"
        "Fix:\n"
        f"  - Pass a real checkpoint folder like: {DEFAULT_POLICY_PATH}\n"
        "  - Or omit `--policy-path` to use the built-in default in this script.\n"
    )


def validate_engine_dir(engine_dir: Path, vit_dtype: str, llm_dtype: str, dit_dtype: str) -> None:
    """Fail fast with a friendly error if required engines are missing."""

    required = [
        f"vit_{vit_dtype}.engine",
        f"llm_{llm_dtype}.engine",
        f"DiT_{dit_dtype}.engine",
        "vlln_vl_self_attention.engine",
        "state_encoder.engine",
        "action_encoder.engine",
        "action_decoder.engine",
    ]
    missing = [name for name in required if not (engine_dir / name).is_file()]
    if not missing:
        return

    missing_lines = "\n".join(f"  - {name}" for name in missing)
    raise FileNotFoundError(
        "Engine directory is missing required TensorRT engine files.\n"
        f"You passed: {engine_dir}\n"
        "Missing:\n"
        f"{missing_lines}\n"
        "Fix:\n"
        "  - Pass the directory that contains all exported `.engine` files.\n"
        f"  - Example: {DEFAULT_ENGINE_DIR_EXAMPLE}\n"
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


def load_pre_post_processors(
    policy_path: Path,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
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
    # Keep identical behavior to `my_devs/groot_trt/compare_torch_onnx.py`.
    vit_embeds = vit_embeds.view(1, -1, vit_embeds.shape[-1])
    if getattr(backbone.eagle_model, "use_pixel_shuffle", False):
        token_count = int(vit_embeds.shape[1])
        side = int(int(token_count) ** 0.5)
        if side * side != token_count:
            raise ValueError(f"Pixel-shuffle expects square token layout, got token_count={token_count}.")
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
    embedding_layer = backbone.eagle_model.language_model.get_input_embeddings()
    image_token_index = int(backbone.eagle_model.image_token_index)

    input_ids = input_ids.to(device=vit_embeds.device, dtype=torch.int64)
    inputs_embeds = embedding_layer(input_ids).to(torch.float16)

    batch, tokens, channels = inputs_embeds.shape
    flat = inputs_embeds.reshape(batch * tokens, channels)
    ids_flat = input_ids.reshape(batch * tokens)
    selected = ids_flat == image_token_index

    vit_flat = vit_embeds.reshape(-1, channels)
    n = int(selected.sum().item())
    if n > 0:
        flat[selected] = vit_flat[:n]
    return flat.reshape(batch, tokens, channels)


class TrtGrootPolicyAdapter:
    """Policy-like wrapper providing `select_action` using TRT engines."""

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

        # Glue modules in fp16 to match engines.
        self.backbone.eagle_model.mlp1.to(device=self.device, dtype=torch.float16)
        self.backbone.eagle_model.language_model.get_input_embeddings().to(device=self.device, dtype=torch.float16)
        self.action_head.future_tokens.to(device=self.device, dtype=torch.float16)
        if getattr(self.action_head.config, "add_pos_embed", False):
            self.action_head.position_embedding.to(device=self.device, dtype=torch.float16)

        self.num_patches = _num_patches(self.backbone)
        self.original_action_dim = int(self.torch_policy.config.output_features[ACTION].shape[0])

        vit_engine = engine_dir / f"vit_{vit_dtype}.engine"
        llm_engine = engine_dir / f"llm_{llm_dtype}.engine"
        dit_engine = engine_dir / f"DiT_{dit_dtype}.engine"
        required = [
            vit_engine,
            llm_engine,
            engine_dir / "vlln_vl_self_attention.engine",
            engine_dir / "state_encoder.engine",
            engine_dir / "action_encoder.engine",
            dit_engine,
            engine_dir / "action_decoder.engine",
        ]
        missing = [p for p in required if not p.is_file()]
        if missing:
            raise FileNotFoundError("Missing required TensorRT engine files:\n" + "\n".join(f"  - {p}" for p in missing))

        self.sess_vit = TrtSession.load(vit_engine, trt_py_dir=tensorrt_py_dir)
        self.sess_llm = TrtSession.load(llm_engine, trt_py_dir=tensorrt_py_dir)
        self.sess_vlln = TrtSession.load(engine_dir / "vlln_vl_self_attention.engine", trt_py_dir=tensorrt_py_dir)
        self.sess_state_encoder = TrtSession.load(engine_dir / "state_encoder.engine", trt_py_dir=tensorrt_py_dir)
        self.sess_action_encoder = TrtSession.load(engine_dir / "action_encoder.engine", trt_py_dir=tensorrt_py_dir)
        self.sess_dit = TrtSession.load(dit_engine, trt_py_dir=tensorrt_py_dir)
        self.sess_action_decoder = TrtSession.load(engine_dir / "action_decoder.engine", trt_py_dir=tensorrt_py_dir)

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
        # Preprocessor outputs these keys for GR00T.
        pixel_values = batch["eagle_pixel_values"].to(device=self.device, dtype=torch.float16).contiguous()
        input_ids = batch["eagle_input_ids"].to(device=self.device, dtype=torch.int64).contiguous()
        attention_mask = batch["eagle_attention_mask"].to(device=self.device, dtype=torch.int64).contiguous()

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

        # Action head.
        embodiment_id = batch["embodiment_id"].to(device=self.device, dtype=torch.int64).contiguous()
        state = batch["state"].to(device=self.device, dtype=torch.float16).contiguous()

        vl_embs = self.sess_vlln.run({"backbone_features": backbone_features})["output"].to(torch.float16)
        state_features = self.sess_state_encoder.run({"state": state, "embodiment_id": embodiment_id})["output"].to(
            torch.float16
        )

        batch_size = int(vl_embs.shape[0])
        action_horizon = int(self.action_head.config.action_horizon)
        action_dim = int(self.action_head.config.action_dim)
        num_steps = int(self._num_denoising_steps) if self._num_denoising_steps is not None else int(
            self.action_head.num_inference_timesteps
        )
        dt = 1.0 / float(num_steps)

        actions = torch.randn(
            size=(batch_size, action_horizon, action_dim),
            dtype=torch.float16,
            device=self.device,
        )

        future_tokens = self.action_head.future_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1).to(torch.float16)
        if getattr(self.action_head.config, "add_pos_embed", False):
            pos_ids = torch.arange(action_horizon, dtype=torch.long, device=self.device)
            pos_embs = self.action_head.position_embedding(pos_ids).unsqueeze(0).to(torch.float16)
        else:
            pos_embs = None

        for step in range(num_steps):
            t_cont = step / float(num_steps)
            t_discretized = int(t_cont * int(self.action_head.num_timestep_buckets))
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

        return actions[:, :, : self.original_action_dim]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Robot inference loop using TensorRT engines for GR00T.")
    parser.add_argument("--robot-id", default=os.getenv("ROBOT_ID", "my_so101"))
    parser.add_argument("--robot-type", default=os.getenv("ROBOT_TYPE", "so101_follower"))
    parser.add_argument(
        "--calib-dir",
        default=os.getenv(
            "CALIB_DIR", "/home/cqy/.cache/huggingface/lerobot/calibration/robots/so101_follower"
        ),
    )
    parser.add_argument("--robot-port", default=os.getenv("ROBOT_PORT", "/dev/ttyACM0"))

    parser.add_argument("--top-cam-index", type=int, default=int(os.getenv("TOP_CAM_INDEX", "4")))
    parser.add_argument("--wrist-cam-index", type=int, default=int(os.getenv("WRIST_CAM_INDEX", "6")))
    parser.add_argument("--img-width", type=int, default=int(os.getenv("IMG_WIDTH", "640")))
    parser.add_argument("--img-height", type=int, default=int(os.getenv("IMG_HEIGHT", "480")))
    parser.add_argument("--fps", type=int, default=int(os.getenv("FPS", "10")))

    parser.add_argument("--policy-path", default=os.getenv("POLICY_PATH", DEFAULT_POLICY_PATH))
    parser.add_argument(
        "--task",
        default=os.getenv("DATASET_TASK", "Put the block in the bin"),
        help="Language instruction passed to policy inference.",
    )
    parser.add_argument(
        "--run-time-s",
        type=float,
        default=float(os.getenv("RUN_TIME_S", "0")),
        help="Total inference duration in seconds. <=0 means run until Ctrl+C.",
    )
    parser.add_argument("--log-interval", type=int, default=int(os.getenv("LOG_INTERVAL", "30")))

    parser.add_argument("--engine-dir", required=True, help="Directory containing TensorRT .engine files.")
    parser.add_argument(
        "--tensorrt-py-dir",
        default=os.getenv("TENSORRT_PY_DIR"),
        help="Optional TensorRT pip --target dir (contains `tensorrt/` and `tensorrt_libs/`).",
    )
    parser.add_argument("--vit-dtype", default=os.getenv("TRT_VIT_DTYPE", "fp16"), choices=["fp16", "fp8"])
    parser.add_argument("--llm-dtype", default=os.getenv("TRT_LLM_DTYPE", "fp16"), choices=["fp16", "fp8", "nvfp4", "nvfp4_full"])
    parser.add_argument("--dit-dtype", default=os.getenv("TRT_DIT_DTYPE", "fp16"), choices=["fp16", "fp8"])
    parser.add_argument("--num-denoising-steps", type=int, default=None)

    parser.add_argument(
        "--mock",
        type=parse_bool,
        nargs="?",
        const=True,
        default=parse_bool(os.getenv("MOCK", "false")) if os.getenv("MOCK") else False,
        help="If true, do NOT send action to robot (safe mode).",
    )
    parser.add_argument(
        "--dry-run",
        type=parse_bool,
        nargs="?",
        const=True,
        default=False,
        help="Print resolved config and exit without connecting robot.",
    )
    return parser


def main() -> None:
    register_third_party_plugins()
    args = build_parser().parse_args()

    # Common shell pitfall: using `--policy-path "$POLICY_PATH"` when POLICY_PATH is unset.
    # In that case argparse receives an empty string, and Path("") becomes "." which then fails later.
    if isinstance(args.policy_path, str) and not args.policy_path.strip():
        print(
            "[WARN] --policy-path is empty. Falling back to DEFAULT_POLICY_PATH. "
            "Did you forget to export POLICY_PATH?"
        )
        args.policy_path = DEFAULT_POLICY_PATH
    if isinstance(args.engine_dir, str) and not args.engine_dir.strip():
        raise ValueError(
            "--engine-dir is empty. Did you forget to export ENGINE_DIR (or pass an explicit path)?"
        )

    policy_path = Path(args.policy_path).expanduser()
    if not policy_path.is_dir():
        raise FileNotFoundError(f"Policy path does not exist: {policy_path}")
    policy_path = resolve_policy_dir(policy_path)

    engine_dir = Path(args.engine_dir).expanduser()
    if not engine_dir.is_dir():
        raise FileNotFoundError(f"Engine dir does not exist: {engine_dir}")
    validate_engine_dir(engine_dir, vit_dtype=args.vit_dtype, llm_dtype=args.llm_dtype, dit_dtype=args.dit_dtype)

    if args.robot_type not in {"so100_follower", "so101_follower"}:
        raise ValueError(f"Unsupported robot_type={args.robot_type!r}. Only so100_follower/so101_follower supported.")

    print(f"[INFO] Robot id: {args.robot_id}")
    print(f"[INFO] Robot type: {args.robot_type}")
    print(f"[INFO] Robot port: {args.robot_port}")
    print(f"[INFO] Policy path: {policy_path}")
    print(f"[INFO] Engine dir: {engine_dir}")
    print(f"[INFO] Task: {args.task}")
    print(f"[INFO] FPS: {args.fps}")
    print(f"[INFO] run_time_s: {args.run_time_s} (<=0 means until Ctrl+C)")
    print(f"[INFO] mock(no_send_action): {args.mock}")
    print(f"[INFO] TRT dtypes: vit={args.vit_dtype}, llm={args.llm_dtype}, dit={args.dit_dtype}")
    if args.num_denoising_steps is not None:
        print(f"[INFO] num_denoising_steps override: {args.num_denoising_steps}")

    if args.dry_run:
        print("[INFO] DRY_RUN=true, exit.")
        return

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

    # Build robot.
    from lerobot.robots import make_robot_from_config

    robot = make_robot_from_config(robot_cfg)

    # Build policy model directly from checkpoint (used for glue modules + config).
    policy_cfg = PreTrainedConfig.from_pretrained(str(policy_path))
    policy_cfg.pretrained_path = policy_path
    policy_class = get_policy_class(policy_cfg.type)
    torch_policy = policy_class.from_pretrained(str(policy_path), strict=False)
    torch_policy.to(policy_cfg.device)
    torch_policy.eval()

    trt_policy = TrtGrootPolicyAdapter(
        torch_policy,
        engine_dir,
        vit_dtype=args.vit_dtype,
        llm_dtype=args.llm_dtype,
        dit_dtype=args.dit_dtype,
        tensorrt_py_dir=args.tensorrt_py_dir,
        num_denoising_steps=args.num_denoising_steps,
    )

    # Load exact checkpoint processors.
    preprocessor, postprocessor = load_pre_post_processors(policy_path)

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

    step = 0
    start_t = time.perf_counter()
    end_t = start_t + args.run_time_s if args.run_time_s > 0 else None

    try:
        robot.connect()
        trt_policy.reset()
        preprocessor.reset()
        postprocessor.reset()

        while True:
            if end_t is not None and time.perf_counter() >= end_t:
                print("[INFO] Reached requested run_time_s. Exiting inference loop.")
                break

            loop_t = time.perf_counter()

            obs = robot.get_observation()
            obs_processed = robot_observation_processor(obs)
            observation_frame = build_dataset_frame(dataset_features, obs_processed, prefix=OBS_STR)

            action_values = predict_action(
                observation=observation_frame,
                policy=trt_policy,  # policy-like adapter with select_action()
                device=get_safe_torch_device(torch_policy.config.device),
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=torch_policy.config.use_amp,
                task=args.task,
                robot_type=robot.robot_type,
            )

            if not args.mock:
                action_dict = make_robot_action(action_values, dataset_features)
                robot_action_to_send = robot_action_processor((action_dict, obs))
                robot.send_action(robot_action_to_send)

            step += 1
            if args.log_interval > 0 and step % args.log_interval == 0:
                elapsed = time.perf_counter() - start_t
                print(f"[INFO] Step {step} | elapsed={elapsed:.2f}s")

            dt_s = time.perf_counter() - loop_t
            precise_sleep(max(1 / float(args.fps) - dt_s, 0.0))
    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt received. Stopping inference.")
    finally:
        if robot.is_connected:
            robot.disconnect()
        print("[INFO] Inference finished.")


if __name__ == "__main__":
    main()
