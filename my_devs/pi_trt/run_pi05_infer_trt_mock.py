#!/usr/bin/env python

"""Safe mock compare: PyTorch vs TensorRT engines for PI0.5.

This script is designed to reduce deployment risk before wiring TensorRT into the
real robot loop:

1) Prepare observations from:
   - deterministic random mock frames (`--source random`), OR
   - a previously recorded frames directory (`--source frames_dir`)
2) Run policy inference on the SAME frames with:
   - PyTorch policy inference (baseline)
   - TensorRT engines for PI0.5's 3-stage topology
3) Compare action outputs and write artifacts to disk

Notes:
- This script never sends actions to a robot.
- It reuses the checkpoint-saved pre/post processors to match runtime behavior.
- TensorRT inference uses engines built from:
  - `vision_encoder_fp16.onnx`
  - `prefix_cache_fp16.onnx`
  - `denoise_step_fp16.onnx`

TODO:
- Add optional robot-frame capture if we decide to reuse this as a staged
  pre-deployment tool. For now we intentionally keep the script focused on the
  no-action path.
"""

from __future__ import annotations

import argparse
import gc
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


def _find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "lerobot").is_dir():
            return candidate
    raise RuntimeError(f"Failed to locate repo root from: {start}")


REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent)
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

os.chdir(REPO_ROOT)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from lerobot import policies  # noqa: F401  # Register policy config classes.
from lerobot.configs.policies import PreTrainedConfig
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.processor.converters import (
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.control_utils import predict_action
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import get_safe_torch_device

from my_devs.groot_trt.trt_utils import TrtSession
from my_devs.pi_trt.common import (
    DEFAULT_POLICY_PATH,
    build_prefix_from_image_embeddings,
    ensure_local_tokenizer_dir,
    load_policy,
    numpy_metrics,
    prepare_batch,
    prepare_policy_for_fp16,
    resolve_policy_dir,
    save_json,
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


def _now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _np(tensor: torch.Tensor) -> np.ndarray:
    if tensor.is_cuda:
        torch.cuda.synchronize()
    return tensor.detach().cpu().contiguous().numpy()


def _save_frame(frame_path: Path, frame: dict[str, np.ndarray]) -> None:
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    with frame_path.open("wb") as file:
        pickle.dump(frame, file, protocol=4)


def _load_frame(frame_path: Path) -> dict[str, np.ndarray]:
    with frame_path.open("rb") as file:
        return pickle.load(file)


def load_pre_post_processors(
    policy_path: Path,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    ensure_local_tokenizer_dir(REPO_ROOT)
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


def record_frames_random(
    frames_dir: Path,
    *,
    policy_cfg: PreTrainedConfig,
    num_steps: int,
    img_height: int,
    img_width: int,
    seed: int,
) -> list[Path]:
    rng = np.random.default_rng(int(seed))
    image_feature_keys = list(policy_cfg.image_features)
    if not image_feature_keys:
        raise ValueError("PI0.5 random mock requires image_features in the checkpoint config.")
    state_dim = int(policy_cfg.input_features[OBS_STATE].shape[0])

    paths: list[Path] = []
    for step in range(num_steps):
        frame: dict[str, np.ndarray] = {
            OBS_STATE: rng.standard_normal(size=(state_dim,)).astype(np.float32),
        }
        for image_key in image_feature_keys:
            frame[image_key] = rng.integers(0, 256, size=(img_height, img_width, 3), dtype=np.uint8)
        frame_path = frames_dir / f"{step:06d}.pkl"
        _save_frame(frame_path, frame)
        paths.append(frame_path)
    return paths


@dataclass
class TrtEnginePaths:
    vision: Path
    prefix: Path
    denoise: Path


class TrtPi05PolicyAdapter:
    """A minimal policy-like adapter that runs PI0.5 with 3 TensorRT engines.

    It reuses the checkpoint-loaded PyTorch policy for the non-exported "glue":
    - image preprocessing
    - language token embedding
    - attention-mask preparation
    - action queue semantics
    """

    def __init__(
        self,
        torch_policy: torch.nn.Module,
        engine_dir: Path,
        *,
        tensorrt_py_dir: str | None = None,
        num_inference_steps: int | None = None,
    ) -> None:
        self.torch_policy = torch_policy
        self.config = getattr(torch_policy, "config", None)
        self.device = next(torch_policy.parameters()).device
        self.original_action_dim = int(self.config.output_features[ACTION].shape[0])
        self.num_inference_steps = (
            int(num_inference_steps) if num_inference_steps is not None else int(self.config.num_inference_steps)
        )

        paths = TrtEnginePaths(
            vision=engine_dir / "vision_encoder_fp16.engine",
            prefix=engine_dir / "prefix_cache_fp16.engine",
            denoise=engine_dir / "denoise_step_fp16.engine",
        )
        missing = [path for path in paths.__dict__.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing required TensorRT engine files:\n" + "\n".join(f"  - {p}" for p in missing))

        self.sess_vision = TrtSession.load(paths.vision, trt_py_dir=tensorrt_py_dir)
        self.sess_prefix = TrtSession.load(paths.prefix, trt_py_dir=tensorrt_py_dir)
        self.sess_denoise = TrtSession.load(paths.denoise, trt_py_dir=tensorrt_py_dir)

        self._action_queue: deque[torch.Tensor] = deque(maxlen=int(self.config.n_action_steps))

    def reset(self) -> None:
        self._action_queue.clear()

    @torch.inference_mode()
    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : int(self.config.n_action_steps)]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.inference_mode()
    def predict_action_chunk(
        self,
        batch: dict[str, torch.Tensor],
        *,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.torch_policy.eval()

        prepared = prepare_batch(self.torch_policy, batch)
        image_embeddings = self.sess_vision.run({"image_tensor": prepared.image_tensor})["image_embeddings"].to(
            torch.float16
        )

        prefix_embs, prefix_pad_masks, _prefix_att_masks, prefix_attention_mask_4d, prefix_position_ids = (
            build_prefix_from_image_embeddings(
                self.torch_policy,
                image_embeddings,
                prepared.image_masks,
                prepared.tokens,
                prepared.masks,
            )
        )
        kv_cache = self.sess_prefix.run(
            {
                "prefix_embs": prefix_embs,
                "prefix_attention_mask_4d": prefix_attention_mask_4d,
                "prefix_position_ids": prefix_position_ids,
            }
        )["kv_cache"].to(torch.float16)

        if noise is None:
            noise = torch.randn(
                (
                    1,
                    int(self.config.chunk_size),
                    int(self.config.max_action_dim),
                ),
                dtype=torch.float16,
                device=self.device,
            )

        x_t = noise.to(device=self.device, dtype=torch.float16).contiguous()
        dt = -1.0 / float(self.num_inference_steps)
        for step in range(self.num_inference_steps):
            time_value = 1.0 + step * dt
            timestep = torch.full((1,), time_value, dtype=torch.float32, device=self.device)
            velocity = self.sess_denoise.run(
                {
                    "x_t": x_t,
                    "timestep": timestep,
                    "prefix_pad_masks": prefix_pad_masks,
                    "kv_cache": kv_cache,
                }
            )["velocity"].to(torch.float16)
            x_t = (x_t + dt * velocity).to(torch.float16)

        return x_t[:, :, : self.original_action_dim]


class SeededPolicyWrapper:
    """Wrap a policy-like object to make chunk-level sampling reproducible."""

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
        if queue is not None and len(queue) == 0:
            chunk_seed = self.seed + self._chunk_idx
            torch.manual_seed(chunk_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(chunk_seed)
            self._chunk_idx += 1
        return self.policy.select_action(batch)


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
    if hasattr(policy, "reset"):
        policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    seeded_policy = SeededPolicyWrapper(policy, seed=seed)
    seeded_policy.reset()

    policy_device = getattr(getattr(policy, "config", None), "device", "cuda")
    device = get_safe_torch_device(policy_device)
    actions: list[np.ndarray] = []
    start = time.perf_counter()
    for index, frame_path in enumerate(frame_paths):
        observation = _load_frame(frame_path)
        action_t = predict_action(
            observation=observation,
            policy=seeded_policy,  # type: ignore[arg-type]
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=use_amp,
            task=task,
            robot_type=robot_type,
        )
        action_np = _np(action_t).astype(np.float32)
        if action_np.ndim == 2 and action_np.shape[0] == 1:
            action_np = action_np[0]
        actions.append(action_np)
        if (index + 1) % 10 == 0:
            elapsed = time.perf_counter() - start
            print(f"[INFO] Inference step {index + 1}/{len(frame_paths)} elapsed={elapsed:.2f}s")

    actions_arr = np.stack(actions, axis=0)
    total_s = time.perf_counter() - start
    save_json(
        json_out,
        {
            "num_frames": len(frame_paths),
            "actions_shape": list(actions_arr.shape),
            "total_seconds": total_s,
            "fps_effective": float(len(frame_paths) / total_s) if total_s > 0 else float("nan"),
        },
    )
    return actions_arr


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mock compare: PI0.5 PyTorch vs TensorRT engine outputs.")
    parser.add_argument("--policy-path", default=os.getenv("POLICY_PATH", DEFAULT_POLICY_PATH))
    parser.add_argument("--engine-dir", required=True, help="Directory containing PI0.5 TensorRT .engine files.")
    parser.add_argument(
        "--tensorrt-py-dir",
        default=os.getenv("TENSORRT_PY_DIR"),
        help="Optional TensorRT pip --target dir (contains `tensorrt/` and `tensorrt_libs/`).",
    )
    parser.add_argument("--task", default=os.getenv("DATASET_TASK", "Put the block in the bin"))
    parser.add_argument("--robot-type", default=os.getenv("ROBOT_TYPE", "so101_follower"))
    parser.add_argument(
        "--source",
        default=os.getenv("MOCK_SOURCE", "random"),
        choices=["random", "frames_dir"],
        help="Where to get observations. This script intentionally omits live robot capture.",
    )
    parser.add_argument(
        "--frames-dir",
        default=os.getenv("FRAMES_DIR"),
        help="When --source=frames_dir, load frames from this directory (pickle files).",
    )
    parser.add_argument("--out-dir", default=os.getenv("OUT_DIR"), help="Default: outputs/pi_trt/mock_compare_<ts>/")
    parser.add_argument("--seed", type=int, default=int(os.getenv("SEED", "20260310")))
    parser.add_argument("--num-steps", type=int, default=int(os.getenv("NUM_STEPS", "32")))
    parser.add_argument("--img-width", type=int, default=int(os.getenv("IMG_WIDTH", "640")))
    parser.add_argument("--img-height", type=int, default=int(os.getenv("IMG_HEIGHT", "480")))
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="Optional override for TRT denoising steps. Default: checkpoint config.",
    )
    parser.add_argument(
        "--dry-run",
        type=parse_bool,
        nargs="?",
        const=True,
        default=False,
        help="Print resolved config and exit without running inference.",
    )
    return parser


def _resolve_out_dir(args: argparse.Namespace) -> Path:
    if args.out_dir:
        return Path(args.out_dir).expanduser()
    return Path("outputs/pi_trt") / f"mock_compare_{_now_ts()}"


def main() -> None:
    register_third_party_plugins()
    args = build_parser().parse_args()

    policy_path = resolve_policy_dir(args.policy_path)
    engine_dir = Path(args.engine_dir).expanduser()
    if not engine_dir.is_dir():
        raise FileNotFoundError(f"--engine-dir does not exist: {engine_dir}")

    out_dir = _resolve_out_dir(args)
    frames_dir = out_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    policy_cfg = PreTrainedConfig.from_pretrained(str(policy_path))
    policy_cfg.pretrained_path = policy_path

    print(f"[INFO] policy_path: {policy_path}")
    print(f"[INFO] engine_dir: {engine_dir}")
    print(f"[INFO] source: {args.source}")
    print(f"[INFO] out_dir: {out_dir}")
    print(f"[INFO] seed: {args.seed}")
    print(f"[INFO] task: {args.task}")
    print(f"[INFO] robot_type: {args.robot_type}")
    print(f"[INFO] num_steps: {args.num_steps}")
    print(f"[INFO] img_size: {args.img_width}x{args.img_height}")
    print(f"[INFO] tensorrt_py_dir: {args.tensorrt_py_dir}")
    if args.num_inference_steps is not None:
        print(f"[INFO] TRT num_inference_steps override: {args.num_inference_steps}")

    if args.dry_run:
        print("[INFO] DRY_RUN=true, exiting.")
        return

    if args.source == "frames_dir":
        if not args.frames_dir:
            raise ValueError("--frames-dir is required when --source=frames_dir")
        frames_dir = Path(args.frames_dir).expanduser()
        frame_paths = sorted(frames_dir.glob("*.pkl"))
        if not frame_paths:
            raise FileNotFoundError(f"No *.pkl frames found in: {frames_dir}")
    else:
        frame_paths = record_frames_random(
            frames_dir,
            policy_cfg=policy_cfg,
            num_steps=int(args.num_steps),
            img_height=int(args.img_height),
            img_width=int(args.img_width),
            seed=int(args.seed),
        )

    print(f"[OK] Prepared {len(frame_paths)} frame(s). frames_dir={frames_dir}")

    torch.cuda.empty_cache()
    _, torch_policy = load_policy(policy_path, device="cuda")
    use_amp = bool(getattr(torch_policy.config, "use_amp", False))
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

    del torch_policy
    gc.collect()
    torch.cuda.empty_cache()

    _, trt_glue_policy = load_policy(policy_path, device="cuda")
    prepare_policy_for_fp16(trt_glue_policy)
    trt_policy = TrtPi05PolicyAdapter(
        trt_glue_policy,
        engine_dir,
        tensorrt_py_dir=args.tensorrt_py_dir,
        num_inference_steps=args.num_inference_steps,
    )

    print("[INFO] Running TensorRT mock inference...")
    actions_trt = run_inference_on_frames(
        policy=trt_policy,
        policy_path=policy_path,
        frame_paths=frame_paths,
        task=args.task,
        robot_type=args.robot_type,
        seed=int(args.seed),
        json_out=out_dir / "trt_infer_meta.json",
        use_amp=False,
    )
    trt_actions_path = out_dir / "actions_trt.npy"
    np.save(trt_actions_path, actions_trt)
    print(f"[OK] Saved TensorRT actions: {trt_actions_path} shape={actions_trt.shape}")

    report = {
        "policy_path": str(policy_path),
        "engine_dir": str(engine_dir),
        "source": args.source,
        "frames_dir": str(frames_dir),
        "num_frames": len(frame_paths),
        "task": args.task,
        "robot_type": args.robot_type,
        "seed": int(args.seed),
        "torch_use_amp": use_amp,
        "trt_num_inference_steps": (
            int(args.num_inference_steps) if args.num_inference_steps is not None else int(policy_cfg.num_inference_steps)
        ),
        "results": {
            "actions": numpy_metrics(actions_torch, actions_trt),
        },
        "artifacts": {
            "torch_actions": str(torch_actions_path),
            "trt_actions": str(trt_actions_path),
            "torch_meta": str(out_dir / "torch_infer_meta.json"),
            "trt_meta": str(out_dir / "trt_infer_meta.json"),
        },
    }
    report_path = out_dir / "compare_metrics_trt_mock.json"
    save_json(report_path, report)
    print(
        "[RESULT] actions: "
        f"cos={report['results']['actions']['cosine']:.8f}, "
        f"rmse={report['results']['actions']['rmse']:.8f}, "
        f"mean_abs={report['results']['actions']['mean_abs']:.8f}, "
        f"max_abs={report['results']['actions']['max_abs']:.8f}"
    )
    print(f"[OK] Mock compare report saved to: {report_path}")


if __name__ == "__main__":
    main()
