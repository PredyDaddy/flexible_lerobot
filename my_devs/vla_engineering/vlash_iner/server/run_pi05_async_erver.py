#!/usr/bin/env python

from __future__ import annotations

import argparse
import os
import pickle
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, Request, Response
import uvicorn

PACKAGE_PARENT = Path(__file__).resolve().parents[2]
if PACKAGE_PARENT.as_posix() not in sys.path:
    sys.path.insert(0, PACKAGE_PARENT.as_posix())

from vlash_iner.common import (
    DEFAULT_POLICY_PATH,
    KNOWN_TASKS,
    ensure_repo_on_path,
    env_bool,
    parse_bool,
    resolve_repo_root,
    validate_policy_artifacts,
)


REPO_ROOT = resolve_repo_root(Path(__file__))
ensure_repo_on_path(REPO_ROOT)

from vlash_iner.policy_loader import Pi05PolicyBundle, load_pi05_bundle, load_pre_post_processors, log_info
from vlash_iner.run_pi05_compile_warmup import load_compiled_policy, warmup_compiled_bundle
from vlash_iner.safety import validate_action_chunk


INFER_LOCK = threading.Lock()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PI0.5 async inference server. Receives robot observations and returns action chunks."
    )
    parser.add_argument("--host", default=os.getenv("SERVER_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("SERVER_PORT", "8008")))
    parser.add_argument("--endpoint", default=os.getenv("SERVER_ENDPOINT", "/infer"))
    parser.add_argument("--policy-path", default=os.getenv("POLICY_PATH", DEFAULT_POLICY_PATH))
    parser.add_argument(
        "--task",
        default=os.getenv("DATASET_TASK"),
        help="Default task instruction. Client may override per request.",
    )
    parser.add_argument(
        "--task-id",
        choices=sorted(KNOWN_TASKS),
        default=os.getenv("DATASET_TASK_ID"),
        help="Known task alias used when --task is not provided.",
    )
    parser.add_argument("--list-tasks", action="store_true", help="Print known task aliases and exit.")
    parser.add_argument("--robot-type", default=os.getenv("ROBOT_TYPE", "so101_follower"))
    parser.add_argument(
        "--backend",
        choices=["torch", "torch_compile", "tensorrt_split"],
        default=os.getenv("PI05_BACKEND"),
        help=(
            "Inference backend. Defaults to torch_compile when --compile-model=true, otherwise torch. "
            "Use tensorrt_split to patch policy.model.sample_actions with split TensorRT engines."
        ),
    )
    parser.add_argument(
        "--device",
        default=os.getenv("DEVICE"),
        help="Override policy device, e.g. cuda or cpu.",
    )
    parser.add_argument(
        "--compile-model",
        type=parse_bool,
        nargs="?",
        const=True,
        default=env_bool("COMPILE_MODEL", False),
        help="Load policy with torch.compile before serving.",
    )
    parser.add_argument("--compile-mode", default=os.getenv("COMPILE_MODE", "reduce-overhead"))
    parser.add_argument(
        "--prefix-engine-path",
        default=os.getenv(
            "PI05_TRT_PREFIX_ENGINE",
            "my_devs/openpi_trt/artifacts/pi05_so101_prefix_cache_b1_fp32.engine",
        ),
        help="TensorRT split backend prefix_cache engine path.",
    )
    parser.add_argument(
        "--denoise-engine-path",
        default=os.getenv(
            "PI05_TRT_DENOISE_ENGINE",
            "my_devs/openpi_trt/artifacts/pi05_so101_denoise_step_b1_fp32.engine",
        ),
        help="TensorRT split backend denoise_step engine path.",
    )
    parser.add_argument(
        "--trt-model-dtype",
        choices=["float32", "bfloat16"],
        default=os.getenv("PI05_TRT_MODEL_DTYPE", "float32"),
        help=(
            "Policy dtype override before patching TensorRT split backend. "
            "Existing FP32 engines expect float32."
        ),
    )
    parser.add_argument("--warmup-steps", type=int, default=int(os.getenv("WARMUP_STEPS", "0")))
    parser.add_argument("--img-width", type=int, default=int(os.getenv("IMG_WIDTH", "640")))
    parser.add_argument("--img-height", type=int, default=int(os.getenv("IMG_HEIGHT", "480")))
    parser.add_argument("--state-dim", type=int, default=int(os.getenv("STATE_DIM", "6")))
    parser.add_argument(
        "--summarize-artifacts",
        type=parse_bool,
        nargs="?",
        const=True,
        default=env_bool("SUMMARIZE_ARTIFACTS", False),
    )
    parser.add_argument(
        "--sync-cuda-for-timing",
        type=parse_bool,
        nargs="?",
        const=True,
        default=env_bool("SYNC_CUDA_FOR_TIMING", False),
        help="Synchronize CUDA before returning infer_time. More accurate but can reduce throughput.",
    )
    parser.add_argument(
        "--access-log",
        type=parse_bool,
        nargs="?",
        const=True,
        default=env_bool("ACCESS_LOG", False),
    )
    return parser


def resolve_task(parser: argparse.ArgumentParser, args: argparse.Namespace) -> str:
    if args.list_tasks:
        print("[INFO] Known task aliases:")
        for task_id, task in KNOWN_TASKS.items():
            print(f"[INFO]   {task_id}: {task}")
        raise SystemExit(0)

    task = args.task if args.task is not None else (KNOWN_TASKS.get(args.task_id) if args.task_id else None)
    if task is None:
        parser.error(
            "A task must be specified for PI0.5 server inference. "
            "Use --task-id eraser_to_box, --task-id cup_to_upper_right, "
            "--task-id eraser_then_cup, or pass --task \"...\"."
        )
    return task


class Pi05InferServer:
    def __init__(
        self,
        *,
        bundle: Pi05PolicyBundle,
        default_task: str,
        default_robot_type: str,
        sync_cuda_for_timing: bool,
        backend: str,
        backend_info: dict[str, Any] | None = None,
    ) -> None:
        self.bundle = bundle
        self.default_task = default_task
        self.default_robot_type = default_robot_type
        self.sync_cuda_for_timing = sync_cuda_for_timing
        self.backend = backend
        self.backend_info = backend_info or {}
        self.request_count = 0
        self.last_infer_s = 0.0

    def health(self) -> dict[str, Any]:
        policy_cfg = self.bundle.policy.config
        return {
            "ready": True,
            "policy_path": str(self.bundle.policy_path),
            "policy_type": getattr(policy_cfg, "type", None),
            "backend": self.backend,
            "backend_info": self.backend_info,
            "device": str(self.bundle.device),
            "chunk_size": getattr(policy_cfg, "chunk_size", None),
            "n_action_steps": getattr(policy_cfg, "n_action_steps", None),
            "default_task": self.default_task,
            "default_robot_type": self.default_robot_type,
            "request_count": self.request_count,
            "last_infer_s": self.last_infer_s,
        }

    def reset(self) -> dict[str, Any]:
        with INFER_LOCK:
            self.bundle.reset()
        return {"ok": True}

    def infer(self, request_data: dict[str, Any]) -> dict[str, Any]:
        from lerobot.policies.utils import prepare_observation_for_inference

        if not isinstance(request_data, dict):
            raise ValueError(f"Expected request dict, got {type(request_data)!r}")
        observation_frame = request_data.get("observation")
        if not isinstance(observation_frame, dict):
            raise ValueError("Request must contain an 'observation' dict.")

        task = request_data.get("task") or self.default_task
        robot_type = request_data.get("robot_type") or self.default_robot_type
        future_state = request_data.get("future_state")
        n_action_steps = request_data.get("n_action_steps")

        observation = dict(observation_frame)
        if future_state is not None:
            observation["observation.state"] = np.asarray(future_state, dtype=np.float32)

        start_t = time.perf_counter()
        with INFER_LOCK:
            observation = prepare_observation_for_inference(observation, self.bundle.device, task, robot_type)
            observation = self.bundle.preprocessor(observation)
            with torch.inference_mode():
                raw_chunk = self.bundle.policy.predict_action_chunk(observation)
                processed = self.bundle.postprocessor(raw_chunk)
            if self.sync_cuda_for_timing and self.bundle.device.type == "cuda":
                torch.cuda.synchronize(self.bundle.device)

        chunk = processed.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        chunk = validate_action_chunk(chunk)
        if n_action_steps is not None:
            chunk = chunk[: int(n_action_steps)]

        infer_s = time.perf_counter() - start_t
        self.request_count += 1
        self.last_infer_s = infer_s
        return {
            "action_chunk": chunk,
            "infer_time_s": infer_s,
            "request_count": self.request_count,
            "chunk_shape": tuple(chunk.shape),
            "backend": self.backend,
        }


def create_app(server: Pi05InferServer, *, endpoint: str) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return server.health()

    @app.post("/reset")
    async def reset() -> Response:
        return Response(
            content=pickle.dumps(server.reset(), protocol=pickle.HIGHEST_PROTOCOL),
            media_type="application/octet-stream",
        )

    @app.post(endpoint)
    async def infer(request: Request) -> Response:
        body = await request.body()
        if not body:
            return Response(content=b"empty request body", status_code=400)
        try:
            data = pickle.loads(body)
            result = server.infer(data)
        except Exception as exc:
            error = {"ok": False, "error": repr(exc)}
            return Response(
                content=pickle.dumps(error, protocol=pickle.HIGHEST_PROTOCOL),
                status_code=500,
                media_type="application/octet-stream",
            )
        return Response(
            content=pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL),
            media_type="application/octet-stream",
        )

    return app


def resolve_backend(args: argparse.Namespace) -> str:
    if args.backend is not None:
        return args.backend
    return "torch_compile" if args.compile_model else "torch"


def resolve_engine_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def patch_tensorrt_split_backend(
    bundle: Pi05PolicyBundle,
    *,
    prefix_engine_path: Path,
    denoise_engine_path: Path,
) -> dict[str, Any]:
    openpi_trt_dir = REPO_ROOT / "my_devs/openpi_trt"
    openpi_trt_str = openpi_trt_dir.as_posix()
    if openpi_trt_str not in sys.path:
        sys.path.insert(0, openpi_trt_str)

    from runtime.pi05_trt_split import patch_sample_actions_with_split_trt

    if not prefix_engine_path.is_file():
        raise FileNotFoundError(f"TensorRT prefix_cache engine does not exist: {prefix_engine_path}")
    if not denoise_engine_path.is_file():
        raise FileNotFoundError(f"TensorRT denoise_step engine does not exist: {denoise_engine_path}")

    log_info(f"Patching PI0.5 sample_actions with TensorRT split backend.")
    log_info(f"TensorRT prefix engine: {prefix_engine_path}")
    log_info(f"TensorRT denoise engine: {denoise_engine_path}")
    runtime = patch_sample_actions_with_split_trt(
        bundle.policy,
        prefix_engine_path,
        denoise_engine_path,
    )
    return {
        "prefix_engine_path": str(prefix_engine_path),
        "denoise_engine_path": str(denoise_engine_path),
        "runtime": runtime.describe(),
    }


def load_bundle(
    args: argparse.Namespace,
    *,
    policy_path: Path,
    backend: str,
) -> tuple[Pi05PolicyBundle, dict[str, Any]]:
    validate_policy_artifacts(policy_path, summarize=args.summarize_artifacts)
    if backend == "torch":
        bundle = load_pi05_bundle(
            policy_path,
            repo_root=REPO_ROOT,
            strict=False,
            check_artifacts=False,
            summarize_artifacts=False,
        )
        return bundle, {}

    if backend == "torch_compile":
        policy, device = load_compiled_policy(policy_path, device=args.device, compile_mode=args.compile_mode)
        preprocessor, postprocessor = load_pre_post_processors(policy_path)
        bundle = Pi05PolicyBundle(
            policy=policy,
            policy_cfg=policy.config,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            device=device,
            policy_path=policy_path,
        )
        if args.warmup_steps > 0:
            warmup_compiled_bundle(
                bundle,
                task=resolve_task(build_parser(), args),
                robot_type=args.robot_type,
                warmup_steps=args.warmup_steps,
                img_height=args.img_height,
                img_width=args.img_width,
                state_dim=args.state_dim,
            )
        return bundle, {"compile_mode": args.compile_mode, "warmup_steps": args.warmup_steps}

    if backend == "tensorrt_split":
        from lerobot.utils.utils import get_safe_torch_device

        openpi_trt_dir = REPO_ROOT / "my_devs/openpi_trt"
        openpi_trt_str = openpi_trt_dir.as_posix()
        if openpi_trt_str not in sys.path:
            sys.path.insert(0, openpi_trt_str)
        from scripts.pi05_onnx_common import load_policy

        trt_device = args.device or "cuda"
        log_info(
            f"Loading TensorRT-backed policy with device={trt_device} "
            f"dtype={args.trt_model_dtype}..."
        )
        policy = load_policy(policy_path, device=trt_device, model_dtype=args.trt_model_dtype)
        device = get_safe_torch_device(policy.config.device)
        preprocessor, postprocessor = load_pre_post_processors(policy_path)
        bundle = Pi05PolicyBundle(
            policy=policy,
            policy_cfg=policy.config,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            device=device,
            policy_path=policy_path,
        )
        backend_info = patch_tensorrt_split_backend(
            bundle,
            prefix_engine_path=resolve_engine_path(args.prefix_engine_path),
            denoise_engine_path=resolve_engine_path(args.denoise_engine_path),
        )
        backend_info["model_dtype"] = args.trt_model_dtype
        return bundle, backend_info

    raise ValueError(f"Unsupported backend: {backend}")


def main() -> None:
    from lerobot.utils.import_utils import register_third_party_plugins

    register_third_party_plugins()
    parser = build_parser()
    args = parser.parse_args()
    os.chdir(REPO_ROOT)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    task = resolve_task(parser, args)
    policy_path = Path(args.policy_path).expanduser().resolve()
    if not policy_path.is_dir():
        raise FileNotFoundError(f"Policy path does not exist: {policy_path}")
    backend = resolve_backend(args)

    log_info("Starting PI0.5 async inference server.")
    log_info(f"Repo root: {REPO_ROOT}")
    log_info(f"Policy path: {policy_path}")
    log_info(f"Task: {task}")
    log_info(f"Robot type: {args.robot_type}")
    log_info(f"backend: {backend}")
    log_info(f"compile_model: {args.compile_model}")
    log_info(f"warmup_steps: {args.warmup_steps}")

    bundle, backend_info = load_bundle(args, policy_path=policy_path, backend=backend)
    server = Pi05InferServer(
        bundle=bundle,
        default_task=task,
        default_robot_type=args.robot_type,
        sync_cuda_for_timing=args.sync_cuda_for_timing,
        backend=backend,
        backend_info=backend_info,
    )
    app = create_app(server, endpoint=args.endpoint)
    print(f"[INFO] PI0.5 server listening on http://{args.host}:{args.port}{args.endpoint}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, access_log=args.access_log)


if __name__ == "__main__":
    main()
