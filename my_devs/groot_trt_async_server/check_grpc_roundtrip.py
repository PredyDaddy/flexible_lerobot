#!/usr/bin/env python

"""Local gRPC smoke test for the GR00T TensorRT async server stack.

This starts the async server in-process, connects to it over a real gRPC channel,
loads the configured backend, submits one synthetic observation, and waits for one
action chunk response.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
import uuid
from concurrent import futures
from pathlib import Path

import grpc
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

from lerobot.async_inference.helpers import TimedObservation
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
from my_devs.groot_trt_async_server.configs import GrootTrtPolicyServerConfig, GrootTrtRemotePolicyConfig
from my_devs.groot_trt_async_server.policy_server import (
    GrootTrtPolicyServer,
    SESSION_ID_METADATA_KEY,
    SESSION_MODE_CLAIM,
    SESSION_MODE_METADATA_KEY,
    SESSION_MODE_RELEASE,
)
from my_devs.groot_trt_async_server.robot_client import build_observation_payload, unpack_action_payload


def _session_metadata(session_id: str, *, mode: str) -> tuple[tuple[str, str], ...]:
    return (
        (SESSION_ID_METADATA_KEY, session_id),
        (SESSION_MODE_METADATA_KEY, mode),
    )


def _synthetic_lerobot_features(*, image_height: int, image_width: int) -> dict[str, dict[str, object]]:
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": [f"joint_{i}" for i in range(6)],
        },
        "observation.images.top": {
            "dtype": "image",
            "shape": (image_height, image_width, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.wrist": {
            "dtype": "image",
            "shape": (image_height, image_width, 3),
            "names": ["height", "width", "channels"],
        },
    }


def _synthetic_image(*, image_height: int, image_width: int, invert: bool) -> np.ndarray:
    y = np.linspace(0, 255, image_height, dtype=np.uint8)[:, None]
    x = np.linspace(0, 255, image_width, dtype=np.uint8)[None, :]
    if invert:
        base = np.bitwise_xor(y, x)
    else:
        base = np.bitwise_or(y, x)
    return np.stack((base, np.flipud(base), np.fliplr(base)), axis=-1).astype(np.uint8, copy=False)


def _synthetic_raw_observation(*, image_height: int, image_width: int, task: str) -> dict[str, object]:
    return {
        **{f"joint_{i}": float(i) / 10.0 for i in range(6)},
        "top": _synthetic_image(image_height=image_height, image_width=image_width, invert=False),
        "wrist": _synthetic_image(image_height=image_height, image_width=image_width, invert=True),
        "task": task,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a local GR00T async gRPC smoke test with synthetic inputs.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--backend", default="tensorrt", choices=["tensorrt", "pytorch"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--actions-per-chunk", type=int, default=4)
    parser.add_argument("--task", default="Put the block in the bin")
    parser.add_argument("--resource-profile", default="default")
    parser.add_argument("--resource-policy-path", default=None)
    parser.add_argument("--resource-engine-dir", default=None)
    parser.add_argument("--resource-tensorrt-py-dir", default=None)
    parser.add_argument("--vit-dtype", default="fp16", choices=["fp16", "fp8"])
    parser.add_argument("--llm-dtype", default="fp16", choices=["fp16", "fp8", "nvfp4", "nvfp4_full"])
    parser.add_argument("--dit-dtype", default="fp16", choices=["fp16", "fp8"])
    parser.add_argument("--num-denoising-steps", type=int, default=2)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--channel-ready-timeout-s", type=float, default=10.0)
    parser.add_argument("--policy-setup-timeout-s", type=float, default=600.0)
    parser.add_argument("--request-timeout-s", type=float, default=120.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.backend == "tensorrt" and not torch.cuda.is_available():
        raise RuntimeError("TensorRT smoke test requires CUDA, but torch.cuda.is_available() returned False.")

    session_id = uuid.uuid4().hex
    claim_metadata = _session_metadata(session_id, mode=SESSION_MODE_CLAIM)
    release_metadata = _session_metadata(session_id, mode=SESSION_MODE_RELEASE)

    config_kwargs = {
        "host": args.host,
        "port": args.port,
        "resource_profile": args.resource_profile,
    }
    if args.resource_policy_path is not None:
        config_kwargs["resource_policy_path"] = args.resource_policy_path
    if args.resource_engine_dir is not None:
        config_kwargs["resource_engine_dir"] = args.resource_engine_dir
    if args.resource_tensorrt_py_dir is not None:
        config_kwargs["resource_tensorrt_py_dir"] = args.resource_tensorrt_py_dir

    cfg = GrootTrtPolicyServerConfig(**config_kwargs)
    server_impl = GrootTrtPolicyServer(cfg)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4), options=grpc_channel_options())
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(server_impl, server)
    server.add_insecure_port(f"{args.host}:{args.port}")

    channel = None
    try:
        print(f"Starting local gRPC server on {args.host}:{args.port}")
        server.start()

        channel = grpc.insecure_channel(
            f"{args.host}:{args.port}",
            options=grpc_channel_options(),
        )
        grpc.channel_ready_future(channel).result(timeout=args.channel_ready_timeout_s)
        stub = services_pb2_grpc.AsyncInferenceStub(channel)

        print(f"Ready(): session_id={session_id}")
        stub.Ready(services_pb2.Empty(), metadata=claim_metadata, timeout=args.request_timeout_s)

        policy_cfg = GrootTrtRemotePolicyConfig(
            policy_type="groot",
            pretrained_name_or_path=f"server-resource://{args.resource_profile}",
            lerobot_features=_synthetic_lerobot_features(
                image_height=args.image_height,
                image_width=args.image_width,
            ),
            actions_per_chunk=args.actions_per_chunk,
            device=args.device,
            resource_profile=args.resource_profile,
            backend=args.backend,
            engine_dir=None,
            tensorrt_py_dir=None,
            vit_dtype=args.vit_dtype,
            llm_dtype=args.llm_dtype,
            dit_dtype=args.dit_dtype,
            num_denoising_steps=args.num_denoising_steps,
        )

        print(
            "SendPolicyInstructions(): "
            f"backend={args.backend} device={args.device} actions_per_chunk={args.actions_per_chunk}"
        )
        setup_start = time.perf_counter()
        stub.SendPolicyInstructions(
            services_pb2.PolicySetup(data=pickle.dumps(policy_cfg)),
            metadata=claim_metadata,
            timeout=args.policy_setup_timeout_s,
        )
        setup_seconds = time.perf_counter() - setup_start
        print(f"Policy setup completed in {setup_seconds:.3f}s")

        observation = TimedObservation(
            timestamp=time.time(),
            timestep=0,
            must_go=True,
            observation=_synthetic_raw_observation(
                image_height=args.image_height,
                image_width=args.image_width,
                task=args.task,
            ),
        )
        request_id = 0
        payload = build_observation_payload(observation, request_id=request_id, session_id=session_id)

        print(f"SendObservations(): request_id={request_id} timestep={observation.get_timestep()}")
        stub.SendObservations(
            send_bytes_in_chunks(
                pickle.dumps(payload),
                services_pb2.Observation,
                log_prefix="[grpc-smoke]",
                silent=True,
            ),
            metadata=claim_metadata,
            timeout=args.request_timeout_s,
        )

        print("GetActions()")
        get_actions_start = time.perf_counter()
        response = stub.GetActions(
            services_pb2.Empty(),
            metadata=claim_metadata,
            timeout=args.request_timeout_s,
        )
        get_actions_seconds = time.perf_counter() - get_actions_start

        if not response.data:
            raise RuntimeError("GetActions() returned an empty payload.")

        decoded = unpack_action_payload(pickle.loads(response.data))
        if decoded.request_state != "ack":
            raise RuntimeError(
                f"Smoke test expected request_state='ack', got {decoded.request_state!r} "
                f"(reason={decoded.request_state_reason!r})."
            )
        if not decoded.timed_actions:
            raise RuntimeError("Smoke test received ack but no actions were returned.")

        first_action = decoded.timed_actions[0].get_action()
        print(
            "Smoke passed: "
            f"request_state={decoded.request_state} "
            f"acked_request_id={decoded.request_id} "
            f"observation_timestep={decoded.observation_timestep} "
            f"action_count={len(decoded.timed_actions)} "
            f"first_action_shape={tuple(first_action.shape)} "
            f"first_action_dtype={first_action.dtype} "
            f"get_actions_seconds={get_actions_seconds:.3f}"
        )

        stub.Ready(services_pb2.Empty(), metadata=release_metadata, timeout=args.request_timeout_s)
        print("Release(): ok")
        return 0
    finally:
        if channel is not None:
            channel.close()
        server.stop(grace=0).wait()


if __name__ == "__main__":
    raise SystemExit(main())
