#!/usr/bin/env python

"""CLI entrypoint for the GR00T TensorRT async policy server.

This file intentionally stays thin. The actual server implementation and config
ownership belong to the async server layer under the same module directory.
"""

from __future__ import annotations

import argparse
import importlib
import pprint
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "lerobot").is_dir():
            return candidate
    raise RuntimeError(f"Failed to locate repo root from: {start}")


REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent)
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the GR00T TensorRT async policy server.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--inference-latency", type=float, default=0.1)
    parser.add_argument("--obs-queue-timeout", type=float, default=2.0)
    parser.add_argument("--resource-profile", default="default")
    parser.add_argument("--resource-policy-path", default=None)
    parser.add_argument("--resource-engine-dir", default=None)
    parser.add_argument("--resource-tensorrt-py-dir", default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve server symbols, validate config construction, print the result, and exit.",
    )
    return parser


def _resolve_server_symbols():
    module = importlib.import_module("my_devs.groot_trt_async_server.policy_server")
    config_module = importlib.import_module("my_devs.groot_trt_async_server.configs")

    serve_fn = (
        getattr(module, "serve", None)
        or getattr(module, "run_server", None)
        or getattr(module, "main", None)
    )
    if serve_fn is None:
        raise RuntimeError(
            "Could not resolve a server entrypoint from my_devs.groot_trt_async_server.policy_server. "
            "Expected one of: serve, run_server, main."
        )

    config_cls = (
        getattr(config_module, "GrootTrtPolicyServerConfig", None)
        or getattr(config_module, "PolicyServerConfig", None)
    )
    if config_cls is None:
        raise RuntimeError(
            "Could not resolve a policy server config class from "
            "my_devs.groot_trt_async_server.configs. "
            "Expected one of: GrootTrtPolicyServerConfig, PolicyServerConfig."
        )

    return config_cls, serve_fn


def main() -> None:
    args = build_parser().parse_args()
    resolved = {
        "host": args.host,
        "port": args.port,
        "fps": args.fps,
        "inference_latency": args.inference_latency,
        "obs_queue_timeout": args.obs_queue_timeout,
        "resource_profile": args.resource_profile,
    }
    if args.resource_policy_path is not None:
        resolved["resource_policy_path"] = args.resource_policy_path
    if args.resource_engine_dir is not None:
        resolved["resource_engine_dir"] = args.resource_engine_dir
    if args.resource_tensorrt_py_dir is not None:
        resolved["resource_tensorrt_py_dir"] = args.resource_tensorrt_py_dir

    config_cls, serve_fn = _resolve_server_symbols()
    cfg = config_cls(**resolved)
    resolved_output = asdict(cfg) if is_dataclass(cfg) else resolved
    if hasattr(cfg, "environment_dt"):
        resolved_output["environment_dt"] = cfg.environment_dt

    print("[INFO] GR00T TRT async server config validated:")
    print(
        pprint.pformat(
            {
                "config_class": f"{config_cls.__module__}.{config_cls.__qualname__}",
                "entrypoint": f"{serve_fn.__module__}.{serve_fn.__name__}",
                "config": resolved_output,
            }
        )
    )
    if args.dry_run:
        print("[INFO] --dry-run only validated imports and config construction. Server was not started.")
        return

    serve_entrypoint = getattr(serve_fn, "__wrapped__", serve_fn)
    serve_entrypoint(cfg)


if __name__ == "__main__":
    main()
