#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import os
import pickle
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests

PACKAGE_PARENT = Path(__file__).resolve().parents[2]
if PACKAGE_PARENT.as_posix() not in sys.path:
    sys.path.insert(0, PACKAGE_PARENT.as_posix())

from vlash_iner.async_manager import AsyncChunkManager
from vlash_iner.common import KNOWN_TASKS, ensure_repo_on_path, parse_bool, resolve_repo_root
from vlash_iner.safety import ActionSafetyChecker, ActionSafetyConfig


REPO_ROOT = resolve_repo_root(Path(__file__))
ensure_repo_on_path(REPO_ROOT)


class RemotePi05Client:
    def __init__(self, *, base_url: str, endpoint: str, timeout_s: float) -> None:
        self.session = requests.Session()
        self.base_url = base_url.rstrip("/")
        self.url = f"{self.base_url}{endpoint}"
        self.timeout_s = timeout_s

    def health(self) -> dict[str, Any]:
        response = self.session.get(f"{self.base_url}/health", timeout=self.timeout_s)
        response.raise_for_status()
        return response.json()

    def reset(self) -> dict[str, Any]:
        response = self.session.post(f"{self.base_url}/reset", timeout=self.timeout_s)
        response.raise_for_status()
        return pickle.loads(response.content)

    def infer(self, payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
        start_t = time.perf_counter()
        response = self.session.post(
            self.url,
            data=pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL),
            headers={"Content-Type": "application/octet-stream"},
            timeout=self.timeout_s,
        )
        request_latency_s = time.perf_counter() - start_t
        response.raise_for_status()
        result = pickle.loads(response.content)
        if isinstance(result, dict) and result.get("ok") is False:
            raise RuntimeError(result.get("error", "remote inference failed"))
        return result, request_latency_s


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Acceptance checks for PI0.5 remote backend and async chunk scheduling."
    )
    parser.add_argument("--server-url", default=os.getenv("SERVER_URL", "http://127.0.0.1:8008"))
    parser.add_argument("--endpoint", default=os.getenv("SERVER_ENDPOINT", "/infer"))
    parser.add_argument(
        "--request-timeout-s",
        type=float,
        default=float(os.getenv("REQUEST_TIMEOUT_S", "30")),
    )
    parser.add_argument(
        "--mode",
        choices=["infer", "async"],
        default="infer",
        help=(
            "infer: repeated /infer requests. "
            "async: AsyncChunkManager mock run against /infer."
        ),
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=100,
        help="Number of /infer requests for --mode infer.",
    )
    parser.add_argument(
        "--run-time-s",
        type=float,
        default=120.0,
        help="Mock async runtime for --mode async.",
    )
    parser.add_argument("--control-fps", type=float, default=45.0)
    parser.add_argument("--n-action-steps", type=int, default=0)
    parser.add_argument("--inference-overlap-steps", type=int, default=8)
    parser.add_argument("--background-inference", type=parse_bool, nargs="?", const=True, default=True)
    parser.add_argument("--chunk-blend-steps", type=int, default=2)
    parser.add_argument("--future-state-aware", type=parse_bool, nargs="?", const=True, default=False)
    parser.add_argument(
        "--task",
        default=os.getenv("DATASET_TASK"),
        help="Exact language instruction. Required unless --task-id is used.",
    )
    parser.add_argument(
        "--task-id",
        choices=sorted(KNOWN_TASKS),
        default=os.getenv("DATASET_TASK_ID", "eraser_to_box"),
    )
    parser.add_argument("--robot-type", default=os.getenv("ROBOT_TYPE", "so101_follower"))
    parser.add_argument("--img-width", type=int, default=int(os.getenv("IMG_WIDTH", "640")))
    parser.add_argument("--img-height", type=int, default=int(os.getenv("IMG_HEIGHT", "480")))
    parser.add_argument("--state-dim", type=int, default=int(os.getenv("STATE_DIM", "6")))
    parser.add_argument("--action-dim", type=int, default=int(os.getenv("ACTION_DIM", "6")))
    parser.add_argument(
        "--max-action-abs",
        type=float,
        default=float(os.getenv("MAX_ACTION_ABS", "1000000")),
    )
    parser.add_argument(
        "--max-action-delta",
        type=float,
        default=float(os.getenv("MAX_ACTION_DELTA", "1000000")),
    )
    parser.add_argument(
        "--max-wait-count",
        type=int,
        default=0,
        help="Maximum accepted AsyncChunkManager wait_count in --mode async.",
    )
    parser.add_argument(
        "--max-request-latency-p95-s",
        type=float,
        default=0.0,
        help="Optional acceptance threshold. <=0 derives overlap/control-fps budget in async mode.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[index]


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "min": min(values),
        "max": max(values),
    }


def resolve_task(args: argparse.Namespace) -> str:
    return args.task if args.task is not None else KNOWN_TASKS[args.task_id]


def make_dummy_observation(args: argparse.Namespace) -> dict[str, np.ndarray]:
    return {
        "observation.state": np.zeros((args.state_dim,), dtype=np.float32),
        "observation.images.top": np.zeros((args.img_height, args.img_width, 3), dtype=np.uint8),
        "observation.images.wrist": np.zeros((args.img_height, args.img_width, 3), dtype=np.uint8),
    }


def validate_chunk(chunk: np.ndarray, *, action_dim: int, n_action_steps: int | None) -> None:
    if chunk.ndim != 2:
        raise ValueError(f"Expected 2D action chunk, got {chunk.shape}")
    if n_action_steps is not None and chunk.shape[0] != n_action_steps:
        raise ValueError(f"Expected chunk length {n_action_steps}, got {chunk.shape[0]}")
    if chunk.shape[1] != action_dim:
        raise ValueError(f"Expected action_dim={action_dim}, got {chunk.shape[1]}")
    if np.isnan(chunk).any() or np.isinf(chunk).any():
        raise ValueError("Action chunk contains NaN or Inf")


def run_infer_acceptance(
    client: RemotePi05Client,
    args: argparse.Namespace,
    *,
    task: str,
    n_action_steps: int,
) -> dict[str, Any]:
    observation = make_dummy_observation(args)
    request_latencies: list[float] = []
    server_infers: list[float] = []
    action_mins: list[float] = []
    action_maxs: list[float] = []

    client.reset()
    for idx in range(1, args.requests + 1):
        result, request_latency_s = client.infer(
            {
                "observation": observation,
                "task": task,
                "robot_type": args.robot_type,
                "n_action_steps": n_action_steps,
            }
        )
        chunk = np.asarray(result["action_chunk"], dtype=np.float32)
        validate_chunk(chunk, action_dim=args.action_dim, n_action_steps=n_action_steps)
        request_latencies.append(request_latency_s)
        server_infers.append(float(result.get("infer_time_s", 0.0)))
        action_mins.append(float(np.min(chunk)))
        action_maxs.append(float(np.max(chunk)))
        if idx == 1 or idx == args.requests or idx % max(args.requests // 10, 1) == 0:
            print(
                f"[INFO] infer {idx}/{args.requests}: "
                f"request_latency={request_latency_s:.3f}s "
                f"server_infer={server_infers[-1]:.3f}s "
                f"shape={chunk.shape}",
                flush=True,
            )

    return {
        "mode": "infer",
        "requests": args.requests,
        "request_latency_s": summarize(request_latencies),
        "server_infer_s": summarize(server_infers),
        "action_min": min(action_mins),
        "action_max": max(action_maxs),
    }


def run_async_acceptance(
    client: RemotePi05Client,
    args: argparse.Namespace,
    *,
    task: str,
    n_action_steps: int,
) -> dict[str, Any]:
    observation = make_dummy_observation(args)
    request_latencies: list[float] = []
    server_infers: list[float] = []
    safety = ActionSafetyChecker(
        ActionSafetyConfig(
            action_dim=args.action_dim,
            max_abs=args.max_action_abs,
            max_delta=args.max_action_delta,
        )
    )

    def predict_chunk(observation_frame: dict, future_state: np.ndarray | None) -> np.ndarray:
        payload = {
            "observation": observation_frame,
            "future_state": future_state,
            "task": task,
            "robot_type": args.robot_type,
            "n_action_steps": n_action_steps,
        }
        result, request_latency_s = client.infer(payload)
        chunk = np.asarray(result["action_chunk"], dtype=np.float32)
        validate_chunk(chunk, action_dim=args.action_dim, n_action_steps=n_action_steps)
        request_latencies.append(request_latency_s)
        server_infers.append(float(result.get("infer_time_s", 0.0)))
        return chunk

    client.reset()
    manager = AsyncChunkManager(
        predict_chunk,
        n_action_steps=n_action_steps,
        overlap_steps=args.inference_overlap_steps,
        action_dim=args.action_dim,
        background_inference=args.background_inference,
        blend_steps=args.chunk_blend_steps,
        future_state_aware=args.future_state_aware,
    )
    step = 0
    start_t = time.perf_counter()
    end_t = start_t + args.run_time_s
    wait_count = 0
    inference_count = 0
    try:
        while time.perf_counter() < end_t:
            loop_t = time.perf_counter()
            action = manager.get_action(observation)
            safety.validate(action)
            step += 1
            if step % max(int(args.control_fps), 1) == 0:
                print(
                    f"[INFO] async step={step} elapsed={time.perf_counter() - start_t:.2f}s "
                    f"requests={manager.stats.inference_count} "
                    f"pending={manager.stats.pending_inference} wait_count={manager.stats.wait_count}",
                    flush=True,
                )
            time.sleep(max(1.0 / args.control_fps - (time.perf_counter() - loop_t), 0.0))
    finally:
        wait_count = manager.stats.wait_count
        inference_count = manager.stats.inference_count
        manager.close()

    return {
        "mode": "async",
        "run_time_s": args.run_time_s,
        "control_fps": args.control_fps,
        "steps": step,
        "observed_hz": step / max(time.perf_counter() - start_t, 1e-6),
        "request_count": len(request_latencies),
        "wait_count": wait_count,
        "manager_inference_count": inference_count,
        "inference_overlap_steps": args.inference_overlap_steps,
        "background_inference": args.background_inference,
        "chunk_blend_steps": args.chunk_blend_steps,
        "request_latency_s": summarize(request_latencies),
        "server_infer_s": summarize(server_infers),
    }


def evaluate_acceptance(report: dict[str, Any], args: argparse.Namespace) -> tuple[bool, list[str]]:
    failures: list[str] = []
    latency_threshold = args.max_request_latency_p95_s
    if latency_threshold <= 0 and report["mode"] == "async" and args.inference_overlap_steps > 0:
        latency_threshold = args.inference_overlap_steps / args.control_fps

    if latency_threshold > 0 and report["request_latency_s"]["p95"] > latency_threshold:
        failures.append(
            f"request_latency p95 {report['request_latency_s']['p95']:.3f}s exceeds "
            f"threshold {latency_threshold:.3f}s"
        )

    if report["mode"] == "async" and report["wait_count"] > args.max_wait_count:
        failures.append(f"wait_count {report['wait_count']} exceeds max_wait_count {args.max_wait_count}")

    return not failures, failures


def main() -> None:
    os.chdir(REPO_ROOT)
    args = build_parser().parse_args()
    task = resolve_task(args)
    client = RemotePi05Client(
        base_url=args.server_url,
        endpoint=args.endpoint,
        timeout_s=args.request_timeout_s,
    )
    health = client.health()
    n_action_steps = args.n_action_steps or int(
        health.get("n_action_steps") or health.get("chunk_size") or 50
    )
    print(f"[INFO] Server health: {health}", flush=True)
    print(f"[INFO] Acceptance mode: {args.mode}", flush=True)
    print(f"[INFO] Task: {task}", flush=True)
    print(f"[INFO] n_action_steps: {n_action_steps}", flush=True)

    if args.mode == "infer":
        report = run_infer_acceptance(client, args, task=task, n_action_steps=n_action_steps)
    else:
        report = run_async_acceptance(client, args, task=task, n_action_steps=n_action_steps)

    report["server_health"] = health
    passed, failures = evaluate_acceptance(report, args)
    report["passed"] = passed
    report["failures"] = failures
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(f"[INFO] Report written: {args.output_json}", flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
