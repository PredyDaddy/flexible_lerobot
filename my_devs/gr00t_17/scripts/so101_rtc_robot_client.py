#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SPEC = importlib.util.spec_from_file_location(
    "so101_robot_client_base", SCRIPT_DIR / "so101_robot_client.py"
)
assert BASE_SPEC is not None and BASE_SPEC.loader is not None
base = importlib.util.module_from_spec(BASE_SPEC)
sys.modules[BASE_SPEC.name] = base
BASE_SPEC.loader.exec_module(base)


@dataclass
class PendingRequest:
    future: Future
    request_step: int
    started_at: float


class TimelineActionQueue:
    """Thread-independent action timeline indexed by the control step."""

    def __init__(self) -> None:
        self._origin = 0
        self._actions: np.ndarray | None = None

    def merge(self, chunk: np.ndarray, request_step: int, current_step: int) -> int:
        chunk = np.asarray(chunk, dtype=np.float32)
        delay_steps = max(current_step - request_step, 0)
        if delay_steps >= len(chunk):
            raise RuntimeError(f"RTC response is stale: delay={delay_steps}, horizon={len(chunk)}")
        self._origin = request_step + delay_steps
        self._actions = chunk[delay_steps:].copy()
        return delay_steps

    def get(self, step: int) -> np.ndarray | None:
        if self._actions is None:
            return None
        index = step - self._origin
        if index < 0 or index >= len(self._actions):
            return None
        return self._actions[index].copy()

    def remaining(self, step: int) -> int:
        if self._actions is None:
            return 0
        return max(len(self._actions) - max(step - self._origin, 0), 0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Asynchronous GR00T N1.7 RTC client for SO101")
    parser.add_argument("--mode", choices=("ping", "predict", "dry-run", "actuate"), required=True)
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=5556)
    parser.add_argument("--request-timeout-s", type=float, default=2.0)
    parser.add_argument("--expected-backend", choices=("any", "pytorch", "tensorrt"), default="any")
    parser.add_argument("--checkpoint-path", type=Path, default=base.DEFAULT_CHECKPOINT)
    parser.add_argument("--robot-id", default="hfy_follower")
    parser.add_argument("--calibration-dir", type=Path, default=base.DEFAULT_CALIBRATION_DIR)
    parser.add_argument("--robot-port", default=base.DEFAULT_ROBOT_PORT)
    parser.add_argument("--top-cam", type=base.parse_camera, default=base.parse_camera("/dev/video4"))
    parser.add_argument("--wrist-cam", type=base.parse_camera, default=base.parse_camera("/dev/video6"))
    parser.add_argument("--top-cam-fourcc", default="YUYV")
    parser.add_argument("--wrist-cam-fourcc", default="MJPG")
    parser.add_argument("--camera-warmup-s", type=float, default=2.0)
    parser.add_argument("--task", default=base.KNOWN_TASKS[0])
    parser.add_argument("--allow-unknown-task", action="store_true")
    parser.add_argument("--prediction-count", type=int, default=3)
    parser.add_argument("--run-time-s", type=float, default=120.0)
    parser.add_argument("--execution-horizon", type=int, default=8)
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument("--rtc-ramp-rate", type=float, default=2.0)
    parser.add_argument(
        "--bounds-mode", choices=("q01_q99", "dataset_minmax", "physical"), default="physical"
    )
    parser.add_argument("--max-command-delta", type=float, default=200.0)
    parser.add_argument("--max-relative-target", type=float, default=200.0)
    parser.add_argument("--max-inference-s", type=float, default=2.0)
    parser.add_argument("--enable-actuation", action="store_true")
    parser.add_argument("--confirm-actuation", default="")
    parser.add_argument("--report", type=Path)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.server_host not in {"127.0.0.1", "localhost"}:
        raise ValueError("RTC client only accepts a localhost policy server")
    if args.task not in base.KNOWN_TASKS and not args.allow_unknown_task:
        raise ValueError(f"Task was not present in training data: {args.task!r}")
    values = (
        args.request_timeout_s,
        args.run_time_s,
        args.control_hz,
        args.rtc_ramp_rate,
        args.max_command_delta,
        args.max_relative_target,
        args.max_inference_s,
    )
    if not all(math.isfinite(value) and value > 0 for value in values):
        raise ValueError("Numeric runtime arguments must be finite and positive")
    if not 1 <= args.execution_horizon < 16:
        raise ValueError("execution_horizon must be in [1, 15]")
    if args.control_hz > 30:
        raise ValueError("control_hz must be in (0, 30]")
    if args.prediction_count < 1:
        raise ValueError("prediction_count must be positive")
    if args.mode == "actuate" and (
        not args.enable_actuation or args.confirm_actuation != base.ACTUATION_CONFIRMATION
    ):
        raise PermissionError(
            f"Actuation requires --enable-actuation and --confirm-actuation {base.ACTUATION_CONFIRMATION}"
        )


def rtc_options(args: argparse.Namespace, frozen_steps: int) -> dict[str, Any]:
    overlap = 16 - args.execution_horizon
    return {
        "rtc_enabled": True,
        "rtc_advance_steps": args.execution_horizon,
        "rtc_frozen_steps": min(max(int(frozen_steps), 0), overlap),
        "rtc_ramp_rate": args.rtc_ramp_rate,
    }


def request_chunk(
    client: Any, observation: dict[str, Any], options: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any], float]:
    started = time.perf_counter()
    result = client.call("get_action", {"observation": observation, "options": options})
    latency = time.perf_counter() - started
    if not isinstance(result, (list, tuple)) or len(result) != 2:
        raise RuntimeError(f"Unexpected policy response: {type(result)!r}")
    return base.validate_action_chunk(result[0]), result[1], latency


def request_chunk_on_new_connection(
    host: str,
    port: int,
    timeout_s: float,
    observation: dict[str, Any],
    options: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any], float]:
    # ZeroMQ sockets are thread-affine. The inference worker owns this short-lived connection.
    worker_client = base.LocalPolicyClient(host, port, timeout_s)
    try:
        return request_chunk(worker_client, observation, options)
    finally:
        worker_client.close()


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    checkpoint = base.ensure_within(args.checkpoint_path, base.GR00T17_ROOT, must_exist=True)
    checkpoint_info = base.validate_checkpoint(checkpoint)
    client = base.LocalPolicyClient(args.server_host, args.server_port, args.request_timeout_s)
    ping_response = client.ping()
    if args.expected_backend != "any" and ping_response.get("inference_backend") != args.expected_backend:
        client.close()
        raise RuntimeError(f"Expected {args.expected_backend} server, got ping response: {ping_response}")
    if args.mode == "ping":
        client.close()
        return {
            "status": "passed",
            "mode": "ping",
            "checkpoint": checkpoint_info,
            "server": ping_response,
        }

    bounds = base.load_action_bounds(checkpoint, args.bounds_mode)
    robot, calibration_file = base.build_robot(args)
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "timestamp": datetime.now().astimezone().isoformat(),
        "mode": args.mode,
        "checkpoint": checkpoint_info,
        "server": ping_response,
        "rtc": {
            "enabled": True,
            "action_horizon": 16,
            "execution_horizon": args.execution_horizon,
            "overlap_steps": 16 - args.execution_horizon,
            "ramp_rate": args.rtc_ramp_rate,
            "control_hz": args.control_hz,
        },
        "command_envelope": {
            "bounds_mode": bounds.source,
            "max_command_delta": args.max_command_delta,
            "max_relative_target": args.max_relative_target,
        },
        "devices": {"calibration_file": str(calibration_file)},
    }
    records: list[dict[str, Any]] = []
    sent = 0
    underruns = 0
    period = 1.0 / args.control_hz
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gr00t-rtc")
    pending: PendingRequest | None = None
    queue = TimelineActionQueue()
    try:
        robot.connect(calibrate=False)
        warmup_start = time.perf_counter()
        while time.perf_counter() - warmup_start < args.camera_warmup_s:
            robot.get_observation()
            time.sleep(period)
        observation = robot.get_observation()
        report["initial_state"] = base.extract_state(observation).tolist()
        report["cameras"] = {
            "top": base.validate_image("top", observation["top"]),
            "wrist": base.validate_image("wrist", observation["wrist"]),
        }
        client.call("reset", {"options": {"reason": "new_rtc_run"}})

        # Warm CUDA kernels before establishing the action timeline. Nothing from this
        # request is queued or sent to the robot, and reset clears the server's RTC state.
        _, warmup_info, warmup_latency = request_chunk(
            client, base.build_model_observation(observation, args.task), rtc_options(args, 0)
        )
        report["model_warmup"] = {"latency_s": warmup_latency, **warmup_info}
        client.call("reset", {"options": {"reason": "post_model_warmup"}})

        first_chunk, first_info, first_latency = request_chunk(
            client, base.build_model_observation(observation, args.task), rtc_options(args, 0)
        )
        queue.merge(first_chunk, 0, 0)
        records.append({"request_step": 0, "delay_steps": 0, "latency_s": first_latency, **first_info})
        frozen_estimate = max(1, math.ceil(first_latency * args.control_hz))

        if args.mode == "predict":
            for index in range(1, args.prediction_count):
                observation = robot.get_observation()
                chunk, info, latency = request_chunk(
                    client,
                    base.build_model_observation(observation, args.task),
                    rtc_options(args, frozen_estimate),
                )
                records.append(
                    {
                        "request_step": index * args.execution_horizon,
                        "delay_steps": math.ceil(latency * args.control_hz),
                        "latency_s": latency,
                        "chunk_shape": list(chunk.shape),
                        **info,
                    }
                )
                frozen_estimate = max(1, math.ceil(latency * args.control_hz))
            report["requests"] = records
            report["status"] = "passed"
            return report

        deadline = time.perf_counter() + args.run_time_s
        step = 0
        next_request_step = args.execution_horizon
        next_tick = time.perf_counter()
        reference = base.extract_state(observation)
        while time.perf_counter() < deadline:
            if pending is not None and pending.future.done():
                chunk, info, latency = pending.future.result()
                if latency > args.max_inference_s:
                    raise TimeoutError(f"RTC inference took {latency:.3f}s")
                delay = queue.merge(chunk, pending.request_step, step)
                records.append(
                    {
                        "request_step": pending.request_step,
                        "delay_steps": delay,
                        "latency_s": latency,
                        **info,
                    }
                )
                frozen_estimate = max(1, math.ceil(latency * args.control_hz))
                pending = None

            if pending is None and step >= next_request_step:
                request_observation = robot.get_observation()
                options = rtc_options(args, frozen_estimate)
                started = time.perf_counter()
                future = executor.submit(
                    request_chunk_on_new_connection,
                    args.server_host,
                    args.server_port,
                    args.request_timeout_s,
                    base.build_model_observation(request_observation, args.task),
                    options,
                )
                pending = PendingRequest(future=future, request_step=step, started_at=started)
                next_request_step += args.execution_horizon

            predicted = queue.get(step)
            if predicted is None:
                underruns += 1
                raise RuntimeError(f"RTC action queue underrun at control step {step}")
            command, safety = base.sanitize_action(predicted, reference, bounds, args.max_command_delta)
            if args.mode == "actuate":
                actual = robot.send_action(base.action_to_robot_dict(command))
                reference = np.asarray([actual[key] for key in base.JOINT_KEYS], dtype=np.float32)
                sent += 1
            step += 1
            next_tick += period
            time.sleep(max(next_tick - time.perf_counter(), 0.0))

        report["requests"] = records
        report["sent_action_count"] = sent
        report["control_step_count"] = step
        report["queue_underruns"] = underruns
        report["achieved_control_hz"] = step / args.run_time_s
        report["status"] = "passed"
        return report
    finally:
        if pending is not None:
            pending.future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        errors = base.disconnect_robot_safely(robot)
        client.close()
        if errors and sys.exc_info()[0] is None:
            raise RuntimeError(f"Device cleanup errors: {errors}")


def main() -> None:
    args = build_parser().parse_args()
    try:
        report = run(args)
    except BaseException as exc:
        report = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        base.write_report(args.report, report)
        raise
    base.write_report(args.report, report)
    print(f"[OK] RTC {args.mode} passed")


if __name__ == "__main__":
    main()
