from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import cv2
import numpy as np

THIS_FILE = Path(__file__).resolve()
BACKEND_ROOT = THIS_FILE.parents[1]
OPENPI_TRAIN_ROOT = BACKEND_ROOT.parent
REPO_ROOT = OPENPI_TRAIN_ROOT.parents[1]
for path in (BACKEND_ROOT, OPENPI_TRAIN_ROOT, REPO_ROOT / "src"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from config import Config
from config import load_config
from http_client import InferClient
from so101_runtime import ACTION_NAMES
from so101_runtime import action_to_robot_action
from so101_runtime import build_dataset_features
from so101_runtime import build_observation
from so101_runtime import build_robot
from so101_runtime import parse_bool
from so101_runtime import patch_motor_bus_retries
from so101_runtime import validate_action_names


@dataclass
class PendingInference:
    raw_observation: dict
    anchor_state: np.ndarray
    started_at: float
    thread: threading.Thread | None = None
    result: dict | None = None
    error: BaseException | None = None

    def join(self) -> dict:
        if self.thread is None:
            raise RuntimeError("Async inference was not started.")
        self.thread.join()
        if self.error is not None:
            raise self.error
        if self.result is None:
            raise RuntimeError("Async inference finished without a result.")
        return self.result


def _encode_jpeg(image: np.ndarray) -> bytes:
    frame = np.asarray(image)
    if frame.ndim != 3:
        raise ValueError(f"Expected image shape=(H,W,C), got {frame.shape}")
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr)
    if not ok:
        raise RuntimeError("Failed to encode image as JPEG.")
    return buf.tobytes()


def _build_payload(observation: dict) -> dict:
    return {
        "images": {
            "top": _encode_jpeg(observation["observation.images.top"]),
            "wrist": _encode_jpeg(observation["observation.images.wrist"]),
        },
        "state": np.asarray(observation["observation.state"], dtype=np.float32),
        "prompt": observation["prompt"],
        "timestamp": time.time(),
    }


def _resample_actions(actions: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[0] <= 1:
        return actions
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("source_fps and target_fps must be > 0 for action interpolation.")
    if abs(source_fps - target_fps) < 1e-6:
        return actions

    target_len = max(1, int(round(actions.shape[0] * target_fps / source_fps)))
    source_t = np.arange(actions.shape[0], dtype=np.float32) / float(source_fps)
    target_t = np.arange(target_len, dtype=np.float32) / float(target_fps)
    target_t = np.clip(target_t, source_t[0], source_t[-1])
    out = np.empty((target_len, actions.shape[1]), dtype=np.float32)
    for dim in range(actions.shape[1]):
        out[:, dim] = np.interp(target_t, source_t, actions[:, dim]).astype(np.float32)
    return out


def _blend_chunk_start(actions: np.ndarray, previous_action: np.ndarray | None, blend_steps: int) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if previous_action is None or blend_steps <= 0 or actions.size == 0:
        return actions
    previous_action = np.asarray(previous_action, dtype=np.float32)
    if previous_action.shape != actions[0].shape:
        return actions

    out = actions.copy()
    steps = min(int(blend_steps), out.shape[0])
    for idx in range(steps):
        alpha = float(idx + 1) / float(steps + 1)
        out[idx] = (1.0 - alpha) * previous_action + alpha * out[idx]
    return out


def _stabilize_actions(
    actions: np.ndarray,
    previous_action: np.ndarray | None,
    max_delta_per_step: float,
    ema_alpha: float,
) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.size == 0:
        return actions
    if previous_action is None:
        return actions

    alpha = float(np.clip(ema_alpha, 0.0, 1.0))
    max_delta = float(max_delta_per_step)
    out = actions.copy()
    prev = np.asarray(previous_action, dtype=np.float32).copy()
    for idx in range(out.shape[0]):
        target = out[idx]
        if alpha < 1.0:
            target = prev + alpha * (target - prev)
        if max_delta > 0.0:
            target = prev + np.clip(target - prev, -max_delta, max_delta)
        out[idx] = target
        prev = target
    return out


def _start_async_infer(
    client: InferClient,
    payload: dict,
    raw_observation: dict,
    anchor_state: np.ndarray,
) -> PendingInference:
    pending = PendingInference(
        raw_observation=raw_observation,
        anchor_state=np.asarray(anchor_state, dtype=np.float32),
        started_at=time.perf_counter(),
    )

    def _run() -> None:
        try:
            pending.result = client.infer(payload)
        except BaseException as exc:
            pending.error = exc

    pending.thread = threading.Thread(target=_run, daemon=True)
    pending.thread.start()
    return pending


def _blocking_infer_observation(
    *,
    robot: Any,
    robot_observation_processor: Any,
    dataset_features: dict,
    task: str,
    client: InferClient,
) -> tuple[dict, dict, float, np.ndarray]:
    infer_t = time.perf_counter()
    observation, raw_observation = build_observation(
        robot,
        robot_observation_processor,
        dataset_features,
        task,
    )
    result = client.infer(_build_payload(observation))
    infer_ms = (time.perf_counter() - infer_t) * 1000.0
    return result, raw_observation, infer_ms, np.asarray(observation["observation.state"], dtype=np.float32)


def _override_config(cfg: Config, args: argparse.Namespace) -> Config:
    from dataclasses import replace

    client = cfg.client
    executor = cfg.executor
    if args.infer_url is not None:
        client = replace(client, infer_url=args.infer_url)
    if args.endpoint is not None:
        client = replace(client, endpoint=args.endpoint)
    if args.timeout_s is not None:
        client = replace(client, timeout_s=args.timeout_s)
    if args.run_time_s is not None:
        client = replace(client, run_time_s=args.run_time_s)
    if args.task is not None:
        client = replace(client, task=args.task)
    if args.execute_actions is not None:
        executor = replace(executor, execute_actions=args.execute_actions)
    if args.dry_run is not None:
        executor = replace(executor, dry_run=args.dry_run)
    if args.action_chunk_steps is not None:
        executor = replace(executor, action_chunk_steps=args.action_chunk_steps)
    if args.control_fps is not None:
        executor = replace(executor, control_fps=args.control_fps)
    if args.policy_fps is not None:
        executor = replace(executor, policy_fps=args.policy_fps)
    if args.interpolate_actions is not None:
        executor = replace(executor, interpolate_actions=args.interpolate_actions)
    if args.boundary_blend_steps is not None:
        executor = replace(executor, boundary_blend_steps=args.boundary_blend_steps)
    if args.max_action_delta_per_step is not None:
        executor = replace(executor, max_action_delta_per_step=args.max_action_delta_per_step)
    if args.action_ema_alpha is not None:
        executor = replace(executor, action_ema_alpha=args.action_ema_alpha)
    if args.async_prefetch is not None:
        executor = replace(executor, async_prefetch=args.async_prefetch)
    if args.prefetch_after_steps is not None:
        executor = replace(executor, prefetch_after_steps=args.prefetch_after_steps)
    return replace(cfg, client=client, executor=executor)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SO101 through the RealtimeVLA-style HTTP backend.")
    parser.add_argument("--config", required=True, help="Path to client YAML config.")
    parser.add_argument("--infer-url", default=os.getenv("REALTIMEVLA_INFER_URL"))
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--timeout-s", type=float, default=None)
    parser.add_argument("--run-time-s", type=float, default=None)
    parser.add_argument("--task", default=os.getenv("DATASET_TASK"))
    parser.add_argument("--execute-actions", type=parse_bool, default=None)
    parser.add_argument("--dry-run", type=parse_bool, default=None)
    parser.add_argument("--action-chunk-steps", type=int, default=None)
    parser.add_argument("--control-fps", type=float, default=None)
    parser.add_argument("--policy-fps", type=float, default=None)
    parser.add_argument("--interpolate-actions", type=parse_bool, default=None)
    parser.add_argument("--boundary-blend-steps", type=int, default=None)
    parser.add_argument("--max-action-delta-per-step", type=float, default=None)
    parser.add_argument("--action-ema-alpha", type=float, default=None)
    parser.add_argument("--async-prefetch", type=parse_bool, default=None)
    parser.add_argument("--prefetch-after-steps", type=int, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cfg = _override_config(load_config(args.config), args)

    print(f"[INFO] Backend root: {BACKEND_ROOT}")
    print(f"[INFO] HTTP infer URL: {cfg.client.infer_url.rstrip('/')}{cfg.client.endpoint}")
    print(f"[INFO] Task: {cfg.client.task}")
    print(f"[INFO] execute_actions: {cfg.executor.execute_actions}")
    print(f"[INFO] dry_run: {cfg.executor.dry_run}")
    print(f"[INFO] action_chunk_steps: {cfg.executor.action_chunk_steps}")
    print(f"[INFO] control_fps: {cfg.executor.control_fps}")
    print(f"[INFO] policy_fps: {cfg.executor.policy_fps}")
    print(f"[INFO] interpolate_actions: {cfg.executor.interpolate_actions}")
    print(f"[INFO] boundary_blend_steps: {cfg.executor.boundary_blend_steps}")
    print(f"[INFO] max_action_delta_per_step: {cfg.executor.max_action_delta_per_step}")
    print(f"[INFO] action_ema_alpha: {cfg.executor.action_ema_alpha}")
    print(f"[INFO] async_prefetch: {cfg.executor.async_prefetch}")
    print(f"[INFO] prefetch_after_steps: {cfg.executor.prefetch_after_steps}")
    print(f"[INFO] run_time_s: {cfg.client.run_time_s} (<=0 means until Ctrl+C)")

    if cfg.executor.action_chunk_steps < 1:
        raise ValueError("executor.action_chunk_steps must be >= 1")
    if cfg.executor.control_fps <= 0:
        raise ValueError("executor.control_fps must be > 0")
    if cfg.executor.policy_fps <= 0:
        raise ValueError("executor.policy_fps must be > 0")

    if cfg.executor.dry_run:
        print("[INFO] dry_run=true, exiting before server or hardware access.")
        return 0

    robot = None
    try:
        robot, robot_config = build_robot(cfg.robot)
        patch_motor_bus_retries(robot, cfg.robot.motor_io_retries)
        dataset_features, robot_action_processor, robot_observation_processor = build_dataset_features(robot)
        validate_action_names(dataset_features)

        print(f"[INFO] Robot config type resolved by current LeRobot registry: {robot_config.type}")
        print(f"[INFO] Robot runtime type: {robot.robot_type}")
        print(f"[INFO] Action names: {dataset_features['action']['names']}")

        client = InferClient(
            base_url=cfg.client.infer_url,
            endpoint=cfg.client.endpoint,
            timeout_s=cfg.client.timeout_s,
        )
        print(f"[INFO] HTTP client ready: {client.url}")

        robot.connect()
        print("[INFO] Robot connected. Starting HTTP remote inference loop.")

        step = 0
        start_t = time.perf_counter()
        end_t = start_t + cfg.client.run_time_s if cfg.client.run_time_s > 0 else None
        pending_infer: PendingInference | None = None
        control_dt_s = 1.0 / float(cfg.executor.control_fps)
        previous_sent_action: np.ndarray | None = None
        requested_prefetch_after_steps = int(cfg.executor.prefetch_after_steps)
        if cfg.executor.async_prefetch and requested_prefetch_after_steps >= int(cfg.executor.action_chunk_steps):
            print(
                "[WARN] prefetch_after_steps is >= action_chunk_steps; "
                "effective prefetch will happen at the last action and will not hide inference latency well."
            )

        while True:
            if end_t is not None and time.perf_counter() >= end_t:
                print("[INFO] Reached requested run_time_s. Exiting.")
                break

            if pending_infer is not None:
                wait_t = time.perf_counter()
                result = pending_infer.join()
                raw_observation = pending_infer.raw_observation
                anchor_state = pending_infer.anchor_state
                infer_ms = (time.perf_counter() - pending_infer.started_at) * 1000.0
                wait_ms = (time.perf_counter() - wait_t) * 1000.0
                pending_infer = None
            else:
                result, raw_observation, infer_ms, anchor_state = _blocking_infer_observation(
                    robot=robot,
                    robot_observation_processor=robot_observation_processor,
                    dataset_features=dataset_features,
                    task=cfg.client.task,
                    client=client,
                )
                wait_ms = infer_ms

            raw_actions = np.asarray(result.get("action_list"), dtype=np.float32)
            if raw_actions.ndim != 2 or raw_actions.shape[1] != 6:
                raise ValueError(f"Expected action chunk shape=(horizon, 6), got shape={raw_actions.shape}")

            policy_steps_to_execute = min(int(cfg.executor.action_chunk_steps), raw_actions.shape[0])
            actions = raw_actions[:policy_steps_to_execute]
            if cfg.executor.interpolate_actions:
                actions = _resample_actions(
                    actions,
                    source_fps=float(cfg.executor.policy_fps),
                    target_fps=float(cfg.executor.control_fps),
                )
            stabilizer_anchor = previous_sent_action
            if stabilizer_anchor is None and anchor_state is not None:
                stabilizer_anchor = np.asarray(anchor_state, dtype=np.float32)
            actions = _blend_chunk_start(actions, stabilizer_anchor, int(cfg.executor.boundary_blend_steps))
            actions = _stabilize_actions(
                actions,
                stabilizer_anchor,
                max_delta_per_step=float(cfg.executor.max_action_delta_per_step),
                ema_alpha=float(cfg.executor.action_ema_alpha),
            )
            steps_to_execute = actions.shape[0]
            prefetch_after_steps = int(cfg.executor.prefetch_after_steps)
            if prefetch_after_steps < 0:
                prefetch_after_steps = max(0, steps_to_execute // 2)
            prefetch_after_steps = min(prefetch_after_steps, max(steps_to_execute - 1, 0))
            chunk_start_t = time.perf_counter()
            chunk_start_step = step
            prefetch_start_ms: float | None = None
            prefetch_started = False
            for chunk_index, action in enumerate(actions[:steps_to_execute]):
                if end_t is not None and time.perf_counter() >= end_t:
                    break

                if (
                    cfg.executor.async_prefetch
                    and pending_infer is None
                    and not prefetch_started
                    and chunk_index >= prefetch_after_steps
                ):
                    try:
                        prefetch_observation, prefetch_raw_observation = build_observation(
                            robot,
                            robot_observation_processor,
                            dataset_features,
                            cfg.client.task,
                        )
                        pending_infer = _start_async_infer(
                            client,
                            _build_payload(prefetch_observation),
                            prefetch_raw_observation,
                            np.asarray(prefetch_observation["observation.state"], dtype=np.float32),
                        )
                        prefetch_start_ms = (time.perf_counter() - chunk_start_t) * 1000.0
                        prefetch_started = True
                    except Exception as exc:
                        print(f"[WARN] Async prefetch start failed: {type(exc).__name__}: {exc}")
                        prefetch_started = True

                step_t = time.perf_counter()
                robot_action = action_to_robot_action(action, dataset_features)
                if cfg.executor.execute_actions:
                    robot_action_to_send = robot_action_processor((robot_action, raw_observation))
                    robot.send_action(robot_action_to_send)

                step += 1
                if cfg.client.log_interval > 0 and step % cfg.client.log_interval == 0:
                    elapsed = time.perf_counter() - start_t
                    action_text = np.array2string(
                        np.array([robot_action[name] for name in ACTION_NAMES], dtype=np.float32),
                        precision=3,
                        suppress_small=True,
                    )
                    print(
                        f"[INFO] Step {step} elapsed={elapsed:.2f}s "
                        f"chunk_index={chunk_index}/{steps_to_execute - 1} "
                        f"infer_ms={infer_ms:.1f} wait_ms={wait_ms:.1f} "
                        f"server_infer_s={result.get('infer_time', 0.0):.3f} "
                        f"prefetch={'on' if pending_infer is not None else 'off'} "
                        f"action={action_text}"
                    )

                dt_s = time.perf_counter() - step_t
                previous_sent_action = np.asarray(action, dtype=np.float32).copy()
                time.sleep(max(control_dt_s - dt_s, 0.0))

            chunk_elapsed_s = time.perf_counter() - chunk_start_t
            chunk_steps = step - chunk_start_step
            actual_control_fps = chunk_steps / chunk_elapsed_s if chunk_elapsed_s > 0 else 0.0
            print(
                f"[INFO] Chunk done steps={chunk_steps}/{steps_to_execute} "
                f"duration_s={chunk_elapsed_s:.3f} actual_control_fps={actual_control_fps:.1f} "
                f"target_control_fps={float(cfg.executor.control_fps):.1f} "
                f"policy_steps={policy_steps_to_execute} interpolate={cfg.executor.interpolate_actions} "
                f"max_delta={float(cfg.executor.max_action_delta_per_step):.3f} "
                f"ema_alpha={float(cfg.executor.action_ema_alpha):.3f} "
                f"effective_prefetch_step={prefetch_after_steps} "
                f"prefetch_start_ms={prefetch_start_ms if prefetch_start_ms is not None else -1:.1f} "
                f"next_ready={pending_infer is not None and pending_infer.result is not None}"
            )

    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt received. Stopping.")
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        return 1
    finally:
        if robot is not None and getattr(robot, "is_connected", False):
            try:
                robot.disconnect()
            except Exception:
                pass
        print("[INFO] Client finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
