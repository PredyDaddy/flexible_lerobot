#!/usr/bin/env python

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import signal
import socketserver
import threading
import time
from typing import Any

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor import make_default_robot_action_processor
from lerobot.robots import make_robot_from_config
from lerobot.utils.constants import ACTION
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging

from my_devs.jz_robot.common import (
    DEFAULT_ROBOT_CONFIG,
    apply_common_robot_overrides,
    load_robot_config,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegisteredReplayAction:
    slot_id: str
    label: str
    dataset_root: str
    episode: int
    fps: int

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "RegisteredReplayAction":
        slot_id = str(payload["slot_id"])
        dataset_root = str(payload["dataset_root"])
        episode = int(payload["episode"])
        label = str(payload.get("label") or f"{Path(dataset_root).name} · Episode {episode}")
        fps = int(payload.get("fps") or 30)
        return cls(
            slot_id=slot_id,
            label=label,
            dataset_root=dataset_root,
            episode=episode,
            fps=fps,
        )


@dataclass(frozen=True)
class PreparedReplayAction:
    spec: RegisteredReplayAction
    action_names: tuple[str, ...]
    action_rows: tuple[tuple[float, ...], ...]


@dataclass
class ActivePlayback:
    prepared: PreparedReplayAction
    started_at: datetime
    stop_event: threading.Event
    stop_reason: str | None = None
    thread: threading.Thread | None = None


class PersistentJZReplayRuntime:
    """Keep replay assets warm across multiple mobile-grid triggers."""

    dataset_repo_id = "local/jz_pick_place_three_rs"
    robot_id = "jz_dual_arm_rs"

    def __init__(self, *, log_path: Path | None = None):
        self.log_path = str(log_path) if log_path else None

        self._state_lock = threading.RLock()
        self._command_lock = threading.Lock()
        self._warmup_lock = threading.Lock()

        self._registered_actions: tuple[RegisteredReplayAction, ...] = ()
        self._prepared_actions: dict[str, PreparedReplayAction] = {}

        self._robot = None
        self._robot_action_processor = None

        self._ready = False
        self._warmup_in_progress = False
        self._warmup_error: str | None = None
        self._active_playback: ActivePlayback | None = None
        self._last_run: dict[str, Any] | None = None
        self._shutdown_requested = False

    def handle_command(self, request: dict[str, Any]) -> dict[str, Any]:
        command = request.get("command")
        if command == "status":
            return self.get_status()
        if command == "ensure_ready":
            return self.ensure_ready(request.get("actions") or [])
        if command == "play":
            slot_id = request.get("slot_id")
            if not slot_id:
                raise ValueError("slot_id is required for play")
            return self.play(str(slot_id))
        if command == "shutdown":
            self.shutdown()
            return self.get_status()
        raise ValueError(f"unsupported command: {command}")

    def ensure_ready(self, actions_payload: list[dict[str, Any]]) -> dict[str, Any]:
        normalized = self._normalize_actions(actions_payload)
        self._register_actions(normalized)

        if self._ready:
            return self.get_status()

        with self._warmup_lock:
            if self._ready:
                return self.get_status()

            with self._state_lock:
                self._warmup_in_progress = True
                self._warmup_error = None

            try:
                prepared_actions = {
                    spec.slot_id: self._prepare_action(spec)
                    for spec in self._registered_actions
                }
                robot_action_processor = make_default_robot_action_processor()
                robot = self._robot or self._build_robot()

                with self._state_lock:
                    self._prepared_actions = prepared_actions
                    self._robot_action_processor = robot_action_processor
                    self._robot = robot
                    self._ready = True
                    self._warmup_error = None
            except Exception as exc:
                with self._state_lock:
                    self._ready = False
                    self._warmup_error = str(exc)
                raise
            finally:
                with self._state_lock:
                    self._warmup_in_progress = False

        return self.get_status()

    def play(self, slot_id: str) -> dict[str, Any]:
        with self._command_lock:
            self.ensure_ready([])

            prepared = self._prepared_actions.get(slot_id)
            if prepared is None:
                raise KeyError(f"unknown replay slot: {slot_id}")

            self._ensure_robot_connected()
            self._stop_active_playback(reason="replaced")

            active = ActivePlayback(
                prepared=prepared,
                started_at=datetime.now(),
                stop_event=threading.Event(),
            )
            thread = threading.Thread(target=self._run_playback, args=(active,), daemon=True)
            active.thread = thread

            with self._state_lock:
                self._active_playback = active

            thread.start()

            return {
                "slot_id": prepared.spec.slot_id,
                "label": prepared.spec.label,
                "dataset_root": prepared.spec.dataset_root,
                "episode": prepared.spec.episode,
                "pid": os.getpid(),
                "worker_pid": os.getpid(),
                "started_at": active.started_at.isoformat(),
                "log_path": self.log_path,
                "worker_ready": True,
                "robot_connected": True,
                "mode": "worker",
            }

    def get_status(self) -> dict[str, Any]:
        with self._state_lock:
            active = self._active_playback
            running = bool(active and active.thread and active.thread.is_alive())
            return {
                "worker_started": True,
                "worker_ready": self._ready,
                "warmup_in_progress": self._warmup_in_progress,
                "warmup_error": self._warmup_error,
                "robot_connected": bool(self._robot and self._robot.is_connected),
                "running": running,
                "active_slot_id": active.prepared.spec.slot_id if running and active else None,
                "active_label": active.prepared.spec.label if running and active else None,
                "started_at": active.started_at.isoformat() if running and active else None,
                "pid": os.getpid(),
                "worker_pid": os.getpid(),
                "log_path": self.log_path,
                "last_run": self._last_run,
                "mode": "worker",
                "shutdown_requested": self._shutdown_requested,
            }

    def shutdown(self) -> None:
        with self._command_lock:
            self._shutdown_requested = True
            self._stop_active_playback(reason="shutdown")
            robot = self._robot
            if robot is not None and robot.is_connected:
                robot.disconnect()

    def _normalize_actions(self, payload: list[dict[str, Any]]) -> tuple[RegisteredReplayAction, ...]:
        if payload:
            return tuple(RegisteredReplayAction.from_payload(item) for item in payload)
        if self._registered_actions:
            return self._registered_actions
        raise ValueError("no replay actions registered")

    def _register_actions(self, normalized: tuple[RegisteredReplayAction, ...]) -> None:
        with self._state_lock:
            if normalized == self._registered_actions:
                return
            self._registered_actions = normalized
            self._prepared_actions = {}
            self._ready = False
            self._warmup_error = None

    def _prepare_action(self, spec: RegisteredReplayAction) -> PreparedReplayAction:
        dataset = LeRobotDataset(
            self.dataset_repo_id,
            root=Path(spec.dataset_root),
            episodes=[spec.episode],
        )
        episode_frames = dataset.hf_dataset.filter(lambda row: row["episode_index"] == spec.episode)
        actions = episode_frames.select_columns(ACTION)
        action_names = tuple(dataset.features[ACTION]["names"])

        action_rows = []
        for index in range(len(actions)):
            action_rows.append(tuple(float(value) for value in actions[index][ACTION]))

        if not action_rows:
            raise RuntimeError(f"no replay frames found for {spec.slot_id}")

        return PreparedReplayAction(
            spec=spec,
            action_names=action_names,
            action_rows=tuple(action_rows),
        )

    def _build_robot(self):
        robot_cfg = load_robot_config(DEFAULT_ROBOT_CONFIG)
        robot_cfg = apply_common_robot_overrides(
            robot_cfg,
            robot_id=self.robot_id,
            left_joint_state_topic=None,
            right_joint_state_topic=None,
            left_command_topic=None,
            right_command_topic=None,
            use_gripper=True,
            left_gripper_state_topic=None,
            right_gripper_state_topic=None,
            left_gripper_command_topic=None,
            right_gripper_command_topic=None,
            init_state_timeout_s=0,
            state_timeout_s=0.2,
            qos_depth=10,
            use_external_commands=False,
            img_width=None,
            img_height=None,
            camera_fps=None,
            warmup_s=None,
        )
        robot_cfg.cameras = {}
        return make_robot_from_config(robot_cfg)

    def _ensure_robot_connected(self) -> None:
        if self._robot is None:
            self._robot = self._build_robot()
        if not self._robot.is_connected:
            self._robot.connect()

    def _stop_active_playback(self, reason: str) -> None:
        with self._state_lock:
            active = self._active_playback
            if active is None:
                return
            active.stop_reason = reason
            active.stop_event.set()
            thread = active.thread

        if thread and thread.is_alive():
            thread.join(timeout=2.0)
            if thread.is_alive():
                raise RuntimeError("current replay thread did not stop in time")

    def _run_playback(self, active: ActivePlayback) -> None:
        prepared = active.prepared
        return_code = 0
        error_message: str | None = None

        try:
            for action_values in prepared.action_rows:
                if active.stop_event.is_set():
                    break

                frame_start = time.perf_counter()
                raw_action = {
                    name: action_values[index]
                    for index, name in enumerate(prepared.action_names)
                }
                robot_obs = self._robot.get_observation()
                processed_action = self._robot_action_processor((raw_action, robot_obs))
                self._robot.send_action(processed_action)

                precise_sleep(max(1.0 / prepared.spec.fps - (time.perf_counter() - frame_start), 0.0))
        except Exception as exc:
            return_code = 1
            error_message = str(exc)
            logger.exception("persistent JZ replay failed for %s", prepared.spec.slot_id)
        finally:
            finished_at = datetime.now()
            last_run = {
                "slot_id": prepared.spec.slot_id,
                "label": prepared.spec.label,
                "return_code": return_code,
                "started_at": active.started_at.isoformat(),
                "finished_at": finished_at.isoformat(),
                "log_path": self.log_path,
            }
            if active.stop_reason:
                last_run["stop_reason"] = active.stop_reason
            if error_message:
                last_run["error"] = error_message

            with self._state_lock:
                self._last_run = last_run
                if self._active_playback is active:
                    self._active_playback = None


class ThreadedUnixStreamServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


class ReplayWorkerRequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        raw_request = self.rfile.readline()
        if not raw_request:
            return

        request: dict[str, Any] | None = None
        try:
            request = json.loads(raw_request.decode("utf-8"))
            response = {
                "ok": True,
                "data": self.server.runtime.handle_command(request),  # type: ignore[attr-defined]
            }
        except Exception as exc:
            logger.exception("worker command failed")
            response = {
                "ok": False,
                "error": str(exc),
                "error_type": exc.__class__.__name__,
            }

        self.wfile.write((json.dumps(response, ensure_ascii=False) + "\n").encode("utf-8"))
        self.wfile.flush()

        if request and request.get("command") == "shutdown" and response.get("ok"):
            threading.Thread(target=self.server.shutdown, daemon=True).start()  # type: ignore[attr-defined]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Persistent JZ replay worker for RoboWeb.")
    parser.add_argument("--socket-path", required=True)
    parser.add_argument("--log-path", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    init_logging()

    socket_path = Path(args.socket_path).expanduser().resolve()
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path.unlink(missing_ok=True)

    log_path = Path(args.log_path).expanduser().resolve() if args.log_path else None
    runtime = PersistentJZReplayRuntime(log_path=log_path)
    server = ThreadedUnixStreamServer(str(socket_path), ReplayWorkerRequestHandler)
    server.runtime = runtime  # type: ignore[attr-defined]

    def handle_signal(signum, _frame) -> None:
        logger.info("received signal %s, shutting down replay worker", signum)
        runtime.shutdown()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        runtime.shutdown()
        server.server_close()
        socket_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
