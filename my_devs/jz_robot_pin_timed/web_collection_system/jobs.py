from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ManagedJob:
    name: str
    command: list[str]
    cwd: Path
    env: dict[str, str] | None = None
    log_limit: int = 800

    process: subprocess.Popen[str] | None = field(default=None, init=False)
    started_at: float | None = field(default=None, init=False)
    finished_at: float | None = field(default=None, init=False)
    returncode: int | None = field(default=None, init=False)
    _logs: deque[str] = field(init=False)
    _log_lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _reader_thread: threading.Thread | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._logs = deque(maxlen=self.log_limit)

    def start(self) -> None:
        if self.is_running():
            raise RuntimeError(f"{self.name} is already running")

        process_env = os.environ.copy()
        if self.env:
            process_env.update(self.env)
        process_env.setdefault("PYTHONUNBUFFERED", "1")

        self.started_at = time.time()
        self.finished_at = None
        self.returncode = None
        self.process = subprocess.Popen(
            self.command,
            cwd=str(self.cwd),
            env=process_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        self._append_log(f"[web] started pid={self.process.pid}: {self.name}")
        self._reader_thread = threading.Thread(target=self._read_output, daemon=True)
        self._reader_thread.start()

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def stop(self, timeout_s: float = 15.0) -> None:
        process = self.process
        if process is None or not self.is_running():
            return

        self._append_log("[web] stopping with SIGINT")
        self._signal_group(signal.SIGINT)
        if self._wait(timeout_s):
            return

        self._append_log("[web] SIGINT timeout; escalating to SIGTERM")
        self._signal_group(signal.SIGTERM)
        if self._wait(3.0):
            return

        self._append_log("[web] SIGTERM timeout; escalating to SIGKILL")
        self._signal_group(signal.SIGKILL)
        self._wait(2.0)

    def status(self) -> dict[str, object]:
        return {
            "name": self.name,
            "running": self.is_running(),
            "pid": None if self.process is None else self.process.pid,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "returncode": self.returncode,
            "logs": self.logs(),
        }

    def logs(self) -> list[str]:
        with self._log_lock:
            return list(self._logs)

    def _append_log(self, line: str) -> None:
        with self._log_lock:
            self._logs.append(line)

    def _read_output(self) -> None:
        process = self.process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            self._append_log(line.rstrip("\n"))
        self.returncode = process.wait()
        self.finished_at = time.time()
        self._append_log(f"[web] finished returncode={self.returncode}")

    def _signal_group(self, sig: signal.Signals) -> None:
        process = self.process
        if process is None:
            return
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return

    def _wait(self, timeout_s: float) -> bool:
        process = self.process
        if process is None:
            return True
        try:
            self.returncode = process.wait(timeout=timeout_s)
            self.finished_at = time.time()
            return True
        except subprocess.TimeoutExpired:
            return False


def idle_status(name: str) -> dict[str, object]:
    return {
        "name": name,
        "running": False,
        "pid": None,
        "started_at": None,
        "finished_at": None,
        "returncode": None,
        "logs": [],
    }
