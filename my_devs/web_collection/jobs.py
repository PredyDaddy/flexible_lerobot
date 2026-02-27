from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass
class SubprocessJob:
    name: str
    cmd: list[str]
    cwd: Path
    env: dict[str, str] | None = None
    id: str | None = None
    on_line: Callable[[str], None] | None = None
    on_finish: Callable[[int | None], None] | None = None

    process: subprocess.Popen[str] | None = field(default=None, init=False)
    started_at: float | None = field(default=None, init=False)
    finished_at: float | None = field(default=None, init=False)
    returncode: int | None = field(default=None, init=False)
    _log_lines: deque[str] = field(default_factory=lambda: deque(maxlen=500), init=False)
    _log_lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _reader_thread: threading.Thread | None = field(default=None, init=False)

    def start(self) -> None:
        if self.process is not None and self.is_running():
            raise RuntimeError(f"Job '{self.name}' is already running.")

        self.started_at = time.time()
        self.finished_at = None
        self.returncode = None

        # Make sure we can import local modules when launching from other cwd.
        env = os.environ.copy()
        if self.env:
            env.update(self.env)
        env.setdefault("PYTHONUNBUFFERED", "1")

        self.process = subprocess.Popen(
            self.cmd,
            cwd=str(self.cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        self._reader_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader_thread.start()

    def _read_stdout(self) -> None:
        assert self.process is not None
        assert self.process.stdout is not None
        for line in self.process.stdout:
            line = line.rstrip("\n")
            with self._log_lock:
                self._log_lines.append(line)
            if self.on_line is not None:
                try:
                    self.on_line(line)
                except Exception:
                    # Never crash the reader thread due to side-effects (e.g. sound).
                    pass

        # Ensure returncode is captured when the stream ends.
        self.returncode = self.process.poll()
        self.finished_at = time.time()
        if self.on_finish is not None:
            try:
                self.on_finish(self.returncode)
            except Exception:
                pass

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def stop(self, timeout_s: float = 10.0) -> None:
        if self.process is None:
            return
        if not self.is_running():
            return

        # Prefer SIGINT so recorder scripts can flush/close files.
        try:
            self.process.send_signal(signal.SIGINT)
        except ProcessLookupError:
            return

        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if not self.is_running():
                self.returncode = self.process.poll()
                self.finished_at = time.time()
                return
            time.sleep(0.1)

        # Escalate to SIGTERM, then SIGKILL.
        try:
            self.process.terminate()
        except ProcessLookupError:
            return

        try:
            self.process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            try:
                self.process.kill()
            except ProcessLookupError:
                pass

        self.returncode = self.process.poll()
        self.finished_at = time.time()

    def logs(self) -> list[str]:
        with self._log_lock:
            return list(self._log_lines)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p
