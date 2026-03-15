from __future__ import annotations

import errno
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


def is_process_alive(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return False
        return True
    return True


def same_process_group(pid: int | None, pgid: int | None) -> bool:
    if pid is None or pgid is None or not is_process_alive(pid):
        return False
    try:
        return os.getpgid(pid) == pgid
    except ProcessLookupError:
        return False


@dataclass(frozen=True)
class LaunchResult:
    pid: int
    pgid: int


@dataclass(frozen=True)
class StopResult:
    requested: bool
    signal_stage: str
    process_still_running: bool


@dataclass
class RunningProcess:
    process: subprocess.Popen[str]
    pgid: int
    reader_thread: threading.Thread


class ProcessSupervisor:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._running: dict[str, RunningProcess] = {}
        self._finished_returncodes: dict[str, int] = {}

    def poll_returncode(self, job_id: str) -> int | None:
        """Return the process returncode if available.

        This is used by the service reconciliation logic to avoid a race where the
        process has exited but the output streaming thread hasn't reached the
        `wait()` call yet (e.g., because a descendant process keeps stdout open).
        """
        with self._lock:
            handle = self._running.get(job_id)
            cached = self._finished_returncodes.get(job_id)
        if cached is not None:
            return cached
        if handle is None:
            return None
        returncode = handle.process.poll()
        if returncode is None:
            return None
        # Best-effort cleanup: avoid leaking handles when stdout never reaches EOF.
        with self._lock:
            self._finished_returncodes[job_id] = returncode
            existing = self._running.get(job_id)
            if existing is handle:
                self._running.pop(job_id, None)
        return returncode

    def launch(
        self,
        *,
        job_id: str,
        cmd: list[str],
        cwd: Path,
        env: dict[str, str],
        log_path: Path,
        on_line: Callable[[str], None] | None = None,
        on_exit: Callable[[int], None] | None = None,
    ) -> LaunchResult:
        with self._lock:
            existing = self._running.get(job_id)
            if existing is not None and existing.process.poll() is None:
                raise RuntimeError(f"Job {job_id} is already running.")
            self._finished_returncodes.pop(job_id, None)

        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        pgid = os.getpgid(process.pid)

        reader_thread = threading.Thread(
            target=self._stream_output,
            kwargs={
                "job_id": job_id,
                "process": process,
                "log_path": log_path,
                "on_line": on_line,
                "on_exit": on_exit,
            },
            daemon=True,
        )
        with self._lock:
            self._running[job_id] = RunningProcess(process=process, pgid=pgid, reader_thread=reader_thread)
        reader_thread.start()
        return LaunchResult(pid=process.pid, pgid=pgid)

    def is_running(self, job_id: str) -> bool:
        with self._lock:
            handle = self._running.get(job_id)
        return handle is not None and handle.process.poll() is None

    def stop(self, job_id: str, *, sigint_timeout: float = 10.0, sigterm_timeout: float = 5.0) -> StopResult:
        with self._lock:
            handle = self._running.get(job_id)
        if handle is None:
            return StopResult(requested=False, signal_stage="missing", process_still_running=False)
        return self.stop_process_group(
            pid=handle.process.pid,
            pgid=handle.pgid,
            sigint_timeout=sigint_timeout,
            sigterm_timeout=sigterm_timeout,
        )

    def stop_process_group(
        self,
        *,
        pid: int | None,
        pgid: int | None,
        sigint_timeout: float = 10.0,
        sigterm_timeout: float = 5.0,
    ) -> StopResult:
        if pid is None:
            return StopResult(requested=False, signal_stage="missing", process_still_running=False)
        if pgid is None:
            try:
                pgid = os.getpgid(pid)
            except ProcessLookupError:
                return StopResult(requested=False, signal_stage="not_running", process_still_running=False)
        if not same_process_group(pid, pgid):
            return StopResult(requested=False, signal_stage="not_running", process_still_running=False)

        for sig, timeout_s, stage in (
            (signal.SIGINT, sigint_timeout, "SIGINT"),
            (signal.SIGTERM, sigterm_timeout, "SIGTERM"),
            (signal.SIGKILL, 1.0, "SIGKILL"),
        ):
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                return StopResult(requested=True, signal_stage=stage, process_still_running=False)
            if not self._wait_for_exit(pid, timeout_s):
                continue
            return StopResult(requested=True, signal_stage=stage, process_still_running=False)

        return StopResult(requested=True, signal_stage="SIGKILL", process_still_running=is_process_alive(pid))

    def _stream_output(
        self,
        *,
        job_id: str,
        process: subprocess.Popen[str],
        log_path: Path,
        on_line: Callable[[str], None] | None,
        on_exit: Callable[[int], None] | None,
    ) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            assert process.stdout is not None
            for raw_line in process.stdout:
                line = raw_line.rstrip("\n")
                handle.write(line)
                handle.write("\n")
                handle.flush()
                if on_line is not None:
                    try:
                        on_line(line)
                    except Exception:
                        pass

        returncode = process.wait()
        with self._lock:
            self._finished_returncodes[job_id] = returncode
            self._running.pop(job_id, None)
        if on_exit is not None:
            try:
                on_exit(returncode)
            except Exception:
                pass

    def _wait_for_exit(self, pid: int, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if not is_process_alive(pid):
                return True
            time.sleep(0.2)
        return not is_process_alive(pid)
