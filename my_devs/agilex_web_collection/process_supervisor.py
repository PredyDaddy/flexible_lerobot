from __future__ import annotations

import errno
import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


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
    argv: list[str]
    log_path: Path


class ProcessSupervisor:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._running: dict[str, RunningProcess] = {}

    def launch(
        self,
        *,
        job_id: str,
        cmd: Sequence[str],
        cwd: Path,
        env: Mapping[str, str],
        log_path: Path,
        on_line: Callable[[str], None] | None = None,
        on_exit: Callable[[int], None] | None = None,
    ) -> LaunchResult:
        argv = [str(part) for part in cmd]
        if not argv:
            raise ValueError("cmd must contain at least one argument")

        with self._lock:
            existing = self._running.get(job_id)
            if existing is not None and existing.process.poll() is None:
                raise RuntimeError(f"Job {job_id} is already running.")

        process = subprocess.Popen(
            argv,
            cwd=str(cwd),
            env=dict(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            shell=False,
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
            name=f"agilex-job-{job_id}",
        )
        with self._lock:
            self._running[job_id] = RunningProcess(
                process=process,
                pgid=pgid,
                reader_thread=reader_thread,
                argv=argv,
                log_path=log_path,
            )
        reader_thread.start()
        return LaunchResult(pid=process.pid, pgid=pgid)

    def is_running(self, job_id: str) -> bool:
        with self._lock:
            handle = self._running.get(job_id)
        return handle is not None and handle.process.poll() is None

    def stop(
        self,
        job_id: str,
        *,
        sigint_timeout: float = 10.0,
        sigterm_timeout: float = 5.0,
    ) -> StopResult:
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

        stages = (
            (signal.SIGINT, sigint_timeout, "SIGINT"),
            (signal.SIGTERM, sigterm_timeout, "SIGTERM"),
            (signal.SIGKILL, 1.0, "SIGKILL"),
        )
        for signum, timeout_s, stage in stages:
            try:
                os.killpg(pgid, signum)
            except ProcessLookupError:
                return StopResult(requested=True, signal_stage=stage, process_still_running=False)
            if self._wait_for_exit(pid, timeout_s):
                return StopResult(requested=True, signal_stage=stage, process_still_running=False)

        return StopResult(
            requested=True,
            signal_stage="SIGKILL",
            process_still_running=is_process_alive(pid),
        )

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
            self._running.pop(job_id, None)
        if on_exit is not None:
            try:
                on_exit(returncode)
            except Exception:
                pass

    def _wait_for_exit(self, pid: int, timeout_s: float) -> bool:
        deadline = time.monotonic() + max(0.0, timeout_s)
        while time.monotonic() < deadline:
            if not is_process_alive(pid):
                return True
            time.sleep(0.2)
        return not is_process_alive(pid)


__all__ = [
    "LaunchResult",
    "ProcessSupervisor",
    "RunningProcess",
    "StopResult",
    "is_process_alive",
    "same_process_group",
]
