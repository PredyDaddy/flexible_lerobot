from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
import os
import signal
import subprocess
import threading


class ManagedProcess:
    def __init__(
        self,
        argv: Sequence[str],
        cwd: Path,
        env: Mapping[str, str] | None = None,
        on_line: Callable[[str], None] | None = None,
        on_exit: Callable[[int | None], None] | None = None,
    ) -> None:
        self.argv = list(argv)
        self.cwd = Path(cwd)
        self.env = dict(env or {})
        self.on_line = on_line
        self.on_exit = on_exit
        self.process: subprocess.Popen[str] | None = None
        self.pid: int | None = None
        self.pgid: int | None = None
        self.returncode: int | None = None
        self._reader_thread: threading.Thread | None = None

    def start(self) -> None:
        if self.process is not None and self.is_running():
            raise RuntimeError("Process is already running.")

        env = os.environ.copy()
        env.update(self.env)
        env.setdefault("PYTHONUNBUFFERED", "1")

        self.process = subprocess.Popen(
            self.argv,
            cwd=str(self.cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            shell=False,
            start_new_session=True,
        )
        self.pid = self.process.pid
        self.pgid = os.getpgid(self.process.pid)
        self._reader_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader_thread.start()

    def _read_stdout(self) -> None:
        assert self.process is not None
        assert self.process.stdout is not None

        for line in self.process.stdout:
            clean_line = line.rstrip("\n")
            if self.on_line is not None:
                self.on_line(clean_line)

        self.returncode = self.process.wait()
        if self.on_exit is not None:
            self.on_exit(self.returncode)

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def stop(self, interrupt_timeout_s: float = 10.0, term_timeout_s: float = 2.0) -> int | None:
        if self.process is None:
            return None

        if not self.is_running():
            self.returncode = self.process.poll()
            self._join_reader()
            return self.returncode

        if self.pgid is None:
            self.pgid = os.getpgid(self.process.pid)

        self._signal_process_group(signal.SIGINT)
        if self._wait(interrupt_timeout_s):
            self._join_reader()
            return self.returncode

        self._signal_process_group(signal.SIGTERM)
        if self._wait(term_timeout_s):
            self._join_reader()
            return self.returncode

        self._signal_process_group(signal.SIGKILL)
        self._wait(1.0)
        self._join_reader()
        return self.returncode

    def _signal_process_group(self, signum: signal.Signals) -> None:
        if self.pgid is None:
            return
        try:
            os.killpg(self.pgid, signum)
        except ProcessLookupError:
            pass

    def _wait(self, timeout_s: float) -> bool:
        if self.process is None:
            return True
        try:
            self.returncode = self.process.wait(timeout=timeout_s)
            return True
        except subprocess.TimeoutExpired:
            return False

    def _join_reader(self) -> None:
        if self._reader_thread is None or self._reader_thread is threading.current_thread():
            return
        self._reader_thread.join(timeout=1.0)
