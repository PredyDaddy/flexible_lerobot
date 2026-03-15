from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import threading

from my_devs.agilex_web_collection.app.api.schemas import JobEvent, JobState, LogChunk
from my_devs.agilex_web_collection.app.services.command_builder import AgileXRuntimePaths


@dataclass(frozen=True)
class JobPaths:
    job_dir: Path
    request_path: Path
    state_path: Path
    stdout_path: Path
    events_path: Path
    record_config_path: Path


class JobStore:
    def __init__(self, jobs_root: Path | None = None) -> None:
        runtime_paths = AgileXRuntimePaths.defaults()
        self.jobs_root = Path(jobs_root or runtime_paths.jobs_root).expanduser().resolve()
        self.locks_dir = self.jobs_root / "locks"
        self._write_lock = threading.Lock()
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self.locks_dir.mkdir(parents=True, exist_ok=True)

    def create_job(self, job_id: str) -> JobPaths:
        job_dir = self.jobs_root / job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        paths = self.job_paths(job_id)
        paths.stdout_path.touch(exist_ok=True)
        paths.events_path.touch(exist_ok=True)
        paths.record_config_path.touch(exist_ok=True)
        return paths

    def job_paths(self, job_id: str) -> JobPaths:
        job_dir = self.jobs_root / job_id
        return JobPaths(
            job_dir=job_dir,
            request_path=job_dir / "request.json",
            state_path=job_dir / "state.json",
            stdout_path=job_dir / "stdout.log",
            events_path=job_dir / "events.jsonl",
            record_config_path=job_dir / "record_config.json",
        )

    def write_request(self, job_id: str, payload: dict[str, object]) -> Path:
        request_path = self.job_paths(job_id).request_path
        with self._write_lock:
            request_path.write_text(
                json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return request_path

    def write_state(self, state: JobState) -> Path:
        state_path = self.job_paths(state.job_id).state_path
        with self._write_lock:
            state_path.write_text(
                json.dumps(state.model_dump(mode="json"), ensure_ascii=True, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return state_path

    def read_state(self, job_id: str) -> JobState:
        state_path = self.job_paths(job_id).state_path
        return JobState.model_validate_json(state_path.read_text(encoding="utf-8"))

    def append_stdout(self, job_id: str, line: str) -> Path:
        stdout_path = self.job_paths(job_id).stdout_path
        with self._write_lock:
            with stdout_path.open("a", encoding="utf-8") as f:
                f.write(f"{line}\n")
        return stdout_path

    def append_event(self, event: JobEvent) -> Path:
        events_path = self.job_paths(event.job_id).events_path
        with self._write_lock:
            with events_path.open("a", encoding="utf-8") as f:
                f.write(event.model_dump_json() + "\n")
        return events_path

    def read_logs(self, job_id: str, cursor: int = 0, max_chars: int = 65536) -> LogChunk:
        stdout_path = self.job_paths(job_id).stdout_path
        if not stdout_path.exists():
            return LogChunk(cursor=cursor, next_cursor=cursor, lines=[], eof=True)

        with stdout_path.open("r", encoding="utf-8") as f:
            f.seek(cursor)
            chunk = f.read(max_chars)
            next_cursor = f.tell()
            f.seek(0, 2)
            eof = next_cursor >= f.tell()

        return LogChunk(
            cursor=cursor,
            next_cursor=next_cursor,
            lines=chunk.splitlines(),
            eof=eof,
        )

    def read_recent_logs(self, job_id: str, max_lines: int = 200) -> list[str]:
        stdout_path = self.job_paths(job_id).stdout_path
        if not stdout_path.exists():
            return []
        with stdout_path.open("r", encoding="utf-8") as f:
            return f.read().splitlines()[-max_lines:]

    def list_recent_states(self, limit: int = 20) -> list[JobState]:
        state_paths = [
            path
            for path in self.jobs_root.glob("*/state.json")
            if path.parent.name != "locks"
        ]
        state_paths.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        return [JobState.model_validate_json(path.read_text(encoding="utf-8")) for path in state_paths[:limit]]

    def write_active_lock(self, job_id: str) -> Path:
        lock_path = self.locks_dir / "record.lock"
        with self._write_lock:
            lock_path.write_text(job_id + "\n", encoding="utf-8")
        return lock_path

    def clear_active_lock(self) -> None:
        lock_path = self.locks_dir / "record.lock"
        with self._write_lock:
            if lock_path.exists():
                lock_path.unlink()

    def read_active_lock(self) -> str | None:
        lock_path = self.locks_dir / "record.lock"
        if not lock_path.exists():
            return None
        return lock_path.read_text(encoding="utf-8").strip() or None
