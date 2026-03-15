from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from .models import JobDetail, JobEvent, JobRequestSnapshot, JobState, utc_now
from .settings import AgilexWebSettings


class FileJobStore:
    def __init__(
        self,
        settings: AgilexWebSettings | None = None,
        *,
        jobs_root: str | Path | None = None,
    ) -> None:
        self.settings = settings or AgilexWebSettings.discover()
        self.jobs_root = (
            Path(jobs_root).expanduser().resolve()
            if jobs_root is not None
            else self.settings.jobs_root
        )
        self.lock_dir = self.jobs_root / "locks"
        self.active_lock_path = self.lock_dir / "record.lock"
        self._write_lock = threading.Lock()

    @classmethod
    def from_settings(
        cls,
        settings: AgilexWebSettings | None = None,
        *,
        jobs_root: str | Path | None = None,
    ) -> "FileJobStore":
        return cls(settings=settings, jobs_root=jobs_root)

    def ensure_layout(self) -> None:
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self.lock_dir.mkdir(parents=True, exist_ok=True)

    def job_dir(self, job_id: str) -> Path:
        return self.jobs_root / job_id

    def request_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "request.json"

    def state_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "state.json"

    def log_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "stdout.log"

    def events_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "events.jsonl"

    def config_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "record_config.json"

    def create_job(self, request_snapshot: JobRequestSnapshot, state: JobState) -> Path:
        self.ensure_layout()
        job_dir = self.job_dir(request_snapshot.job_id)
        job_dir.mkdir(parents=True, exist_ok=False)
        self._write_json_atomic(self.request_path(request_snapshot.job_id), request_snapshot.model_dump(mode="json"))
        self._write_json_atomic(self.state_path(state.job_id), state.model_dump(mode="json"))
        self.log_path(state.job_id).touch(exist_ok=True)
        self.events_path(state.job_id).touch(exist_ok=True)
        self.config_path(state.job_id).touch(exist_ok=True)
        return job_dir

    def write_request(self, job_id: str, payload: JobRequestSnapshot | dict[str, Any]) -> Path:
        content = payload.model_dump(mode="json") if isinstance(payload, JobRequestSnapshot) else payload
        path = self.request_path(job_id)
        self._write_json_atomic(path, content)
        return path

    def load_request(self, job_id: str) -> JobRequestSnapshot:
        return JobRequestSnapshot.model_validate(self._read_json(self.request_path(job_id)))

    def load_state(self, job_id: str) -> JobState:
        return JobState.model_validate(self._read_json(self.state_path(job_id)))

    def load_job(self, job_id: str) -> JobDetail:
        return JobDetail(request=self.load_request(job_id), state=self.load_state(job_id))

    def write_state(self, state: JobState) -> Path:
        path = self.state_path(state.job_id)
        self._write_json_atomic(path, state.model_dump(mode="json"))
        return path

    def append_event(
        self,
        job_id: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
        *,
        message: str | None = None,
        at: str | None = None,
    ) -> Path:
        event = JobEvent(
            at=at or utc_now(),
            job_id=job_id,
            type=event_type,
            message=message,
            payload=payload or {},
        )
        path = self.events_path(job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._write_lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(event.model_dump_json())
                handle.write("\n")
        return path

    def list_job_ids(self, limit: int | None = None) -> list[str]:
        self.ensure_layout()
        job_ids = sorted(
            [path.name for path in self.jobs_root.iterdir() if path.is_dir() and path.name != "locks"],
            reverse=True,
        )
        if limit is None:
            return job_ids
        return job_ids[: max(1, limit)]

    def list_jobs(self, limit: int | None = None) -> list[JobDetail]:
        jobs: list[JobDetail] = []
        for job_id in self.list_job_ids():
            try:
                jobs.append(self.load_job(job_id))
            except FileNotFoundError:
                continue
        jobs.sort(key=lambda detail: detail.state.created_at, reverse=True)
        if limit is None:
            return jobs
        return jobs[: max(1, limit)]

    def read_logs(self, job_id: str, cursor: int = 0, limit_bytes: int = 65536) -> tuple[list[str], int, bool]:
        path = self.log_path(job_id)
        if not path.exists():
            return [], 0, False

        file_size = path.stat().st_size
        start = max(0, min(cursor, file_size))
        with path.open("rb") as handle:
            handle.seek(start)
            chunk = handle.read(max(1, limit_bytes))

        next_cursor = start + len(chunk)
        truncated = next_cursor < file_size
        if truncated and b"\n" in chunk:
            last_newline = chunk.rfind(b"\n") + 1
            chunk = chunk[:last_newline]
            next_cursor = start + last_newline

        text = chunk.decode("utf-8", errors="replace")
        return text.splitlines(), next_cursor, truncated

    def read_recent_logs(self, job_id: str, max_lines: int = 200) -> list[str]:
        path = self.log_path(job_id)
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return handle.read().splitlines()[-max(1, max_lines) :]

    def write_active_job_id(self, job_id: str) -> Path:
        self.ensure_layout()
        with self._write_lock:
            self.active_lock_path.write_text(f"{job_id}\n", encoding="utf-8")
        return self.active_lock_path

    def read_active_job_id(self) -> str | None:
        if not self.active_lock_path.exists():
            return None
        value = self.active_lock_path.read_text(encoding="utf-8").strip()
        return value or None

    def clear_active_job_id(self, expected_job_id: str | None = None) -> None:
        with self._write_lock:
            if not self.active_lock_path.exists():
                return
            if expected_job_id is not None:
                current = self.active_lock_path.read_text(encoding="utf-8").strip()
                if current and current != expected_job_id:
                    return
            self.active_lock_path.unlink()

    def _read_json(self, path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _write_json_atomic(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        with self._write_lock:
            with tmp_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
                handle.write("\n")
            tmp_path.replace(path)


JobStore = FileJobStore
