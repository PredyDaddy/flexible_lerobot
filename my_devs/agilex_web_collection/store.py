from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from .models import JobDetail, JobRequestSnapshot, JobState
from .paths import AgilexWebPaths


class FileJobStore:
    def __init__(self, paths: AgilexWebPaths) -> None:
        self.paths = paths
        self._lock = threading.Lock()

    def ensure_layout(self) -> None:
        self.paths.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.paths.jobs_root.mkdir(parents=True, exist_ok=True)

    def job_dir(self, job_id: str) -> Path:
        return self.paths.job_dir(job_id)

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

    def create_job(self, request_snapshot: JobRequestSnapshot, state: JobState) -> None:
        job_dir = self.job_dir(request_snapshot.job_id)
        job_dir.mkdir(parents=True, exist_ok=False)
        self._write_json_atomic(self.request_path(request_snapshot.job_id), request_snapshot.model_dump(mode="json"))
        self._write_json_atomic(self.state_path(state.job_id), state.model_dump(mode="json"))
        self.log_path(state.job_id).touch()
        self.events_path(state.job_id).touch()

    def load_request(self, job_id: str) -> JobRequestSnapshot:
        return JobRequestSnapshot.model_validate(self._read_json(self.request_path(job_id)))

    def load_state(self, job_id: str) -> JobState:
        return JobState.model_validate(self._read_json(self.state_path(job_id)))

    def load_job(self, job_id: str) -> JobDetail:
        return JobDetail(request=self.load_request(job_id), state=self.load_state(job_id))

    def write_state(self, state: JobState) -> None:
        self._write_json_atomic(self.state_path(state.job_id), state.model_dump(mode="json"))

    def append_event(self, job_id: str, event_type: str, payload: dict) -> None:
        event = {"type": event_type, **payload}
        path = self.events_path(job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=True, sort_keys=True))
                handle.write("\n")

    def list_job_ids(self) -> list[str]:
        self.ensure_layout()
        return sorted([path.name for path in self.paths.jobs_root.iterdir() if path.is_dir()], reverse=True)

    def list_jobs(self) -> list[JobDetail]:
        jobs: list[JobDetail] = []
        for job_id in self.list_job_ids():
            try:
                jobs.append(self.load_job(job_id))
            except FileNotFoundError:
                continue
        jobs.sort(key=lambda detail: detail.state.created_at, reverse=True)
        return jobs

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

    def _read_json(self, path: Path) -> dict:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _write_json_atomic(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        with self._lock:
            with tmp_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
            tmp_path.replace(path)

