from __future__ import annotations

import os
import re
import shlex
import shutil
import threading
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .job_store import FileJobStore
from .models import (
    DEFAULT_REQUEST_VALUES,
    FIRST_TRIAL_TEMPLATE,
    JobArtifacts,
    JobDetail,
    JobRequestSnapshot,
    JobState,
    JobStatus,
    JobSummary,
    LogReadResult,
    NormalizedRecordRequest,
    PreflightResponse,
    RESUME_TEMPLATE,
    RecordRequest,
    RuntimeInfo,
    StopJobResponse,
    format_number,
    utc_now,
)
from .process_supervisor import ProcessSupervisor, StopResult, is_process_alive, same_process_group
from .settings import AgilexWebSettings


START_EPISODE_RE = re.compile(r"^\[record\] Start episode (?P<current>\d+)/(?P<total>\d+) for .+$")
START_RESET_RE = re.compile(r"^\[record\] Start reset after episode (?P<current>\d+)/(?P<total>\d+) for .+$")
SAVING_EPISODE_RE = re.compile(r"^\[record\] Saving episode (?P<current>\d+)/(?P<total>\d+)$")
SAVED_EPISODE_RE = re.compile(r"^\[record\] Saved episode (?P<current>\d+)/(?P<total>\d+)$")
DATASET_OUTPUT_RE = re.compile(r"^Dataset output=(?P<path>.+)$")
CONFIG_FILE_RE = re.compile(r"^Config file=(?P<path>.+)$")


class JobNotFoundError(FileNotFoundError):
    pass


class JobConflictError(RuntimeError):
    def __init__(self, message: str, *, conflicts: list[str], preflight: PreflightResponse) -> None:
        super().__init__(message)
        self.conflicts = conflicts
        self.preflight = preflight


class JobStartError(RuntimeError):
    pass


@dataclass(frozen=True)
class ServiceComponents:
    settings: AgilexWebSettings
    store: FileJobStore
    supervisor: ProcessSupervisor
    service: "RecordJobService"


class RecordJobService:
    def __init__(
        self,
        settings: AgilexWebSettings | None = None,
        *,
        store: FileJobStore | None = None,
        supervisor: ProcessSupervisor | None = None,
    ) -> None:
        self.settings = settings or AgilexWebSettings.discover()
        self.store = store or FileJobStore(self.settings)
        self.supervisor = supervisor or ProcessSupervisor()
        self._lock = threading.RLock()
        self.APP_NAME = self.settings.app_name
        self.APP_VERSION = self.settings.app_version
        self.CONDA_ENV = self.settings.conda_env

    def initialize(self) -> None:
        self.store.ensure_layout()
        self._reconcile_all_jobs()

    def runtime_info(self) -> RuntimeInfo:
        self.store.ensure_layout()
        active_job = self.get_active_job()
        limitations = [
            "Only one active recording job is allowed at a time.",
            "Stop is best-effort and does not guarantee the current episode was fully saved.",
            "Closing or refreshing the browser does not stop the recording job.",
            "resume=true only validates dataset directory existence in this MVP.",
        ]
        if not self.settings.static_index_path.is_file():
            limitations.append(
                "Static homepage is missing. The API is usable, but GET / expects static/index.html."
            )
        return RuntimeInfo(
            app_name=self.APP_NAME,
            app_version=self.APP_VERSION,
            conda_env=self.CONDA_ENV,
            current_conda_env=self.settings.current_conda_env,
            script_path=str(self.settings.script_path),
            output_root=str(self.settings.outputs_root),
            dataset_root=str(self.settings.dataset_root),
            jobs_root=str(self.settings.jobs_root),
            static_index=str(self.settings.static_index_path),
            static_index_available=self.settings.static_index_path.is_file(),
            config_path_template=str(self.settings.config_path_template),
            per_job_config_supported=True,
            single_active_job_only=True,
            active_job_id=None if active_job is None else active_job.state.job_id,
            defaults={
                **DEFAULT_REQUEST_VALUES,
                "dataset_name_placeholder": "agilex_record_demo_video",
                "templates": {
                    "first_trial": FIRST_TRIAL_TEMPLATE,
                    "script_defaults": DEFAULT_REQUEST_VALUES,
                    "resume_existing": RESUME_TEMPLATE,
                },
            },
            limitations=limitations,
        )

    def preflight(self, request: RecordRequest | Mapping[str, Any]) -> PreflightResponse:
        raw_request = self._coerce_request(request)
        normalized = self._normalize_request(raw_request)
        with self._lock:
            return self._preflight_locked(raw_request, normalized)

    def create_job(self, request: RecordRequest | Mapping[str, Any]) -> JobDetail:
        raw_request = self._coerce_request(request)
        normalized = self._normalize_request(raw_request)
        with self._lock:
            preflight = self._preflight_locked(raw_request, normalized)
            if not preflight.ok:
                raise JobConflictError(
                    preflight.conflicts[0] if preflight.conflicts else "Job creation rejected.",
                    conflicts=preflight.conflicts,
                    preflight=preflight,
                )

            created_at = utc_now()
            job_id = self._new_job_id()
            artifacts = JobArtifacts(
                request_path=str(self.store.request_path(job_id)),
                state_path=str(self.store.state_path(job_id)),
                log_path=str(self.store.log_path(job_id)),
                events_path=str(self.store.events_path(job_id)),
                config_path=str(self.store.config_path(job_id)),
            )
            command = self._build_command(normalized)
            env_overrides = self.settings.env_for_job(job_id)
            request_snapshot = JobRequestSnapshot(
                job_id=job_id,
                created_at=created_at,
                request=raw_request.model_dump(mode="json"),
                normalized_request=normalized.model_dump(mode="json"),
                dataset_name=normalized.dataset_name,
                dataset_dir=str(self._dataset_dir(normalized.dataset_name)),
                output_root=str(self.settings.outputs_root),
                job_dir=str(self.store.job_dir(job_id)),
                command=command,
                command_text=shlex.join(command),
                env_overrides=env_overrides,
                estimated_duration_s=self._estimate_duration_s(normalized),
                warnings=list(preflight.warnings),
                notes=[
                    "record.sh is launched with a fixed 6-argument contract.",
                    "CONFIG_PATH is isolated per job via environment injection.",
                    "Stop is best-effort and does not guarantee the current episode was fully saved.",
                ],
                artifacts=artifacts,
            )
            initial_state = JobState(
                job_id=job_id,
                status="starting",
                phase="booting",
                active=True,
                can_stop=True,
                created_at=created_at,
                updated_at=created_at,
                total_episodes=normalized.num_episodes,
                status_message="Launching record.sh",
                warnings=list(preflight.warnings),
            )
            self.store.create_job(request_snapshot, initial_state)
            self.store.write_active_job_id(job_id)
            self.store.append_event(
                job_id,
                "job_created",
                {
                    "dataset_name": normalized.dataset_name,
                    "status": initial_state.status,
                    "phase": initial_state.phase,
                },
                at=created_at,
            )

            try:
                launch = self.supervisor.launch(
                    job_id=job_id,
                    cmd=command,
                    cwd=self.settings.repo_root,
                    env=self._merged_env(env_overrides),
                    log_path=self.store.log_path(job_id),
                    on_line=lambda line, current_job_id=job_id: self._handle_log_line(current_job_id, line),
                    on_exit=lambda returncode, current_job_id=job_id: self._handle_process_exit(
                        current_job_id,
                        returncode,
                    ),
                )
            except Exception as exc:
                failed_at = utc_now()
                failed_state = initial_state.model_copy(
                    update={
                        "status": "failed",
                        "phase": "finalizing",
                        "active": False,
                        "can_stop": False,
                        "finished_at": failed_at,
                        "updated_at": failed_at,
                        "status_message": f"Failed to launch record.sh: {exc}",
                        "conflicts": [],
                    }
                )
                self.store.write_state(failed_state)
                self.store.clear_active_job_id(expected_job_id=job_id)
                self.store.append_event(
                    job_id,
                    "job_start_failed",
                    {"error": str(exc)},
                    message=str(exc),
                    at=failed_at,
                )
                raise JobStartError(str(exc)) from exc

            started_at = utc_now()
            running_state = initial_state.model_copy(
                update={
                    "status": "running",
                    "phase": "booting",
                    "updated_at": started_at,
                    "started_at": started_at,
                    "pid": launch.pid,
                    "pgid": launch.pgid,
                    "status_message": "record.sh is running",
                }
            )
            self.store.write_state(running_state)
            self.store.append_event(
                job_id,
                "job_started",
                {"pid": launch.pid, "pgid": launch.pgid},
                at=started_at,
            )
            return JobDetail(request=request_snapshot, state=running_state)

    def get_active_job(self) -> JobDetail | None:
        with self._lock:
            return self._active_job_locked()

    def get_recent_jobs(self, limit: int = 20) -> list[JobSummary]:
        with self._lock:
            details = [self._reconcile_detail_locked(detail) for detail in self.store.list_jobs(limit=limit)]
            return [self._to_summary(detail) for detail in details]

    def list_jobs(self, limit: int = 20) -> list[JobSummary]:
        return self.get_recent_jobs(limit=limit)

    def get_job_detail(self, job_id: str) -> JobDetail:
        try:
            detail = self.store.load_job(job_id)
        except FileNotFoundError as exc:
            raise JobNotFoundError(job_id) from exc
        with self._lock:
            return self._reconcile_detail_locked(detail)

    def get_job(self, job_id: str) -> JobDetail:
        return self.get_job_detail(job_id)

    def get_logs(self, job_id: str, cursor: int = 0, limit_bytes: int = 65536) -> LogReadResult:
        detail = self.get_job_detail(job_id)
        lines, next_cursor, truncated = self.store.read_logs(job_id, cursor=cursor, limit_bytes=limit_bytes)
        return LogReadResult(
            job_id=job_id,
            cursor=cursor,
            next_cursor=next_cursor,
            lines=lines,
            truncated=truncated,
            status=detail.state.status,
            active=detail.state.active,
        )

    def stop_job(self, job_id: str) -> StopJobResponse:
        with self._lock:
            try:
                detail = self._reconcile_detail_locked(self.store.load_job(job_id))
            except FileNotFoundError as exc:
                raise JobNotFoundError(job_id) from exc

            if not detail.state.active:
                return StopJobResponse(
                    requested=False,
                    signal_stage="not_active",
                    message="Job is not active.",
                    job=detail,
                )

            stop_requested_at = utc_now()
            stop_state = detail.state.model_copy(
                update={
                    "status": "stop_requested",
                    "active": True,
                    "can_stop": False,
                    "stop_requested_at": stop_requested_at,
                    "updated_at": stop_requested_at,
                    "status_message": "Stop requested. Waiting for the process group to exit.",
                }
            )
            self.store.write_state(stop_state)
            self.store.append_event(job_id, "job_stop_requested", at=stop_requested_at)

            stop_result = self.supervisor.stop(job_id)
            if not stop_result.requested:
                stop_result = self.supervisor.stop_process_group(
                    pid=stop_state.pid,
                    pgid=stop_state.pgid,
                )

            refreshed = self._reconcile_detail_locked(self.store.load_job(job_id))
            if stop_result.requested and not stop_result.process_still_running and refreshed.state.active:
                refreshed = self._finalize_stopped_job(refreshed, returncode=refreshed.state.returncode)

            return StopJobResponse(
                requested=stop_result.requested,
                signal_stage=stop_result.signal_stage,
                message=self._stop_message(stop_result),
                job=refreshed,
            )

    def _preflight_locked(
        self,
        request: RecordRequest,
        normalized: NormalizedRecordRequest,
    ) -> PreflightResponse:
        dataset_dir = self._dataset_dir(normalized.dataset_name)
        dataset_exists = dataset_dir.exists()
        active_job = self._active_job_locked()

        conflicts: list[str] = []
        warnings: list[str] = []
        notes = [
            "The backend always builds the full 6-argument command and does not rely on record.sh positional defaults.",
            "CONFIG_PATH is isolated per job to avoid the shared outputs/record_config.json collision.",
            "Closing the browser does not stop the recording job.",
        ]

        if request.num_episodes == 1 and request.reset_time_s != 0:
            warnings.append("reset_time_s was normalized to 0 because num_episodes=1.")
        if dataset_exists and normalized.resume:
            warnings.append("The target dataset directory already exists. This run will append to it.")

        conflicts.extend(self._runtime_conflicts())
        if active_job is not None:
            conflicts.append(
                f"An active recording job already exists: {active_job.state.job_id}. Only one active job is allowed."
            )
        if dataset_exists and not normalized.resume:
            conflicts.append(
                f"Dataset directory already exists: {dataset_dir}. Set resume=true or choose another dataset_name."
            )
        if not dataset_exists and normalized.resume:
            conflicts.append(
                f"resume=true requires an existing dataset directory, but none was found at {dataset_dir}."
            )

        command = self._build_command(normalized)
        return PreflightResponse(
            ok=not conflicts,
            request=request.model_dump(mode="json"),
            normalized_request=normalized.model_dump(mode="json"),
            dataset_exists=dataset_exists,
            dataset_dir=str(dataset_dir),
            output_root=str(self.settings.outputs_root),
            runtime_jobs_root=str(self.settings.jobs_root),
            config_path_template=str(self.settings.config_path_template),
            command=command,
            command_text=shlex.join(command),
            env_overrides=self.settings.preview_env(),
            estimated_duration_s=self._estimate_duration_s(normalized),
            active_job_id=None if active_job is None else active_job.state.job_id,
            conflicts=conflicts,
            warnings=warnings,
            notes=notes,
        )

    def _active_job_locked(self) -> JobDetail | None:
        active_job_id = self.store.read_active_job_id()
        if active_job_id:
            try:
                detail = self._reconcile_detail_locked(self.store.load_job(active_job_id))
            except FileNotFoundError:
                self.store.clear_active_job_id(expected_job_id=active_job_id)
            else:
                if detail.state.active:
                    return detail
                self.store.clear_active_job_id(expected_job_id=active_job_id)

        for detail in self.store.list_jobs():
            reconciled = self._reconcile_detail_locked(detail)
            if reconciled.state.active:
                self.store.write_active_job_id(reconciled.state.job_id)
                return reconciled

        self.store.clear_active_job_id()
        return None

    def _reconcile_all_jobs(self) -> None:
        with self._lock:
            for detail in self.store.list_jobs():
                self._reconcile_detail_locked(detail)

    def _reconcile_detail_locked(self, detail: JobDetail) -> JobDetail:
        state = detail.state
        if not state.active:
            if self.store.read_active_job_id() == state.job_id:
                self.store.clear_active_job_id(expected_job_id=state.job_id)
            return detail

        alive = self.supervisor.is_running(state.job_id)
        if not alive and state.pid is not None:
            alive = is_process_alive(state.pid)
            if alive and state.pgid is not None:
                alive = same_process_group(state.pid, state.pgid)

        if alive:
            if state.status == "starting":
                updated_at = utc_now()
                running_state = state.model_copy(
                    update={
                        "status": "running",
                        "updated_at": updated_at,
                        "status_message": "record.sh is running",
                    }
                )
                self.store.write_state(running_state)
                self.store.append_event(state.job_id, "job_marked_running", at=updated_at)
                self.store.write_active_job_id(state.job_id)
                return JobDetail(request=detail.request, state=running_state)
            self.store.write_active_job_id(state.job_id)
            return detail

        if state.status == "stop_requested" or state.stop_requested_at is not None:
            return self._finalize_stopped_job(detail, returncode=state.returncode)
        return self._finalize_finished_job(detail, returncode=state.returncode)

    def _finalize_stopped_job(self, detail: JobDetail, returncode: int | None) -> JobDetail:
        finished_at = utc_now()
        stopped_state = detail.state.model_copy(
            update={
                "status": "stopped",
                "phase": "finalizing",
                "active": False,
                "can_stop": False,
                "returncode": returncode,
                "finished_at": detail.state.finished_at or finished_at,
                "updated_at": finished_at,
                "status_message": "Stop requested and the process group is no longer running.",
            }
        )
        self.store.write_state(stopped_state)
        self.store.clear_active_job_id(expected_job_id=detail.state.job_id)
        self.store.append_event(
            detail.state.job_id,
            "job_stopped",
            {"returncode": returncode},
            at=finished_at,
        )
        return JobDetail(request=detail.request, state=stopped_state)

    def _finalize_finished_job(self, detail: JobDetail, returncode: int | None) -> JobDetail:
        finished_at = utc_now()
        status: JobStatus
        message: str
        if returncode == 0:
            status = "succeeded"
            message = "record.sh exited successfully."
        else:
            status = "failed"
            if returncode is None:
                message = "Process is no longer running and no exit code was captured."
            else:
                message = f"record.sh exited with return code {returncode}."
        final_state = detail.state.model_copy(
            update={
                "status": status,
                "phase": "finalizing",
                "active": False,
                "can_stop": False,
                "returncode": returncode,
                "finished_at": detail.state.finished_at or finished_at,
                "updated_at": finished_at,
                "status_message": message,
            }
        )
        self.store.write_state(final_state)
        self.store.clear_active_job_id(expected_job_id=detail.state.job_id)
        self.store.append_event(
            detail.state.job_id,
            "job_finished",
            {"returncode": returncode, "status": status},
            at=finished_at,
        )
        return JobDetail(request=detail.request, state=final_state)

    def _handle_log_line(self, job_id: str, line: str) -> None:
        with self._lock:
            try:
                detail = self.store.load_job(job_id)
            except FileNotFoundError:
                return

            state = detail.state
            updates: dict[str, object] = {}
            event_type: str | None = None
            event_payload: dict[str, Any] | None = None
            status_message: str | None = None

            if match := START_EPISODE_RE.match(line):
                current_episode = int(match.group("current"))
                total_episodes = int(match.group("total"))
                updates.update(
                    {
                        "phase": "recording",
                        "current_episode": current_episode,
                        "total_episodes": total_episodes,
                    }
                )
                event_type = "job_progress"
                event_payload = {
                    "phase": "recording",
                    "current_episode": current_episode,
                    "total_episodes": total_episodes,
                }
                status_message = f"Recording episode {current_episode}/{total_episodes}."
            elif match := START_RESET_RE.match(line):
                current_episode = int(match.group("current"))
                total_episodes = int(match.group("total"))
                updates.update(
                    {
                        "phase": "resetting",
                        "current_episode": current_episode,
                        "total_episodes": total_episodes,
                    }
                )
                event_type = "job_progress"
                event_payload = {
                    "phase": "resetting",
                    "current_episode": current_episode,
                    "total_episodes": total_episodes,
                }
                status_message = f"Resetting after episode {current_episode}/{total_episodes}."
            elif match := SAVING_EPISODE_RE.match(line):
                current_episode = int(match.group("current"))
                total_episodes = int(match.group("total"))
                updates.update(
                    {
                        "phase": "saving",
                        "current_episode": current_episode,
                        "total_episodes": total_episodes,
                    }
                )
                event_type = "job_progress"
                event_payload = {
                    "phase": "saving",
                    "current_episode": current_episode,
                    "total_episodes": total_episodes,
                }
                status_message = f"Saving episode {current_episode}/{total_episodes}."
            elif match := SAVED_EPISODE_RE.match(line):
                current_episode = int(match.group("current"))
                total_episodes = int(match.group("total"))
                updates.update(
                    {
                        "phase": "saving",
                        "current_episode": current_episode,
                        "total_episodes": total_episodes,
                    }
                )
                event_type = "job_progress"
                event_payload = {
                    "phase": "saving",
                    "current_episode": current_episode,
                    "total_episodes": total_episodes,
                }
                status_message = f"Saved episode {current_episode}/{total_episodes}."
            elif match := DATASET_OUTPUT_RE.match(line):
                event_type = "dataset_output"
                event_payload = {"dataset_dir": match.group("path")}
            elif match := CONFIG_FILE_RE.match(line):
                event_type = "config_path"
                event_payload = {"config_path": match.group("path")}
            elif state.phase == "booting" and line.strip():
                updates["phase"] = "recording"
                status_message = "record.sh emitted output and entered recording flow."

            if updates or status_message:
                updates["updated_at"] = utc_now()
                if status_message is not None:
                    updates["status_message"] = status_message
                updated_state = state.model_copy(update=updates)
                self.store.write_state(updated_state)
                detail = JobDetail(request=detail.request, state=updated_state)

            if event_type is not None:
                payload = event_payload or {}
                self.store.append_event(
                    job_id,
                    event_type,
                    payload,
                    message=line if len(line) <= 512 else line[:509] + "...",
                )

    def _handle_process_exit(self, job_id: str, returncode: int) -> None:
        with self._lock:
            try:
                detail = self.store.load_job(job_id)
            except FileNotFoundError:
                return

            if detail.state.status == "stop_requested" or detail.state.stop_requested_at is not None:
                self._finalize_stopped_job(detail, returncode=returncode)
                return
            self._finalize_finished_job(detail, returncode=returncode)

    def _coerce_request(self, request: RecordRequest | Mapping[str, Any]) -> RecordRequest:
        if isinstance(request, RecordRequest):
            return request
        if isinstance(request, Mapping):
            return RecordRequest.model_validate(dict(request))
        raise TypeError(f"Unsupported request type: {type(request)!r}")

    def _normalize_request(self, request: RecordRequest) -> NormalizedRecordRequest:
        reset_time_s = 0.0 if request.num_episodes == 1 else float(request.reset_time_s)
        return NormalizedRecordRequest(
            dataset_name=request.dataset_name,
            episode_time_s=float(request.episode_time_s),
            num_episodes=int(request.num_episodes),
            fps=int(request.fps),
            reset_time_s=reset_time_s,
            resume=bool(request.resume),
        )

    def _build_command(self, request: NormalizedRecordRequest) -> list[str]:
        return [
            "bash",
            str(self.settings.script_path),
            request.dataset_name,
            format_number(request.episode_time_s),
            str(request.num_episodes),
            str(request.fps),
            format_number(request.reset_time_s),
            "true" if request.resume else "false",
        ]

    def _merged_env(self, overrides: dict[str, str]) -> dict[str, str]:
        env = os.environ.copy()
        env.update(overrides)
        env.setdefault("PYTHONUNBUFFERED", "1")
        return env

    def _runtime_conflicts(self) -> list[str]:
        conflicts: list[str] = []
        if not self.settings.script_path.is_file():
            conflicts.append(f"record.sh was not found at {self.settings.script_path}.")
        if self.settings.current_conda_env != self.settings.conda_env and shutil.which("conda") is None:
            conflicts.append(
                f"Conda executable was not found, but current env is '{self.settings.current_conda_env or ''}' "
                f"instead of required '{self.settings.conda_env}'."
            )
        if not self._is_writable_target(self.settings.outputs_root):
            conflicts.append(f"Output root is not writable: {self.settings.outputs_root}.")
        if not self._is_writable_target(self.settings.jobs_root):
            conflicts.append(f"Runtime jobs root is not writable: {self.settings.jobs_root}.")
        return conflicts

    def _dataset_dir(self, dataset_name: str) -> Path:
        return self.settings.dataset_root / dataset_name

    def _estimate_duration_s(self, request: NormalizedRecordRequest) -> float:
        reset_segments = max(0, request.num_episodes - 1)
        return round((request.episode_time_s * request.num_episodes) + (request.reset_time_s * reset_segments), 3)

    def _new_job_id(self) -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"job_{stamp}_{uuid.uuid4().hex[:8]}"

    def _is_writable_target(self, path: Path) -> bool:
        candidate = path
        while not candidate.exists() and candidate != candidate.parent:
            candidate = candidate.parent
        return os.access(candidate, os.W_OK)

    def _to_summary(self, detail: JobDetail) -> JobSummary:
        return JobSummary(
            job_id=detail.state.job_id,
            dataset_name=detail.request.dataset_name,
            dataset_dir=detail.request.dataset_dir,
            status=detail.state.status,
            phase=detail.state.phase,
            active=detail.state.active,
            can_stop=detail.state.can_stop,
            created_at=detail.state.created_at,
            started_at=detail.state.started_at,
            finished_at=detail.state.finished_at,
            returncode=detail.state.returncode,
            estimated_duration_s=detail.request.estimated_duration_s,
            status_message=detail.state.status_message,
        )

    def _stop_message(self, stop_result: StopResult) -> str:
        if not stop_result.requested:
            if stop_result.signal_stage == "not_running":
                return "The job process was already gone before the stop request reached it."
            if stop_result.signal_stage == "not_active":
                return "The job is not active."
            return "No running process group could be found for this job."
        if stop_result.process_still_running:
            return (
                "Stop signals were sent but the process group is still running. The job remains in stop_requested "
                "until the next reconciliation."
            )
        return f"Stop request sent via {stop_result.signal_stage}."


def build_service_components(settings: AgilexWebSettings | None = None) -> ServiceComponents:
    resolved_settings = settings or AgilexWebSettings.discover()
    store = FileJobStore(resolved_settings)
    supervisor = ProcessSupervisor()
    service = RecordJobService(
        resolved_settings,
        store=store,
        supervisor=supervisor,
    )
    return ServiceComponents(
        settings=resolved_settings,
        store=store,
        supervisor=supervisor,
        service=service,
    )


def build_record_job_service(settings: AgilexWebSettings | None = None) -> RecordJobService:
    return build_service_components(settings).service


__all__ = [
    "JobConflictError",
    "JobNotFoundError",
    "JobStartError",
    "RecordJobService",
    "ServiceComponents",
    "build_record_job_service",
    "build_service_components",
]
