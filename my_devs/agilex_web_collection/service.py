from __future__ import annotations

import os
import re
import shlex
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .models import (
    ACTIVE_JOB_STATUSES,
    JobArtifacts,
    JobDetail,
    JobRequestSnapshot,
    JobState,
    JobSummary,
    LogReadResult,
    NormalizedRecordRequest,
    PreflightResponse,
    RecordRequest,
    RuntimeInfo,
    StopJobResponse,
    format_number,
    utc_now,
)
from .paths import AgilexWebPaths
from .store import FileJobStore
from .supervisor import ProcessSupervisor, StopResult, is_process_alive


class JobNotFoundError(FileNotFoundError):
    pass


class JobConflictError(RuntimeError):
    def __init__(self, message: str, *, conflicts: list[str], preflight: PreflightResponse) -> None:
        super().__init__(message)
        self.conflicts = conflicts
        self.preflight = preflight


class JobStartError(RuntimeError):
    pass


class RecordJobService:
    APP_NAME = "agilex_web_collection"
    APP_VERSION = "0.1.0"
    CONDA_ENV = "lerobot_flex"

    def __init__(
        self,
        *,
        paths: AgilexWebPaths,
        store: FileJobStore,
        supervisor: ProcessSupervisor,
    ) -> None:
        self.paths = paths
        self.store = store
        self.supervisor = supervisor
        self._lock = threading.Lock()
        self._episode_pattern = re.compile(r"episode[^0-9]*(\d+)(?:[^0-9]+(\d+))?", re.IGNORECASE)

    def initialize(self) -> None:
        self.store.ensure_layout()
        self._reconcile_all_jobs()

    def runtime_info(self) -> RuntimeInfo:
        self.store.ensure_layout()
        active_job = self.get_active_job()
        limitations = [
            "Only one active recording job is allowed at a time.",
            "Stop is best-effort and does not guarantee the current episode was fully saved.",
            "resume=true only checks directory existence in this MVP; dataset structure is not deeply validated.",
            "single_task_text is recorded as the language task for each capture job.",
            "repo_prefix defaults to dummy and forms the dataset repo_id namespace.",
        ]
        static_index = self.paths.static_dir / "index.html"
        if not static_index.is_file():
            limitations.append(
                "Frontend static files are not part of this backend task. GET / expects static/index.html and will "
                "return 503 until the frontend files are added."
            )
        return RuntimeInfo(
            app_name=self.APP_NAME,
            app_version=self.APP_VERSION,
            conda_env=self.CONDA_ENV,
            current_conda_env=os.environ.get("CONDA_DEFAULT_ENV"),
            script_path=str(self.paths.script_path),
            output_root=str(self.paths.outputs_root),
            dataset_root=str(self.paths.dataset_root),
            jobs_root=str(self.paths.jobs_root),
            static_index=str(static_index),
            static_index_available=static_index.is_file(),
            config_path_template=str(self.paths.jobs_root / "{job_id}" / "record_config.json"),
            per_job_config_supported=True,
            single_active_job_only=True,
            active_job_id=None if active_job is None else active_job.state.job_id,
            defaults={
                "repo_prefix": "dummy",
                "dataset_name_placeholder": "agilex_record_demo_video",
                "episode_time_s": 8,
                "num_episodes": 1,
                "fps": 10,
                "reset_time_s": 0,
                "resume": False,
                "single_task_text": "agilex static record test",
            },
            limitations=limitations,
        )

    def preflight(self, request: RecordRequest) -> PreflightResponse:
        normalized = self._normalize_request(request)
        with self._lock:
            return self._preflight_locked(request, normalized)

    def create_job(self, request: RecordRequest) -> JobDetail:
        normalized = self._normalize_request(request)
        with self._lock:
            preflight = self._preflight_locked(request, normalized)
            if not preflight.ok:
                raise JobConflictError(
                    preflight.conflicts[0] if preflight.conflicts else "Job creation rejected.",
                    conflicts=preflight.conflicts,
                    preflight=preflight,
                )

            created_at = utc_now()
            job_id = self._new_job_id()
            job_dir = self.paths.job_dir(job_id)
            artifacts = JobArtifacts(
                request_path=str(self.store.request_path(job_id)),
                state_path=str(self.store.state_path(job_id)),
                log_path=str(self.store.log_path(job_id)),
                events_path=str(self.store.events_path(job_id)),
                config_path=str(self.store.config_path(job_id)),
            )
            command = self._build_command(normalized)
            env_overrides = self._build_env(job_id, normalized)
            notes = [
                "CONFIG_PATH is isolated per job via environment injection.",
                "DATASET_REPO_ID is composed from repo_prefix/dataset_name and injected via environment.",
                "Stopping is best-effort and cannot guarantee the current episode boundary is fully persisted.",
            ]
            request_snapshot = JobRequestSnapshot(
                job_id=job_id,
                created_at=created_at,
                request=request.model_dump(mode="json"),
                normalized_request=normalized.model_dump(mode="json"),
                dataset_name=normalized.dataset_name,
                dataset_dir=str(self._dataset_dir(normalized)),
                output_root=str(self.paths.outputs_root),
                job_dir=str(job_dir),
                command=command,
                command_text=shlex.join(command),
                env_overrides=env_overrides,
                estimated_duration_s=self._estimate_duration_s(normalized),
                warnings=list(preflight.warnings),
                notes=notes,
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
                started_at=None,
                total_episodes=normalized.num_episodes,
                status_message="Launching record.sh",
                warnings=list(preflight.warnings),
            )
            self.store.create_job(request_snapshot, initial_state)
            self.store.append_event(
                job_id,
                "job_created",
                {
                    "at": created_at,
                    "dataset_name": normalized.dataset_name,
                    "status": "starting",
                },
            )

            try:
                launch = self.supervisor.launch(
                    job_id=job_id,
                    cmd=command,
                    cwd=self.paths.repo_root,
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
                    }
                )
                self.store.write_state(failed_state)
                self.store.append_event(
                    job_id,
                    "job_start_failed",
                    {"at": failed_at, "message": str(exc)},
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
                {"at": started_at, "pid": launch.pid, "pgid": launch.pgid},
            )
            return JobDetail(request=request_snapshot, state=running_state)

    def list_jobs(self, limit: int = 20) -> list[JobSummary]:
        self._reconcile_all_jobs()
        items: list[JobSummary] = []
        for detail in self.store.list_jobs()[: max(1, limit)]:
            items.append(self._to_summary(detail))
        return items

    def get_active_job(self) -> JobDetail | None:
        with self._lock:
            for detail in self.store.list_jobs():
                detail = self._reconcile_detail_locked(detail)
                if detail.state.active:
                    return detail
        return None

    def get_job(self, job_id: str) -> JobDetail:
        try:
            detail = self.store.load_job(job_id)
        except FileNotFoundError as exc:
            raise JobNotFoundError(job_id) from exc
        with self._lock:
            return self._reconcile_detail_locked(detail)

    def get_logs(self, job_id: str, cursor: int = 0, limit_bytes: int = 65536) -> LogReadResult:
        detail = self.get_job(job_id)
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
            self.store.append_event(job_id, "job_stop_requested", {"at": stop_requested_at})

            stop_result = self.supervisor.stop(job_id)
            if not stop_result.requested:
                stop_result = self.supervisor.stop_process_group(pid=stop_state.pid, pgid=stop_state.pgid)

            if stop_result.requested and not stop_result.process_still_running:
                refreshed = self._reconcile_detail_locked(self.store.load_job(job_id))
                if refreshed.state.active:
                    refreshed = self._finalize_stopped_job(refreshed, returncode=refreshed.state.returncode)
            else:
                refreshed = self._reconcile_detail_locked(self.store.load_job(job_id))

            message = self._stop_message(stop_result)
            return StopJobResponse(
                requested=stop_result.requested,
                signal_stage=stop_result.signal_stage,
                message=message,
                job=refreshed,
            )

    def _preflight_locked(
        self,
        request: RecordRequest,
        normalized: NormalizedRecordRequest,
    ) -> PreflightResponse:
        dataset_dir = self._dataset_dir(normalized)
        dataset_exists = dataset_dir.exists()
        active_job = self._active_job_locked()

        conflicts: list[str] = []
        warnings: list[str] = []
        notes = [
            "resume=true only validates directory existence in this MVP.",
            "record.sh is launched with a per-job CONFIG_PATH and a fixed 7-argument command contract.",
            "repo_prefix and dataset_name are combined into DATASET_REPO_ID via environment injection.",
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

        env_overrides = self._preview_env(normalized)
        command = self._build_command(normalized)
        return PreflightResponse(
            ok=not conflicts,
            request=request.model_dump(mode="json"),
            normalized_request=normalized.model_dump(mode="json"),
            dataset_exists=dataset_exists,
            dataset_dir=str(dataset_dir),
            output_root=str(self.paths.outputs_root),
            runtime_jobs_root=str(self.paths.jobs_root),
            config_path_template=str(self.paths.jobs_root / "{job_id}" / "record_config.json"),
            command=command,
            command_text=shlex.join(command),
            env_overrides=env_overrides,
            estimated_duration_s=self._estimate_duration_s(normalized),
            active_job_id=None if active_job is None else active_job.state.job_id,
            conflicts=conflicts,
            warnings=warnings,
            notes=notes,
        )

    def _runtime_conflicts(self) -> list[str]:
        conflicts: list[str] = []
        if not self.paths.script_path.is_file():
            conflicts.append(f"record.sh was not found at {self.paths.script_path}.")
        if not self._is_writable_target(self.paths.outputs_root):
            conflicts.append(f"Output root is not writable: {self.paths.outputs_root}.")
        if not self._is_writable_target(self.paths.jobs_root):
            conflicts.append(f"Runtime jobs root is not writable: {self.paths.jobs_root}.")
        return conflicts

    def _normalize_request(self, request: RecordRequest) -> NormalizedRecordRequest:
        reset_time_s = 0.0 if request.num_episodes == 1 else float(request.reset_time_s)
        return NormalizedRecordRequest(
            repo_prefix=request.repo_prefix,
            dataset_name=request.dataset_name,
            episode_time_s=float(request.episode_time_s),
            num_episodes=int(request.num_episodes),
            fps=int(request.fps),
            reset_time_s=reset_time_s,
            resume=bool(request.resume),
            single_task_text=request.single_task_text,
        )

    def _build_command(self, request: NormalizedRecordRequest) -> list[str]:
        return [
            "bash",
            str(self.paths.script_path),
            request.dataset_name,
            format_number(request.episode_time_s),
            str(request.num_episodes),
            str(request.fps),
            format_number(request.reset_time_s),
            "true" if request.resume else "false",
            request.single_task_text,
        ]

    def _build_env(self, job_id: str, request: NormalizedRecordRequest) -> dict[str, str]:
        return {
            "CONDA_ENV": self.CONDA_ENV,
            "HF_LEROBOT_HOME": str(self.paths.outputs_root),
            "DATASET_REPO_ID": request.repo_id,
            "CONFIG_PATH": str(self.store.config_path(job_id)),
            "PYTHONUNBUFFERED": "1",
        }

    def _preview_env(self, request: NormalizedRecordRequest) -> dict[str, str]:
        return {
            "CONDA_ENV": self.CONDA_ENV,
            "HF_LEROBOT_HOME": str(self.paths.outputs_root),
            "DATASET_REPO_ID": request.repo_id,
            "CONFIG_PATH": str(self.paths.jobs_root / "{job_id}" / "record_config.json"),
            "PYTHONUNBUFFERED": "1",
        }

    def _merged_env(self, overrides: dict[str, str]) -> dict[str, str]:
        env = os.environ.copy()
        env.update(overrides)
        return env

    def _dataset_dir(self, request: NormalizedRecordRequest) -> Path:
        return self.paths.dataset_root / request.repo_prefix / request.dataset_name

    def _estimate_duration_s(self, request: NormalizedRecordRequest) -> float:
        reset_segments = max(0, request.num_episodes - 1)
        return round((request.episode_time_s * request.num_episodes) + (request.reset_time_s * reset_segments), 3)

    def _active_job_locked(self) -> JobDetail | None:
        for detail in self.store.list_jobs():
            detail = self._reconcile_detail_locked(detail)
            if detail.state.active:
                return detail
        return None

    def _reconcile_all_jobs(self) -> None:
        with self._lock:
            for detail in self.store.list_jobs():
                self._reconcile_detail_locked(detail)

    def _reconcile_detail_locked(self, detail: JobDetail) -> JobDetail:
        state = detail.state
        if not state.active:
            # Best-effort correction for a rare race:
            # the job was already marked inactive but the exit code had not been
            # persisted yet. If the supervisor still has a handle, reconcile it.
            if state.returncode is None:
                returncode = self.supervisor.poll_returncode(state.job_id)
                if returncode is not None:
                    if state.status == "stop_requested" or state.stop_requested_at is not None:
                        return self._finalize_stopped_job(detail, returncode=returncode)
                    return self._finalize_finished_job(detail, returncode=returncode)
            return detail

        alive = self.supervisor.is_running(state.job_id)
        if not alive and state.pid is not None:
            alive = is_process_alive(state.pid)

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
                self.store.append_event(state.job_id, "job_marked_running", {"at": updated_at})
                return JobDetail(request=detail.request, state=running_state)
            return detail

        returncode = state.returncode
        if returncode is None:
            # Avoid a reconciliation race:
            # - the process already exited (pid is gone / poll != None)
            # - but the streaming thread hasn't written the exit code back yet
            #   (e.g., because a descendant process keeps stdout open).
            returncode = self.supervisor.poll_returncode(state.job_id)

        if state.status == "stop_requested" or state.stop_requested_at is not None:
            return self._finalize_stopped_job(detail, returncode=returncode)
        return self._finalize_finished_job(detail, returncode=returncode)

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
        self.store.append_event(detail.state.job_id, "job_stopped", {"at": finished_at, "returncode": returncode})
        return JobDetail(request=detail.request, state=stopped_state)

    def _finalize_finished_job(self, detail: JobDetail, returncode: int | None) -> JobDetail:
        finished_at = utc_now()
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
        self.store.append_event(detail.state.job_id, "job_finished", {"at": finished_at, "returncode": returncode})
        return JobDetail(request=detail.request, state=final_state)

    def _handle_log_line(self, job_id: str, line: str) -> None:
        with self._lock:
            try:
                detail = self.store.load_job(job_id)
            except FileNotFoundError:
                return

            updates: dict[str, object] = {}
            phase = self._phase_from_log_line(detail.state.phase, line)
            if phase != detail.state.phase:
                updates["phase"] = phase
            current_episode, total_episodes = self._episode_progress_from_log_line(line)
            if current_episode is not None:
                updates["current_episode"] = current_episode
            if total_episodes is not None:
                updates["total_episodes"] = total_episodes
            if not updates:
                return

            updates["updated_at"] = utc_now()
            updated_state = detail.state.model_copy(update=updates)
            self.store.write_state(updated_state)
            self.store.append_event(
                job_id,
                "job_progress",
                {
                    "at": updated_state.updated_at,
                    "phase": updated_state.phase,
                    "current_episode": updated_state.current_episode,
                    "total_episodes": updated_state.total_episodes,
                },
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

    def _phase_from_log_line(self, current_phase: str, line: str) -> str:
        lowered = line.lower()
        if "reset" in lowered:
            return "resetting"
        if any(token in lowered for token in ("saving", "saved", "write", "flush")):
            return "saving"
        if any(token in lowered for token in ("final", "completed", "complete", "finished")):
            return "finalizing"
        if current_phase == "booting" and line.strip():
            return "recording"
        return current_phase

    def _episode_progress_from_log_line(self, line: str) -> tuple[int | None, int | None]:
        match = self._episode_pattern.search(line)
        if not match:
            return None, None
        current_episode = int(match.group(1))
        total_episodes = int(match.group(2)) if match.group(2) else None
        return current_episode, total_episodes

    def _new_job_id(self) -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"job_{stamp}_{uuid.uuid4().hex[:8]}"

    def _is_writable_target(self, path: Path) -> bool:
        target = path if path.exists() else path.parent
        return os.access(target, os.W_OK)

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
