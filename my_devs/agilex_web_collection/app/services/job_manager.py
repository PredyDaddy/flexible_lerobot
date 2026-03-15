from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re
import threading
import time
import uuid

from my_devs.agilex_web_collection.app.api.schemas import (
    JobArtifacts,
    JobEvent,
    JobPhase,
    JobProgress,
    JobState,
    JobStatus,
    LogChunk,
    RecordRequest,
    RecordValidationResult,
)
from my_devs.agilex_web_collection.app.services.command_builder import (
    AgileXRuntimePaths,
    build_process_launch_spec,
    build_record_command_preview,
)
from my_devs.agilex_web_collection.app.services.job_store import JobStore
from my_devs.agilex_web_collection.app.services.param_validator import (
    RecordRequestValidationError,
    validate_and_normalize_record_request,
)
from my_devs.agilex_web_collection.app.services.process_runner import ManagedProcess


START_EPISODE_RE = re.compile(r"^\[record\] Start episode (?P<current>\d+)/(?P<total>\d+) for .+$")
START_RESET_RE = re.compile(r"^\[record\] Start reset after episode (?P<current>\d+)/(?P<total>\d+) for .+$")
SAVING_EPISODE_RE = re.compile(r"^\[record\] Saving episode (?P<current>\d+)/(?P<total>\d+)$")
SAVED_EPISODE_RE = re.compile(r"^\[record\] Saved episode (?P<current>\d+)/(?P<total>\d+)$")
DATASET_OUTPUT_RE = re.compile(r"^Dataset output=(?P<path>.+)$")
CONFIG_FILE_RE = re.compile(r"^Config file=(?P<path>.+)$")


TERMINAL_JOB_STATUSES = {
    JobStatus.REJECTED,
    JobStatus.STOPPED,
    JobStatus.SUCCEEDED,
    JobStatus.FAILED,
}


class RecordJobManagerError(RuntimeError):
    pass


class ActiveJobConflictError(RecordJobManagerError):
    pass


class JobNotFoundError(RecordJobManagerError):
    pass


class RecordJobManager:
    def __init__(
        self,
        store: JobStore | None = None,
        runtime_paths: AgileXRuntimePaths | None = None,
    ) -> None:
        self.runtime_paths = runtime_paths or AgileXRuntimePaths.defaults()
        self.store = store or JobStore(self.runtime_paths.jobs_root)
        self._lock = threading.RLock()
        self._states: dict[str, JobState] = {}
        self._active_job_id: str | None = None
        self._active_process: ManagedProcess | None = None

    def validate_record_request(
        self,
        request: RecordRequest | Mapping[str, object],
    ) -> RecordValidationResult:
        predicted_dataset_dir = self.runtime_paths.predicted_dataset_dir(
            request["dataset_name"] if isinstance(request, Mapping) else request.dataset_name
        )
        normalized, dataset_exists, warnings = validate_and_normalize_record_request(
            request=request,
            predicted_dataset_dir=predicted_dataset_dir,
        )
        return RecordValidationResult(
            request=normalized,
            command_preview=build_record_command_preview(normalized, self.runtime_paths),
            predicted_dataset_dir=str(predicted_dataset_dir),
            dataset_exists=dataset_exists,
            warnings=warnings,
            outputs_root=str(self.runtime_paths.outputs_root),
            record_script_path=str(self.runtime_paths.script_path),
            conda_env=self.runtime_paths.conda_env,
        )

    def create_record_job(self, request: RecordRequest | Mapping[str, object]) -> JobState:
        validation = self.validate_record_request(request)
        now = time.time()

        with self._lock:
            self._ensure_no_active_job_locked()

            job_id = self._new_job_id()
            job_paths = self.store.create_job(job_id)
            launch_spec = build_process_launch_spec(
                request=validation.request,
                job_dir=job_paths.job_dir,
                runtime_paths=self.runtime_paths,
            )

            state = JobState(
                job_id=job_id,
                status=JobStatus.STARTING,
                phase=JobPhase.BOOTING,
                request=validation.request,
                progress=JobProgress(
                    total_episodes=validation.request.num_episodes,
                    expected_total_time_s=validation.request.expected_total_time_s,
                ),
                artifacts=JobArtifacts(
                    job_dir=str(job_paths.job_dir),
                    request_path=str(job_paths.request_path),
                    state_path=str(job_paths.state_path),
                    stdout_log_path=str(job_paths.stdout_path),
                    events_path=str(job_paths.events_path),
                    config_snapshot_path=str(job_paths.record_config_path),
                    predicted_dataset_dir=str(launch_spec.predicted_dataset_dir),
                ),
                warnings=validation.warnings,
                created_at=now,
                updated_at=now,
            )
            self._states[job_id] = state
            self.store.write_request(
                job_id,
                {
                    "request": validation.request.model_dump(mode="json"),
                    "warnings": validation.warnings,
                    "command_preview": validation.command_preview,
                    "predicted_dataset_dir": validation.predicted_dataset_dir,
                },
            )
            self._persist_state_locked(
                state,
                event_type="job.created",
                payload={"status": state.status.value, "phase": state.phase.value},
            )

            process = ManagedProcess(
                argv=launch_spec.argv,
                cwd=launch_spec.cwd,
                env=launch_spec.env,
                on_line=lambda line: self._handle_stdout_line(job_id, line),
                on_exit=lambda returncode: self._handle_process_exit(job_id, returncode),
            )

            try:
                process.start()
            except Exception as exc:
                failed_state = state.model_copy(
                    update={
                        "status": JobStatus.FAILED,
                        "phase": JobPhase.FINALIZING,
                        "updated_at": time.time(),
                        "finished_at": time.time(),
                        "error": f"Failed to start process: {exc}",
                    }
                )
                self._states[job_id] = failed_state
                self._persist_state_locked(
                    failed_state,
                    event_type="job.start_failed",
                    message=str(exc),
                    payload={"error": str(exc)},
                )
                raise RecordJobManagerError(str(exc)) from exc

            running_state = state.model_copy(
                update={
                    "status": JobStatus.RUNNING,
                    "phase": JobPhase.BOOTING,
                    "updated_at": time.time(),
                    "started_at": time.time(),
                    "pid": process.pid,
                    "pgid": process.pgid,
                }
            )
            self._states[job_id] = running_state
            self._active_job_id = job_id
            self._active_process = process
            self.store.write_active_lock(job_id)
            self._persist_state_locked(
                running_state,
                event_type="process.started",
                payload={
                    "pid": process.pid,
                    "pgid": process.pgid,
                    "command_preview": launch_spec.command_preview,
                    "config_path": str(launch_spec.config_path),
                },
            )
            return running_state

    def get_active_job(self) -> JobState | None:
        with self._lock:
            self._refresh_active_process_locked()
            if self._active_job_id is None:
                return None
            return self._states.get(self._active_job_id) or self.store.read_state(self._active_job_id)

    def get_job(self, job_id: str) -> JobState:
        with self._lock:
            if job_id in self._states:
                return self._states[job_id]
        try:
            state = self.store.read_state(job_id)
        except FileNotFoundError as exc:
            raise JobNotFoundError(f"Unknown job: {job_id}") from exc
        with self._lock:
            self._states[job_id] = state
        return state

    def list_jobs(self, limit: int = 20) -> list[JobState]:
        return self.store.list_recent_states(limit=limit)

    def read_job_logs(self, job_id: str, cursor: int = 0, max_chars: int = 65536) -> LogChunk:
        self.get_job(job_id)
        return self.store.read_logs(job_id, cursor=cursor, max_chars=max_chars)

    def read_recent_job_logs(self, job_id: str, max_lines: int = 200) -> list[str]:
        self.get_job(job_id)
        return self.store.read_recent_logs(job_id, max_lines=max_lines)

    def stop_job(self, job_id: str) -> JobState:
        with self._lock:
            state = self.get_job(job_id)
            if state.status in TERMINAL_JOB_STATUSES:
                return state

            if job_id != self._active_job_id or self._active_process is None:
                raise ActiveJobConflictError(f"Job is not active: {job_id}")

            process = self._active_process
            stop_requested_state = state.model_copy(
                update={
                    "status": JobStatus.STOP_REQUESTED,
                    "updated_at": time.time(),
                    "stop_requested_at": time.time(),
                }
            )
            self._states[job_id] = stop_requested_state
            self._persist_state_locked(
                stop_requested_state,
                event_type="job.stop_requested",
                payload={"pid": process.pid, "pgid": process.pgid},
            )

        process.stop()
        return self.get_job(job_id)

    def _new_job_id(self) -> str:
        return uuid.uuid4().hex

    def _ensure_no_active_job_locked(self) -> None:
        self._refresh_active_process_locked()
        if self._active_job_id is not None:
            raise ActiveJobConflictError(f"Another recording job is active: {self._active_job_id}")

    def _refresh_active_process_locked(self) -> None:
        if self._active_process is None or self._active_job_id is None:
            return
        if self._active_process.is_running():
            return
        state = self._states.get(self._active_job_id)
        if state is not None and state.status in TERMINAL_JOB_STATUSES:
            self._active_process = None
            self._active_job_id = None
            self.store.clear_active_lock()

    def _handle_stdout_line(self, job_id: str, line: str) -> None:
        with self._lock:
            try:
                state = self._states[job_id]
            except KeyError:
                return

            self.store.append_stdout(job_id, line)
            updated_state, event_type, payload = self._state_from_log_line(state, line)
            self._states[job_id] = updated_state
            self._persist_state_locked(updated_state, event_type=event_type, message=line, payload=payload)

    def _handle_process_exit(self, job_id: str, returncode: int | None) -> None:
        with self._lock:
            state = self._states.get(job_id)
            if state is None:
                return
            if state.status in TERMINAL_JOB_STATUSES:
                return

            if state.status == JobStatus.STOP_REQUESTED:
                final_status = JobStatus.STOPPED
                error = None
            elif returncode == 0:
                final_status = JobStatus.SUCCEEDED
                error = None
            else:
                final_status = JobStatus.FAILED
                error = f"Process exited with return code {returncode}."

            finished_state = state.model_copy(
                update={
                    "status": final_status,
                    "phase": JobPhase.FINALIZING,
                    "updated_at": time.time(),
                    "finished_at": time.time(),
                    "returncode": returncode,
                    "error": error,
                }
            )
            self._states[job_id] = finished_state
            self._persist_state_locked(
                finished_state,
                event_type="process.finished",
                payload={"returncode": returncode, "status": final_status.value},
            )
            if job_id == self._active_job_id:
                self._active_process = None
                self._active_job_id = None
                self.store.clear_active_lock()

    def _persist_state_locked(
        self,
        state: JobState,
        *,
        event_type: str,
        payload: dict[str, object] | None = None,
        message: str | None = None,
    ) -> None:
        self.store.write_state(state)
        self.store.append_event(
            JobEvent(
                ts=time.time(),
                job_id=state.job_id,
                type=event_type,
                message=message,
                payload=payload or {},
            )
        )

    def _state_from_log_line(
        self,
        state: JobState,
        line: str,
    ) -> tuple[JobState, str, dict[str, object]]:
        payload: dict[str, object] = {}
        updated = state.model_copy(update={"updated_at": time.time()})

        dataset_match = DATASET_OUTPUT_RE.match(line)
        if dataset_match:
            dataset_dir = dataset_match.group("path")
            updated = updated.model_copy(
                update={
                    "artifacts": updated.artifacts.model_copy(update={"actual_dataset_dir": dataset_dir}),
                }
            )
            return updated, "log.dataset_output", {"actual_dataset_dir": dataset_dir}

        config_match = CONFIG_FILE_RE.match(line)
        if config_match:
            config_path = config_match.group("path")
            updated = updated.model_copy(
                update={
                    "artifacts": updated.artifacts.model_copy(update={"actual_config_path": config_path}),
                }
            )
            return updated, "log.config_file", {"actual_config_path": config_path}

        episode_match = START_EPISODE_RE.match(line)
        if episode_match:
            current_episode = int(episode_match.group("current"))
            total_episodes = int(episode_match.group("total"))
            updated = updated.model_copy(
                update={
                    "phase": JobPhase.RECORDING,
                    "progress": updated.progress.model_copy(
                        update={
                            "current_episode": current_episode,
                            "total_episodes": total_episodes,
                        }
                    ),
                }
            )
            payload = {"current_episode": current_episode, "total_episodes": total_episodes}
            return updated, "log.recording_started", payload

        reset_match = START_RESET_RE.match(line)
        if reset_match:
            current_episode = int(reset_match.group("current"))
            total_episodes = int(reset_match.group("total"))
            updated = updated.model_copy(
                update={
                    "phase": JobPhase.RESETTING,
                    "progress": updated.progress.model_copy(
                        update={
                            "current_episode": current_episode,
                            "total_episodes": total_episodes,
                        }
                    ),
                }
            )
            payload = {"current_episode": current_episode, "total_episodes": total_episodes}
            return updated, "log.reset_started", payload

        saving_match = SAVING_EPISODE_RE.match(line)
        if saving_match:
            current_episode = int(saving_match.group("current"))
            total_episodes = int(saving_match.group("total"))
            updated = updated.model_copy(
                update={
                    "phase": JobPhase.SAVING,
                    "progress": updated.progress.model_copy(
                        update={
                            "current_episode": current_episode,
                            "total_episodes": total_episodes,
                        }
                    ),
                }
            )
            payload = {"current_episode": current_episode, "total_episodes": total_episodes}
            return updated, "log.saving_started", payload

        saved_match = SAVED_EPISODE_RE.match(line)
        if saved_match:
            saved_episodes = int(saved_match.group("current"))
            total_episodes = int(saved_match.group("total"))
            updated = updated.model_copy(
                update={
                    "phase": JobPhase.SAVING,
                    "progress": updated.progress.model_copy(
                        update={
                            "saved_episodes": saved_episodes,
                            "total_episodes": total_episodes,
                        }
                    ),
                }
            )
            payload = {"saved_episodes": saved_episodes, "total_episodes": total_episodes}
            return updated, "log.episode_saved", payload

        return updated, "log.line", {}
