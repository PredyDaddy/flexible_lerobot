from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from typing_extensions import Annotated


DATASET_NAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$"
REPO_PREFIX_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$"

DEFAULT_REQUEST_VALUES = {
    "repo_prefix": "dummy",
    "dataset_name": "",
    "episode_time_s": 8,
    "num_episodes": 1,
    "fps": 10,
    "reset_time_s": 0,
    "resume": False,
    "single_task_text": "agilex static record test",
}
FIRST_TRIAL_TEMPLATE = {
    "repo_prefix": "dummy",
    "dataset_name": "",
    "episode_time_s": 8,
    "num_episodes": 1,
    "fps": 30,
    "reset_time_s": 0,
    "resume": False,
    "single_task_text": "agilex static record test",
}
RESUME_TEMPLATE = {
    "repo_prefix": "dummy",
    "dataset_name": "",
    "episode_time_s": 8,
    "num_episodes": 1,
    "fps": 30,
    "reset_time_s": 0,
    "resume": True,
    "single_task_text": "agilex static record test",
}

JobStatus = Literal[
    "created",
    "validating",
    "rejected",
    "starting",
    "running",
    "stop_requested",
    "stopped",
    "succeeded",
    "failed",
]
JobPhase = Literal["preflight", "booting", "recording", "resetting", "saving", "finalizing"]

ACTIVE_JOB_STATUSES = {"starting", "running", "stop_requested"}
TERMINAL_JOB_STATUSES = {"rejected", "stopped", "succeeded", "failed"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def format_number(value: int | float) -> str:
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:.15g}"


class RecordRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    repo_prefix: Annotated[
        str,
        StringConstraints(pattern=REPO_PREFIX_PATTERN, min_length=1, max_length=64),
    ] = "dummy"
    dataset_name: Annotated[str, StringConstraints(pattern=DATASET_NAME_PATTERN, min_length=1, max_length=64)]
    episode_time_s: float = Field(gt=0, le=3600)
    num_episodes: int = Field(ge=1, le=1000)
    fps: int = Field(ge=1, le=120)
    reset_time_s: float = Field(ge=0, le=600)
    resume: bool
    single_task_text: Annotated[str, StringConstraints(min_length=1, max_length=512)]


class NormalizedRecordRequest(RecordRequest):
    @property
    def repo_id(self) -> str:
        return f"{self.repo_prefix}/{self.dataset_name}"


class JobArtifacts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_path: str
    state_path: str
    log_path: str
    events_path: str
    config_path: str


class JobRequestSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    kind: Literal["record"] = "record"
    created_at: str
    request: dict[str, Any]
    normalized_request: dict[str, Any]
    dataset_name: str
    dataset_dir: str
    output_root: str
    job_dir: str
    command: list[str]
    command_text: str
    env_overrides: dict[str, str]
    estimated_duration_s: float
    warnings: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    artifacts: JobArtifacts


class JobState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    kind: Literal["record"] = "record"
    status: JobStatus
    phase: JobPhase
    active: bool
    can_stop: bool
    created_at: str
    updated_at: str
    started_at: str | None = None
    finished_at: str | None = None
    stop_requested_at: str | None = None
    pid: int | None = None
    pgid: int | None = None
    returncode: int | None = None
    current_episode: int | None = None
    total_episodes: int | None = None
    status_message: str = ""
    warnings: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)


class JobDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request: JobRequestSnapshot
    state: JobState


class JobSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    dataset_name: str
    dataset_dir: str
    status: JobStatus
    phase: JobPhase
    active: bool
    can_stop: bool
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    returncode: int | None = None
    estimated_duration_s: float
    status_message: str = ""


class JobEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    at: str
    job_id: str
    type: str
    message: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class RuntimeInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    app_name: str
    app_version: str
    conda_env: str
    current_conda_env: str | None = None
    script_path: str
    output_root: str
    dataset_root: str
    jobs_root: str
    static_index: str
    static_index_available: bool
    config_path_template: str
    per_job_config_supported: bool
    single_active_job_only: bool
    active_job_id: str | None = None
    defaults: dict[str, Any]
    limitations: list[str] = Field(default_factory=list)


class PreflightResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    request: dict[str, Any]
    normalized_request: dict[str, Any]
    dataset_exists: bool
    dataset_dir: str
    output_root: str
    runtime_jobs_root: str
    config_path_template: str
    command: list[str]
    command_text: str
    env_overrides: dict[str, str]
    estimated_duration_s: float
    active_job_id: str | None = None
    conflicts: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class LogReadResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    cursor: int
    next_cursor: int
    lines: list[str]
    truncated: bool
    status: JobStatus
    active: bool


class StopJobResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requested: bool
    signal_stage: str
    message: str
    job: JobDetail
