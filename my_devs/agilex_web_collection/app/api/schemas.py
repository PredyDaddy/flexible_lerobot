from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class JobKind(str, Enum):
    RECORD = "record"


class JobStatus(str, Enum):
    CREATED = "created"
    VALIDATING = "validating"
    REJECTED = "rejected"
    STARTING = "starting"
    RUNNING = "running"
    STOP_REQUESTED = "stop_requested"
    STOPPED = "stopped"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class JobPhase(str, Enum):
    PREFLIGHT = "preflight"
    BOOTING = "booting"
    RECORDING = "recording"
    RESETTING = "resetting"
    SAVING = "saving"
    FINALIZING = "finalizing"


class RecordRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_name: str
    episode_time_s: float = Field(gt=0, le=3600)
    num_episodes: int = Field(ge=1, le=1000)
    fps: int = Field(ge=1, le=120)
    reset_time_s: float = Field(ge=0, le=600)
    resume: bool


class NormalizedRecordRequest(RecordRequest):
    model_config = ConfigDict(extra="forbid", frozen=True)

    repo_id: str
    expected_total_time_s: float = Field(ge=0)


class RecordValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request: NormalizedRecordRequest
    command_preview: str
    predicted_dataset_dir: str
    dataset_exists: bool
    warnings: list[str] = Field(default_factory=list)
    outputs_root: str
    record_script_path: str
    conda_env: str


class JobProgress(BaseModel):
    model_config = ConfigDict(extra="forbid")

    current_episode: int = 0
    total_episodes: int = 0
    saved_episodes: int = 0
    expected_total_time_s: float = 0.0


class JobArtifacts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_dir: str
    request_path: str
    state_path: str
    stdout_log_path: str
    events_path: str
    config_snapshot_path: str
    predicted_dataset_dir: str
    actual_dataset_dir: str | None = None
    actual_config_path: str | None = None


class JobState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    kind: JobKind = JobKind.RECORD
    status: JobStatus
    phase: JobPhase
    request: NormalizedRecordRequest
    progress: JobProgress
    artifacts: JobArtifacts
    warnings: list[str] = Field(default_factory=list)
    created_at: float
    updated_at: float
    started_at: float | None = None
    finished_at: float | None = None
    stop_requested_at: float | None = None
    pid: int | None = None
    pgid: int | None = None
    returncode: int | None = None
    error: str | None = None


class JobEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ts: float
    job_id: str
    type: str
    message: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class LogChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cursor: int
    next_cursor: int
    lines: list[str] = Field(default_factory=list)
    eof: bool
