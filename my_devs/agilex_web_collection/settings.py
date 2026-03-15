from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .paths import AgilexWebPaths


DEFAULT_APP_NAME = "agilex_web_collection"
DEFAULT_APP_VERSION = "0.1.0"
DEFAULT_CONDA_ENV = "lerobot_flex"

ENV_APP_NAME = "AGILEX_WEB_APP_NAME"
ENV_APP_VERSION = "AGILEX_WEB_APP_VERSION"
ENV_CONDA_ENV = "AGILEX_WEB_CONDA_ENV"
ENV_REPO_ROOT = "AGILEX_WEB_REPO_ROOT"
ENV_AGILEX_ROOT = "AGILEX_WEB_AGILEX_ROOT"
ENV_STATIC_DIR = "AGILEX_WEB_STATIC_DIR"
ENV_RUNTIME_ROOT = "AGILEX_WEB_RUNTIME_ROOT"
ENV_RUNTIME_DIR = "AGILEX_WEB_RUNTIME_DIR"
ENV_JOBS_ROOT = "AGILEX_WEB_JOBS_ROOT"
ENV_SCRIPT_PATH = "AGILEX_WEB_SCRIPT_PATH"
ENV_OUTPUT_ROOT = "AGILEX_WEB_OUTPUT_ROOT"
ENV_DATASET_ROOT = "AGILEX_WEB_DATASET_ROOT"


@dataclass(frozen=True)
class AgilexWebSettings:
    app_name: str
    app_version: str
    conda_env: str
    repo_root: Path
    package_root: Path
    agilex_root: Path
    static_dir: Path
    runtime_dir: Path
    jobs_root: Path
    script_path: Path
    outputs_root: Path
    dataset_root: Path

    @classmethod
    def discover(cls) -> "AgilexWebSettings":
        package_root = Path(__file__).resolve().parent
        default_repo_root = package_root.parents[1]
        repo_root = _resolve_path(os.environ.get(ENV_REPO_ROOT), default_repo_root)
        agilex_root = _resolve_path(
            os.environ.get(ENV_AGILEX_ROOT),
            repo_root / "my_devs" / "add_robot" / "agilex",
            base_dir=repo_root,
        )
        static_dir = _resolve_path(
            os.environ.get(ENV_STATIC_DIR),
            package_root / "static",
            base_dir=repo_root,
        )
        runtime_dir = _resolve_path(
            os.environ.get(ENV_RUNTIME_ROOT) or os.environ.get(ENV_RUNTIME_DIR),
            package_root / "runtime",
            base_dir=repo_root,
        )
        jobs_root = _resolve_path(
            os.environ.get(ENV_JOBS_ROOT),
            runtime_dir / "agilex_jobs",
            base_dir=runtime_dir,
        )
        outputs_root = _resolve_path(
            os.environ.get(ENV_OUTPUT_ROOT),
            agilex_root / "outputs",
            base_dir=repo_root,
        )
        dataset_root = _resolve_path(
            os.environ.get(ENV_DATASET_ROOT),
            outputs_root / "dummy",
            base_dir=outputs_root,
        )
        script_path = _resolve_path(
            os.environ.get(ENV_SCRIPT_PATH),
            agilex_root / "record.sh",
            base_dir=repo_root,
        )
        return cls(
            app_name=os.environ.get(ENV_APP_NAME, DEFAULT_APP_NAME),
            app_version=os.environ.get(ENV_APP_VERSION, DEFAULT_APP_VERSION),
            conda_env=os.environ.get(ENV_CONDA_ENV, DEFAULT_CONDA_ENV),
            repo_root=repo_root,
            package_root=package_root,
            agilex_root=agilex_root,
            static_dir=static_dir,
            runtime_dir=runtime_dir,
            jobs_root=jobs_root,
            script_path=script_path,
            outputs_root=outputs_root,
            dataset_root=dataset_root,
        )

    @property
    def current_conda_env(self) -> str | None:
        return os.environ.get("CONDA_DEFAULT_ENV")

    @property
    def static_index_path(self) -> Path:
        return self.static_dir / "index.html"

    @property
    def lock_dir(self) -> Path:
        return self.jobs_root / "locks"

    @property
    def active_lock_path(self) -> Path:
        return self.lock_dir / "record.lock"

    @property
    def config_path_template(self) -> Path:
        return self.jobs_root / "{job_id}" / "record_config.json"

    def job_dir(self, job_id: str) -> Path:
        return self.jobs_root / job_id

    def config_path_for_job(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "record_config.json"

    def preview_env(self) -> dict[str, str]:
        return {
            "CONDA_ENV": self.conda_env,
            "HF_LEROBOT_HOME": str(self.outputs_root),
            "CONFIG_PATH": str(self.config_path_template),
            "PYTHONUNBUFFERED": "1",
        }

    def env_for_job(self, job_id: str) -> dict[str, str]:
        return {
            "CONDA_ENV": self.conda_env,
            "HF_LEROBOT_HOME": str(self.outputs_root),
            "CONFIG_PATH": str(self.config_path_for_job(job_id)),
            "PYTHONUNBUFFERED": "1",
        }

    def ensure_runtime_layout(self) -> None:
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self.lock_dir.mkdir(parents=True, exist_ok=True)

    def to_paths(self) -> AgilexWebPaths:
        return AgilexWebPaths(
            repo_root=self.repo_root,
            package_root=self.package_root,
            static_dir=self.static_dir,
            runtime_dir=self.runtime_dir,
            jobs_root=self.jobs_root,
            agilex_root=self.agilex_root,
            script_path=self.script_path,
            outputs_root=self.outputs_root,
            dataset_root=self.dataset_root,
        )


def _resolve_path(raw_value: str | None, default: Path, *, base_dir: Path | None = None) -> Path:
    candidate = default if raw_value in (None, "") else Path(raw_value)
    if not candidate.is_absolute():
        anchor = base_dir or Path.cwd()
        candidate = anchor / candidate
    return candidate.expanduser().resolve()
