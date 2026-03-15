from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex

from my_devs.agilex_web_collection.app.api.schemas import NormalizedRecordRequest


DEFAULT_CONDA_ENV = "lerobot_flex"
DEFAULT_RECORD_SCRIPT_RELATIVE = Path("my_devs/add_robot/agilex/record.sh")
DEFAULT_OUTPUTS_ROOT_RELATIVE = Path("my_devs/add_robot/agilex/outputs")
DEFAULT_RUNTIME_JOBS_RELATIVE = Path("my_devs/agilex_web_collection/runtime/jobs")


def _format_shell_float(value: float) -> str:
    text = f"{value:.12g}"
    if "." in text or "e" in text.lower():
        return text
    return f"{text}.0"


@dataclass(frozen=True)
class AgileXRuntimePaths:
    repo_root: Path
    script_path: Path
    outputs_root: Path
    jobs_root: Path
    conda_env: str = DEFAULT_CONDA_ENV

    @classmethod
    def defaults(cls) -> "AgileXRuntimePaths":
        repo_root = Path(__file__).resolve().parents[4]
        return cls(
            repo_root=repo_root,
            script_path=repo_root / DEFAULT_RECORD_SCRIPT_RELATIVE,
            outputs_root=repo_root / DEFAULT_OUTPUTS_ROOT_RELATIVE,
            jobs_root=repo_root / DEFAULT_RUNTIME_JOBS_RELATIVE,
        )

    def predicted_dataset_dir(self, dataset_name: str) -> Path:
        return self.outputs_root / "dummy" / dataset_name


@dataclass(frozen=True)
class ProcessLaunchSpec:
    argv: list[str]
    env: dict[str, str]
    cwd: Path
    command_preview: str
    predicted_dataset_dir: Path
    config_path: Path


def build_record_command_preview(
    request: NormalizedRecordRequest,
    runtime_paths: AgileXRuntimePaths | None = None,
) -> str:
    runtime_paths = runtime_paths or AgileXRuntimePaths.defaults()
    argv = [
        "bash",
        str(runtime_paths.script_path),
        request.dataset_name,
        _format_shell_float(request.episode_time_s),
        str(request.num_episodes),
        str(request.fps),
        _format_shell_float(request.reset_time_s),
        "true" if request.resume else "false",
    ]
    return shlex.join(argv)


def build_process_launch_spec(
    request: NormalizedRecordRequest,
    job_dir: Path,
    runtime_paths: AgileXRuntimePaths | None = None,
) -> ProcessLaunchSpec:
    runtime_paths = runtime_paths or AgileXRuntimePaths.defaults()
    config_path = job_dir / "record_config.json"
    argv = [
        "bash",
        str(runtime_paths.script_path),
        request.dataset_name,
        _format_shell_float(request.episode_time_s),
        str(request.num_episodes),
        str(request.fps),
        _format_shell_float(request.reset_time_s),
        "true" if request.resume else "false",
    ]
    env = {
        "CONDA_ENV": runtime_paths.conda_env,
        "HF_LEROBOT_HOME": str(runtime_paths.outputs_root),
        "CONFIG_PATH": str(config_path),
        "PYTHONUNBUFFERED": "1",
    }
    return ProcessLaunchSpec(
        argv=argv,
        env=env,
        cwd=runtime_paths.repo_root,
        command_preview=shlex.join(argv),
        predicted_dataset_dir=runtime_paths.predicted_dataset_dir(request.dataset_name),
        config_path=config_path,
    )
