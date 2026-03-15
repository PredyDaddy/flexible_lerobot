from __future__ import annotations

from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys
import uuid

from fastapi.testclient import TestClient

from my_devs.agilex_web_collection.paths import AgilexWebPaths
from my_devs.agilex_web_collection.service import RecordJobService
from my_devs.agilex_web_collection.store import FileJobStore
from my_devs.agilex_web_collection.supervisor import ProcessSupervisor


def _load_backend_app_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "my_devs" / "agilex_web_collection" / "app.py"
    module_name = f"my_devs.agilex_web_collection._backend_app_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _write_script(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _build_paths(root: Path, script_content: str) -> AgilexWebPaths:
    package_root = root / "package"
    static_dir = package_root / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    (static_dir / "index.html").write_text("<html><body>ok</body></html>\n", encoding="utf-8")

    agilex_root = root / "my_devs" / "add_robot" / "agilex"
    script_path = agilex_root / "record.sh"
    _write_script(script_path, script_content)

    outputs_root = agilex_root / "outputs"
    outputs_root.mkdir(parents=True, exist_ok=True)

    runtime_dir = package_root / "runtime"
    jobs_root = runtime_dir / "agilex_jobs"
    return AgilexWebPaths(
        repo_root=root,
        package_root=package_root,
        static_dir=static_dir,
        runtime_dir=runtime_dir,
        jobs_root=jobs_root,
        agilex_root=agilex_root,
        script_path=script_path,
        outputs_root=outputs_root,
        dataset_root=outputs_root / "dummy",
    )


@contextmanager
def _backend_client(tmp_path: Path, script_content: str):
    root = tmp_path / uuid.uuid4().hex
    paths = _build_paths(root, script_content)
    store = FileJobStore(paths)
    supervisor = ProcessSupervisor()
    service = RecordJobService(paths=paths, store=store, supervisor=supervisor)
    module = _load_backend_app_module()
    app = module.create_app(paths=paths, service=service)

    with TestClient(app) as client:
        yield client, paths


def _noop_script() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail
mkdir -p "$(dirname "${CONFIG_PATH}")"
mkdir -p "${HF_LEROBOT_HOME}/dummy/${1}"
printf '{"dataset_name":"%s"}\n' "${1}" > "${CONFIG_PATH}"
echo "Dataset output=${HF_LEROBOT_HOME}/dummy/${1}"
echo "Task=${7}"
echo "Config file=${CONFIG_PATH}"
echo "[record] Start episode 1/${3} for ${2}s"
echo "[record] Saved episode 1/${3}"
"""


def test_preflight_validates_shape_and_normalizes_single_episode_reset(tmp_path: Path) -> None:
    with _backend_client(tmp_path, _noop_script()) as (client, paths):
        invalid_name = client.post(
            "/api/preflight",
            json={
                "dataset_name": "bad name",
                "episode_time_s": 8,
                "num_episodes": 1,
                "fps": 10,
                "reset_time_s": 0,
                "resume": False,
                "single_task_text": "Pick up the black cups and place them in the orange box.",
            },
        )
        assert invalid_name.status_code == 422

        extra_field = client.post(
            "/api/preflight",
            json={
                "dataset_name": "demo_api",
                "episode_time_s": 8,
                "num_episodes": 1,
                "fps": 10,
                "reset_time_s": 0,
                "resume": False,
                "single_task_text": "Pick up the black cups and place them in the orange box.",
                "unexpected": "value",
            },
        )
        assert extra_field.status_code == 422

        response = client.post(
            "/api/preflight",
            json={
                "dataset_name": "demo_api",
                "episode_time_s": 8,
                "num_episodes": 1,
                "fps": 10,
                "reset_time_s": 5,
                "resume": False,
                "single_task_text": "Pick up the black cups and place them in the orange box.",
            },
        )
        response.raise_for_status()
        payload = response.json()

        assert payload["ok"] is True
        assert payload["normalized_request"]["reset_time_s"] == 0.0
        assert payload["warnings"] == ["reset_time_s was normalized to 0 because num_episodes=1."]
        assert payload["dataset_dir"] == str(paths.dataset_root / "demo_api")
        assert payload["command"][0:2] == ["bash", str(paths.script_path)]
        assert payload["command"][2:] == [
            "demo_api",
            "8",
            "1",
            "10",
            "0",
            "false",
            "Pick up the black cups and place them in the orange box.",
        ]
        assert "Pick up the black cups and place them in the orange box." in payload["command_text"]


def test_preflight_reports_directory_conflicts_and_resume_semantics(tmp_path: Path) -> None:
    with _backend_client(tmp_path, _noop_script()) as (client, paths):
        existing_dir = paths.dataset_root / "existing_set"
        existing_dir.mkdir(parents=True, exist_ok=True)

        existing_without_resume = client.post(
            "/api/preflight",
            json={
                "dataset_name": "existing_set",
                "episode_time_s": 8,
                "num_episodes": 1,
                "fps": 10,
                "reset_time_s": 0,
                "resume": False,
                "single_task_text": "Pick up the black cups and place them in the orange box.",
            },
        )
        existing_without_resume.raise_for_status()
        payload = existing_without_resume.json()
        assert payload["ok"] is False
        assert payload["dataset_exists"] is True
        assert any("Dataset directory already exists" in item for item in payload["conflicts"])

        existing_with_resume = client.post(
            "/api/preflight",
            json={
                "dataset_name": "existing_set",
                "episode_time_s": 8,
                "num_episodes": 2,
                "fps": 10,
                "reset_time_s": 2,
                "resume": True,
                "single_task_text": "Pick up the black cups and place them in the orange box.",
            },
        )
        existing_with_resume.raise_for_status()
        payload = existing_with_resume.json()
        assert payload["ok"] is True
        assert payload["dataset_exists"] is True
        assert payload["warnings"] == [
            "The target dataset directory already exists. This run will append to it."
        ]
        assert payload["command"][2:] == [
            "existing_set",
            "8",
            "2",
            "10",
            "2",
            "true",
            "Pick up the black cups and place them in the orange box.",
        ]

        missing_resume = client.post(
            "/api/preflight",
            json={
                "dataset_name": "missing_resume",
                "episode_time_s": 8,
                "num_episodes": 1,
                "fps": 10,
                "reset_time_s": 0,
                "resume": True,
                "single_task_text": "Pick up the black cups and place them in the orange box.",
            },
        )
        missing_resume.raise_for_status()
        payload = missing_resume.json()
        assert payload["ok"] is False
        assert payload["dataset_exists"] is False
        assert any("resume=true requires an existing dataset directory" in item for item in payload["conflicts"])


def test_create_job_rejects_directory_conflict_with_http_409(tmp_path: Path) -> None:
    with _backend_client(tmp_path, _noop_script()) as (client, paths):
        dataset_dir = paths.dataset_root / "demo_conflict"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        response = client.post(
            "/api/jobs",
            json={
                "dataset_name": "demo_conflict",
                "episode_time_s": 8,
                "num_episodes": 1,
                "fps": 10,
                "reset_time_s": 0,
                "resume": False,
                "single_task_text": "Pick up the black cups and place them in the orange box.",
            },
        )

        assert response.status_code == 409
        payload = response.json()
        assert payload["error"]["code"] == "job_conflict"
        assert payload["error"]["message"].startswith("Dataset directory already exists")
        assert any(
            "Dataset directory already exists" in item
            for item in payload["error"]["details"]["conflicts"]
        )
        assert payload["error"]["details"]["preflight"]["dataset_exists"] is True

        jobs = client.get("/api/jobs")
        jobs.raise_for_status()
        assert jobs.json() == []

        runtime = client.get("/api/runtime")
        runtime.raise_for_status()
        runtime_payload = runtime.json()
        assert runtime_payload["single_active_job_only"] is True
        assert runtime_payload["conda_env"] == "lerobot_flex"
