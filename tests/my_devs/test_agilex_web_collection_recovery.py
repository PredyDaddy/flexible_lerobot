from __future__ import annotations

from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys
import time
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
def _backend_client_for_paths(paths: AgilexWebPaths):
    store = FileJobStore(paths)
    supervisor = ProcessSupervisor()
    service = RecordJobService(paths=paths, store=store, supervisor=supervisor)
    module = _load_backend_app_module()
    app = module.create_app(paths=paths, service=service)

    with TestClient(app) as client:
        try:
            yield client, service
        finally:
            active_job = service.get_active_job()
            if active_job is not None:
                service.stop_job(active_job.state.job_id)


def _wait_for_status(
    client: TestClient, job_id: str, expected_statuses: set[str], timeout_s: float = 5.0
) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        response = client.get(f"/api/jobs/{job_id}")
        response.raise_for_status()
        payload = response.json()
        if payload["state"]["status"] in expected_statuses:
            return payload
        time.sleep(0.05)
    raise AssertionError(f"Job {job_id} did not reach one of {sorted(expected_statuses)}.")


def _long_running_script() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail
trap 'echo "[record] Stop requested"; exit 130' INT TERM
mkdir -p "$(dirname "${CONFIG_PATH}")"
DATASET_DIR="${HF_LEROBOT_HOME}/dummy/${1}"
mkdir -p "${DATASET_DIR}"
printf '{"dataset_name":"%s","single_task_text":"%s"}\n' "${1}" "${7}" > "${CONFIG_PATH}"
echo "Dataset output=${DATASET_DIR}"
echo "Task=${7}"
echo "Config file=${CONFIG_PATH}"
echo "[record] Start episode 1/${3} for ${2}s"
i=0
while true; do
  echo "recovery-heartbeat-${i}"
  i=$((i + 1))
  sleep 0.2
done
"""


def test_active_job_is_recovered_from_job_store_after_service_restart(tmp_path: Path) -> None:
    root = tmp_path / "recovery_workspace"
    paths = _build_paths(root, _long_running_script())

    with _backend_client_for_paths(paths) as (client_one, _service_one):
        create = client_one.post(
            "/api/jobs",
            json={
                "dataset_name": "recover_me",
                "episode_time_s": 8,
                "num_episodes": 1,
                "fps": 10,
                "reset_time_s": 0,
                "resume": False,
                "single_task_text": "Pick up the black cups and place them in the orange box.",
            },
        )
        assert create.status_code == 201
        job_id = create.json()["state"]["job_id"]

        with _backend_client_for_paths(paths) as (client_two, service_two):
            active = client_two.get("/api/jobs/active")
            active.raise_for_status()
            active_payload = active.json()

            assert active_payload is not None
            assert active_payload["state"]["job_id"] == job_id
            assert active_payload["state"]["active"] is True
            assert service_two.get_active_job() is not None

            listed = client_two.get("/api/jobs")
            listed.raise_for_status()
            assert listed.json()[0]["job_id"] == job_id

            stop = client_two.post(f"/api/jobs/{job_id}/stop")
            stop.raise_for_status()
            assert stop.json()["requested"] is True

            stopped = _wait_for_status(client_two, job_id, {"stopped"})
            assert stopped["state"]["active"] is False
            assert stopped["state"]["returncode"] == 130

            active_after_stop = client_two.get("/api/jobs/active")
            active_after_stop.raise_for_status()
            assert active_after_stop.json() is None
