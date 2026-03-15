from __future__ import annotations

from contextlib import contextmanager
import importlib.util
import json
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
def _backend_client(tmp_path: Path, script_content: str):
    root = tmp_path / uuid.uuid4().hex
    paths = _build_paths(root, script_content)
    store = FileJobStore(paths)
    supervisor = ProcessSupervisor()
    service = RecordJobService(paths=paths, store=store, supervisor=supervisor)
    module = _load_backend_app_module()
    app = module.create_app(paths=paths, service=service)

    with TestClient(app) as client:
        try:
            yield client, paths
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


def _script_for_finished_job() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail
mkdir -p "$(dirname "${CONFIG_PATH}")"
DATASET_DIR="${HF_LEROBOT_HOME}/dummy/${1}"
mkdir -p "${DATASET_DIR}"
printf '{"dataset_name":"%s","single_task_text":"%s"}\n' "${1}" "${7}" > "${CONFIG_PATH}"
echo "Dataset output=${DATASET_DIR}"
echo "Task=${7}"
echo "Config file=${CONFIG_PATH}"
for i in $(seq 1 30); do
  echo "trace-line-${i}=abcdefghijklmnopqrstuvwxyz0123456789"
done
echo "[record] Start episode 1/${3} for ${2}s"
echo "[record] Saving episode 1/${3}"
echo "[record] Saved episode 1/${3}"
"""


def _script_for_long_running_job() -> str:
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
  echo "heartbeat-${i}"
  i=$((i + 1))
  sleep 0.2
done
"""


def test_job_creation_persists_job_files_and_supports_incremental_log_reads(tmp_path: Path) -> None:
    with _backend_client(tmp_path, _script_for_finished_job()) as (client, paths):
        create = client.post(
            "/api/jobs",
            json={
                "dataset_name": "runtime_demo",
                "episode_time_s": 8,
                "num_episodes": 1,
                "fps": 10,
                "reset_time_s": 0,
                "resume": False,
                "single_task_text": "Pick up the black cups and place them in the orange box.",
            },
        )
        assert create.status_code == 201
        detail = create.json()
        job_id = detail["state"]["job_id"]

        finished = _wait_for_status(client, job_id, {"succeeded"})
        job_dir = Path(finished["request"]["job_dir"])
        stdout_log = Path(finished["request"]["artifacts"]["log_path"])
        request_path = Path(finished["request"]["artifacts"]["request_path"])
        state_path = Path(finished["request"]["artifacts"]["state_path"])
        events_path = Path(finished["request"]["artifacts"]["events_path"])
        config_path = Path(finished["request"]["artifacts"]["config_path"])

        assert job_dir.is_dir()
        assert request_path.is_file()
        assert state_path.is_file()
        assert stdout_log.is_file()
        assert events_path.is_file()
        assert config_path.is_file()

        first_chunk = client.get(f"/api/jobs/{job_id}/logs", params={"cursor": 0, "limit_bytes": 1024})
        first_chunk.raise_for_status()
        first_payload = first_chunk.json()
        assert first_payload["truncated"] is True
        assert first_payload["lines"][0] == "Dataset output=" + str(paths.dataset_root / "runtime_demo")
        assert "[record] Saved episode 1/1" not in first_payload["lines"]

        second_chunk = client.get(
            f"/api/jobs/{job_id}/logs",
            params={"cursor": first_payload["next_cursor"], "limit_bytes": 1024},
        )
        second_chunk.raise_for_status()
        second_payload = second_chunk.json()
        assert second_payload["lines"] != []
        all_lines = first_payload["lines"] + second_payload["lines"]

        assert any(line.startswith("Config file=") for line in all_lines)
        assert "Task=Pick up the black cups and place them in the orange box." in all_lines
        assert any(line.startswith("trace-line-") for line in all_lines)

        all_log_lines = stdout_log.read_text(encoding="utf-8").splitlines()
        assert "[record] Start episode 1/1 for 8s" in all_log_lines
        assert "[record] Saving episode 1/1" in all_log_lines
        assert "[record] Saved episode 1/1" in all_log_lines
        assert json.loads(request_path.read_text(encoding="utf-8"))["dataset_name"] == "runtime_demo"
        assert (
            json.loads(request_path.read_text(encoding="utf-8"))["single_task_text"]
            == "Pick up the black cups and place them in the orange box."
        )
        assert json.loads(state_path.read_text(encoding="utf-8"))["status"] == "succeeded"
        assert json.loads(config_path.read_text(encoding="utf-8")) == {
            "dataset_name": "runtime_demo",
            "single_task_text": "Pick up the black cups and place them in the orange box.",
        }
        assert events_path.read_text(encoding="utf-8").strip() != ""


def test_only_one_active_job_is_allowed_and_stop_endpoint_stops_it(tmp_path: Path) -> None:
    with _backend_client(tmp_path, _script_for_long_running_job()) as (client, _paths):
        first = client.post(
            "/api/jobs",
            json={
                "dataset_name": "active_job",
                "episode_time_s": 8,
                "num_episodes": 1,
                "fps": 10,
                "reset_time_s": 0,
                "resume": False,
                "single_task_text": "Pick up the black cups and place them in the orange box.",
            },
        )
        assert first.status_code == 201
        first_job = first.json()
        job_id = first_job["state"]["job_id"]

        second = client.post(
            "/api/jobs",
            json={
                "dataset_name": "blocked_job",
                "episode_time_s": 8,
                "num_episodes": 1,
                "fps": 10,
                "reset_time_s": 0,
                "resume": False,
                "single_task_text": "Pick up the black cups and place them in the orange box.",
            },
        )
        assert second.status_code == 409
        assert any(
            "Only one active job is allowed"
            in item
            for item in second.json()["error"]["details"]["conflicts"]
        )

        stop = client.post(f"/api/jobs/{job_id}/stop")
        stop.raise_for_status()
        stop_payload = stop.json()
        assert stop_payload["requested"] is True
        assert stop_payload["signal_stage"] in {"SIGINT", "SIGTERM", "SIGKILL"}

        stopped = _wait_for_status(client, job_id, {"stopped"})
        assert stopped["state"]["stop_requested_at"] is not None

        active_job = client.get("/api/jobs/active")
        active_job.raise_for_status()
        assert active_job.json() is None
