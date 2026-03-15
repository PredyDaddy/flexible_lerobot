from __future__ import annotations

from pathlib import Path

from .conftest import (
    asgi_request,
    build_service,
    build_test_paths,
    dataset_dir_for,
    dataset_repo_id_for,
    patched_app,
)


def test_root_runtime_and_preflight_endpoints(tmp_path: Path, monkeypatch) -> None:
    paths = build_test_paths(tmp_path, script_kind="success")
    service = build_service(paths)

    with patched_app(monkeypatch, paths, service) as app:
        root_response = asgi_request(app, "GET", "/")
        assert root_response.status_code == 200
        assert "AgileX" in root_response.text

        runtime_response = asgi_request(app, "GET", "/api/runtime")
        assert runtime_response.status_code == 200
        runtime_payload = runtime_response.json()
        assert runtime_payload["app_name"] == "agilex_web_collection"
        assert runtime_payload["script_path"] == str(paths.script_path)
        assert runtime_payload["output_root"] == str(paths.outputs_root)
        assert runtime_payload["single_active_job_only"] is True

        invalid_preflight = asgi_request(
            app,
            "POST",
            "/api/preflight",
            json_body={
                "repo_prefix": "dummy",
                "dataset_name": "../bad",
                "episode_time_s": 8,
                "num_episodes": 1,
                "fps": 30,
                "reset_time_s": 0,
                "resume": False,
                "single_task_text": "Pick up the black cups and place them in the orange box.",
            },
        )
        assert invalid_preflight.status_code == 422

        ok_preflight = asgi_request(
            app,
            "POST",
            "/api/preflight",
            json_body={
                "repo_prefix": "dummy",
                "dataset_name": "api_demo",
                "episode_time_s": 8,
                "num_episodes": 1,
                "fps": 30,
                "reset_time_s": 5,
                "resume": False,
                "single_task_text": "Pick up the black cups and place them in the orange box.",
            },
        )
        assert ok_preflight.status_code == 200
        ok_payload = ok_preflight.json()
        assert ok_payload["ok"] is True
        assert ok_payload["normalized_request"]["repo_prefix"] == "dummy"
        assert ok_payload["normalized_request"]["reset_time_s"] == 0.0
        assert ok_payload["dataset_dir"] == str(dataset_dir_for(paths, "dummy", "api_demo"))
        assert ok_payload["env_overrides"]["DATASET_REPO_ID"] == dataset_repo_id_for("dummy", "api_demo")
        assert ok_payload["command"][-1] == "Pick up the black cups and place them in the orange box."
        assert "Pick up the black cups and place them in the orange box." in ok_payload["command_text"]

        custom_prefix = asgi_request(
            app,
            "POST",
            "/api/preflight",
            json_body={
                "repo_prefix": "agilex_lab",
                "dataset_name": "demo_Data",
                "episode_time_s": 8,
                "num_episodes": 1,
                "fps": 30,
                "reset_time_s": 0,
                "resume": False,
                "single_task_text": "Pick up the black cups and place them in the orange box.",
            },
        )
        assert custom_prefix.status_code == 200
        custom_payload = custom_prefix.json()
        assert custom_payload["ok"] is True
        assert custom_payload["normalized_request"]["repo_prefix"] == "agilex_lab"
        assert custom_payload["dataset_dir"] == str(dataset_dir_for(paths, "agilex_lab", "demo_Data"))
        assert custom_payload["env_overrides"]["DATASET_REPO_ID"] == dataset_repo_id_for("agilex_lab", "demo_Data")
        assert custom_payload["command"][2] == "demo_Data"


def test_job_endpoints_cover_lifecycle_and_conflict(tmp_path: Path, monkeypatch) -> None:
    paths = build_test_paths(tmp_path, script_kind="long_running")
    service = build_service(paths)
    payload = {
        "repo_prefix": "dummy",
        "dataset_name": "api_job",
        "episode_time_s": 8,
        "num_episodes": 1,
        "fps": 30,
        "reset_time_s": 0,
        "resume": False,
        "single_task_text": "Pick up the black cups and place them in the orange box.",
    }

    with patched_app(monkeypatch, paths, service) as app:
        create_response = asgi_request(app, "POST", "/api/jobs", json_body=payload)
        assert create_response.status_code == 201
        job = create_response.json()
        job_id = job["state"]["job_id"]
        assert job["request"]["dataset_name"] == "api_job"
        assert job["request"]["request"]["repo_prefix"] == "dummy"
        assert job["request"]["normalized_request"]["repo_prefix"] == "dummy"
        assert job["request"]["request"]["single_task_text"] == payload["single_task_text"]
        assert job["request"]["normalized_request"]["single_task_text"] == payload["single_task_text"]
        assert job["request"]["env_overrides"]["DATASET_REPO_ID"] == dataset_repo_id_for("dummy", "api_job")
        assert payload["single_task_text"] in job["request"]["command_text"]
        assert job["state"]["active"] is True

        active_response = asgi_request(app, "GET", "/api/jobs/active")
        assert active_response.status_code == 200
        assert active_response.json()["state"]["job_id"] == job_id

        detail_response = asgi_request(app, "GET", f"/api/jobs/{job_id}")
        assert detail_response.status_code == 200
        assert detail_response.json()["request"]["dataset_dir"] == str(dataset_dir_for(paths, "dummy", "api_job"))

        logs_response = asgi_request(app, "GET", f"/api/jobs/{job_id}/logs")
        assert logs_response.status_code == 200
        assert logs_response.json()["job_id"] == job_id

        conflict_response = asgi_request(
            app,
            "POST",
            "/api/jobs",
            json_body={**payload, "dataset_name": "api_job_conflict"},
        )
        assert conflict_response.status_code == 409
        conflict_payload = conflict_response.json()
        if "detail" in conflict_payload:
            message = conflict_payload["detail"]["message"]
            conflicts = conflict_payload["detail"]["conflicts"]
        else:
            error = conflict_payload["error"]
            message = error["message"]
            conflicts = error["details"]["conflicts"]
        assert "Only one active job is allowed" in message
        assert conflicts

        stop_response = asgi_request(app, "POST", f"/api/jobs/{job_id}/stop")
        assert stop_response.status_code == 200
        stop_payload = stop_response.json()
        assert stop_payload["requested"] is True
        assert stop_payload["job"]["state"]["job_id"] == job_id

        recent_response = asgi_request(app, "GET", "/api/jobs")
        assert recent_response.status_code == 200
        assert recent_response.json()[0]["job_id"] == job_id
