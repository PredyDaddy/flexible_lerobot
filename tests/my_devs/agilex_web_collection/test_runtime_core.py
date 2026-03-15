from __future__ import annotations

from pathlib import Path

import pytest

from .conftest import (
    build_service,
    build_test_paths,
    dataset_dir_for,
    dataset_repo_id_for,
    sample_request,
    wait_for_terminal_state,
)


def test_preflight_normalizes_single_episode_and_blocks_resume_conflicts(tmp_path: Path) -> None:
    paths = build_test_paths(tmp_path)
    service = build_service(paths)

    response = service.preflight(sample_request(reset_time_s=5))
    assert response.ok is True
    assert response.normalized_request["repo_prefix"] == "dummy"
    assert response.normalized_request["reset_time_s"] == 0.0
    assert response.estimated_duration_s == 8.0
    assert response.dataset_dir == str(dataset_dir_for(paths, "dummy", "demo"))
    assert response.env_overrides["DATASET_REPO_ID"] == dataset_repo_id_for("dummy", "demo")
    assert response.warnings == ["reset_time_s was normalized to 0 because num_episodes=1."]

    existing_dataset = dataset_dir_for(paths, "dummy", "demo")
    existing_dataset.mkdir(parents=True)
    conflict = service.preflight(sample_request())
    assert conflict.ok is False
    assert any("Dataset directory already exists" in item for item in conflict.conflicts)

    missing_resume = service.preflight(sample_request(dataset_name="resume_missing", resume=True))
    assert missing_resume.ok is False
    assert any("resume=true requires an existing dataset directory" in item for item in missing_resume.conflicts)

    custom_prefix = service.preflight(sample_request(repo_prefix="agilex_lab", dataset_name="demo_Data"))
    assert custom_prefix.ok is True
    assert custom_prefix.normalized_request["repo_prefix"] == "agilex_lab"
    assert custom_prefix.dataset_dir == str(dataset_dir_for(paths, "agilex_lab", "demo_Data"))
    assert custom_prefix.env_overrides["DATASET_REPO_ID"] == dataset_repo_id_for("agilex_lab", "demo_Data")


def test_create_job_persists_job_artifacts_and_logs(tmp_path: Path) -> None:
    paths = build_test_paths(tmp_path, script_kind="success")
    service = build_service(paths)

    created = service.create_job(sample_request())
    finished = wait_for_terminal_state(service, created.state.job_id)
    job_dir = paths.job_dir(created.state.job_id)

    assert finished.state.status == "succeeded"
    assert finished.state.current_episode == 1
    assert finished.state.total_episodes == 1
    assert finished.request.request["repo_prefix"] == "dummy"
    assert finished.request.normalized_request["repo_prefix"] == "dummy"
    assert finished.request.env_overrides["DATASET_REPO_ID"] == dataset_repo_id_for("dummy", "demo")
    assert finished.request.command == [
        "bash",
        str(paths.script_path),
        "demo",
        "8",
        "1",
        "30",
        "0",
        "false",
        "Pick up the black cups and place them in the orange box.",
    ]
    assert (job_dir / "request.json").is_file()
    assert (job_dir / "state.json").is_file()
    assert (job_dir / "stdout.log").is_file()
    assert (job_dir / "events.jsonl").is_file()
    assert (job_dir / "record_config.json").is_file()
    assert (dataset_dir_for(paths, "dummy", "demo") / "meta").is_dir()
    assert service.get_active_job() is None

    logs = service.get_logs(created.state.job_id, cursor=0)
    assert "Episode 1/1" in logs.lines
    assert "Saving episode 1/1" in logs.lines
    assert "Task=Pick up the black cups and place them in the orange box." in logs.lines
    assert "Dataset repo_id=dummy/demo" in logs.lines
    follow_up_logs = service.get_logs(created.state.job_id, cursor=logs.next_cursor)
    assert follow_up_logs.lines == []


def test_single_active_job_conflict_and_best_effort_stop(tmp_path: Path) -> None:
    paths = build_test_paths(tmp_path, script_kind="long_running")
    service = build_service(paths)

    active = service.create_job(sample_request(dataset_name="demo_stop"))

    conflicting_preflight = service.preflight(sample_request(dataset_name="demo_other"))
    assert conflicting_preflight.ok is False
    assert any("Only one active job is allowed" in item for item in conflicting_preflight.conflicts)

    with pytest.raises(Exception, match="Only one active job is allowed"):
        service.create_job(sample_request(dataset_name="demo_other"))

    stop_response = service.stop_job(active.state.job_id)
    finished = wait_for_terminal_state(service, active.state.job_id)

    assert stop_response.requested is True
    assert stop_response.signal_stage in {"SIGINT", "SIGTERM", "SIGKILL"}
    assert finished.state.status == "stopped"
    assert finished.state.stop_requested_at is not None
    assert service.get_active_job() is None
