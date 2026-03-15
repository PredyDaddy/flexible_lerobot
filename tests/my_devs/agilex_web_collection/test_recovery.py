from __future__ import annotations

from pathlib import Path

from .conftest import build_service, build_test_paths, sample_request, wait_for_terminal_state


def test_service_recovery_restores_active_job_and_allows_pid_based_stop(tmp_path: Path) -> None:
    paths = build_test_paths(tmp_path, script_kind="long_running")
    original_service = build_service(paths)
    created = original_service.create_job(sample_request(dataset_name="recover_demo"))

    recovered_service = build_service(paths)

    active = recovered_service.get_active_job()
    assert active is not None
    assert active.state.job_id == created.state.job_id
    assert active.state.pid == created.state.pid

    stop_response = recovered_service.stop_job(created.state.job_id)
    finished = wait_for_terminal_state(recovered_service, created.state.job_id)

    assert stop_response.requested is True
    assert finished.state.status == "stopped"
    assert recovered_service.get_active_job() is None

    recent = recovered_service.list_jobs(limit=5)
    assert recent
    assert recent[0].job_id == created.state.job_id


def test_recovery_can_read_incremental_logs_from_existing_job(tmp_path: Path) -> None:
    paths = build_test_paths(tmp_path, script_kind="success")
    service = build_service(paths)
    created = service.create_job(sample_request(dataset_name="recover_logs"))
    finished = wait_for_terminal_state(service, created.state.job_id)

    reloaded_service = build_service(paths)
    logs = reloaded_service.get_logs(finished.state.job_id, cursor=0)

    assert logs.status == "succeeded"
    assert logs.active is False
    assert any("Completed" in line for line in logs.lines)
