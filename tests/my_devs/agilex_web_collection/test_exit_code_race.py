from __future__ import annotations

from pathlib import Path

import pytest

from .conftest import MODULES, build_service, build_test_paths, sample_request


def test_reconcile_uses_supervisor_exit_code_before_marking_job_failed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Regression test for successful jobs that briefly lose the persisted exit code.

    The live bug happens when reconcile sees:
    - the process is no longer running
    - state.returncode is still None
    - the supervisor already knows the real exit code

    Before the fix, service.get_job() finalizes this as:
    status='failed', returncode=None

    After the fix, the service should consult the supervisor-provided exit code and
    finalize the job as succeeded.
    """

    paths = build_test_paths(tmp_path, script_kind="long_running")
    service = build_service(paths)
    created = service.create_job(sample_request(dataset_name="race_exit_code"))
    job_id = created.state.job_id

    # Keep the real process around so cleanup can still stop it after the assertion.
    stop_pid = created.state.pid
    stop_pgid = created.state.pgid

    monkeypatch.setattr(service.supervisor, "is_running", lambda _job_id: False)
    monkeypatch.setattr(MODULES.service, "is_process_alive", lambda _pid: False)
    monkeypatch.setattr(service.supervisor, "poll_returncode", lambda _job_id: 0)

    try:
        detail = service.get_job(job_id)
        assert detail.state.status == "succeeded"
        assert detail.state.returncode == 0
        assert detail.state.active is False
    finally:
        service.supervisor.stop_process_group(pid=stop_pid, pgid=stop_pgid)
