from __future__ import annotations

import asyncio
from contextlib import contextmanager
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import time
import types
from urllib.parse import urlencode

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_DIR = REPO_ROOT / "my_devs" / "agilex_web_collection"
TEST_PACKAGE_NAME = "_agilex_web_collection_testpkg"


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _bootstrap_test_modules():
    package = types.ModuleType(TEST_PACKAGE_NAME)
    package.__path__ = [str(PACKAGE_DIR)]
    sys.modules[TEST_PACKAGE_NAME] = package

    models_module = _load_module(f"{TEST_PACKAGE_NAME}.models", PACKAGE_DIR / "models.py")
    paths_module = _load_module(f"{TEST_PACKAGE_NAME}.paths", PACKAGE_DIR / "paths.py")
    store_module = _load_module(f"{TEST_PACKAGE_NAME}.store", PACKAGE_DIR / "store.py")
    supervisor_module = _load_module(f"{TEST_PACKAGE_NAME}.supervisor", PACKAGE_DIR / "supervisor.py")
    service_module = _load_module(f"{TEST_PACKAGE_NAME}.service", PACKAGE_DIR / "service.py")
    app_module = _load_module(f"{TEST_PACKAGE_NAME}.app_file", PACKAGE_DIR / "app.py")

    return types.SimpleNamespace(
        app=app_module,
        models=models_module,
        paths=paths_module,
        store=store_module,
        supervisor=supervisor_module,
        service=service_module,
    )


MODULES = _bootstrap_test_modules()
app_module = MODULES.app
AgilexWebPaths = MODULES.paths.AgilexWebPaths


def _write_executable(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)
    return path


def _success_script() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail
exec python - "$@" <<'PY'
import json
import os
from pathlib import Path
import sys
import time

dataset_name, episode_time_s, num_episodes, fps, reset_time_s, resume = sys.argv[1:7]
single_task_text = " ".join(sys.argv[7:]).strip()
config_path = Path(os.environ["CONFIG_PATH"])
config_path.parent.mkdir(parents=True, exist_ok=True)
dataset_repo_id = os.environ.get("DATASET_REPO_ID", f"dummy/{dataset_name}")
dataset_dir = Path(os.environ["HF_LEROBOT_HOME"]) / Path(dataset_repo_id)
(dataset_dir / "meta").mkdir(parents=True, exist_ok=True)
config_path.write_text(
    json.dumps(
        {
            "dataset_name": dataset_name,
            "dataset_repo_id": dataset_repo_id,
            "single_task_text": single_task_text,
        }
    ),
    encoding="utf-8",
)
print(f"Dataset output={dataset_dir}", flush=True)
print(f"Dataset repo_id={dataset_repo_id}", flush=True)
print(f"Resume={resume}", flush=True)
print(f"Task={single_task_text}", flush=True)
print(f"Episode 1/{num_episodes}", flush=True)
time.sleep(0.05)
print(f"Saving episode 1/{num_episodes}", flush=True)
time.sleep(0.05)
print("Completed", flush=True)
PY
"""


def _long_running_script() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail
exec python - "$@" <<'PY'
import json
import os
from pathlib import Path
import signal
import sys
import time

dataset_name, episode_time_s, num_episodes, fps, reset_time_s, resume = sys.argv[1:7]
single_task_text = " ".join(sys.argv[7:]).strip()
config_path = Path(os.environ["CONFIG_PATH"])
config_path.parent.mkdir(parents=True, exist_ok=True)
dataset_repo_id = os.environ.get("DATASET_REPO_ID", f"dummy/{dataset_name}")
dataset_dir = Path(os.environ["HF_LEROBOT_HOME"]) / Path(dataset_repo_id)
(dataset_dir / "meta").mkdir(parents=True, exist_ok=True)
config_path.write_text(
    json.dumps(
        {
            "dataset_name": dataset_name,
            "dataset_repo_id": dataset_repo_id,
            "single_task_text": single_task_text,
        }
    ),
    encoding="utf-8",
)
stop_requested = False

def _handle_signal(signum, _frame):
    global stop_requested
    print(f"Signal {signum} received", flush=True)
    stop_requested = True

signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)

print(f"Dataset output={dataset_dir}", flush=True)
print(f"Dataset repo_id={dataset_repo_id}", flush=True)
print(f"Task={single_task_text}", flush=True)
print(f"Episode 1/{num_episodes}", flush=True)
while not stop_requested:
    print("Recording heartbeat", flush=True)
    time.sleep(0.1)
print("Stopping", flush=True)
raise SystemExit(130)
PY
"""


def build_test_paths(tmp_path: Path, *, script_kind: str = "success") -> AgilexWebPaths:
    repo_root = tmp_path
    package_root = repo_root / "my_devs" / "agilex_web_collection"
    static_dir = package_root / "static"
    runtime_dir = package_root / "runtime"
    jobs_root = runtime_dir / "agilex_jobs"
    agilex_root = repo_root / "my_devs" / "add_robot" / "agilex"
    outputs_root = agilex_root / "outputs"
    dataset_root = outputs_root
    static_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    agilex_root.mkdir(parents=True, exist_ok=True)
    outputs_root.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)
    (static_dir / "index.html").write_text("<!doctype html><title>AgileX</title>", encoding="utf-8")

    script_path = agilex_root / "record.sh"
    if script_kind == "success":
        script_content = _success_script()
    elif script_kind == "long_running":
        script_content = _long_running_script()
    else:
        raise ValueError(f"Unknown script_kind: {script_kind}")
    _write_executable(script_path, script_content)

    init_kwargs = {
        "repo_root": repo_root,
        "package_root": package_root,
        "static_dir": static_dir,
        "runtime_dir": runtime_dir,
        "jobs_root": jobs_root,
        "agilex_root": agilex_root,
        "script_path": script_path,
        "outputs_root": outputs_root,
        "dataset_root": dataset_root,
    }
    supported = inspect.signature(AgilexWebPaths).parameters
    filtered_kwargs = {key: value for key, value in init_kwargs.items() if key in supported}
    return AgilexWebPaths(**filtered_kwargs)


def dataset_dir_for(paths: AgilexWebPaths, repo_prefix: str, dataset_name: str) -> Path:
    return paths.outputs_root / repo_prefix / dataset_name


def dataset_repo_id_for(repo_prefix: str, dataset_name: str) -> str:
    return f"{repo_prefix}/{dataset_name}"


def build_service(paths: AgilexWebPaths):
    store = app_module.STORE.__class__(paths)
    supervisor = app_module.SUPERVISOR.__class__()
    service = app_module.SERVICE.__class__(
        paths=paths,
        store=store,
        supervisor=supervisor,
    )
    service.initialize()
    return service


def sample_request(**overrides: object):
    payload = {
        "repo_prefix": "dummy",
        "dataset_name": "demo",
        "episode_time_s": 8,
        "num_episodes": 1,
        "fps": 30,
        "reset_time_s": 0,
        "resume": False,
        "single_task_text": "Pick up the black cups and place them in the orange box.",
    }
    payload.update(overrides)
    return MODULES.models.RecordRequest.model_validate(payload)


def wait_for_terminal_state(
    service,
    job_id: str,
    *,
    timeout_s: float = 8.0,
):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        detail = service.get_job(job_id)
        if not detail.state.active:
            return detail
        time.sleep(0.05)
    raise AssertionError(f"Job {job_id} did not reach a terminal state within {timeout_s} seconds.")


@contextmanager
def patched_app(monkeypatch: pytest.MonkeyPatch, paths: AgilexWebPaths, service):
    del monkeypatch
    yield app_module.create_app(paths=paths, service=service)


class ASGIResponse:
    def __init__(self, status_code: int, headers: dict[str, str], body: bytes) -> None:
        self.status_code = status_code
        self.headers = headers
        self.body = body
        self.text = body.decode("utf-8", errors="replace")

    def json(self):
        return json.loads(self.body.decode("utf-8"))


def asgi_request(
    app,
    method: str,
    path: str,
    *,
    json_body: dict | None = None,
    query_params: dict[str, str | int] | None = None,
) -> ASGIResponse:
    body = b""
    headers: list[tuple[bytes, bytes]] = []
    if json_body is not None:
        body = json.dumps(json_body).encode("utf-8")
        headers.append((b"content-type", b"application/json"))
        headers.append((b"content-length", str(len(body)).encode("ascii")))

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method.upper(),
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("utf-8"),
        "query_string": urlencode(query_params or {}, doseq=True).encode("utf-8"),
        "headers": headers,
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
        "root_path": "",
    }

    messages: list[dict] = []
    request_sent = False

    async def receive():
        nonlocal request_sent
        if request_sent:
            return {"type": "http.disconnect"}
        request_sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        messages.append(message)

    asyncio.run(app(scope, receive, send))

    start = next(message for message in messages if message["type"] == "http.response.start")
    body_chunks = [message.get("body", b"") for message in messages if message["type"] == "http.response.body"]
    decoded_headers = {
        key.decode("latin-1"): value.decode("latin-1") for key, value in start.get("headers", [])
    }
    return ASGIResponse(start["status"], decoded_headers, b"".join(body_chunks))
