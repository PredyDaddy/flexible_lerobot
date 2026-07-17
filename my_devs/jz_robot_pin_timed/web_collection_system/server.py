from __future__ import annotations

import argparse
import json
import math
import mimetypes
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from my_devs.jz_robot_pin_timed.web_collection_system.jobs import ManagedJob, idle_status

REPO_ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = APP_ROOT / "static"
OUTPUTS_ROOT = (
    Path(os.environ.get("JZ_WEB_DATASET_ROOT", REPO_ROOT / "tests" / "outputs")).expanduser().resolve()
)
VISUALIZATION_URL = os.environ.get("JZ_WEB_VISUALIZATION_URL", "http://10.1.42.3:7000/static/")
ARMED_ACTIONS_ENABLED = os.environ.get("JZ_WEB_ARMED_ACTIONS") == "1"
MOCK_COMMANDS = os.environ.get("JZ_WEB_MOCK_COMMANDS") == "1"
CONTROL_TOKEN = os.environ.get("JZ_WEB_CONTROL_TOKEN", "").strip()

DATASET_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
VISUALIZATION_SCRIPT = REPO_ROOT / "my_devs/jz_robot_pin_timed/x86/start_pin_joystick.sh"
VISUALIZATION_STOP_SCRIPT = REPO_ROOT / "my_devs/jz_robot_pin_timed/x86/stop_pin_teleop.sh"
RECORD_SCRIPT = REPO_ROOT / "my_devs/jz_robot_pin_timed/record.sh"
REPLAY_SCRIPT = REPO_ROOT / "my_devs/jz_robot_pin_timed/x86/start_pin_replay.sh"

RECORD_DEFAULTS: dict[str, str] = {
    "NUM_EPISODES": "10",
    "EPISODE_TIME_S": "10",
    "RESET_TIME_S": "5",
    "RECORD_FPS": "20",
    "VIDEO": "true",
    "VIDEO_CRF": "18",
    "VIDEO_ENCODING_BATCH_SIZE": "10",
    "RESUME": "false",
    "EXECUTION": "armed",
    "SEND_ACTION_TRANSPORT": "udp",
    "ZMQ_PRESET": "jz_three_zmq",
    "RTSP_PRESET": "none",
    "TIMING_SIDECAR": "true",
    "REQUIRE_STATE_SOURCE_TIMING": "true",
    "REQUIRE_STATE_ADVANCE_PER_OBSERVATION": "true",
    "STATE_ADVANCE_TIMEOUT_S": "0.1",
    "MAX_CAMERA_STATE_RECEIVE_SKEW_MS": "200.0",
    "LEFT_GRIPPER_OBSERVATION_SOURCE": "measured_opening",
    "RIGHT_GRIPPER_OBSERVATION_SOURCE": "commanded_opening",
    "MAX_INITIAL_JOINT_DELTA_RAD": "10.0",
    "MAX_JOINT_STEP_RAD": "10.0",
    "JZ_ROBOT_PIN_ARMED": "1",
    "I_UNDERSTAND_JZ_ROBOT_PIN_MOVES_ROBOT": "1",
}

REPLAY_DEFAULTS: dict[str, str] = {
    "EXECUTION": "armed",
    "SEND_ACTION_TRANSPORT": "udp",
    "REPLAY_FPS": "20",
    "MAX_INITIAL_JOINT_DELTA_RAD": "0.5",
    "MAX_JOINT_STEP_RAD": "0.05",
    "PLAY_SOUNDS": "true",
    "JZ_ROBOT_PIN_ARMED": "1",
    "I_UNDERSTAND_JZ_ROBOT_PIN_MOVES_ROBOT": "1",
}

RECORD_OPTION_FIELDS = {
    "NUM_EPISODES",
    "EPISODE_TIME_S",
    "RESET_TIME_S",
    "RECORD_FPS",
    "VIDEO_CRF",
    "STATE_ADVANCE_TIMEOUT_S",
    "MAX_CAMERA_STATE_RECEIVE_SKEW_MS",
    "MAX_INITIAL_JOINT_DELTA_RAD",
    "MAX_JOINT_STEP_RAD",
}
REPLAY_OPTION_FIELDS = {
    "REPLAY_FPS",
    "MAX_INITIAL_JOINT_DELTA_RAD",
    "MAX_JOINT_STEP_RAD",
    "PLAY_SOUNDS",
}


class ApiError(Exception):
    def __init__(self, status: HTTPStatus, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    path: str
    total_episodes: int
    total_frames: int
    fps: int | float
    modified_at: float

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "path": self.path,
            "total_episodes": self.total_episodes,
            "total_frames": self.total_frames,
            "fps": self.fps,
            "modified_at": self.modified_at,
        }


def load_dataset(path: Path) -> DatasetInfo:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        raise ApiError(HTTPStatus.BAD_REQUEST, "回放数据集路径必须是绝对路径")
    root = candidate.resolve()
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise ApiError(HTTPStatus.BAD_REQUEST, f"不是 LeRobot 数据集：缺少 {info_path}")
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ApiError(HTTPStatus.BAD_REQUEST, f"无法读取数据集 metadata：{exc}") from exc
    if info.get("robot_type") != "jz_robot_pin_timed":
        raise ApiError(HTTPStatus.BAD_REQUEST, f"数据集 robot_type 不是 jz_robot_pin_timed：{root}")

    total_episodes = int(info.get("total_episodes", 0))
    total_frames = int(info.get("total_frames", 0))
    if total_episodes <= 0:
        raise ApiError(HTTPStatus.BAD_REQUEST, f"数据集没有已保存 episode：{root}")
    return DatasetInfo(
        name=root.name,
        path=str(root),
        total_episodes=total_episodes,
        total_frames=total_frames,
        fps=info.get("fps", 0),
        modified_at=info_path.stat().st_mtime,
    )


def discover_datasets(root: Path = OUTPUTS_ROOT) -> list[DatasetInfo]:
    if not root.is_dir():
        return []
    datasets: list[DatasetInfo] = []
    for info_path in root.glob("*/meta/info.json"):
        try:
            datasets.append(load_dataset(info_path.parents[1]))
        except ApiError:
            continue
    return sorted(datasets, key=lambda item: (item.modified_at, item.name), reverse=True)


def validate_record_root(value: object) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise ApiError(HTTPStatus.BAD_REQUEST, "请填写录制数据集的完整保存路径")
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        raise ApiError(HTTPStatus.BAD_REQUEST, "录制保存路径必须是绝对路径")
    root = candidate.resolve()
    if root == Path("/"):
        raise ApiError(HTTPStatus.BAD_REQUEST, "录制保存路径不能是文件系统根目录")
    if not DATASET_NAME_PATTERN.fullmatch(root.name):
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            "数据集目录名只能包含字母、数字、点、下划线和连字符",
        )
    if root.exists():
        raise ApiError(HTTPStatus.CONFLICT, f"录制保存路径已存在；RESUME=false 不会覆盖：{root}")
    if not root.parent.is_dir():
        raise ApiError(HTTPStatus.BAD_REQUEST, f"录制保存路径的父目录不存在：{root.parent}")
    return root


def validate_options(
    raw_options: object,
    defaults: dict[str, str],
    allowed_fields: set[str],
) -> dict[str, str]:
    options = dict(defaults)
    if raw_options is None:
        return options
    if not isinstance(raw_options, dict):
        raise ApiError(HTTPStatus.BAD_REQUEST, "options 必须是对象")
    unexpected = sorted(set(raw_options) - allowed_fields)
    if unexpected:
        raise ApiError(HTTPStatus.BAD_REQUEST, f"不支持的参数：{', '.join(unexpected)}")

    for key, value in raw_options.items():
        string_value = str(value).strip()
        if not string_value:
            raise ApiError(HTTPStatus.BAD_REQUEST, f"{key} 不能为空")
        if key == "PLAY_SOUNDS":
            if string_value not in {"true", "false"}:
                raise ApiError(HTTPStatus.BAD_REQUEST, "PLAY_SOUNDS 必须是 true 或 false")
        else:
            try:
                number = float(string_value)
            except ValueError as exc:
                raise ApiError(HTTPStatus.BAD_REQUEST, f"{key} 必须是数字") from exc
            if not math.isfinite(number):
                raise ApiError(HTTPStatus.BAD_REQUEST, f"{key} 必须是有限数字")
            if number < 0:
                raise ApiError(HTTPStatus.BAD_REQUEST, f"{key} 不能小于 0")
            positive_fields = {
                "MAX_CAMERA_STATE_RECEIVE_SKEW_MS",
                "MAX_INITIAL_JOINT_DELTA_RAD",
                "MAX_JOINT_STEP_RAD",
            }
            if key in positive_fields and number <= 0:
                raise ApiError(HTTPStatus.BAD_REQUEST, f"{key} 必须大于 0")
            integer_fields = {"NUM_EPISODES", "RECORD_FPS", "REPLAY_FPS", "VIDEO_CRF"}
            if key in integer_fields and not number.is_integer():
                raise ApiError(HTTPStatus.BAD_REQUEST, f"{key} 必须是整数")
            if key == "NUM_EPISODES" and number <= 0:
                raise ApiError(HTTPStatus.BAD_REQUEST, "NUM_EPISODES 必须是正整数")
        options[key] = string_value
    return options


def build_record_environment(dataset_root: Path, raw_options: object = None) -> dict[str, str]:
    env = validate_options(raw_options, RECORD_DEFAULTS, RECORD_OPTION_FIELDS)
    # Encode once after the requested collection is complete. LeRobot's
    # VideoEncodingManager also flushes a final partial batch on clean exit.
    env["VIDEO_ENCODING_BATCH_SIZE"] = env["NUM_EPISODES"]
    env["DATASET_NAME"] = dataset_root.name
    env["DATASET_ROOT"] = str(dataset_root)
    env["DATASET_REPO_ID"] = f"local/{dataset_root.name}"
    return env


def build_replay_environment(
    dataset: DatasetInfo,
    episode: object,
    raw_options: object = None,
) -> dict[str, str]:
    try:
        episode_index = int(episode)
    except (TypeError, ValueError) as exc:
        raise ApiError(HTTPStatus.BAD_REQUEST, "请选择要回放的 episode") from exc
    if episode_index < 0 or episode_index >= dataset.total_episodes:
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            f"episode {episode_index} 超出范围 0..{dataset.total_episodes - 1}",
        )

    env = validate_options(raw_options, REPLAY_DEFAULTS, REPLAY_OPTION_FIELDS)
    env["DATASET_NAME"] = dataset.name
    env["DATASET_ROOT"] = dataset.path
    env["DATASET_REPO_ID"] = f"local/{dataset.name}"
    env["EPISODE"] = str(episode_index)
    return env


def command_preview(script: Path, env: dict[str, str]) -> str:
    lines = [f"{key}={json.dumps(value, ensure_ascii=False)} \\" for key, value in env.items()]
    lines.append(f"bash {script.relative_to(REPO_ROOT)}")
    return "\n".join(lines)


def visualization_running() -> bool:
    try:
        completed = subprocess.run(
            ["pgrep", "-af", str(VISUALIZATION_SCRIPT)],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    current_pid = os.getpid()
    return any(
        line.strip() and not line.startswith(f"{current_pid} ") for line in completed.stdout.splitlines()
    )


def mock_command(name: str) -> list[str]:
    duration_s = 3600 if name == "visualization" else 2
    code = (
        "import sys,time;"
        f"print('[mock] {name} started', flush=True);"
        "print('[mock] no robot command will be executed', flush=True);"
        f"time.sleep({duration_s});"
        f"print('[mock] {name} finished', flush=True)"
    )
    return [sys.executable, "-c", code]


class Application:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.visualization_job: ManagedJob | None = None
        self.operation_job: ManagedJob | None = None
        self.operation_kind: str | None = None
        self.last_command_preview = ""

    def bootstrap(self) -> dict[str, object]:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        suggested_root = OUTPUTS_ROOT / f"jz_robot_pin_timed_real_10eps_{stamp}"
        return {
            "repo_root": str(REPO_ROOT),
            "outputs_root": str(OUTPUTS_ROOT),
            "visualization_url": VISUALIZATION_URL,
            "armed_actions_enabled": ARMED_ACTIONS_ENABLED,
            "mock_commands": MOCK_COMMANDS,
            "control_token_required": bool(CONTROL_TOKEN),
            "suggested_record_root": str(suggested_root),
            "record_defaults": RECORD_DEFAULTS,
            "replay_defaults": REPLAY_DEFAULTS,
            "record_episode_count_mode": "flexible",
        }

    def status(self) -> dict[str, object]:
        with self.lock:
            visualization = (
                idle_status("visualization")
                if self.visualization_job is None
                else self.visualization_job.status()
            )
            externally_running = visualization_running()
            visualization["running"] = bool(visualization["running"] or externally_running)
            visualization["external"] = bool(
                externally_running
                and (self.visualization_job is None or not self.visualization_job.is_running())
            )
            operation = (
                idle_status("operation") if self.operation_job is None else self.operation_job.status()
            )
            operation["kind"] = self.operation_kind
            return {
                "visualization": visualization,
                "operation": operation,
                "armed_actions_enabled": ARMED_ACTIONS_ENABLED,
                "mock_commands": MOCK_COMMANDS,
                "last_command_preview": self.last_command_preview,
                "server_time": time.time(),
            }

    def start_visualization(self) -> dict[str, object]:
        with self.lock:
            if visualization_running() or (
                self.visualization_job is not None and self.visualization_job.is_running()
            ):
                raise ApiError(HTTPStatus.CONFLICT, "可视化已经在运行")
            command = mock_command("visualization") if MOCK_COMMANDS else ["bash", str(VISUALIZATION_SCRIPT)]
            job = ManagedJob("visualization", command, REPO_ROOT)
            job.start()
            self.visualization_job = job
            self.last_command_preview = f"bash {VISUALIZATION_SCRIPT.relative_to(REPO_ROOT)}"
            return {"ok": True, "url": VISUALIZATION_URL, "command": self.last_command_preview}

    def stop_visualization(self) -> dict[str, object]:
        with self.lock:
            job = self.visualization_job
        if job is not None and job.is_running():
            job.stop()
        if not MOCK_COMMANDS:
            completed = subprocess.run(
                ["bash", str(VISUALIZATION_STOP_SCRIPT)],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=15.0,
                check=False,
            )
            output = "\n".join(part for part in (completed.stdout.strip(), completed.stderr.strip()) if part)
            if completed.returncode != 0:
                raise ApiError(HTTPStatus.INTERNAL_SERVER_ERROR, output or "停止可视化失败")
        else:
            output = "[mock] visualization stopped"
        return {"ok": True, "output": output}

    def start_record(self, body: dict[str, object]) -> dict[str, object]:
        self._require_armed(body)
        dataset_root = validate_record_root(body.get("dataset_root"))
        env = build_record_environment(dataset_root, body.get("options"))
        with self.lock:
            self._require_idle_operation()
            if not MOCK_COMMANDS and not visualization_running():
                raise ApiError(HTTPStatus.CONFLICT, "录制前必须先启动可视化")
            command = mock_command("record") if MOCK_COMMANDS else ["bash", str(RECORD_SCRIPT)]
            job = ManagedJob("record", command, REPO_ROOT, env=env)
            job.start()
            self.operation_job = job
            self.operation_kind = "record"
            self.last_command_preview = command_preview(RECORD_SCRIPT, env)
        return {
            "ok": True,
            "dataset_root": str(dataset_root),
            "episodes": int(env["NUM_EPISODES"]),
            "command": self.last_command_preview,
        }

    def start_replay(self, body: dict[str, object]) -> dict[str, object]:
        self._require_armed(body)
        dataset = load_dataset(Path(str(body.get("dataset_root", ""))))
        env = build_replay_environment(dataset, body.get("episode"), body.get("options"))
        with self.lock:
            self._require_idle_operation()
            command = mock_command("replay") if MOCK_COMMANDS else ["bash", str(REPLAY_SCRIPT)]
            job = ManagedJob("replay", command, REPO_ROOT, env=env)
            job.start()
            self.operation_job = job
            self.operation_kind = "replay"
            self.last_command_preview = command_preview(REPLAY_SCRIPT, env)
        return {
            "ok": True,
            "dataset_root": dataset.path,
            "episode": int(env["EPISODE"]),
            "command": self.last_command_preview,
        }

    def stop_operation(self) -> dict[str, object]:
        with self.lock:
            job = self.operation_job
            kind = self.operation_kind
        if job is None or not job.is_running():
            return {"ok": True, "stopped": False, "kind": kind}
        job.stop()
        return {"ok": True, "stopped": True, "kind": kind, "returncode": job.returncode}

    def _require_idle_operation(self) -> None:
        if self.operation_job is not None and self.operation_job.is_running():
            raise ApiError(HTTPStatus.CONFLICT, f"{self.operation_kind} 正在运行，请先停止")

    @staticmethod
    def _require_armed(body: dict[str, object]) -> None:
        if not (ARMED_ACTIONS_ENABLED or MOCK_COMMANDS):
            raise ApiError(
                HTTPStatus.FORBIDDEN,
                "Web armed 操作未解锁；请使用 JZ_WEB_ARMED_ACTIONS=1 启动服务",
            )
        if body.get("confirmed") is not True:
            raise ApiError(HTTPStatus.BAD_REQUEST, "必须确认 Orin、急停和机器人工作区状态")


APPLICATION = Application()


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "JZTimedCollection/1.0"

    def do_GET(self) -> None:
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/api/bootstrap":
                self._write_json(APPLICATION.bootstrap())
                return
            if parsed.path == "/api/status":
                self._write_json(APPLICATION.status())
                return
            if parsed.path == "/api/datasets":
                self._write_json({"datasets": [item.as_dict() for item in discover_datasets()]})
                return
            if parsed.path == "/api/dataset":
                query = parse_qs(parsed.query)
                value = query.get("path", [""])[0]
                dataset = load_dataset(Path(value))
                self._write_json(dataset.as_dict())
                return
            self._serve_static(parsed.path)
        except ApiError as exc:
            self._write_error(exc.status, exc.message)
        except Exception as exc:
            self._write_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"服务端错误：{exc}")

    def do_POST(self) -> None:
        try:
            self._check_control_token()
            body = self._read_json()
            path = urlparse(self.path).path
            if path == "/api/visualization/start":
                response = APPLICATION.start_visualization()
            elif path == "/api/visualization/stop":
                response = APPLICATION.stop_visualization()
            elif path == "/api/record/start":
                response = APPLICATION.start_record(body)
            elif path == "/api/replay/start":
                response = APPLICATION.start_replay(body)
            elif path == "/api/operation/stop":
                response = APPLICATION.stop_operation()
            else:
                raise ApiError(HTTPStatus.NOT_FOUND, f"未知 API：{path}")
            self._write_json(response)
        except ApiError as exc:
            self._write_error(exc.status, exc.message)
        except Exception as exc:
            self._write_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"服务端错误：{exc}")

    def log_message(self, format_string: str, *args: object) -> None:
        print(f"[web] {self.address_string()} {format_string % args}")

    def _check_control_token(self) -> None:
        if CONTROL_TOKEN and self.headers.get("X-JZ-Control-Token", "") != CONTROL_TOKEN:
            raise ApiError(HTTPStatus.UNAUTHORIZED, "控制令牌无效")

    def _read_json(self) -> dict[str, object]:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, "Content-Length 无效") from exc
        if content_length > 1_000_000:
            raise ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "请求体过大")
        raw = self.rfile.read(content_length) if content_length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, "请求 JSON 无效") from exc
        if not isinstance(body, dict):
            raise ApiError(HTTPStatus.BAD_REQUEST, "请求 JSON 必须是对象")
        return body

    def _serve_static(self, request_path: str) -> None:
        relative = "index.html" if request_path in {"", "/"} else unquote(request_path.lstrip("/"))
        if relative.startswith("static/"):
            relative = relative.removeprefix("static/")
        candidate = (STATIC_ROOT / relative).resolve()
        if STATIC_ROOT not in candidate.parents and candidate != STATIC_ROOT:
            raise ApiError(HTTPStatus.NOT_FOUND, "文件不存在")
        if not candidate.is_file():
            raise ApiError(HTTPStatus.NOT_FOUND, "文件不存在")
        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        data = candidate.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header(
            "Content-Type",
            f"{content_type}; charset=utf-8" if content_type.startswith("text/") else content_type,
        )
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _write_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _write_error(self, status: HTTPStatus, message: str) -> None:
        self._write_json({"ok": False, "error": message}, status)


def main() -> None:
    parser = argparse.ArgumentParser(description="JZ Robot Pin Timed 数采系统 Web 服务")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), RequestHandler)
    print(f"[web] repo={REPO_ROOT}")
    print(f"[web] datasets={OUTPUTS_ROOT}")
    print(f"[web] armed_actions_enabled={ARMED_ACTIONS_ENABLED} mock_commands={MOCK_COMMANDS}")
    print(f"[web] listening=http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[web] stopping")
    finally:
        APPLICATION.stop_operation()
        if APPLICATION.visualization_job is not None:
            APPLICATION.visualization_job.stop()
        server.server_close()


if __name__ == "__main__":
    main()
