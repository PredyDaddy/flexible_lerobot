from __future__ import annotations

import sys
import time
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import yaml

from my_devs.web_collection.jobs import SubprocessJob
from my_devs.web_collection.sound_player import SoundPlayer


REPO_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).resolve().parent / "static"
CONFIG_DIR = Path(__file__).resolve().parent / "configs"
APP_CONFIG_PATH = CONFIG_DIR / "app.yaml"
SOUNDS_DIR = REPO_ROOT / "media" / "sounds"


def _next_episode_name(task_dir: Path) -> str:
    task_dir.mkdir(parents=True, exist_ok=True)
    max_idx = -1
    for p in task_dir.glob("episode_*.hdf5"):
        stem = p.stem
        try:
            idx = int(stem.split("_", 1)[1])
        except Exception:
            continue
        max_idx = max(max_idx, idx)
    return f"episode_{max_idx + 1:06d}"


def _normalize_instruction(value: str | None) -> str:
    if value is None:
        return ""
    return value.strip()


class RecordStartRequest(BaseModel):
    dataset_dir: str | None = None
    config_name: str | None = None
    task_name: str | None = None
    episode_name: str | None = None
    num_episodes: int = Field(default=1, ge=1)
    fps: int | None = None
    max_frames: int | None = None
    reset_time_s: float | None = Field(default=None, ge=0)
    use_depth: bool = False
    instruction: str | None = None


class ConvertRequest(BaseModel):
    input_hdf5: str
    output_dir: str | None = None
    repo_id: str
    robot_type: str | None = None
    fps: int | None = None
    use_videos: bool = False
    swap_rb: bool = False
    instruction: str | None = None


class ConvertDatasetRequest(BaseModel):
    input_dataset_dir: str
    output_dir: str | None = None
    repo_id: str
    robot_type: str | None = None
    fps: int | None = None
    use_videos: bool = False
    swap_rb: bool = False
    instruction: str | None = None
    no_base: bool = False


class JobStatus(BaseModel):
    running: bool
    started_at: float | None
    finished_at: float | None
    returncode: int | None
    logs: list[str]


app = FastAPI(title="web_collection", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
if SOUNDS_DIR.is_dir():
    app.mount("/sounds", StaticFiles(directory=str(SOUNDS_DIR)), name="sounds")


_lock = Lock()
_record_job: SubprocessJob | None = None
_convert_job: SubprocessJob | None = None
_sound: SoundPlayer | None = None


def _load_yaml(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _settings() -> dict:
    cfg = _load_yaml(APP_CONFIG_PATH)
    defaults = cfg.get("defaults", {}) or {}
    sounds = cfg.get("sounds", {}) or {}

    # Hard defaults (safe if app.yaml is missing).
    out = {
        "defaults": {
            "dataset_dir": str(defaults.get("dataset_dir", "/tmp/web_collection_data")),
            "lerobot_output_dir": str(defaults.get("lerobot_output_dir", "/tmp/web_collection_lerobot")),
            "config_name": str(defaults.get("config_name", "default.yaml")),
            "task_name": str(defaults.get("task_name", "aloha_mobile_dummy")),
            "repo_id": str(defaults.get("repo_id", "local/web_collection_demo")),
            "num_episodes": int(defaults.get("num_episodes", 1)),
            "reset_time_s": float(defaults.get("reset_time_s", 3.0)),
        },
        "sounds": {
            "enabled": bool(sounds.get("enabled", True)),
            "volume": float(sounds.get("volume", 1.0)),
            "files": dict(sounds.get("files", {}) or {}),
        },
    }
    return out


@app.on_event("startup")
def _startup() -> None:
    global _sound
    s = _settings()
    snd_cfg = s["sounds"]
    files = snd_cfg.get("files") or {
        "ready": "ready.wav",
        "start_record": "start_record.wav",
        "reset_env": "reset_env.wav",
        "finish": "finish.wav",
    }
    _sound = SoundPlayer(
        sounds_dir=SOUNDS_DIR,
        files=files,
        enabled=bool(snd_cfg.get("enabled", True)),
        volume=float(snd_cfg.get("volume", 1.0)),
    )
    st = _sound.status()
    if st.enabled:
        print(f"[sound] enabled, loaded: {st.sounds_loaded}")
    else:
        print(f"[sound] disabled (initialized={st.initialized}) err={st.init_error} loaded={st.sounds_loaded}")


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/configs")
def list_configs() -> list[str]:
    if not CONFIG_DIR.exists():
        return []
    # Only expose recorder configs in the dropdown (exclude app.yaml UI settings).
    return sorted([p.name for p in CONFIG_DIR.glob("*.yaml") if p.name != APP_CONFIG_PATH.name])


@app.get("/api/settings")
def get_settings():
    s = _settings()
    snd = _sound.status() if _sound is not None else None
    return {
        **s,
        "sound_status": None if snd is None else {"enabled": snd.enabled, "initialized": snd.initialized, "init_error": snd.init_error, "sounds_loaded": snd.sounds_loaded},
    }


class SoundPlayRequest(BaseModel):
    name: str
    block: bool = False


@app.post("/api/sound/play")
def sound_play(req: SoundPlayRequest):
    if _sound is None:
        raise HTTPException(status_code=503, detail="Sound system not initialized.")
    played = _sound.play(req.name, block=bool(req.block))
    st = _sound.status()
    return {
        "ok": True,
        "played": played,
        "status": {"enabled": st.enabled, "initialized": st.initialized, "init_error": st.init_error, "sounds_loaded": st.sounds_loaded},
    }


@app.get("/api/episodes")
def list_episodes(dataset_dir: str, task_name: str) -> list[str]:
    task_dir = Path(dataset_dir).expanduser().resolve() / task_name
    if not task_dir.exists():
        return []
    return sorted([str(p) for p in task_dir.glob("episode_*.hdf5")])


def _job_status(job: SubprocessJob | None) -> JobStatus:
    if job is None:
        return JobStatus(running=False, started_at=None, finished_at=None, returncode=None, logs=[])
    return JobStatus(
        running=job.is_running(),
        started_at=job.started_at,
        finished_at=job.finished_at,
        returncode=job.returncode,
        logs=job.logs(),
    )


@app.get("/api/status")
def status():
    with _lock:
        return {
            "record": _job_status(_record_job).model_dump(),
            "convert": _job_status(_convert_job).model_dump(),
            "server_time": time.time(),
        }


@app.post("/api/record/start")
def record_start(req: RecordStartRequest):
    s = _settings()
    defaults = s["defaults"]
    config_name = req.config_name or defaults.get("config_name") or "default.yaml"
    cfg_path = (CONFIG_DIR / config_name).resolve()
    if not cfg_path.is_file() or cfg_path.parent != CONFIG_DIR.resolve():
        raise HTTPException(status_code=400, detail=f"Invalid config_name: {config_name}")
    cfg = _load_yaml(cfg_path)

    instruction = _normalize_instruction(req.instruction)
    if not instruction:
        instruction = _normalize_instruction(cfg.get("instruction"))
    if not instruction:
        raise HTTPException(
            status_code=400,
            detail=(
                "instruction 不能为空。VLA 数据采集必须提供语言指令；"
                "请在页面填写“指令 / 任务”，或在配置文件中设置非空 instruction。"
            ),
        )

    dataset_dir = Path((req.dataset_dir or defaults.get("dataset_dir") or "/tmp/web_collection_data")).expanduser().resolve()
    task_name = req.task_name or (defaults.get("task_name") or "task")
    task_dir = dataset_dir / task_name

    episode_name = req.episode_name or _next_episode_name(task_dir)
    out_hdf5 = task_dir / f"{episode_name}.hdf5"
    if out_hdf5.exists():
        raise HTTPException(status_code=409, detail=f"Episode already exists: {out_hdf5}")

    cmd = [
        sys.executable,
        "-m",
        "my_devs.web_collection.record_hdf5",
        "--config",
        str(cfg_path),
        "--dataset_dir",
        str(dataset_dir),
        "--task_name",
        str(task_name),
        "--episode_name",
        str(episode_name),
    ]
    if req.num_episodes and int(req.num_episodes) > 1:
        cmd += ["--num_episodes", str(int(req.num_episodes))]
    if req.fps is not None:
        cmd += ["--fps", str(int(req.fps))]
    if req.max_frames is not None:
        cmd += ["--max_frames", str(int(req.max_frames))]
    if req.reset_time_s is not None:
        cmd += ["--reset_time_s", str(float(req.reset_time_s))]
    if req.use_depth:
        cmd += ["--use_depth"]
    cmd += ["--instruction", instruction]

    with _lock:
        global _record_job
        if _record_job is not None and _record_job.is_running():
            raise HTTPException(status_code=409, detail="A record job is already running.")

        def _on_line(line: str) -> None:
            if _sound is None:
                return
            if line.startswith("[episode]"):
                print("[sound] start_record")
                _sound.play("start_record", block=False)
            elif line.startswith("[reset]"):
                print("[sound] reset_env")
                _sound.play("reset_env", block=False)

        def _on_finish(_rc: int | None) -> None:
            if _sound is None:
                return
            print("[sound] finish")
            _sound.play("finish", block=False)

        job = SubprocessJob(name="record", cmd=cmd, cwd=REPO_ROOT, on_line=_on_line, on_finish=_on_finish)
        job.start()
        _record_job = job

    return {"ok": True, "out_hdf5": str(out_hdf5), "cmd": cmd}


@app.post("/api/record/stop")
def record_stop():
    with _lock:
        job = _record_job
    if job is None or not job.is_running():
        return {"ok": True, "stopped": False}

    job.stop(timeout_s=15.0)
    return {"ok": True, "stopped": True, "returncode": job.returncode}


@app.post("/api/convert/start")
def convert_start(req: ConvertRequest):
    if not req.input_hdf5 or not req.input_hdf5.strip():
        raise HTTPException(
            status_code=400,
            detail="input_hdf5 为空。请先在下拉框选择一个 episode_*.hdf5，或使用“转换整个任务目录”。",
        )

    in_path = Path(req.input_hdf5).expanduser().resolve()
    if not in_path.is_file():
        raise HTTPException(status_code=400, detail=f"input_hdf5 not found: {in_path}")

    s = _settings()
    defaults = s["defaults"]
    out_dir = str(Path((req.output_dir or defaults.get("lerobot_output_dir") or "/tmp/web_collection_lerobot")).expanduser().resolve())

    cmd = [
        sys.executable,
        "-m",
        "my_devs.web_collection.convert_to_lerobot",
        "--input_hdf5",
        str(in_path),
        "--output_dir",
        out_dir,
        "--repo_id",
        req.repo_id,
    ]
    if req.robot_type is not None:
        cmd += ["--robot_type", req.robot_type]
    if req.fps is not None:
        cmd += ["--fps", str(int(req.fps))]
    if req.use_videos:
        cmd += ["--use_videos"]
    if req.swap_rb:
        cmd += ["--swap_rb"]
    instruction_override = _normalize_instruction(req.instruction)
    if req.instruction is not None and not instruction_override:
        raise HTTPException(status_code=400, detail="instruction 为空白。请填写非空指令，或留空以使用 HDF5 内的 instruction。")
    if instruction_override:
        cmd += ["--instruction", instruction_override]

    with _lock:
        global _convert_job
        if _convert_job is not None and _convert_job.is_running():
            raise HTTPException(status_code=409, detail="A convert job is already running.")

        job = SubprocessJob(name="convert", cmd=cmd, cwd=REPO_ROOT)
        job.start()
        _convert_job = job

    return {"ok": True, "cmd": cmd}


@app.post("/api/convert_dataset/start")
def convert_dataset_start(req: ConvertDatasetRequest):
    in_dir = Path(req.input_dataset_dir).expanduser().resolve()
    if not in_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"input_dataset_dir not found: {in_dir}")

    s = _settings()
    defaults = s["defaults"]
    out_dir = str(Path((req.output_dir or defaults.get("lerobot_output_dir") or "/tmp/web_collection_lerobot")).expanduser().resolve())

    cmd = [
        sys.executable,
        "-m",
        "my_devs.web_collection.convert_dataset_to_lerobot",
        "--input_dataset_dir",
        str(in_dir),
        "--output_dir",
        out_dir,
        "--repo_id",
        req.repo_id,
    ]
    if req.robot_type is not None:
        cmd += ["--robot_type", req.robot_type]
    if req.fps is not None:
        cmd += ["--fps", str(int(req.fps))]
    if req.use_videos:
        cmd += ["--use_videos"]
    if req.swap_rb:
        cmd += ["--swap_rb"]
    instruction_override = _normalize_instruction(req.instruction)
    if req.instruction is not None and not instruction_override:
        raise HTTPException(status_code=400, detail="instruction 为空白。请填写非空指令，或留空以使用 HDF5 内的 instruction。")
    if instruction_override:
        cmd += ["--instruction", instruction_override]
    if req.no_base:
        cmd += ["--no_base"]

    with _lock:
        global _convert_job
        if _convert_job is not None and _convert_job.is_running():
            raise HTTPException(status_code=409, detail="A convert job is already running.")

        job = SubprocessJob(name="convert_dataset", cmd=cmd, cwd=REPO_ROOT)
        job.start()
        _convert_job = job

    return {"ok": True, "cmd": cmd}


@app.post("/api/convert/stop")
def convert_stop():
    with _lock:
        job = _convert_job
    if job is None or not job.is_running():
        return {"ok": True, "stopped": False}

    job.stop(timeout_s=30.0)
    return {"ok": True, "stopped": True, "returncode": job.returncode}
