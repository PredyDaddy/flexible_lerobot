# 怎么使用
```bash
bash my_devs/agilex_web_collection/run.sh
```

# AgileX Web Collection Backend

This package contains the MVP backend and runtime layer for the AgileX web collection page.
It only wraps `my_devs/add_robot/agilex/record.sh` and does not modify the script or the static frontend files.

## Environment

- Python runtime and all checks must use `conda run -n lerobot_flex ...`.
- The runtime assumes the repository root is the working directory when launching `uvicorn`.
- The recording script stays unchanged and is launched as:

```bash
bash my_devs/add_robot/agilex/record.sh <dataset_name> <episode_time_s> <num_episodes> <fps> <reset_time_s> <resume> <single_task_text>
```

## Start

Run the backend with the required Conda environment:

```bash
conda run --no-capture-output -n lerobot_flex uvicorn my_devs.agilex_web_collection:app --host 0.0.0.0 --port 8000
```

`conda run` defaults to capturing stdout/stderr. Use `--no-capture-output` so `uvicorn` startup and request logs are visible in the terminal while the service is running.

You can also use the helper launcher from the repository root:

```bash
bash my_devs/agilex_web_collection/run.sh
```

## Runtime layout

- Job ledger root: `my_devs/agilex_web_collection/runtime/agilex_jobs/`
- Per-job files:
  - `request.json`
  - `state.json`
  - `stdout.log`
  - `events.jsonl`
  - `record_config.json`
- AgileX output root: `my_devs/add_robot/agilex/outputs`
- Dataset root: `my_devs/add_robot/agilex/outputs/<repo_prefix>/<dataset_name>`

## API

- `GET /`
  - Serves `my_devs/agilex_web_collection/static/index.html`.
  - The backend does not create or edit frontend files. If `index.html` is still missing, this endpoint returns `503`.
- `GET /api/runtime`
  - Returns readonly runtime paths, defaults, active job id, and current backend limitations.
- `POST /api/preflight`
  - Accepts only the 8 core fields:
    - `repo_prefix`
    - `dataset_name`
    - `episode_time_s`
    - `num_episodes`
    - `fps`
    - `reset_time_s`
    - `resume`
    - `single_task_text`
  - Validates types, ranges, dataset directory occupancy, `resume` semantics, runtime writability, and single-active-job conflicts.
  - Returns the normalized request, command preview, output path, and warnings/conflicts.
- `POST /api/jobs`
  - Creates a recording job if preflight passes.
- `GET /api/jobs`
  - Lists recent jobs.
- `GET /api/jobs/active`
  - Returns the current active job, or `null`.
- `GET /api/jobs/{job_id}`
  - Returns the request snapshot plus mutable state for one job.
- `GET /api/jobs/{job_id}/logs?cursor=<byte_offset>`
  - Returns incremental logs from `stdout.log`. `cursor` is a byte offset and the response includes `next_cursor`.
- `POST /api/jobs/{job_id}/stop`
  - Sends a best-effort stop request using the job process group.

## Behavior notes

- Only one active recording job is allowed at a time.
- Command execution always uses an argument array and never uses `shell=True`.
- `single_task_text` is forwarded to `record.sh` as the 7th positional argument and becomes the dataset task text.
- `repo_prefix` defaults to `dummy` and is combined with `dataset_name` into `DATASET_REPO_ID=<repo_prefix>/<dataset_name>`.
- This implementation does support per-job config isolation because `record.sh` already honors `CONFIG_PATH`.
- Stop uses `SIGINT -> SIGTERM -> SIGKILL` against the process group, but it is still best-effort. A stopped job does not guarantee that the current episode was fully saved.
- `resume=true` only validates that the dataset directory exists. The MVP does not deeply validate whether the directory is a healthy LeRobot dataset.
