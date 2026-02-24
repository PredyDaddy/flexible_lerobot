# 数据网页端可视化 Plan（代码拆解与技术细节）

本文档是 `docs/data_vis_docs/数据网页端可视化最终方案.md` 的“工程落地版”：按模块拆解需要开发哪些代码、每块代码的职责、核心技术细节与验收标准。

> 约束复述（必须满足）：不修改 `src/lerobot/`；新增独立 `vis/`；离线可用；视频用浏览器 `<video>` + HTTP Range；v3.0 视频为“多 episode 拼接 mp4”，必须用 `from_timestamp/to_timestamp` 裁剪播放。

---

## 1. 交付物清单（最终会新增/修改哪些文件）

### 1.1 `vis/` 代码（P0/P1）

建议采用（轻量一体化）结构：

```
vis/
  __init__.py
  README.md
  requirements.txt
  web/
    __init__.py
    main.py
    api.py
    dataset_access.py
    cache.py
    qc.py
    templates/
      index.html
      dataset.html
      episode.html
    static/
      app.css
      app.js
      vendor/            # 可选：plotly.min.js / echarts.min.js
  scripts/
    __init__.py
    precompute.py
```

> 可选：如果你更偏“方案1”的工程拆分，可以把 `dataset_access.py` 拆成
`core/dataset_manager.py` 与 `core/video_streamer.py`，再把路由拆到 `api/` 包下。
本 plan 默认走轻量方案，避免文件过多影响迭代速度。

### 1.2 测试（建议 P1 引入）

```
tests/vis/
  test_dataset_root.py
  test_media_range.py
  test_timeseries_downsample.py
  test_path_traversal.py
```

### 1.3 文档（P0）

- `vis/README.md`：安装、启动、FAQ（Range/离线/远程端口转发）

---

## 2. Phase 划分与验收标准

### Phase 1（P0：能在浏览器刷完整数据集）

**必须具备：**

- 以 URL 方式打开：列表页 → episode 详情页
- 详情页多相机播放：自动 seek 到 `from_timestamp`，播放到 `to_timestamp` 自动停止
- 支持上一条/下一条/自动下一条（快捷键）
- 后端提供 mp4 静态服务并正确支持 HTTP Range（拖动进度条可用）

**验收：**

- 访问 `http://127.0.0.1:8000/` 能看到数据集/episode 列表
- 进入任意 episode，视频从正确片段开始播放，不串到其他 episode
- 拖动进度条不会重新下载整文件（浏览器网络面板能看到 Range/206）

### Phase 2（P1：QC 产品化 + 曲线）

**必须具备：**

- `timeseries` API（强制 `stride/downsample`）+ 前端曲线显示
- 预计算脚本：生成缩略图 + QC + summary
- 列表页可按 QC/任务/长度筛选

**验收：**

- 大 episode 不会因 JSON 过大导致前端卡死（默认点数限制生效）
- 列表页缩略图秒开（走缓存），异常 episode 有 badge 且可筛选

### Phase 3（P2：增强）

- 导出 Rerun `.rrd`，逐帧/标注，多数据集管理等按需实现

---

## 3. 后端（FastAPI）详细设计：模块与关键技术点

后端目标：**只读**访问 dataset root，提供：

- 页面（Jinja2 模板）
- JSON API（episodes / timeseries / qc / thumbnail）
- 媒体（mp4 Range）

### 3.1 `vis/web/main.py`（启动入口）

**职责：**

- 解析 CLI 参数：`--root`、`--repo-id`、`--host`、`--port`、`--cache-dir`、`--read-only`
- 创建 FastAPI app（调用 `vis.web.api:create_app(...)`）
- 启动 Uvicorn

**技术细节：**

- `--root` 既可传 dataset root，也可传 HF cache root（`~/.cache/huggingface/lerobot`）
- 默认 root：`lerobot.utils.constants.HF_LEROBOT_HOME`
- `--repo-id` 建议 P0 强制要求（先把“单数据集浏览”做好）；P1 再加 `/api/datasets` 自动发现

**推荐形态：**

- `python -m vis.web.main --root ... --repo-id user/repo --host 127.0.0.1 --port 8000`

### 3.2 `vis/web/dataset_access.py`（数据访问层）

这层是整个系统的“正确性核心”，负责把 v3.0 元数据变成前端可用的结构。

#### 3.2.1 dataset root 识别与发现

**函数建议：**

- `resolve_dataset_root(root: Path, repo_id: str | None) -> Path`
  - 若 `root/meta/info.json` 存在 → return root
  - 否则若 `repo_id` 且 `root/repo_id/meta/info.json` 存在 → return root/repo_id
  - 否则 raise（错误信息要把“期望的 meta/info.json 路径”打印出来）

- `discover_datasets(root: Path) -> list[DatasetRef]`（P1）
  - 扫描 `root/<user>/<repo>/meta/info.json`
  - 只要存在 `meta/info.json` 就认为是 dataset root

> 注意：`repo_id` 是 `user/repo`（两级目录）；不建议支持更深层路径，防止误扫描。

#### 3.2.2 元数据加载（复用 lerobot 工具函数）

**复用：**

- `lerobot.datasets.utils.load_info(local_dir)`
- `lerobot.datasets.utils.load_episodes(local_dir)`（会过滤掉 `stats/` 列，加载更快）
- `lerobot.datasets.utils.load_tasks(local_dir)`（可选）
- `lerobot.datasets.utils.load_stats(local_dir)`（可选）

**缓存：**

- 用 `functools.lru_cache` 缓存 `(dataset_root, file_mtime)` 粒度的结果，避免频繁 IO
- 但注意：episodes 数据集可能很大；缓存对象要控制 size，或只缓存“轻量摘要”（例如 episodes 的列裁剪结果）

#### 3.2.3 episode 列表（用于列表页）

**输入：**

- `offset/limit`（分页）
- `task`（按 tasks 字符串筛选）
- `min_len/max_len`（长度筛选）
- `has_issue`（结合 qc 缓存筛选）

**输出字段建议（每条 episode）：**

- `episode_index`
- `length`
- `duration_s = length / fps`
- `tasks`（列表）
- `thumbnail_url`（如果缓存有就给，没有就给空或占位）
- `qc_summary = {status, reasons}`（如果缓存有）

#### 3.2.4 episode 详情（视频裁剪信息：核心）

必须返回“按 video_key 的裁剪播放信息”，格式建议：

```json
{
  "repo_id": "user/repo",
  "episode_index": 12,
  "fps": 30,
  "videos": {
    "observation.images.camera_front": {
      "url": "/media/datasets/user/repo/videos/observation.images.camera_front/chunk-000/file-001.mp4",
      "from_timestamp": 123.456,
      "to_timestamp": 138.789,
      "duration_s": 15.333
    }
  }
}
```

**技术细节：**

- mp4 URL 用 `chunk_index/file_index` 做 canonical，利于浏览器缓存复用
- `from/to` 必须来自 episodes 元数据：`videos/<video_key>/from_timestamp` / `to_timestamp`
- `video_keys` 可从 `info["features"]` 推导：`dtype == "video"`

#### 3.2.5 timeseries API（列裁剪 + 下采样）

**目标：**输出曲线必须“可用且不卡”，默认点数要有上限。

**实现建议：**

- 读取：优先 `pyarrow.dataset`（对多 parquet 文件更友好）
  - `ds = pyarrow.dataset.dataset(dataset_root/"data", format="parquet")`
  - `table = ds.to_table(filter=field("episode_index") == ep, columns=[...])`
  - columns 至少含 `timestamp, frame_index`，外加请求 keys

- 下采样：
  - 支持显式 `stride`（例如 `stride=5`）
  - 若未提供 `stride`，根据 `max_points` 自动计算：
    - `stride = max(1, ceil(num_rows / max_points))`
  - `max_points` 默认建议 2000～5000（可配置）

- vector 维度裁剪（可选但强烈推荐）：
  - `dims` 参数解析规则：
    - `action:0-6,observation.state:0-13`
  - 对 list/np.ndarray 列做维度 slice，减少 JSON 体积

**输出建议：**

- 返回 `stride` 与 `num_points_raw/num_points_returned`，便于前端提示“已下采样”

**兼容性兜底：**

- v3.0 默认有 `episode_index`（见 `src/lerobot/datasets/utils.py:DEFAULT_FEATURES`）
- 如果遇到没有该列的数据（理论上不应发生），兜底方案：
  - 从 episodes 元数据取 `dataset_from_index/dataset_to_index`
  - 用 data 表里的 `index`（全局 index）做范围过滤：`from <= index < to`

### 3.3 `vis/web/api.py`（FastAPI 路由与 app 工厂）

建议用工厂函数：

- `create_app(root: Path, repo_id: str | None, cache_dir: Path) -> FastAPI`
  - app.state 挂载 `DatasetRegistry`/`DatasetAccess` 对象
  - 挂载静态 `/static`
  - 配置模板目录（Jinja2）

#### 3.3.1 页面路由

- `GET /`：数据集列表或默认数据集入口
- `GET /datasets/{user}/{repo}`：episode 列表页
- `GET /datasets/{user}/{repo}/episodes/{ep}`：episode 详情页

页面渲染只负责“首屏 HTML”，真正数据走 JSON API 拉取（更灵活）。

#### 3.3.2 JSON API 路由

（与最终方案一致，建议加版本前缀也可：`/api/v1/...`）

- `GET /api/datasets`（P1）
- `GET /api/datasets/{user}/{repo}/info`
- `GET /api/datasets/{user}/{repo}/episodes`
- `GET /api/datasets/{user}/{repo}/episodes/{ep}`
- `GET /api/datasets/{user}/{repo}/episodes/{ep}/timeseries`
- `GET /api/datasets/{user}/{repo}/episodes/{ep}/thumbnail`

#### 3.3.3 Media 路由：mp4 Range（最容易踩坑）

目标路由（canonical）：

- `GET /media/datasets/{user}/{repo}/videos/{video_key}/chunk-{chunk_index}/file-{file_index}.mp4`

**实现策略优先级：**

1) 优先使用 Starlette/FastAPI 内置 `FileResponse` 的 Range 支持（若版本支持）  
2) 若不支持或行为不一致，则实现手工 Range：
   - 解析 `Range: bytes=start-end`
   - 校验范围：`0 <= start <= end < file_size`
   - 返回 `StreamingResponse`（generator 每次读 1–4MB）
   - 头部：
     - `Accept-Ranges: bytes`
     - `Content-Range: bytes start-end/size`
     - `Content-Length: end-start+1`
     - `Content-Type: video/mp4`
   - 无 Range 则返回完整 `FileResponse`
   - 非法 Range 返回 `416 Range Not Satisfiable`

**安全要求：**

- 禁止目录穿越：只允许访问 `dataset_root/videos/...`
  - 生成真实路径后做 `path.resolve()`
  - 校验 `resolved_path.is_relative_to(dataset_root.resolve())`（py<3.9 可用手写）
- `video_key` 不能用于直接拼接任意路径：
  - 只允许 `.` `_` `-` 字母数字（或至少 reject `..` 和 `/`）
  - 或者不依赖 URL 中的 `video_key` 做路径拼接：从 episodes/info 推导允许集合，再比对

### 3.4 `vis/web/cache.py`（缓存与文件布局）

缓存目标：提升列表页体验（缩略图、QC、summary），避免每次启动/浏览都扫 parquet 或抽帧。

**建议缓存根：**

- 默认：`vis/.cache/`（在 repo 内，便于开发）
- 可配置：`--cache-dir /path/to/cache`（避免污染 repo）

**dataset_id 生成：**

- 推荐：`"{user}__{repo}"`（可读）
- 若 root 非标准 HF cache，可加短 hash：`"{user}__{repo}__{sha1(root)[:8]}"`

**文件建议：**

- `thumbs/episode-000123.jpg`
- `episodes_summary.json`
- `qc.json`

**API：**

- `get_cache_paths(dataset_id) -> CachePaths`
- `load_qc(dataset_id) -> dict[int, ...] | None`
- `load_summary(dataset_id) -> dict | None`
- `thumbnail_path(dataset_id, ep) -> Path`

注意并发：懒生成缩略图时，写文件要用临时文件 + rename（原子替换）。

### 3.5 `vis/web/qc.py`（质检规则）

把 `docs/train_agilex/数据检查指南.md` 抽象成“自动化可计算”的最小集合。

**建议实现的 QC 项（P1）：**

1) timestamp gap：
   - `expected = 1/fps`
   - `intervals = diff(timestamps)`
   - `gap_count = sum(intervals > 2*expected)`
2) NaN/Inf：
   - action/state 任意维度 `nan/inf` 计数
3) 静止段：
   - `diff(action)` 的绝对值 < `eps`（如 `1e-6`）判静止
   - 统计最长连续静止帧数（阈值如 60）
4) 视频可解码（可选，建议在 precompute 阶段做）：
   - `ffprobe -v error <video>` returncode 检查

**输出结构：**

- `QCResult = {status: "ok"|"warn"|"bad", reasons: [..], metrics: {...}}`
- 在列表页展示 `status` 与简短 reasons（最多 2–3 条，避免噪声）

---

## 4. 前端（模板 + 原生 JS）详细设计

前端核心目标：**快速刷完整数据集**，其次才是“深度分析”。

### 4.1 模板文件

- `templates/index.html`
  - 若 P0 只支持单 dataset：可直接重定向到 `/datasets/<user>/<repo>`
  - 若 P1 支持多 dataset：显示数据集列表（repo_id、episodes 数、fps 等）

- `templates/dataset.html`
  - 容器：筛选栏 + episode 列表（分页）+ 右侧预览（可选）
  - 列表项：缩略图、ep index、tasks、duration、QC badge

- `templates/episode.html`
  - 上方：多相机 video grid
  - 下方：曲线区（Plotly/Chart.js）
  - 右上角：上一条/下一条、自动下一条开关、导出 rrd（P2）

### 4.2 `static/app.js`（核心交互逻辑）

#### 4.2.1 列表页（dataset.html）

- 拉取 `GET /api/datasets/<user>/<repo>/episodes?...`
- 渲染列表项：
  - 缩略图：`<img src="/api/.../thumbnail?...">`（无则占位）
  - badge：`ok/warn/bad`
- 筛选：
  - task 下拉/搜索
  - has_issue toggle
  - min/max length
- 分页：
  - offset/limit 或 page/page_size

#### 4.2.2 详情页（episode.html）：裁剪播放 + 同步

**数据加载：**

- `GET /api/datasets/<user>/<repo>/episodes/<ep>` 获取 `videos` 字典
- 为每个 `video_key` 创建 `<video>`
  - `video.src = url`（整段 mp4）
  - 等 `loadedmetadata` 后：`video.currentTime = from_timestamp`

**播放边界：**

- 监听 master video 的 `timeupdate`：
  - 若 `currentTime >= to_timestamp - epsilon`：
    - `pause()`
    - 若开启 auto-next：跳转下一条 episode

**多相机同步策略（简单版）：**

- master: `t_rel = master.currentTime - from_master`
- 对其他相机：
  - `target = from + t_rel`
  - 若 `abs(video.currentTime - target) > 0.12`：`video.currentTime = target`
  - 只在播放状态下做校正，暂停时不抖动

> 注意：频繁强制 seek 会造成卡顿；阈值要大一点，且校正频率可用 `requestAnimationFrame`
或 `setInterval(250ms)`，不要每个 `timeupdate` 都校正。

**快捷键：**

- `J/←`：上一条（URL ep-1）
- `K/→`：下一条（URL ep+1）
- `Space`：播放/暂停 master（并同步其他相机）
- `A`：切换 auto-next

#### 4.2.3 曲线（P1）

- 拉取 `GET /api/.../timeseries?keys=...&stride=...`
- 绘制：
  - action/state 默认只画 1–3 维（或提供维度选择）
  - done/reward/success 画在单独 y 轴或子图
- 与视频联动：
  - 用 `timestamp` 对齐：当前 `t_rel` 对应曲线 x
  - 显示一个竖线指示当前播放位置（无需逐点高亮）

### 4.3 `static/app.css`（基础布局）

- 多相机 grid：`display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));`
- 列表页：左筛选 + 右列表，或单列卡片
- badge：ok/warn/bad 三色（不要依赖图片）

---

## 5. 预计算脚本（`vis/scripts/precompute.py`）详细设计

目标：把“每次浏览都要抽帧/跑 QC”变成一次性离线任务，输出缓存文件。

### 5.1 CLI 设计

- `python -m vis.scripts.precompute --root ... --repo-id ...`
- 参数：
  - `--thumb-video-key`（默认第一个 video_key）
  - `--thumb-width 320`
  - `--qc`（开关）
  - `--thumbs`（开关）
  - `--max-episodes`（调试用）
  - `--overwrite`

### 5.2 缩略图生成策略

每个 episode：

1) 从 episodes 元数据拿到：
   - 该 `video_key` 的 `chunk_index/file_index`
   - `from_timestamp`
2) 定位 mp4：`dataset_root / info["video_path"].format(...)`
3) 抽帧：
   - 优先 `ffmpeg`：
     - `ffmpeg -ss <from_timestamp> -i <mp4> -frames:v 1 -vf scale=<w>:-1 -q:v 2 <out>.jpg`
   - 失败则回退到 `lerobot.datasets.video_utils.decode_video_frames`
4) 输出到 `vis/.cache/<dataset_id>/thumbs/episode-XXXXXX.jpg`

> 抽帧不需要完全精确；关键是“快速 + 可用”。首帧略偏差可接受。

### 5.3 QC 计算策略

对每个 episode 读取必要列（建议不下采样或 stride=1）：

- `timestamp`
- `action`（若存在）
- `observation.state`（若存在）
- 可选：`next.reward/next.done/next.success`

计算：

- gap_count、nan_count、inf_count、max_static_run
- status：
  - bad：存在 nan/inf 或 gap_count>0 或 max_static_run 超阈值（阈值可配置）
  - warn：轻微异常（例如 gap_count 小于某阈值）
  - ok：无异常

写入：

- `qc.json`：`{ "<ep>": {status,reasons,metrics} }`

### 5.4 summary 输出

- `episodes_summary.json`：包含列表页需要的全部字段（避免启动后再拼）
  - 建议包含：`episode_index,length,duration_s,tasks,thumbnail_relpath,qc_status,qc_reasons`

---

## 6. 依赖与环境约定

### 6.1 `vis/requirements.txt`（建议）

最小集合（尽量复用仓库已有依赖）：

- `fastapi`
- `uvicorn`
- `jinja2`
- `pyarrow`（通常已是依赖，但可写上以保证）
- （可选）`orjson`（更快 JSON 序列化）

缩略图：

- 优先依赖系统 `ffmpeg/ffprobe`（不写进 requirements），并提供无 ffmpeg 的回退路径

### 6.2 离线与环境变量

- 运行与开发都在 `(my_lerobot)` conda 环境
- 支持 `HF_HUB_OFFLINE=1`
- 默认数据根：`HF_LEROBOT_HOME`（见 `src/lerobot/utils/constants.py`）

---

## 7. 安全与鲁棒性清单（必须做）

- media 路由防目录穿越（`resolve()` + root 校验）
- Range header 严格解析与 416 返回
- timeseries 强制 stride/max_points（避免 DoS/浏览器崩溃）
- API 参数校验（ep 越界、key 不存在、dims 非法）
- 不在服务端执行任何写数据到 dataset_root（只读）

---

## 8. 测试与验证（建议）

### 8.1 单元测试（P1）

- dataset root 识别：
  - 传 dataset root / 传 HF cache root+repo-id 两种
- path traversal：
  - `video_key=../...` 必须 400/404，不能读到 videos 之外
- Range：
  - 请求 `Range: bytes=0-1` 返回 206 且 Content-Range 正确
- timeseries downsample：
  - 默认 `max_points` 生效；stride 参数优先生效

### 8.2 手动验证脚本

- 启服务后：
  - 列表页翻页/筛选是否正确
  - episode 详情页拖动进度条是否顺滑（Range 生效）
  - 多相机是否大致同步（容忍轻微漂移）

---

## 9. 实施顺序建议（从“最容易跑通”到“最有价值”）

1) 后端：dataset root 识别 + episodes 详情 API（返回 from/to + canonical mp4 URL）
2) 后端：media Range（确保 `<video>` seek 可用）
3) 前端：episode 详情页裁剪播放 + 上/下一条
4) 后端：episodes 列表 API（分页）+ 模板列表页
5) precompute：缩略图（列表页体验）
6) timeseries：列裁剪 + stride + 曲线（P1）
7) QC：计算 + badge + 筛选（P1）

