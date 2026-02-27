# Web Collection 工作报告（2026-02-26）

## 1. 项目目标

本次工作目标是交付一条可用的数据采集与转换 pipeline：

1. 从 ROS 话题录制数据到 HDF5。
2. 将 HDF5 数据集目录（多个 `episode_*.hdf5`）合并转换为一个 LeRobot v3 数据集。
3. 提供简洁中文 Web UI，支持录制和整目录转换。
4. 录制流程加入语音提示与倒计时，支持批量录制与重置等待时间。

## 2. 交付结果

### 2.1 后端能力（`my_devs/web_collection`）

1. 录制服务 `record_hdf5.py`
   - 支持多相机 RGB、可选深度、双臂关节、底盘里程计写入 HDF5。
   - 支持批量录制：`num_episodes`。
   - 支持录制间隔重置时间：`reset_time_s`。
   - 支持保存语言指令：`instruction`（VLA 需要）。
   - 数据命名自动递增：`episode_000000.hdf5`、`episode_000001.hdf5`。

2. 转换服务
   - 单文件转换脚本：`convert_to_lerobot.py`（保留，前端不再使用）。
   - 整目录合并转换脚本：`convert_dataset_to_lerobot.py`。
   - 整目录模式会读取目录下全部 `episode_*.hdf5`，逐条 `save_episode()`，最终 `finalize()` 为一个 LeRobot 数据集。
   - 安全保护：禁止将 `output_dir` 设为 `input_dataset_dir` 内部，避免误操作原始数据。
   - 修复了整目录转换脚本中的缩进问题，确保脚本可执行。

3. Web API（`app.py`）
   - 录制：`/api/record/start`、`/api/record/stop`。
   - 转换：`/api/convert_dataset/start`、`/api/convert/stop`。
   - 状态：`/api/status`。
   - 设置：`/api/settings`（返回 UI 默认路径、任务名、声音配置）。
   - 声音：`/api/sound/play`（服务端播放）。

4. 声音系统（`sound_player.py`）
   - 按用户要求使用 `pygame` 在服务端播放音频。
   - 录制前：播放 `ready.wav`，前端显示 3-2-1 倒计时。
   - 每条开始：检测日志 `[episode]` 播放 `start_record.wav`。
   - 条目间重置：检测日志 `[reset]` 播放 `reset_env.wav`。
   - 全部完成：任务结束播放 `finish.wav`。
   - 声音文件路径来自配置，可在 `app.yaml` 修改。

### 2.2 前端 UI（`my_devs/web_collection/static`）

1. 录制页
   - 中文字段。
   - 支持录制条数（episode 数）与重置等待秒数。
   - 支持 VLA 指令输入。
   - 黑白简洁风格。

2. 转换页（按最新要求已调整）
   - 已移除“单个 HDF5 选择/转换”入口。
   - 固定为“整目录合并转换”模式。
   - 输入一个任务目录，例如：
     `/home/agilex/cqy/flexible_lerobot/datasets/hdf5_datasets/aloha_mobile_dummy`
   - 一键转换该目录下所有 `episode_*.hdf5`。

### 2.3 配置文件

1. 录制配置：`my_devs/web_collection/configs/default.yaml`
   - 采样 `fps`、`max_frames`、`reset_time_s`、话题名映射等。

2. 应用配置：`my_devs/web_collection/configs/app.yaml`
   - `defaults.dataset_dir`
   - `defaults.lerobot_output_dir`
   - `defaults.task_name`
   - `defaults.repo_id`
   - `sounds.enabled/volume/files`

## 3. 关键流程说明

### 3.1 录制流程

1. 点击“开始录制”。
2. 服务端播放“准备开始”，前端显示 3-2-1。
3. 启动录制子进程，进入 episode 录制循环。
4. 若设置 `num_episodes > 1`，每条完成后按 `reset_time_s` 等待并提示重置环境。
5. 全部录制完成后播放“结束录制”。

### 3.2 整目录转换流程

1. 输入任务目录（包含多个 `episode_*.hdf5`）。
2. 点击“开始转换”。
3. 后端调用 `convert_dataset_to_lerobot.py`。
4. 按 episode 逐条写入同一 LeRobot 数据集并 finalize。
5. 输出到 `lerobot_output_dir` 下新目录，不改动原始 HDF5。

## 4. 验证记录

1. 语法检查通过：
   - `conda run -n lerobot_flex python -m py_compile my_devs/web_collection/app.py my_devs/web_collection/jobs.py my_devs/web_collection/record_hdf5.py my_devs/web_collection/convert_to_lerobot.py my_devs/web_collection/convert_dataset_to_lerobot.py my_devs/web_collection/sound_player.py`

2. 环境版本确认：
   - `pygame 2.6.1`
   - `numpy 2.2.6`

3. 功能验证结论：
   - 录制/转换 API 通路可用。
   - 前端已切换为整目录转换模式。
   - 录制与转换日志可在页面实时查看。

## 5. 环境变更记录

为支持服务端音频播放，已在 `lerobot_flex` 环境安装 pygame（通过 `.[pygame-dep]`）。

当前环境中 `numpy` 为 `2.2.6`。如后续遇到第三方依赖兼容问题，需要结合具体报错再做版本回退或锁定。

## 6. 已知风险与注意事项

1. 声音播放依赖主机音频设备与权限。
   - 若系统层无可用音频设备或权限不足，服务仍可运行，但声音会被禁用并在启动日志提示。

2. ROS 话题准备要求
   - 录制开始前需要对应话题已发布，否则会等待并可能超时。

3. 批量录制约束
   - 当 `num_episodes > 1` 时，需要 `max_frames > 0` 才能自动分条录制。

## 7. 当前建议使用方式

1. 启动服务：

```bash
cd /home/agilex/cqy/flexible_lerobot
conda run -n lerobot_flex --no-capture-output uvicorn my_devs.web_collection.app:app --host 0.0.0.0 --port 8008 --log-level info
```

2. 浏览器访问：
   - `http://<机器IP>:8008/`

3. 转换整目录：
   - 在“任务目录（整目录合并转换）”填写：
     `/home/agilex/cqy/flexible_lerobot/datasets/hdf5_datasets/aloha_mobile_dummy`
   - 点击“开始转换”。

