# Web Collection 最终技术方案（简版）

目标：在真实机器人 ROS 系统上，形成一条稳定可重复的闭环：

1. **Web UI 一键录制**（ROS topics -> 单个 episode `.hdf5`）
2. **Web UI 一键转换**（`.hdf5` -> LeRobot v3 数据集目录）
3. **转换结果可直接被 LeRobot 训练/加载代码读取**

本方案重点保证 pipeline 可用、可调试、可扩展（后续接入更多 topic/“60 节点”也不推翻架构），UI 保持极简。

## 1. 数据结构与接口约定

### 1.1 录制输出：HDF5（episode 粒度）

每次录制输出一个文件：`{dataset_dir}/{task_name}/{episode_name}.hdf5`。

核心键（与现有 ALOHA 风格兼容）：
- `observations/images/<cam>`: `(T, H, W, 3)` `uint8`，统一保存为 RGB
- `observations/qpos`: `(T, D)` `float32`
- `observations/qvel`: `(T, D)` `float32`（缺失则写 0）
- `observations/effort`: `(T, D)` `float32`（缺失则写 0）
- `action`: `(T, A)` `float32`
- `base_action`: `(T, 2)` `float32`（无里程计则写 0）
- `timestamp`: `(T,)` `float64`

元信息（attrs）：
- `fps`, `task_name`, `episode_name`, `instruction`, `config_yaml`, `schema=web_collection_hdf5_v1`

### 1.2 转换输出：LeRobot v3 数据集

转换产生一个 LeRobot 数据集根目录（不覆盖已有目录，自动生成唯一输出目录）：
- `meta/info.json`, `meta/tasks.parquet`, `meta/episodes/...`
- `data/.../*.parquet`
- `videos/.../*.mp4`（若勾选“Encode videos”）

关键 feature（最小可训练集合）：
- `observation.state`: `float32` `(D,)`
- `action`: `float32` `(A,)`
- `observation.images.<cam>`: `image` 或 `video`
- `observation.environment_state`: `float32` `(2,)`（base）
- `language_instruction`: `string`（可选，但当前默认写入）

此外每帧必须包含 LeRobot 约定的 `task` 字段：
- 本方案将 `instruction` 同时写入 `task`（用于 task-conditioned 训练）

## 2. 系统组件

### 2.1 Recorder（ROS -> HDF5）

实现：`my_devs/web_collection/record_hdf5.py`
- 订阅配置中列出的 topics（RGB/Depth/Image、JointState、Odometry）
- 以固定 `fps` 采样“最新消息”（静止录制也能稳定落盘）
- 捕获 `SIGINT/SIGTERM`，**随时 stop 都会保存已录到的部分**
- 支持一次录制多条 episode：`--num_episodes N`，并在每条之间等待 `--reset_time_s` 秒用于“重置环境”

### 2.2 Converter（HDF5 -> LeRobot）

实现：`my_devs/web_collection/convert_to_lerobot.py`
- 逐帧从 HDF5 读取并写入 `LeRobotDataset.add_frame()`
- `task` 使用 `instruction`（如果为空则用 `task_name`）
- 支持 `--swap_rb` 修正通道顺序（避免 BGR/RGB 误差）
- 转换过程对输入 HDF5 是**只读**的：不会修改、移动或删除原始数据文件

### 2.3 Web UI（极简）

实现：`my_devs/web_collection/app.py` + `my_devs/web_collection/static/*`
- 单页 UI：Record / Convert 两块
- 后端用 subprocess 启动 recorder/converter，提供 stop，展示日志
- 不做复杂 job 队列（先把闭环打通）

## 3. 面向“VLA/语言指令”的处理

问题：录制回放本身不带指令，但 VLA 训练需要指令。

方案：
- UI 提供 `Instruction/Task` 输入框
- 写入 HDF5 attrs：`instruction`
- 转换时：
  - `frame["task"] = instruction`
  - `frame["language_instruction"] = instruction`

这样即使录制时机器人静止，仍能验证“带语言字段的数据集 pipeline”可用。

## 4. 当前实现与下一步扩展

已实现：
- `my_devs/web_collection` 下完整可运行代码与默认 config
- UI、录制、转换闭环（在 ROS master/topic 可用时）

建议下一步（不阻塞当前闭环）：
1. 增加“topic 自动发现/选择”（读取 ROS graph 展示 topic 列表），方便接入“60 节点”
2. 统一时序对齐策略（最新采样 -> 近似同步 -> 严格同步可选）
3. 支持更多 msg 类型（例如 `Float64MultiArray`、自定义状态），并落到 `observation.*` 的结构化 feature
