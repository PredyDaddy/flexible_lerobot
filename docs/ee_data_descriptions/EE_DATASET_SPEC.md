# EE Pose LeRobot Dataset v3.0 规范（AgileX + Pinocchio/Placo）

> 目标：给“拿到数据要训练/解析”的同学一份**以仓库实现为准**的规格说明，避免仅凭经验/旧文档导致的时序错位、视频对齐错误、EE 坐标系误解等问题。

本规范覆盖：
- LeRobot Dataset `v3.0` 的磁盘组织与索引（`meta/episodes`、chunk/file 切分、mp4 串联 episode）。
- AgileX 机器人数据的 `action` / `observation.state` 的真实语义（以 `lerobot-record` 录制实现为准）。
- `record_ee_pose=true` 时 EE pose 的字段、单位、计算方式（Pinocchio / Placo）。

---

## 0. TL;DR（最容易踩坑的 5 件事）

1. **v3.0 不是“每个 episode 一个文件”**：一个 `data/chunk-xxx/file-yyy.parquet` / `videos/.../file-yyy.mp4`
   通常包含多个 episode。episode 的边界必须看 `meta/episodes/*/*.parquet`。
2. **视频对齐靠 `from_timestamp + timestamp`**：`LeRobotDataset` 解码视频帧使用
   `video_timestamp = videos/<video_key>/from_timestamp + timestamp`。
3. **`timestamp` 默认是 `frame_index / fps`**：如果录制时未提供真实时间戳，会自动生成等间隔离散时间。
4. **`action` 在录制时优先取“执行后观测 next_obs 里能抽出来的关节值”**：
   对 AgileX 来说，这通常是 follower 实际关节位置（接近“下一时刻关节位置”），不是纯粹的“下发目标”。
5. **Pinocchio EE pose 不是直接用 URDF 里现成的末端 link**：
   实现会在模型里新增 `ee` frame（挂到 `joint6`，带 `ee_rotation_rpy/ee_translation`）。

---

## 1. 推荐读取方式（强烈建议）

优先用 `LeRobotDataset`，它会自动处理：
- parquet 分片拼接；
- mp4 串联 episode 的时间片段定位；
- 视频解码并返回 `float32`、`[0,1]`、channel-first 的张量；
- `task_index -> task` 文本映射。

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

repo_id = "cqy/agilex_vla_demo_ee_pinocchio_apple_grasp"
ds = LeRobotDataset(repo_id)  # 默认 root = HF_LEROBOT_HOME / repo_id

sample = ds[0]
print(sample["observation.state"].shape, sample["action"].shape)
print("state names:", ds.meta.features["observation.state"]["names"])
print("action names:", ds.meta.features["action"]["names"])
print("task:", sample["task"])
```

### 1.1 数据集根目录 `root` 的定义（很重要）

在本仓库实现中，`LeRobotDataset(repo_id, root=...)` 的 `root` 指向**数据集根目录本身**，
也就是包含 `meta/ data/ videos/` 的那个目录；默认路径是：

- `HF_LEROBOT_HOME`（默认 `~/.cache/huggingface/lerobot`）下的 `${HF_LEROBOT_HOME}/${repo_id}`

对应实现见：`src/lerobot/utils/constants.py`、`src/lerobot/datasets/lerobot_dataset.py`。

---

## 2. v3.0 目录结构（规范 + 切分原则）

数据集根目录 `DATASET_ROOT/`：

```
DATASET_ROOT/
├── meta/
│   ├── info.json
│   ├── stats.json
│   ├── tasks.parquet
│   └── episodes/
│       └── chunk-000/
│           ├── file-000.parquet
│           └── ...
├── data/
│   └── chunk-000/
│       ├── file-000.parquet
│       └── ...
└── videos/
    ├── observation.images.<camera_key>/
    │   └── chunk-000/
    │       ├── file-000.mp4
    │       └── ...
    └── ...
```

### 2.1 chunk / file 的含义（不要误解）

- chunk 目录与 file 文件是**为了控制文件数量与单文件大小**而切分的：
  - `chunks_size`：一个 chunk 目录下最多 `N` 个 file（默认 1000）。
  - `data_files_size_in_mb`：parquet 文件超过阈值会滚动到新 file。
  - `video_files_size_in_mb`：mp4 文件超过阈值会滚动到新 file。
- 因此一个 `file-xxx.parquet`/`file-xxx.mp4` 通常包含多个 episode（串联存储）。

这些规则由 `src/lerobot/datasets/lerobot_dataset.py` 的 `_save_episode_data` / `_save_episode_video`
实现。

---

## 3. `meta/info.json`：最权威的 schema 描述

`meta/info.json` 是数据集 schema 的“单一事实来源”，至少需要理解这些字段：

- `codebase_version`：本仓库期望为 `v3.0`
- `robot_type`：例如 `agilex`
- `fps`、`chunks_size`、`data_files_size_in_mb`、`video_files_size_in_mb`
- `data_path`、`video_path`：路径模板（v3.0 固定为 `chunk-{xxx}/file-{yyy}` 风格）
- `features`：关键模态与其 `dtype/shape/names`（训练/解析以此为准）

### 3.1 特征的生成方式（为什么 names 的顺序很重要）

本仓库录制时的特征来自 `robot.observation_features` 与 `robot.action_features`，经 pipeline 处理后写入
`features`：

- 1D float 观测会被合并成 `observation.state` 向量；
- 1D float action 会被合并成 `action` 向量；
- 相机图像会变成 `observation.images.<camera_key>`（默认存为 `video`，mp4）。

**`names` 的顺序 = Python dict 的插入顺序**（AgileX 的实现是先 left 再 right，再 EE，再 cameras）。
所以解析时应优先按 `names` 查索引，而不是硬编码切片（除非你明确固定了 schema）。

相关实现见：`src/lerobot/robots/agilex/agilex.py`、`src/lerobot/datasets/utils.py`。

---

## 4. `data/*/*.parquet`：逐帧数值数据（不含图像帧）

### 4.1 典型列结构（v3.0 默认录制）

在默认录制脚本（`lerobot-record`）下，parquet 里通常包含：

| 列名 | 含义 |
|---|---|
| `observation.state` | float32 向量：关节/末端位姿等低维状态 |
| `action` | float32 向量：动作标签（注意其语义，见第 7 节） |
| `timestamp` | 每帧时间戳（默认 `frame_index / fps`） |
| `frame_index` | episode 内帧号，从 0 开始 |
| `episode_index` | episode 编号，从 0 开始 |
| `index` | 全局帧号，从 0 开始，跨 episode 递增 |
| `task_index` | 用于映射到 `meta/tasks.parquet` 的任务文本 |

### 4.2 `timestamp` 的默认生成规则

如果写入 frame 时没有显式提供 `timestamp`，实现会自动用：

```
timestamp = frame_index / fps
```

见 `src/lerobot/datasets/lerobot_dataset.py` 的 `add_frame()`。

---

## 5. `meta/episodes/*/*.parquet`：episode 定位索引（训练/手工解析必读）

`meta/episodes` 每行一个 episode，回答这些问题：

- 这个 episode 有多少帧？（`length`）
- 它对应全局帧号区间？（`dataset_from_index`、`dataset_to_index`）
- 它落在哪个 data parquet 分片？（`data/chunk_index`、`data/file_index`）
- 它落在哪个 mp4 文件？并在该 mp4 的哪个时间片段？（`videos/<video_key>/from_timestamp`、`to_timestamp`）

### 5.1 `dataset_from_index/to_index` 是半开区间

实现中 `dataset_to_index = dataset_from_index + episode_length`，因此区间语义为：

```
[dataset_from_index, dataset_to_index)
```

`dataset_to_index` 是**排他的**。

---

## 6. `videos/<video_key>/*/*.mp4`：视频数据与对齐方式

### 6.1 为什么需要 `from_timestamp/to_timestamp`

v3.0 会把多个 episode 的视频帧按时间顺序拼到同一个 mp4 里减少文件数。
因此某个 episode 的视频起止必须用 `meta/episodes` 提供的时间片段定位。

### 6.2 精确对齐公式（以实现为准）

对于某个样本（帧）：

```
video_timestamp = from_timestamp + timestamp
```

其中 `from_timestamp` 来自当前 episode 在 `meta/episodes` 对应行里的
`videos/<video_key>/from_timestamp`，`timestamp` 来自 parquet 里的 `timestamp`。

对应实现见 `src/lerobot/datasets/lerobot_dataset.py` 的 `_query_videos()`。

### 6.3 编码/解码注意事项（默认 AV1）

- 默认编码：`libsvtav1`（AV1），像素格式通常为 `yuv420p`（见 `src/lerobot/datasets/video_utils.py`）。
- 解码后张量：`float32`、范围 `[0,1]`。
- 解码后布局：`LeRobotDataset` 返回单帧时通常是 `(3, H, W)`（channel-first）。

---

## 7. `action` 的录制语义（必须读，否则训练容易“时序错位”）

### 7.1 录制循环的真实时序（核心）

`lerobot-record` 的控制循环逻辑（简化）是：

1. 读取当前观测 `obs`（初始为 `robot.get_observation()`，后续复用上一次 `next_obs`）。
2. 构造 `observation_frame = build_dataset_frame(..., obs_processed, prefix="observation")`。
3. 计算并下发动作：`sent_action = robot.send_action(robot_action_to_send)`（内部可能做裁剪/限幅）。
4. 再读一次观测：`next_obs = robot.get_observation()`。
5. 尝试从 `next_obs` 中**提取**与 `action.names` 同名的键，作为 `executed_action_values`；
   如果提取失败才 fallback 到 `sent_action`。
6. 将当前帧写入 dataset：`frame = {observation_frame, action_frame, task}`。

对应实现见 `src/lerobot/scripts/lerobot_record.py`。

### 7.2 对 AgileX 的含义（强制理解）

AgileX 的 `get_observation()` 里的关节值来自 follower/puppet 真实关节状态。
因此在典型配置下：

- `observation.state` 对应 **obs(t)** 的 follower 关节 +（可选）EE pose +（可选）视频帧；
- `action` 对应 **next_obs(t+1)** 里抽取出来的 follower 关节位置（更像“下一步实际到达的关节位置”）。

这在“关节位置控制”的模仿学习里是合理的（预测下一步关节位置/目标），但如果你在做严格的
`obs(t) -> command(t)` 对齐，请务必自行确认/重对齐。

---

## 8. `observation.state` 的构成与索引规则（AgileX）

### 8.1 严格规则：以 `meta/info.json -> features["observation.state"]["names"]` 为准

不要硬编码索引。即使你认为“肯定是前 14 维 joint、后 12 维 EE”，也建议在代码里通过 `names`
查找字段位置。

### 8.2 AgileX 的常见顺序（由 `AgileXRobot.observation_features` 的插入顺序决定）

AgileX 的观测 key 插入顺序是：

1. 左臂 7 个 joint：`left_<joint>.pos`
2. 右臂 7 个 joint：`right_<joint>.pos`
3. （可选）左 EE 6 个量：`left_ee.*`
4. （可选）右 EE 6 个量：`right_ee.*`
5. 相机图像（不进入 `observation.state`，而是 `observation.images.<camera_key>`）

对应实现见 `src/lerobot/robots/agilex/agilex.py`。

### 8.3 单位约定（以实现与配置为准）

- joint（除 gripper）：通常为弧度 `rad`
- gripper：在 AgileX 配置里是行程，通常为米 `m`
- EE position：米 `m`
- EE rotation：
  - Pinocchio：`roll/pitch/yaw`（`rad`）
  - Placo：`wx/wy/wz`（rotation vector，轴角向量，单位 `rad`）

---

## 9. EE pose 的定义（`record_ee_pose=true`）

### 9.1 开关与字段

- 开关：`AgileXConfig.record_ee_pose=true`
- 后端：`AgileXConfig.kinematics_backend in {"pinocchio","placo"}`
- 字段：
  - pinocchio：`{x,y,z,roll,pitch,yaw}`
  - placo：`{x,y,z,wx,wy,wz}`

对应实现见 `src/lerobot/robots/agilex/agilex.py`。

### 9.2 Pinocchio 后端（RPY）

#### FK 输入（ROS -> URDF 的映射）

AgileX 读到的 follower 关节状态是 ROS topic 的关节角，FK 使用前 6 个关节（不含 gripper）：

- `q_ros = puppet_state.position[:6]`
- 映射：`q_urdf = q_ros * signs + offsets`
  - `signs/offsets` 来自配置 `left_joint_signs/right_joint_signs` 与 `left_joint_offsets_rad/right_joint_offsets_rad`

#### EE frame 的定义（不是直接用 URDF 的末端 link）

Pinocchio FK 实现会：
- 从 URDF 构建模型，并锁定夹爪关节（默认 `joint7/joint8`）。
- 在模型里新增一个名为 `ee` 的 frame，挂到 `joint6` 上，并叠加：
  - `ee_rotation_rpy`（rad）
  - `ee_translation`（m）

然后输出 `ee` frame 的 `[x,y,z,roll,pitch,yaw]`（ZYX 欧拉角提取）。

对应实现见 `my_ik/piper_pinocchio_kinematics.py`。

#### TCP offset（可选）

AgileX 在 Pinocchio FK 结果上还支持叠加 TCP offset：

- `left_tcp_offset_xyzrpy/right_tcp_offset_xyzrpy`（m/rad）
- 实现为 4x4 矩阵右乘，然后再转回 xyzrpy。

#### 参考坐标系（相对谁）

Pinocchio 输出是相对 URDF 根坐标系的位姿。以本仓库默认 URDF
`my_sim/piper_ros-noetic/src/piper_description/urdf/piper_description.urdf` 为例：

- 根 link 为 `dummy_link`
- `dummy_link -> base_link` 是零位姿固定关节

因此可将 EE pose 理解为“相对 base_link（与 dummy_link 同位姿）”。

> 重要限制：当前实现对左右臂各用一套同构 URDF 做 FK，**不包含两臂基座之间的外参**。
> 因此如果你要把左右臂 EE 放到同一全局坐标系，需要自行提供外参标定/定义。

### 9.3 Placo 后端（RotVec）

Placo FK 的关键差异：
- 输入单位是 **degrees**（实现会从 rad 转 deg）。
- `RobotKinematics.forward_kinematics()` 输出是原始 4x4 变换矩阵。
- rotation vector（`wx,wy,wz`）的提取在 `AgileXRobot.get_observation()` 中完成
  （见 `src/lerobot/robots/agilex/agilex.py`），而非在 FK 函数内部。
- target frame 来自 `AgileXConfig.kinematics_target_frame`（默认 `gripper_base`），关节名来自
  `AgileXConfig.kinematics_joint_names`（默认 `joint1..joint6`）。

---

## 10. `meta/stats.json`：统计与归一化（实现约定）

- 统计由录制时每个 episode 计算后再聚合得到（实现：`src/lerobot/datasets/compute_stats.py`）。
- 对数值向量（如 `action`、`observation.state`）：
  - `min/max/mean/std/q01/q10/q50/q90/q99` 都是**逐维**统计。
- 对 image/video：
  - 会对采样帧做 per-channel 统计，并归一化到 `[0,1]`（shape 常见为 `(3,1,1)`）。

> 建议：规范文档里不要手抄固定统计数值；如需范围说明，应以实际 `meta/stats.json` 为准。

---

## 11. 不依赖 `LeRobotDataset` 的手工解析流程（可选）

如果你要在别的训练框架里手工读取：

1. 读 `meta/info.json`：拿到 `fps`、`features`、`data_path`、`video_path`。
2. 读 `meta/episodes/*/*.parquet`：拿到每个 episode 的 `dataset_from_index/to_index` 与视频时间片段索引。
3. 读 `data/*/*.parquet`：按行取 `observation.state/action/timestamp/...`。
4. 解码 `videos/<video_key>/*/*.mp4`：用 `from_timestamp + timestamp` 取帧（注意 tolerance）。

核心难点就是“一个 mp4 里串了多个 episode”，因此第 2 步是硬要求。

---

## 12. 常见坑与自检清单

- schema 不一致：`record_ee_pose=false`（14 维）与 `true`（多 12 维）不要混录/混训。
- backend 不一致：Pinocchio 是 `roll/pitch/yaw`，Placo 是 `wx/wy/wz`；以 `names` 判断最稳。
- 视频对齐错误：不要用“episode 内 frame_index”直接当 mp4 时间；必须加 `from_timestamp`。
- AV1 解码问题：离线环境若不支持 AV1，建议使用 `LeRobotDataset`（或准备对应 ffmpeg/pyav 支持）。
- 欧拉角差分问题：`roll/pitch/yaw` 在 pitch 接近 ±90° 会奇异；几何运算建议转矩阵/四元数。
- 左右臂坐标系：当前实现没有两臂基座外参，跨臂几何约束需要你自己定义外参。

---

## 13. 数据集 README（推荐模板，可直接复制）

> 你可以把这一节复制到每个数据集的 README 或单独的 `DATASET_CARD.md` 里。

### 13.1 基本信息
- `repo_id`：
- `codebase_version`：
- `robot_type`：
- `fps`：
- `episode_time_s`：
- `total_episodes/total_frames`：
- `task(s)`（来自 `meta/tasks.parquet`）：

### 13.2 录制配置（与 schema 强相关）
- `record_ee_pose`：
- `kinematics_backend`：
- `kinematics_urdf_path`：
- `pinocchio_package_dirs` / `pinocchio_lock_joints`（若使用 pinocchio）：
- `ee_rotation_rpy` / `ee_translation`：
- `left/right_joint_signs`、`left/right_joint_offsets_rad`（若有改动必须写清）：
- `left/right_tcp_offset_xyzrpy`（若有改动必须写清）：
- 相机 `camera_key -> topic/分辨率/fps`：

### 13.3 特征（以 `meta/info.json` 为准）
- `observation.state`：shape、names、单位说明
- `action`：shape、names、单位说明 + **action 语义（next_obs 抽取）说明**
- `observation.images.*`：相机 keys、shape、codec、对齐方式（from_timestamp + timestamp）

### 13.4 已知限制
- mp4 串联 episode（手工解析必须使用 `meta/episodes`）
- 左右臂 EE 坐标系未对齐（无基座外参）

