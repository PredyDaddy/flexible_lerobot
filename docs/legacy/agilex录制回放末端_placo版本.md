# AgileX 末端(EE)录制与回放说明

> **兼容性说明 (lerobot 0.4.3 baseline)**  
> 本文档依赖旧仓库中的 AgileX/末端 IK 回放实现（例如 `src/lerobot/robots/agilex/`、`src/lerobot/scripts/lerobot_replay_ee.py`、以及相关录制参数与可视化脚本）。  
> 当前仓库的 0.4.3 baseline 默认代码树不包含这些内容，因此本文不能直接照着跑；建议仅作为历史记录/移植参考（路线 B）。  

本文档总结旧仓库里 **AgileX Piper 双臂**“末端位姿（EE pose）录制 + IK 回放”的完整工作流：怎么录、怎么放、代码里到底录了什么/放了什么，以及如何用可视化脚本做离线核验。

> 约定：这里的 “EE pose” 指 `x,y,z,wx,wy,wz`（其中 `w*` 为 **rotvec 旋转向量**，单位 rad），对应 `observation.state` 中的 `left_ee.* / right_ee.*` 字段。

---

## 1. 前置条件与依赖

### 1.1 Python/环境

- 所有 Python 相关命令建议在 conda 环境 `my_lerobot` 中运行：
  - `conda activate my_lerobot`
- 运行本仓库脚本建议加上：
  - `PYTHONPATH=src:$PYTHONPATH`

### 1.2 IK/FK 依赖（placo）

末端录制（FK）和末端回放（IK）都依赖 `placo`（`RobotKinematics` 后端）：

- 录制端：`--robot.record_ee_pose=true` 会在 `get_observation()` 中做 FK 并写入 dataset。
- 回放端：`lerobot_replay_ee.py` 每帧做 IK 得到关节目标。

安装方式（任选其一，按你的环境习惯）：

- `pip install -e ".[kinematics]"`
- 或者确保 `placo` 已可被 `import placo`。

### 1.3 可视化依赖（meshcat + trimesh）

用于 `my_sim/visualize_dataset_right_ee.py` 和 `my_sim/visualize_urdf.py`：

- `pip install meshcat trimesh`

### 1.4 ROS 环境（硬件回放）

`replay_ee.sh` 已包含：

- `source /opt/ros/noetic/setup.bash`

如果你在录制时也用到 ROS camera/rospy，建议同样确保 ROS 环境已 source（具体依赖取决于你机器的 ROS 安装方式）。

---

## 2. 录制 EE：record_ee.sh 做了什么

### 2.1 如何运行

脚本：`record_ee.sh`

- 直接运行：`bash record_ee.sh`
- 关键开关：
  - `--robot.record_ee_pose=true`
  - `--robot.kinematics_urdf_path=.../piper_description.urdf`
  - `--robot.kinematics_target_frame=gripper_base`

注意：

- 一旦开启 `record_ee_pose`，**observation schema 会改变**（多了 `*_ee.*`），不要在旧数据集上 `--resume=true` 继续写入（脚本里也有提醒）。

### 2.2 数据到底录了什么（避免“其实录的是关节角”这类担心）

录制的关键链路如下：

1. **机器人观测（observation.state）来自从臂/puppet（follower）关节**  
   - `AgileXRobot.get_observation()` 读取的是 `get_puppet_state()`  
   - 对应代码：`src/lerobot/robots/agilex/agilex.py:426`

2. **开启 `record_ee_pose` 后，EE 是“由从臂关节做 FK 计算出来的”并写入 observation**  
   - FK 输入：从臂当前 6 关节（ROS 角度）+ `signs/offsets` 映射到 URDF  
   - FK 输出：`T_world_frame(target_frame)` 再乘 TCP offset（默认全 0）  
   - 写入的字段：`left_ee.{x,y,z,wx,wy,wz}` / `right_ee.{x,y,z,wx,wy,wz}`  
   - 对应代码：`src/lerobot/robots/agilex/agilex.py:437`

3. **dataset 的 `action` 不是 master 臂角度，而是“执行后的从臂关节位置”**  
   - 录制循环在 `send_action()` 之后会再次读 `next_obs`，并从 observation 中抽取关节作为 `action` 写入（相当于“实际执行到的位置”）。  
   - 对应代码：`src/lerobot/scripts/lerobot_record.py:391`

总结一句：  
**你这个 EE 数据是“从臂关节 → FK 计算的末端位姿”，不是随便填的，也不是主臂角度冒充的。**

---

## 3. 回放 EE：replay_ee.sh 做了什么

### 3.1 如何运行

脚本：`replay_ee.sh`

- 直接运行：`bash replay_ee.sh`
- 关键开关：
  - `python -m lerobot.scripts.lerobot_replay_ee`
  - `--dataset.repo_id=...`
  - `--dataset.episode=...`
  - `--dataset.fps=30`
  - `--robot.max_relative_target=0.05`（你后来调到 `0.2` 会更“跟得上”）
  - `--ik.position_weight=1.0`
  - `--ik.orientation_weight=0.01`

脚本里还加了一个实用设置：

- `HF_DATASETS_CACHE=/tmp/hf_datasets_cache`  
  用于规避某些环境下 huggingface datasets cache 的锁/权限问题。

### 3.2 回放逻辑（确认 “确实在做 IK”，而不是回放关节角）

`lerobot_replay_ee.py` 的核心逻辑是：

1. 每帧读取 `observation.state`（不是 `action`）  
   - `state = ds.hf_dataset[k]["observation.state"]`  
   - 对应代码：`src/lerobot/scripts/lerobot_replay_ee.py:292`

2. 从 `state` 里取出 `*_ee.*` 构造目标位姿 `T_target`  
   - 对应代码：`src/lerobot/scripts/lerobot_replay_ee.py:299`

3. 调用 IK 求 6 关节（placo 求解）  
   - `kin_right.inverse_kinematics(...)`  
   - 对应代码：`src/lerobot/scripts/lerobot_replay_ee.py:320`

4. 将 URDF 关节映射回 ROS 关节并下发 `robot.send_action()`  
   - 对应代码：`src/lerobot/scripts/lerobot_replay_ee.py:342`

也就是说：  
**replay_ee 回放的是“末端轨迹”，每一帧都实时求 IK 得到关节目标；不是把数据集里的关节 action 直接发回去。**

---

## 4. 为什么 replay_ee 会更“丝滑”，以及 `max_relative_target` 的影响

你实际观察到的两个现象很典型：

1. **更丝滑/抖动被“过滤”**  
   - `max_relative_target` 在 `send_action()` 内部会按当前关节状态对“单步关节变化”做裁剪，相当于一个强限速器/低通效果。  
   - 对应代码：`src/lerobot/robots/agilex/agilex.py:523`、`src/lerobot/robots/agilex/agilex.py:558`
   - IK 还会用上一帧解作为初值（warm start），也天然倾向解的连续性（`src/lerobot/scripts/lerobot_replay_ee.py:274`）。

2. **“放不完/追不上”，把 `max_relative_target` 从 0.05 拉到 0.2 后改善明显**  
   - 当 IK 输出的目标关节变化较大时，过小的 `max_relative_target` 会导致机器人一直追不上目标，最终看起来像“这集没放完”。  
   - 把 `max_relative_target` 调大，机器人更容易追上每帧目标，因此轨迹更完整。

实用建议（上机安全优先）：

- 先用小 `max_relative_target` 做安全试跑，确认无异常后再逐步调大。
- 如果仍“追不上”，除了调大 `max_relative_target`，也可以把 `--dataset.fps` 调小（回放时间会变长，机器人更有时间跟上）。  
  注意：`lerobot_replay_ee.py` 的 `--dataset.fps` 是“命令下发频率/时间缩放”，**不会做下采样**（`src/lerobot/scripts/lerobot_replay_ee.py:225`）。

---

## 5. 离线核验与可视化

### 5.1 可视化数据集右臂 EE：my_sim/visualize_dataset_right_ee.py

脚本：`my_sim/visualize_dataset_right_ee.py`

功能：

- 读取数据集里的 `right_ee.*`，在 meshcat 中播放末端位姿。
- 可选：用数据集里的右臂 6 关节做 FK，再叠加一条轨迹并输出误差统计（用于验证“录进去的 EE 是否自洽”）。

#### 5.1.1 只跑分析（不打开浏览器）

```bash
conda activate my_lerobot
HF_DATASETS_CACHE=/tmp/hf_datasets_cache PYTHONPATH=src \
python my_sim/visualize_dataset_right_ee.py \
  --repo-id cqy/agilex_vla_demo_ee_test \
  --episode 1 \
  --compare-joint-fk \
  --compare-from observation \
  --analysis-only
```

说明：

- `--compare-joint-fk`：计算 FK 并与 recorded EE 做误差对比。
- `--compare-from observation`：用 `observation.state` 的右臂关节做 FK（更像“从臂实际状态”）。
- `--analysis-only`：只打印预检查报告，不启动 meshcat。

#### 5.1.2 打开 meshcat 播放 + 叠加 FK 对比轨迹

```bash
conda activate my_lerobot
HF_DATASETS_CACHE=/tmp/hf_datasets_cache PYTHONPATH=src \
python my_sim/visualize_dataset_right_ee.py \
  --repo-id cqy/agilex_vla_demo_ee_test \
  --episode 1 \
  --show-trajectory \
  --compare-joint-fk \
  --compare-from action \
  --loop
```

会输出一个 URL（`Open browser: ...`），用浏览器打开即可看到：

- 紫色 EE：`right_ee.*`（数据集里记录的末端）
- 黄色 EE（对比模式）：用关节做 FK 得到的末端（用于核验）
- 可选 `--show-error-line`：每帧画一条红线连接两者，直观看误差。

关于 `--compare-lag`（重要）：

- 脚本默认：
  - `compare-from=action` → 默认 `lag=1`
  - `compare-from=observation` → 默认 `lag=0`
- 这是为了适配“控制回路时延”：`action` 往往对应“下一帧/执行后”的状态。

### 5.2 可视化 URDF 网格：my_sim/visualize_urdf.py

脚本：`my_sim/visualize_urdf.py`

功能：把 URDF 的 mesh 静态显示出来（零位姿），用于确认 mesh 路径解析/坐标系大体正确。

运行方式：

```bash
conda activate my_lerobot
python my_sim/visualize_urdf.py
```

注意：

- 该脚本当前在 `main()` 内部 **硬编码了 `urdf_path`**（例如 `.../new_aloha.urdf`）。  
  如果要看 Piper 的 URDF，需要把 `urdf_path` 改成你的 `piper_description.urdf` 路径后再运行。

---

## 6. 快速对照：replay.sh（关节回放） vs replay_ee.sh（末端回放）

- `replay.sh`（`lerobot-replay`）：回放 **dataset.action（关节目标/执行关节）**  
  适合验证“关节层数据是否能完整复现”。
- `replay_ee.sh`（`lerobot_replay_ee.py`）：回放 **dataset.observation.state 里的 `*_ee.*`（末端轨迹）**  
  适合验证“末端轨迹 + IK 控制链路”以及后续基于 EE 的方法。

---

## 7. 常见现象与排查要点（基于本仓库实现）

- 现象：`replay_ee` 轨迹趋势对，但不完全一致  
  - 典型原因：IK 分支选择/初值、时延、`max_relative_target` 限速、姿态权重较低（`orientation_weight=0.01`）导致姿态抖动被弱化。
- 现象：`replay_ee` “追不上/像没放完”  
  - 优先检查：`--robot.max_relative_target` 是否过小、`--dataset.fps` 是否过高、机器人起始位姿是否接近该 episode 的首帧。
- 想严格验证“EE 字段确实来自 FK”  
  - 用 `my_sim/visualize_dataset_right_ee.py --compare-joint-fk --compare-from observation`，看误差统计是否接近 0（正常录制应非常小）。

---

## 8. LeRobot 数据集格式（Agilex EE 版本，示例：`cqy/agilex_vla_demo_ee_test`）

> 通用 v3.0 规范请参考：`docs/lerobot数据格式解析.md`。本节只讲 **Agilex + EE pose** 这个数据集在文件层面和字段层面的“落地长相”，以及每个字段的语义/单位/坐标系约定。

### 8.1 数据集在磁盘上的位置与目录结构

LeRobot 默认把数据集放在：

- `HF_LEROBOT_HOME/{repo_id}`
- 默认 `HF_LEROBOT_HOME=~/.cache/huggingface/lerobot`（可通过环境变量 `HF_LEROBOT_HOME` 改写）

以本数据集为例（你的机器上）：

- `/home/agilex/.cache/huggingface/lerobot/cqy/agilex_vla_demo_ee_test`

目录结构（v3.0）：

```
cqy/agilex_vla_demo_ee_test/
├── meta/
│   ├── info.json                  # 数据集总体信息 + feature schema（最重要）
│   ├── stats.json                 # 全局统计（归一化/可视化用）
│   ├── tasks.parquet              # task_index ↔ 自然语言任务文本
│   └── episodes/
│       └── chunk-000/file-000.parquet  # 每个 episode 的索引范围、视频时间段、episode 统计
├── data/
│   └── chunk-000/file-000.parquet # 每一帧的矢量数据（action/state/timestamp/...）
└── videos/
    ├── observation.images.camera_left/chunk-000/file-000.mp4
    ├── observation.images.camera_right/chunk-000/file-000.mp4
    └── observation.images.camera_front/chunk-000/file-000.mp4
```

### 8.2 全局信息：`meta/info.json`

本数据集的关键全局信息（摘自 `meta/info.json`）：

- `codebase_version`: `v3.0`
- `robot_type`: `agilex`
- `fps`: `30`
- `total_episodes`: `2`
- `total_frames`: `480`
- `total_tasks`: `1`
- `splits.train`: `"0:2"`（episode 0 和 1 都属于 train）

更重要的是 `features`，它定义了 **每一帧（frame）到底有哪些字段、dtype、shape、names 顺序**。

### 8.3 每帧数据（Frame）字段总览

对 `cqy/agilex_vla_demo_ee_test`，每一帧至少包含：

- `observation.state`：`float32[26]`（关节 + 末端位姿）
- `action`：`float32[14]`（关节位置动作）
- `timestamp`：`float32`（单位：秒，episode 内相对时间，从 0 开始）
- `frame_index`：`int64`（episode 内帧序号，从 0 开始）
- `episode_index`：`int64`（episode 序号，从 0 开始）
- `index`：`int64`（全数据集扁平化后的全局帧序号，从 0 开始）
- `task_index`：`int64`（任务 ID，对应 `meta/tasks.parquet`）
- `observation.images.*`：视频模态（不在 `data/*.parquet` 里逐行存像素，而是在 `videos/*.mp4` 里存整段视频）

### 8.4 `action`（关节位置动作，14 维）

来源与语义（非常重要）：

- 录制时每个控制周期会 `robot.send_action(...)`，然后立刻读一次 `next_obs = robot.get_observation()`。
- 数据集里的 `action` **优先写入 next_obs 中的关节位置**（即“执行后的从臂关节位置”，不是 teleop 输入也不是未执行的目标）。
- 这意味着：在实现上，`action[t]` 往往更接近 “执行后/下一拍”的关节状态（位置控制系统里通常也可视作下一拍目标）。
- 对应代码：`src/lerobot/scripts/lerobot_record.py`（`_extract_action_from_observation(next_obs)`）。

维度顺序（摘自 `meta/info.json -> features.action.names`）：

```
0  left_shoulder_pan.pos
1  left_shoulder_lift.pos
2  left_shoulder_roll.pos
3  left_elbow.pos
4  left_wrist_pitch.pos
5  left_wrist_roll.pos
6  left_gripper.pos
7  right_shoulder_pan.pos
8  right_shoulder_lift.pos
9  right_shoulder_roll.pos
10 right_elbow.pos
11 right_wrist_pitch.pos
12 right_wrist_roll.pos
13 right_gripper.pos
```

单位约定：

- 6 个关节：**弧度 rad**（来自 ROS JointState `position`）
- gripper：**米 m**（夹爪行程，见 `src/lerobot/robots/agilex/config_agilex.py` 的 `gripper` limit `0.0~0.085`）

### 8.5 `observation.state`（关节 + EE pose，26 维）

`observation.state` 在本数据集里把“关节位置”与“末端位姿”拼在同一个向量里：

- `0..13`：与 `action` 同顺序的 14 个关节位置（同单位）
- `14..19`：左臂末端位姿 `left_ee.*`（6 维）
- `20..25`：右臂末端位姿 `right_ee.*`（6 维）

维度顺序（摘自 `meta/info.json -> features["observation.state"].names`）：

```
0  left_shoulder_pan.pos
1  left_shoulder_lift.pos
2  left_shoulder_roll.pos
3  left_elbow.pos
4  left_wrist_pitch.pos
5  left_wrist_roll.pos
6  left_gripper.pos
7  right_shoulder_pan.pos
8  right_shoulder_lift.pos
9  right_shoulder_roll.pos
10 right_elbow.pos
11 right_wrist_pitch.pos
12 right_wrist_roll.pos
13 right_gripper.pos
14 left_ee.x
15 left_ee.y
16 left_ee.z
17 left_ee.wx
18 left_ee.wy
19 left_ee.wz
20 right_ee.x
21 right_ee.y
22 right_ee.z
23 right_ee.wx
24 right_ee.wy
25 right_ee.wz
```

#### 8.5.1 末端位姿 `*_ee.{x,y,z,wx,wy,wz}` 的定义

这些字段在 `--robot.record_ee_pose=true` 时由 `AgileXRobot.get_observation()` 计算并写入：

- 位置：`x,y,z`（单位：**米 m**）
- 姿态：`wx,wy,wz`（单位：**弧度 rad**），是 **rotvec 旋转向量**（axis-angle 的向量形式）：
  - 向量方向 = 旋转轴
  - 向量模长 = 旋转角

FK 的关键约定（决定“坐标系是什么”）：

- FK 模型：`--robot.kinematics_urdf_path` 指定的 URDF（默认 Piper URDF）
- 末端帧：`--robot.kinematics_target_frame`（本仓库默认 `gripper_base`）
- pose 表达的坐标系：**URDF 的根链接（base link）坐标系** 下的 `target_frame` 位姿  
  （更严格地说：`T_root_target = FK(q) @ T_tcp_offset`）
- TCP 偏置：`left_tcp_offset_xyzrpy / right_tcp_offset_xyzrpy`（默认全 0；先 FK 得到 target_frame，再右乘 TCP offset）
- 关节角映射：录制时会把 ROS 读到的 6 关节角做符号与零偏变换再喂给 FK（见下段）

对应代码：

- EE 计算：`src/lerobot/robots/agilex/agilex.py`（`record_ee_pose` 分支）
- 关节映射参数：`src/lerobot/robots/agilex/config_agilex.py`（`*_joint_signs/*_joint_offsets_rad`）

> 注意：本实现里 FK 输入会把 `q` 从 rad 转为 deg 再送入 `RobotKinematics.forward_kinematics()`；这是 placo 后端的接口约定（内部以 deg 工作）。

### 8.6 Episode 切分方式：`meta/episodes/*.parquet`

LeRobot v3.0 在 `data/*.parquet` 里按 frame 扁平化存储，所以需要 episode 元数据来切分序列。

本数据集的 episode 概览（来自 `meta/episodes/chunk-000/file-000.parquet`）：

```
episode 0: length=240, dataset_from_index=0,   dataset_to_index=240
episode 1: length=240, dataset_from_index=240, dataset_to_index=480
```

含义：

- `dataset_from_index`：该 episode 在扁平帧序列中的起始 index（含）
- `dataset_to_index`：结束 index（不含）
- `length = dataset_to_index - dataset_from_index`

### 8.7 视频（多相机）存储方式：`videos/*.mp4` + episode 时间段

关键点：**每个相机不是“每个 episode 一个 mp4”，而是多个 episode 顺序拼在同一个 mp4 里**（减少文件数）。

因此 episode 元数据还包含每个相机在 mp4 中对应的时间段：

- `videos/{video_key}/from_timestamp`
- `videos/{video_key}/to_timestamp`

以 `observation.images.camera_left` 为例：

- episode 0: `from_timestamp=0.0`, `to_timestamp=8.0`
- episode 1: `from_timestamp=8.0`, `to_timestamp=16.0`

这样，某一帧的图像可通过下面的“时间戳对齐”拿到：

- 该帧的 episode 内时间：`timestamp`（来自 `data/*.parquet`，从 0 开始）
- 该 episode 在 mp4 内的起始时间：`from_timestamp`（来自 episodes 元数据）
- mp4 查询时间：`from_timestamp + timestamp`

本数据集视频参数（摘自 `meta/info.json`）：

- 分辨率：`640x480`
- fps：`30`
- codec：`av1`
- 无音频（`has_audio=false`）

> 训练/可视化时如果遇到“视频解码失败”，通常是因为环境里的 ffmpeg/torchvision 编解码不支持 AV1。此时可考虑用更通用的 codec 重新编码数据集视频，或切换可用的解码后端（见 `lerobot.datasets.video_utils`）。

### 8.8 Task 文本：`meta/tasks.parquet` 与 `task_index`

`meta/tasks.parquet` 维护 `task_index → 文本` 的映射，本数据集只有 1 个任务：

- `task_index=0`
- 文本：`Pick up the black cups and place them in the orange box.`

在 `data/*.parquet` 里每一帧记录的是 `task_index`；使用 `LeRobotDataset` 读取时，会自动补一个 `task` 字符串字段（见 `LeRobotDataset.__getitem__()`）。

### 8.9 最小读取示例（推荐给“只想用数据训练”的同学）

1) 用 LeRobotDataset 读取（能自动取视频帧/自动拿 task 文本）：

```python
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset(
    "cqy/agilex_vla_demo_ee_test",
    root="/home/agilex/.cache/huggingface/lerobot/cqy/agilex_vla_demo_ee_test",  # 或者不传 root 用默认
    download_videos=False,
)

frame = ds.hf_dataset[0]
state = np.asarray(frame["observation.state"], dtype=np.float32)  # (26,)
action = np.asarray(frame["action"], dtype=np.float32)            # (14,)
ts = float(frame["timestamp"])
```

2) 只读 parquet（完全不依赖 LeRobot 代码）：

```python
import pyarrow.parquet as pq

root = "/home/agilex/.cache/huggingface/lerobot/cqy/agilex_vla_demo_ee_test"
table = pq.read_table(f"{root}/data/chunk-000/file-000.parquet")

# 每列都是一个数组/列表
action = table["action"][0].as_py()               # len=14
state = table["observation.state"][0].as_py()     # len=26
timestamp = float(table["timestamp"][0].as_py())
```
