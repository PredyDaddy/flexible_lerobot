# 末端 LeRobot 数据解析（v4，Pinocchio EE Pose 版本）

本文档面向**拿到数据、希望用于训练**的同学：解释 `record_ee_pinocchio.sh` 录制出来的数据集在磁盘上的组织方式（LeRobot Dataset v3.0），以及其中 **关节角（joint）/末端位姿（EE pose）/多相机视频** 等字段的**精确含义、单位与索引规则**。

> 说明：
> - 本文以你当前这份数据集为例（`--dataset.repo_id="cqy/agilex_vla_demo_ee_pinocchio_apple_grasp"`），但结构/字段对所有 **AgileX + `--robot.record_ee_pose=true --robot.kinematics_backend=pinocchio`** 录制的数据集通用。
> - 更完整的 LeRobot Dataset v3.0 通用规范见：`docs/lerobot数据格式解析.md`。本文会在**不依赖外部文档**的前提下，把训练必需的信息讲清楚。

---

## 0. 快速上手（推荐读取方式）

强烈建议直接用 `lerobot` 提供的 `LeRobotDataset` 读取：它会自动处理

- Parquet 的时序数据拼接
- 多个 episode 串在同一个 MP4 里的时间片段定位（`from_timestamp/to_timestamp`）
- 视频解码后返回 **CHW、float32、[0,1]** 的张量
- 将 `task_index` 映射回 `task` 文本

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

repo_id = "cqy/agilex_vla_demo_ee_pinocchio_apple_grasp"
ds = LeRobotDataset(repo_id)  # 默认从 HF_LEROBOT_HOME / repo_id 读取

sample = ds[0]
print(sample.keys())
print("state:", sample["observation.state"].shape, sample["observation.state"].dtype)  # (26,), float32
print("action:", sample["action"].shape, sample["action"].dtype)                      # (14,), float32
print("img:", sample["observation.images.camera_left"].shape, sample["observation.images.camera_left"].dtype)
print("task:", sample["task"])
```

如果你把数据拷贝到别的机器/目录，最稳妥的两种方式：

1. **保持默认目录结构**：把整个 `cqy/agilex_vla_demo_ee_pinocchio_apple_grasp` 放到
   `${HF_LEROBOT_HOME}/cqy/agilex_vla_demo_ee_pinocchio_apple_grasp`（默认 `HF_LEROBOT_HOME=~/.cache/huggingface/lerobot`），然后 `LeRobotDataset(repo_id)` 即可。
2. **指定数据集根目录**：如果你的数据集根目录就是某个绝对路径 `DATASET_ROOT`：

```python
ds = LeRobotDataset(repo_id, root="DATASET_ROOT")
```

> 注意：`root` 传的是**数据集根目录本身**（含 `meta/ data/ videos/` 的那个目录），而不是 `HF_LEROBOT_HOME`。

---

## 1. 数据集在磁盘上的位置与总览

### 1.1 默认本地路径（未 push_to_hub）

`record_ee_pinocchio.sh` 没有显式设置 `--dataset.root`，因此数据默认写入：

```
~/.cache/huggingface/lerobot/<repo_id>/
```

以本数据集为例：

```
~/.cache/huggingface/lerobot/cqy/agilex_vla_demo_ee_pinocchio_apple_grasp/
```

对应常量：`src/lerobot/utils/constants.py` 的 `HF_LEROBOT_HOME`。

### 1.2 本数据集的关键元信息（来自 `meta/info.json`）

- `codebase_version`: `v3.0`
- `robot_type`: `agilex`
- `fps`: `30`
- `features`（关键信号）：
  - `action`: `float32[14]`（双臂关节目标）
  - `observation.state`: `float32[26]`（双臂关节 + 双臂 EE pose）
  - `observation.images.camera_left/right/front`: `video[480,640,3]`（RGB 视频）

---

## 2. 目录结构（v3.0）

数据集根目录（`DATASET_ROOT`）下包含三大部分：

```
DATASET_ROOT/
├── meta/                          # 元数据：info / stats / tasks / episodes 索引
│   ├── info.json
│   ├── stats.json
│   ├── tasks.parquet
│   └── episodes/
│       └── chunk-000/
│           ├── file-000.parquet
│           ├── file-001.parquet
│           └── ...
├── data/                          # 时序数值数据（不含图像帧）
│   └── chunk-000/
│       ├── file-000.parquet
│       ├── file-001.parquet
│       └── ...
└── videos/                        # 每个相机一个目录，MP4 内部串联多个 episode
    ├── observation.images.camera_left/
    │   └── chunk-000/
    │       ├── file-000.mp4
    │       ├── file-001.mp4
    │       └── ...
    ├── observation.images.camera_right/
    └── observation.images.camera_front/
```

### 2.1 chunk / file 的含义（容易踩坑）

v3.0 的切分原则不是“每个 episode 一个文件”，而是为了减少小文件数量、提高读写性能：

- `chunk-XYZ/`：一个目录下最多放 `chunks_size` 个文件（默认 1000）。
- `file-ABC.parquet` / `file-ABC.mp4`：按“文件大小上限”滚动切分（默认 `data_files_size_in_mb=100MB`、`video_files_size_in_mb=200MB`），因此：
  - 一个 parquet/mp4 文件里**通常包含多个 episode**（拼接在一起）
  - 某个 episode 在文件中的起止位置需要看 `meta/episodes/*.parquet` 里的索引

---

## 3. 数值时序数据：`data/*/*.parquet`

### 3.1 Parquet Schema（列与类型）

以 `data/chunk-000/file-000.parquet` 为例，列结构为：

- `action`: `fixed_size_list<float>[14]`
- `observation.state`: `fixed_size_list<float>[26]`
- `timestamp`: `float32`（秒）
- `frame_index`: `int64`（episode 内帧号，从 0 开始）
- `episode_index`: `int64`（episode 编号，从 0 开始）
- `index`: `int64`（全局帧号，从 0 开始，跨 episode 递增）
- `task_index`: `int64`（任务编号，用于去 `meta/tasks.parquet` 查文本任务）

> 注意：当数据集启用 `video=true` 时，**图像不在 parquet 里**，只在 `videos/` 里；parquet 里只放状态、动作、索引等数值信息。

### 3.2 `timestamp` 的定义

默认情况下（录制代码没有显式传入真实时间戳），`timestamp` 是用 `frame_index / fps` 生成的离散时间：

- 第 0 帧：`0.0`
- 第 1 帧：`1/fps`
- …

这使得每个 episode 内时间轴**严格等间隔**，更利于训练对齐。

---

## 4. `action` 与 `observation.state` 的字段顺序与含义

字段顺序由 `meta/info.json` 的 `features.*.names` 决定（也就是模型训练时 `action[i]` / `state[j]` 的语义）。

### 4.1 `action`（float32[14]）

`action` 是**双臂从臂（follower/puppet）关节目标位置**，顺序如下：

| index | name | 说明 | 单位 |
|---:|---|---|---|
| 0 | `left_shoulder_pan.pos` | 左臂关节 1 | rad |
| 1 | `left_shoulder_lift.pos` | 左臂关节 2 | rad |
| 2 | `left_shoulder_roll.pos` | 左臂关节 3 | rad |
| 3 | `left_elbow.pos` | 左臂关节 4 | rad |
| 4 | `left_wrist_pitch.pos` | 左臂关节 5 | rad |
| 5 | `left_wrist_roll.pos` | 左臂关节 6 | rad |
| 6 | `left_gripper.pos` | 左夹爪开合量（行程） | m |
| 7 | `right_shoulder_pan.pos` | 右臂关节 1 | rad |
| 8 | `right_shoulder_lift.pos` | 右臂关节 2 | rad |
| 9 | `right_shoulder_roll.pos` | 右臂关节 3 | rad |
| 10 | `right_elbow.pos` | 右臂关节 4 | rad |
| 11 | `right_wrist_pitch.pos` | 右臂关节 5 | rad |
| 12 | `right_wrist_roll.pos` | 右臂关节 6 | rad |
| 13 | `right_gripper.pos` | 右夹爪开合量（行程） | m |

> action 的来源（录制时）：`lerobot-record` 循环中由 teleop 产生目标，再经过 `robot.send_action()` 可能被限幅/裁剪；最终写入的是“实际下发的关节目标”（`executed_action_values`）。

### 4.2 `observation.state`（float32[26]）

`observation.state` = **双臂关节位置（14） + 双臂末端位姿（12）**，顺序如下：

#### (A) joint 部分：`state[0:14]`

`state[0:14]` 的 14 个 joint 字段与 `action` 完全同序同名（含左右 6 关节 + gripper）。

#### (B) EE pose 部分：`state[14:26]`

| index | name | 说明 | 单位 |
|---:|---|---|---|
| 14 | `left_ee.x` | 左臂 EE 位置 x | m |
| 15 | `left_ee.y` | 左臂 EE 位置 y | m |
| 16 | `left_ee.z` | 左臂 EE 位置 z | m |
| 17 | `left_ee.roll` | 左臂 EE 欧拉角 roll | rad |
| 18 | `left_ee.pitch` | 左臂 EE 欧拉角 pitch | rad |
| 19 | `left_ee.yaw` | 左臂 EE 欧拉角 yaw | rad |
| 20 | `right_ee.x` | 右臂 EE 位置 x | m |
| 21 | `right_ee.y` | 右臂 EE 位置 y | m |
| 22 | `right_ee.z` | 右臂 EE 位置 z | m |
| 23 | `right_ee.roll` | 右臂 EE 欧拉角 roll | rad |
| 24 | `right_ee.pitch` | 右臂 EE 欧拉角 pitch | rad |
| 25 | `right_ee.yaw` | 右臂 EE 欧拉角 yaw | rad |

---

## 5. 末端位姿（EE pose）到底是什么（Pinocchio 版本）

这一节回答三个关键问题：

1. 这个 EE pose 是怎么从关节角算出来的？
2. 它的坐标系是什么？（相对谁）
3. roll/pitch/yaw 的约定是什么？

### 5.1 计算入口与开关

录制脚本（示例）`record_ee_pinocchio.sh` 打开：

- `--robot.record_ee_pose=true`
- `--robot.kinematics_backend=pinocchio`
- `--robot.kinematics_urdf_path=.../piper_description.urdf`
- `--robot.pinocchio_package_dirs=[...]`
- `--robot.pinocchio_lock_joints=["joint7","joint8"]`
- `--robot.ee_rotation_rpy=[0,-1.57,-1.57]`
- `--robot.ee_translation=[0,0,0]`

对应实现：`src/lerobot/robots/agilex/agilex.py` 的 `AgileXRobot.get_observation()`。

### 5.2 FK 输入：ROS 关节角到 URDF 关节角的映射

Pinocchio FK 的输入是 6 个关节（不含 gripper），单位 **rad**。

代码里会做一个“ROS → URDF”的线性映射（逐关节）：

```
q_urdf = q_ros * signs + offsets
```

- `q_ros`：从 ROS topic 读到的 `puppet` 关节位置（`sensor_msgs/JointState.position[:6]`）
- `signs/offsets`：来自 `AgileXConfig.left_joint_signs/right_joint_signs` 与 `*_joint_offsets_rad`

> 本仓库默认 `signs=[1,1,1,1,1,1]`、`offsets=[0,0,0,0,0,0]`，但如果你改过这些参数，EE pose 的语义也会随之变化。

### 5.3 EE 参考坐标系（“相对谁”）

Pinocchio 输出的是 **URDF 模型根坐标系（`dummy_link/base_link`，见 URDF）到 EE 坐标系** 的位姿：

- 使用的 URDF：`my_sim/piper_ros-noetic/src/piper_description/urdf/piper_description.urdf`
- URDF 根部：`<link name="dummy_link"/>`，`base_link` 通过固定关节连接且零位姿
- 因此可理解为：EE pose 相对于 `base_link`（与 `dummy_link` 同位姿）

> 重要提醒（训练时要知道但不一定要处理）：
> - 当前实现对左右臂各用一套同构 URDF 做 FK，**没有额外加“左臂基座到右臂基座”的外参变换**。
> - 因此，如果你想把左右臂的 EE 放到同一全局坐标系，需要你自己提供两臂基座之间的外参；数据集中不包含该外参。

### 5.4 EE 坐标系的定义（为什么需要 `ee_rotation_rpy`）

为了让“末端坐标系”与 IK/控制习惯一致，Pinocchio 代码会在模型里**新增一个 frame**：`ee`，
挂在 `joint6` 上，并施加一个固定的旋转/平移偏置（来自 `ik_demo.py` 的约定）：

- `ee_rotation_rpy = [roll, pitch, yaw]`（rad）
- `ee_translation = [x, y, z]`（m）

默认值为：

- `ee_rotation_rpy = [0.0, -1.57, -1.57]`
- `ee_translation = [0.0, 0.0, 0.0]`

> 换句话说：`left_ee.* / right_ee.*` 并不一定等于 URDF 里已有的 `gripper_base` link 位姿，
> 而是你定义的这个 `ee` frame（再叠加可选的 TCP offset，见下一节）。

### 5.5 可选：TCP offset（本数据集默认是零）

在 FK 算出 `T_base^ee` 后，还支持再乘一个 TCP 偏置：

```
T_base^tcp = T_base^ee · T_ee^tcp
```

其中 `T_ee^tcp` 来自：

- `AgileXConfig.left_tcp_offset_xyzrpy`
- `AgileXConfig.right_tcp_offset_xyzrpy`

默认是全 0（单位 m/rad），即 `T_ee^tcp = I`，因此本数据集记录的就是 `T_base^ee`。

### 5.6 roll/pitch/yaw 的数学约定（非常关键）

本数据集记录的欧拉角遵循 **ZYX（yaw-pitch-roll）** 约定：

```
R = Rz(yaw) · Ry(pitch) · Rx(roll)
```

从旋转矩阵 `R` 提取时使用：

- `roll  = atan2(R[2,1], R[2,2])`
- `pitch = asin(-R[2,0])`
- `yaw   = atan2(R[1,0], R[0,0])`

这与 `my_ik/piper_pinocchio_kinematics.py` 和 `src/lerobot/robots/agilex/agilex.py` 的实现一致。

> 提醒：欧拉角在 pitch 接近 ±90° 时会有万向节锁问题；如果你要做几何运算（差分、插值、损失函数），更建议转换成 rotation matrix 或 quaternion 再处理。

---

## 6. Episode 元数据：`meta/episodes/*/*.parquet`

`meta/episodes/` 里每一行对应一个 episode，用于回答：

- 这个 episode 有多少帧？（`length`）
- 它在全局帧序列中的起止位置？（`dataset_from_index/to_index`）
- 它落在哪个 data parquet 文件里？（`data/chunk_index/file_index`）
- 它落在哪个 video mp4 文件里？并且在该 mp4 的哪个时间段？（`videos/.../from_timestamp/to_timestamp`）

### 6.1 关键列（训练/解析最常用）

- `episode_index`: episode 编号
- `length`: 帧数（通常 `fps * episode_time_s`）
- `dataset_from_index`, `dataset_to_index`：
  - 表示该 episode 在全局帧序列中的半开区间 `[from, to)`
  - 该 episode 的全局帧号 `index` 满足：`from <= index < to`
- `data/chunk_index`, `data/file_index`：该 episode 的帧数据在哪个 parquet 文件里
- 对每个相机 `video_key`（如 `observation.images.camera_left`）：
  - `videos/<video_key>/chunk_index`
  - `videos/<video_key>/file_index`
  - `videos/<video_key>/from_timestamp`
  - `videos/<video_key>/to_timestamp`

### 6.2 为什么需要 `from_timestamp/to_timestamp`（MP4 里串了多个 episode）

v3.0 会把多个 episode 的视频帧**按时间顺序拼到同一个 mp4** 里减少文件数。

因此：

- 同一个 mp4 里可能连续包含 N 个 episode 的视频
- `from_timestamp/to_timestamp` 给出了某个 episode 在该 mp4 中的起止秒数
- `LeRobotDataset` 读取时会用 `from_timestamp + timestamp` 来定位对应帧（详见 `src/lerobot/datasets/lerobot_dataset.py`）

---

## 7. 视频数据：`videos/<video_key>/*/*.mp4`

### 7.1 video_key 与相机对应关系

本数据集包含三个相机：

- `observation.images.camera_left`
- `observation.images.camera_right`
- `observation.images.camera_front`

每个 `video_key` 对应一个目录，目录下按 chunk/file 切分 mp4。

### 7.2 解码后的张量格式（`LeRobotDataset` 返回）

通过 `LeRobotDataset[idx]["observation.images.camera_left"]` 读出来的图像帧：

- 类型：`torch.Tensor`
- 形状：`(C, H, W)`，即 `(3, 480, 640)`
- dtype：`float32`
- 数值范围：`[0, 1]`

### 7.3 编解码注意事项（AV1）

`meta/info.json` 里记录了视频编码信息（例如 `video.codec: "av1"`）。

如果你不使用 `LeRobotDataset` 而要自己用 ffmpeg/pyav 解码，确保你的环境支持 AV1 解码（否则会出现无法读取 mp4 的问题）。

---

## 8. 不用 LeRobotDataset 的“手动解析”指南（可选）

如果你在别的训练框架里使用该数据（不想引入 `lerobot`），可以按下面逻辑手动读取：

1. 读 `meta/info.json` 获取：
   - `fps`
   - `features.action.shape/names`
   - `features.observation.state.shape/names`
   - `video_path` 模板与相机 key
2. 用 `pyarrow/pandas` 读 `data/*/*.parquet`：
   - 每行一帧，拿到 `action` 与 `observation.state`
   - 通过 `episode_index/frame_index` 还原 episode
3. 读 `meta/episodes/*/*.parquet`：
   - 找到目标 episode 的 `videos/.../file_index` 与 `from_timestamp/to_timestamp`
4. 解码对应 mp4：
   - 截取 `[from_timestamp, to_timestamp)` 的视频片段（或在解码时用时间戳精确索引）
   - 按 `fps` 对齐到每个 `timestamp = frame_index/fps`

> 这套手动流程的核心难点就是“一个 mp4 里串了多个 episode”，因此一定要用 `meta/episodes` 提供的时间片段索引来对齐。

---

## 9. 训练时如何理解“一个样本”

在 LeRobot 的约定中，训练数据通常是监督式模仿学习：

- 输入（observation）：
  - `observation.images.*`（多视角 RGB）
  - `observation.state`（关节 + EE pose）
  - `task`（自然语言任务描述，单任务数据集也会保留该字段）
- 标签（action）：
  - `action`（双臂关节目标）

你可以选择：

- 只用 `observation.state[:14]`（关节）作为低维状态
- 或把 `state[14:26]`（EE pose）作为额外状态输入（冗余但可能更利于某些策略）
- 如果你要训练“末端空间动作”，通常需要自己构造 `ΔEE`（例如 `ee(t+1)-ee(t)` 或 SE(3) 差分），本数据集**没有直接提供 EE action**。

---

## 10. 常见坑与排查

1. **不要把不同 schema 的数据集混着 `--resume=true` 录制**  
   开启 `record_ee_pose=true` 会让 `observation.state` 维度从 14 变成 26；schema 不一致会导致后续加载/训练混乱。

2. **只看 parquet 会误以为没有图像**  
   开启 `video=true` 时，图像都在 `videos/`，parquet 不存图像列。

3. **欧拉角不要直接做 L2 差分当角度距离**  
   需要几何一致性时，用 rotation matrix / quaternion。

4. **左右臂 EE 不在统一全局坐标系**  
   当前实现每个臂各自用同构 URDF 做 FK，没有提供两臂基座外参；如果你要跨臂几何约束，需要自行标定/定义外参。

---

## 11. 附：录制脚本与关键参数（便于复现实验）

录制入口脚本：`record_ee_pinocchio.sh`

关键参数（与 EE 字段强相关）：

- `--robot.record_ee_pose=true`
- `--robot.kinematics_backend=pinocchio`
- `--robot.kinematics_urdf_path=.../piper_description.urdf`
- `--robot.pinocchio_package_dirs=[...]`
- `--robot.pinocchio_lock_joints=["joint7","joint8"]`
- `--robot.ee_rotation_rpy=[0,-1.57,-1.57]`
- `--robot.ee_translation=[0,0,0]`

