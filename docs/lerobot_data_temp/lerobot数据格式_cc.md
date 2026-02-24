# LeRobot 数据格式技术文档

## 目录
1. [概述](#概述)
2. [核心概念](#核心概念)
3. [文件结构详解](#文件结构详解)
4. [数据类型与特征系统](#数据类型与特征系统)
5. [元数据架构](#元数据架构)
6. [时序数据处理](#时序数据处理)
7. [视频与图像处理](#视频与图像处理)
8. [数据存储优化](#数据存储优化)
9. [API 使用指南](#api-使用指南)
10. [高级功能](#高级功能)
11. [最佳实践](#最佳实践)
12. [版本兼容性](#版本兼容性)

---

## 概述

LeRobot Dataset 是 Hugging Face 为机器人学习设计的标准化数据格式,专为存储和加载机器人操作数据而优化。该格式支持:

- **多模态数据**: 同时存储状态向量、图像、视频和动作数据
- **时序对齐**: 精确的时间戳系统确保所有传感器数据同步
- **高效存储**: 智能分块和压缩策略,支持大规模数据集
- **流式加载**: 支持从云端流式传输,无需完整下载
- **灵活查询**: 基于时间戳的帧选择和多帧历史/未来数据访问

当前版本: **v3.0**

---

## 核心概念

### 数据集基本组成

LeRobot 数据集由以下核心元素构成:

1. **Episode (剧集)**: 一次完整的机器人任务执行过程,包含从开始到结束的所有帧
2. **Frame (帧)**: 在特定时间点采集的完整数据快照,包含所有传感器的数据
3. **Feature (特征)**: 数据的特定模态,如 `observation.images.cam_front`、`action`、`observation.state`
4. **Task (任务)**: 自然语言描述的任务目标,如 "把苹果放到盘子里"

### 数据流向

```
机器人传感器 → Frame → Episode → Dataset
   ↓              ↓        ↓         ↓
状态、图像、     时间戳    任务标签   元数据、统计
动作、深度
```

---

## 文件结构详解

### 目录树结构

一个典型的 LeRobot 数据集具有以下结构:

```
dataset_root/
├── data/                           # 主要数据存储
│   ├── chunk-000/                  # 数据分块目录
│   │   ├── file-000.parquet       # 包含多 episodes 的 Parquet 文件
│   │   ├── file-001.parquet
│   │   └── ...
│   ├── chunk-001/
│   │   └── ...
│   └── ...
├── meta/                           # 元数据存储
│   ├── episodes/                   # Episode 级元数据
│   │   ├── chunk-000/
│   │   │   ├── file-000.parquet   # Episode 信息
│   │   │   └── ...
│   │   └── ...
│   ├── info.json                   # 数据集描述
│   ├── stats.json                  # 统计信息
│   └── tasks.parquet              # 任务描述
└── videos/                        # 视频文件存储
    ├── observation.images.cam1/
    │   ├── chunk-000/
    │   │   ├── file-000.mp4       # 多 episodes 合并的视频
    │   │   └── ...
    │   └── ...
    └── observation.images.cam2/
        └── ...
```

### 核心文件说明

#### 1. `meta/info.json`

数据集的核心元数据文件,包含:

```json
{
    "codebase_version": "v3.0",           // LeRobot 版本
    "robot_type": "so100",                // 机器人类型
    "total_episodes": 150,                // 总剧集数
    "total_frames": 45000,                // 总帧数
    "total_tasks": 5,                     // 任务数量
    "chunks_size": 1000,                  // 每个 chunk 的最大文件数
    "data_files_size_in_mb": 100,         // 数据文件大小限制
    "video_files_size_in_mb": 200,        // 视频文件大小限制
    "fps": 30,                            // 采集帧率
    "splits": {                           // 数据集划分
        "train": "0:120",
        "val": "120:150"
    },
    "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
    "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    "features": {                         // 特征定义
        "observation.images.cam_front": {
            "dtype": "video",
            "shape": [240, 320, 3],
            "names": ["height", "width", "channels"]
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [7],
            "names": ["joint_0", "joint_1", ..., "gripper"]
        },
        "action": {
            "dtype": "float32",
            "shape": [7],
            "names": ["joint_0", "joint_1", ..., "gripper"]
        },
        "timestamp": {
            "dtype": "float32",
            "shape": [1],
            "names": null
        },
        // ... 其他特征
    }
}
```

**关键字段说明:**

- `codebase_version`: 数据集格式版本,用于版本兼容性检查
- `robot_type`: 机器人类型,用于区分不同机器人平台
- `chunks_size`: 控制存储结构,影响文件数量和加载性能
- `*_files_size_in_mb`: 文件大小限制,达到限制时会创建新文件
- `fps`: 所有数据的时间基准,用于时序对齐
- `features`: 定义数据集中所有特征的属性

#### 2. `data/**/*.parquet`

包含所有帧级数据,使用 Apache Parquet 格式:

**数据列示例:**

| 列名                         | 类型         | 描述                              |
|------------------------------|--------------|-----------------------------------|
| index                        | int64        | 全局唯一帧索引                    |
| episode_index               | int64        | 所属剧集索引                      |
| frame_index                 | int64        | 剧集内帧索引                      |
| task_index                  | int64        | 任务索引                          |
| timestamp                   | float32      | 时间戳(秒)                        |
| observation.images.cam_front| PIL Image    | 相机图像                          |
| observation.state           | float32[7]   | 机器人状态向量                    |
| action                      | float32[7]   | 动作向量                          |
| next.reward                 | float32      | 奖励(可选)                        |
| next.done                   | bool         | 剧集结束标记                      |

**Parquet 文件特性:**

- **列式存储**: 高效压缩和查询
- **嵌套结构**: 支持复杂数据类型
- **统计信息**: 每个列块包含最小/最大值,支持谓词下推
- **跨文件分区**: 多 episodes 存储在单个文件,减少文件数量

#### 3. `meta/episodes/**/*.parquet`

Episode 级元数据:

| 列名                         | 类型         | 描述                              |
|------------------------------|--------------|-----------------------------------|
| episode_index               | int64        | 剧集索引                          |
| tasks                       | list<string> | 任务标签列表                      |
| length                      | int64        | 剧集长度(帧数)                    |
| dataset_from_index          | int64        | 在数据文件中的起始索引            |
| dataset_to_index            | int64        | 在数据文件中的结束索引            |
| data/chunk_index            | int64        | 数据 chunk 索引                   |
| data/file_index             | int64        | 数据文件索引                      |
| videos/*/chunk_index        | int64        | 视频 chunk 索引                   |
| videos/*/file_index         | int64        | 视频文件索引                      |
| videos/*/from_timestamp     | float32      | 视频片段起始时间                  |
| videos/*/to_timestamp       | float32      | 视频片段结束时间                  |

#### 4. `meta/tasks.parquet`

任务描述表:

| task_index | task_name                |
|------------|--------------------------|
| 0          | "pick up the apple"     |
| 1          | "place apple in bowl"   |
| 2          | "push the red block"    |
| ...        | ...                      |

#### 5. `meta/stats.json`

数据集统计信息:

```json
{
    "observation.images.cam_front": {
        "mean": [128.5, 128.5, 128.5],
        "std": [58.3, 57.1, 59.2],
        "min": [0, 0, 0],
        "max": [255, 255, 255],
        "q01": [...],
        "q99": [...]
    },
    "observation.state": {
        "mean": [0.0, 0.1, -0.05, 0.8, 0.2, 0.0, 0.5],
        "std": [0.5, 0.3, 0.2, 0.1, 0.4, 0.3, 0.05],
        "min": [-1.5, -1.0, -0.8, 0.5, -0.5, -1.0, 0.0],
        "max": [1.5, 1.0, 0.8, 1.0, 1.0, 1.0, 1.0]
    }
}
```

**统计类型:**
- `mean`: 平均值
- `std`: 标准差
- `min`/`max`: 最小/最大值
- `q01`/`q99`: 1% 和 99% 分位数(用于鲁棒归一化)

---

## 数据类型与特征系统

### 支持的 dtype

| dtype     | 描述                  | 存储格式              | 典型用途                      |
|-----------|----------------------|---------------------|------------------------------|
| `float32` | 32位浮点数            | Parquet 列          | 状态、动作、奖励              |
| `int64`   | 64位整数              | Parquet 列          | 索引、计数                    |
| `bool`    | 布尔值                | Parquet 列          | 完成标记                      |
| `image`   | 单张图像              | Parquet (嵌入)      | 静态图像                      |
| `video`   | 视频流                | MP4 文件            | 相机视频流                    |
| `string`  | 字符串                | Parquet 列          | 任务描述、元信息              |

### Feature 定义格式

每个特征的定义遵循以下结构:

```python
{
    "dtype": str,           # 数据类型
    "shape": tuple,         # 张量形状
    "names": list | None,   # 维度名称(可选)
    "info": dict            # 额外信息(视频专用)
}
```

#### 示例 1: 状态向量

```python
"observation.state": {
    "dtype": "float32",
    "shape": (7,),                               # 7 个关节
    "names": ["joint_0", "joint_1", ..., "gripper"]
}
```

#### 示例 2: 图像特征

```python
"observation.images.cam_front": {
    "dtype": "video",                          # 或 "image"
    "shape": (240, 320, 3),                    # H, W, C
    "names": ["height", "width", "channels"],
    "info": {
        "codec": "av1",                        # 编码格式
        "fps": 30,                             # 视频帧率
        "duration": 45.5                       # 视频时长(秒)
    }
}
```

#### 示例 3: 单值特征

```python
"timestamp": {
    "dtype": "float32",
    "shape": (1,),
    "names": None                              # 标量值
}
```

### 默认特征

所有 LeRobot 数据集自动包含以下特征:

```python
DEFAULT_FEATURES = {
    "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
    "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
    "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
    "index": {"dtype": "int64", "shape": (1,), "names": None},
    "task_index": {"dtype": "int64", "shape": (1,), "names": None},
}
```

这些特征由系统自动管理,无需用户手动添加。

---

## 元数据架构

### Episode 数据结构

每个 episode 包含:

```python
{
    "episode_index": int,                          # 剧集索引
    "tasks": List[str],                           # 任务标签
    "length": int,                                # 剧集长度
    "dataset_from_index": int,                     # 数据起始索引
    "dataset_to_index": int,                       # 数据结束索引
    "data/chunk_index": int,                       # 数据 chunk
    "data/file_index": int,                        # 数据文件
    "stats": {                                     # 剧集统计
        "observation.state/mean": [...],
        "observation.state/std": [...],
        ...
    },
    # 视频元数据
    "videos/{key}/chunk_index": int,
    "videos/{key}/file_index": int,
    "videos/{key}/from_timestamp": float,
    "videos/{key}/to_timestamp": float
}
```

### 特征命名约定

LeRobot 采用层次化命名:

```
{prefix}.{category}.{key}
```

**Prefixes:**
- `observation`: 观测数据
- `action`: 动作数据
- `next`: 下一时间步数据(奖励、完成标志)
- `info`: 额外信息

**Categories:**
- `images`: 视觉数据
- `state`: 状态向量
- `environment`: 环境信息

**示例:**
- `observation.images.cam_front`: 前相机图像
- `observation.state`: 机器人状态
- `action`: 动作命令
- `next.reward`: 奖励
- `next.done`: 完成标志

---

## 时序数据处理

### 时间戳系统

LeRobot 使用 **基于帧率的时间戳系统**:

```python
timestamp = frame_index / fps
```

**保证:**
- 时间戳精确对齐
- 满足 `|timestamp[i+1] - timestamp[i] - 1/fps| < tolerance_s`
- 支持多传感器同步

### Delta Timestamps 机制

`delta_timestamps` 允许查询历史和未来的多帧数据:

```python
delta_timestamps = {
    "observation.images.cam_front": [-1.0, -0.5, 0.0],    # 1秒前, 0.5秒前, 现在
    "observation.state": [-1.5, -1.0, -0.5, 0.0],         # 历史状态
    "action": [t / fps for t in range(64)]                # 未来64帧动作
}
```

**约束条件:**
- 所有 delta 必须是 `1/fps` 的倍数
- 误差容限 `tolerance_s` (默认 1e-4 秒)

**实现原理:**

1. **时间到索引转换:**
```python
delta_indices = {
    key: [round(d * fps) for d in delta_ts]
    for key, delta_ts in delta_timestamps.items()
}
```

2. **边界处理:**
```python
# Episode 边界内插
query_indices = [
    max(ep_start, min(ep_end - 1, idx + delta))
    for delta in delta_indices
]

# Padding 标记
is_pad = [
    (idx + delta < ep_start) | (idx + delta >= ep_end)
    for delta in delta_indices
]
```

### 多帧查询示例

**单帧索引访问:**
```python
dataset = LeRobotDataset("lerobot/aloha")
frame_100 = dataset[100]  # 返回第100帧
```

**多帧历史访问:**
```python
dataset = LeRobotDataset(
    "lerobot/aloha",
    delta_timestamps={
        "observation.images.cam_front": [-0.5, -0.2, 0.0],  # 3帧
        "observation.state": [-1.0, -0.5, -0.2, 0.0]         # 4帧
    }
)

# dataset[100] 返回:
{
    "observation.images.cam_front": torch.Tensor(3, C, H, W),  # 3个时间点
    "observation.state": torch.Tensor(4, state_dim),           # 4个时间点
    "observation.images.cam_front_is_pad": torch.BoolTensor(3), # padding 标记
    "observation.state_is_pad": torch.BoolTensor(4),
    # ... 其他单帧特征
}
```

---

## 视频与图像处理

### 存储策略

LeRobot 支持两种视觉数据存储模式:

#### 1. 图像模式 (`dtype: "image"`)

**适用场景:**
- 小规模数据集
- 需要随机访问单帧
- 数据量 < 10GB

**存储结构:**
```
images/{image_key}/episode-{episode_index:06d}/frame-{frame_index:06d}.png
```

**特点:**
- 每帧独立 PNG 文件
- 加载速度快(随机访问)
- 存储空间较大

#### 2. 视频模式 (`dtype: "video"`)

**适用场景:**
- 大规模数据集
- 顺序访问为主
- 数据量 > 10GB

**存储结构:**
```
videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4
```

**特点:**
- 多 episodes 合并为 MP4 文件
- 高压缩率(使用 AV1/H.264 编码)
- 支持硬件加速解码
- 需要时序查询

### 视频编码参数

```python
# 编码设置
fps: int = 30
codec: str = "av1"  # 或 "h264"
bitrate: str = "1M"
keyframe_interval: int = 30  # 每30帧一个关键帧
```

**关键帧策略:**
- 关键帧间隔影响随机访问性能
- 较短间隔 = 更快的随机访问 + 较大的文件
- 推荐: 1-2 秒一个关键帧

### 视频解码实现

LeRobot 支持多种视频解码后端:

#### 1. TorchCodec (推荐)

```python
dataset = LeRobotDataset(
    "lerobot/aloha",
    video_backend="torchcodec"  # 默认
)
```

**特点:**
- 高性能,接近原生速度
- 精确的时间戳定位
- 支持 GPU 解码

#### 2. PyAV (备用)

```python
dataset = LeRobotDataset(
    "lerobot/aloha",
    video_backend="pyav"
)
```

**特点:**
- 兼容性最好
- 纯 CPU 解码
- 需要加载关键帧到目标帧的所有帧

### 视频帧查询优化

**时间戳匹配算法:**

```python
# 1. 加载查询时间戳周围的视频段
first_ts = min(timestamps)
last_ts = max(timestamps)

# 2. 定位到关键帧
reader.seek(first_ts, keyframes_only=True)

# 3. 加载所有帧直到最后一个查询时间戳
loaded_frames = []
loaded_timestamps = []
for frame in reader:
    current_ts = frame["pts"]
    loaded_frames.append(frame["data"])
    loaded_timestamps.append(current_ts)
    if current_ts >= last_ts:
        break

# 4. 最近邻匹配
query_ts_tensor = torch.tensor(timestamps)
loaded_ts_tensor = torch.tensor(loaded_timestamps)
dist = torch.cdist(query_ts_tensor[:, None], loaded_ts_tensor[:, None], p=1)
_, argmin = dist.min(1)

# 5. 验证容差
assert all(dist_min < tolerance_s for dist_min in dist.min(1)[0])
```

### 图像变换

LeRobot 集成 torchvision 变换系统:

```python
from torchvision.transforms import v2
from lerobot.datasets.transforms import ImageTransforms, ImageTransformsConfig

# 基础变换
transforms = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToTensor(),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = LeRobotDataset(
    "lerobot/aloha",
    image_transforms=transforms
)

# 随机增强变换
config = ImageTransformsConfig(
    enable=True,
    max_num_transforms=3,
    random_order=False,
    tfs={
        "brightness": {
            "weight": 1.0,
            "type": "ColorJitter",
            "kwargs": {"brightness": (0.8, 1.2)}
        },
        "contrast": {
            "weight": 1.0,
            "type": "ColorJitter",
            "kwargs": {"contrast": (0.8, 1.2)}
        },
        "sharpness": {
            "weight": 1.0,
            "type": "SharpnessJitter",
            "kwargs": {"sharpness": (0.5, 1.5)}
        }
    }
)

transforms = ImageTransforms(config)
```

**支持的变换类型:**
- `ColorJitter`: 亮度、对比度、饱和度、色调
- `SharpnessJitter`: 锐度
- `RandomAffine`: 仿射变换(平移、旋转、缩放)
- `GaussianBlur`: 高斯模糊
- `RandomCrop`: 随机裁剪
- `Resize`: 尺寸调整

---

## 数据存储优化

### 智能分块 (Chunking)

LeRobot 采用多级分块策略优化存储:

#### Chunk 层级

```
dataset_root/
├── data/chunk-000/          # Chunk 级别
│   ├── file-000.parquet    # 文件级别
│   ├── file-001.parquet
│   └── ...
├── data/chunk-001/
└── ...
```

**分块参数:**

```python
chunks_size: int = 1000              # 每个 chunk 的最大文件数
data_files_size_in_mb: int = 100     # 数据文件大小限制
video_files_size_in_mb: int = 200    # 视频文件大小限制
```

**分块逻辑:**

```python
def update_chunk_file_indices(chunk_idx, file_idx, chunks_size):
    if file_idx == chunks_size - 1:
        file_idx = 0                     # 重置文件索引
        chunk_idx += 1                   # 进入新 chunk
    else:
        file_idx += 1                    # 下一个文件
    return chunk_idx, file_idx
```

**优势:**
- 限制单个目录文件数量(性能优化)
- 控制单个文件大小(便于传输)
- 支持并行写入

### 文件大小管理

#### 数据文件切割

```python
# 监控文件大小
latest_size_in_mb = get_file_size_in_mb(latest_path)
av_size_per_frame = latest_size_in_mb / frames_in_current_file

# 如果添加新数据会超出限制,创建新文件
if latest_size_in_mb + av_size_per_frame * new_frames >= data_files_size_in_mb:
    chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, chunks_size)
```

#### 视频文件合并

多个 episodes 可以合并到单个视频文件:

```python
# Episode 边界在视频内部
videos/cam_front/chunk-000/file-000.mp4:
    [Episode 0 - 0:10s][Episode 1 - 10:25s][Episode 2 - 25:40s]

# 通过时间戳偏移定位
from_timestamp = episode_metadata["videos/cam_front/from_timestamp"]
shifted_query_ts = [from_timestamp + ts for ts in query_timestamps]
```

### 异步图像写入

```python
# 启动异步写入器
dataset.start_image_writer(num_processes=0, num_threads=4)

# 收集阶段(非阻塞)
for frame in episode:
    dataset.add_frame(frame)  # 图像异步写入

# 等待完成
dataset._wait_image_writer()

# 停止写入器
dataset.stop_image_writer()
```

**适用场景:**
- 实时数据采集
- 需要平滑帧率的情况
- 多摄像头同时采集

### 视频批量编码

```python
# 单 episode 立即编码
batch_encoding_size = 1

# 批量编码(延迟编码)
batch_encoding_size = 10

# 在 10 个 episodes 后批量编码
dataset.save_episode()
# ... 重复 9 次 ...
dataset.save_episode()  # 触发批量编码
```

**优势:**
- 延迟编码减少计算开销
- 更好的压缩效率(更多帧参考)
- 适合离线数据处理

---

## API 使用指南

### LeRobotDataset 类

#### 初始化

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 从 Hugging Face Hub 加载
dataset = LeRobotDataset(
    repo_id="lerobot/aloha_sim_transfer_cube_human",
    root="~/.cache/huggingface/lerobot",  # 可选
    episodes=None,                         # 加载全部或指定 episodes
    image_transforms=None,                 # 图像变换
    delta_timestamps=None,                 # 时序查询配置
    tolerance_s=1e-4,                      # 时间容差
    revision="v3.0",                       # 版本标签
    force_cache_sync=False,                # 强制同步缓存
    download_videos=True,                  # 下载视频
    video_backend=None,                    # 视频后端
)

# 从本地加载
dataset = LeRobotDataset(
    repo_id="my_local_dataset",
    root="./path/to/dataset"
)
```

#### 数据集创建

```python
# 定义特征
features = {
    "observation.images.cam_front": {
        "dtype": "video",
        "shape": [240, 320, 3],
        "names": ["height", "width", "channels"]
    },
    "observation.state": {
        "dtype": "float32",
        "shape": [7],
        "names": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "gripper"]
    },
    "action": {
        "dtype": "float32",
        "shape": [7],
        "names": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "gripper"]
    }
}

# 创建空数据集
dataset = LeRobotDataset.create(
    repo_id="my_new_dataset",
    fps=30,
    features=features,
    robot_type="so100",
    root="./data",
    use_videos=True
)
```

#### 数据访问

```python
# 数据集长度
len(dataset)  # 帧数

# Episode 数量
dataset.num_episodes

# 采样
dataset[100]  # 第100帧

# 批量加载
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    shuffle=True
)

for batch in dataloader:
    # batch 包含所有特征
    images = batch["observation.images.cam_front"]
    states = batch["observation.state"]
    actions = batch["action"]
    break
```

#### 数据属性

```python
# 元数据访问
meta = dataset.meta

meta.total_episodes      # 剧集总数
meta.total_frames        # 帧总数
meta.fps                 # 帧率
meta.robot_type          # 机器人类型
meta.features            # 特征定义
meta.camera_keys         # 相机键列表
meta.video_keys          # 视频键列表
meta.image_keys          # 图像键列表
meta.tasks               # 任务 DataFrame
meta.stats               # 统计数据
```

### LeRobotDatasetMetadata 类

#### 创建元数据

```python
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

# 创建空数据集元数据
meta = LeRobotDatasetMetadata.create(
    repo_id="my_dataset",
    fps=30,
    features=features,
    robot_type="so100",
    root="./data",
    use_videos=True
)

# 检查数据集信息(不下载数据)
meta = LeRobotDatasetMetadata("lerobot/aloha")
print(f"剧集: {meta.total_episodes}")
print(f"帧数: {meta.total_frames}")
print(f"相机: {meta.camera_keys}")
```

#### 任务管理

```python
# 保存 episode 任务
meta.save_episode_tasks(["pick up", "place object"])

# 获取任务索引
task_idx = meta.get_task_index("pick up")  # 返回 0

# 添加新任务
meta.save_episode_tasks(["new task"])
```

### MultiLeRobotDataset 类

合并多个数据集:

```python
from lerobot.datasets.lerobot_dataset import MultiLeRobotDataset

dataset = MultiLeRobotDataset(
    repo_ids=[
        "lerobot/aloha_sim_transfer_cube_human",
        "lerobot/aloha_sim_insertion_human"
    ],
    root="~/.cache/huggingface/lerobot",
    episodes=None,
    image_transforms=None,
    delta_timestamps=None
)

# 自动合并统计信息
stats = dataset.stats  # 跨数据集聚合

# 数据集索引
batch = dataset[10000]
batch["dataset_index"]  # 源数据集索引
```

**限制:**
- 只保留共有的特征
- 所有数据集的 fps 必须相同
- 视频后端必须兼容

---

## 高级功能

### 数据过滤与选择

#### 按 Episode 加载

```python
# 只加载指定 episodes
dataset = LeRobotDataset(
    "lerobot/aloha",
    episodes=[0, 5, 10, 15]
)
```

#### 按帧数过滤

```python
from lerobot.datasets.dataset_tools import filter_by_frames

# 只保留长度 >= 100 帧的 episodes
filtered = filter_by_frames(dataset, min_frames=100, repo_id="filtered_dataset")
```

#### 按任务过滤

```python
# 只保留特定任务的 episodes
task_mask = dataset.meta.tasks.index == "pick up object"
episode_indices = dataset.meta.episodes[dataset.meta.episodes["tasks"].apply(lambda x: "pick up object" in x)]
```

### 数据集拆分

```python
from lerobot.datasets.dataset_tools import split_dataset

# 按 episode 比例拆分
splits = split_dataset(
    dataset,
    splits={
        "train": 0.8,    # 80% episodes
        "val": 0.2       # 20% episodes
    },
    seed=42
)

train_dataset = splits["train"]
val_dataset = splits["val"]

# 按帧数拆分
splits = split_dataset(
    dataset,
    splits={
        "train": 0.9,    # 90% 帧
        "val": 0.1
    },
    by_episodes=False
)
```

### 特征添加与修改

#### 添加新特征

```python
from lerobot.datasets.dataset_tools import add_features
import numpy as np

# 添加奖励特征
reward_values = np.random.randn(dataset.meta.total_frames).astype(np.float32)

# 添加计算特征
def compute_success(row, episode_index, frame_index):
    return float(frame_index >= len(episode) - 10)

dataset_with_features = add_features(
    dataset,
    features={
        "reward": (
            reward_values,
            {"dtype": "float32", "shape": (1,), "names": None}
        ),
        "success": (
            compute_success,  # 函数
            {"dtype": "float32", "shape": (1,), "names": None}
        )
    },
    repo_id="dataset_with_features"
)
```

#### 删除特征

```python
from lerobot.datasets.dataset_tools import remove_feature

dataset_cleaned = remove_feature(
    dataset,
    feature_names=["success", "reward"],
    repo_id="dataset_cleaned"
)
```

#### 修改特征

```python
from lerobot.datasets.dataset_tools import modify_features

dataset_modified = modify_features(
    dataset,
    add_features={
        "discount": (
            np.ones(dataset.meta.total_frames, dtype=np.float32) * 0.99,
            {"dtype": "float32", "shape": (1,), "names": None}
        )
    },
    remove_features="reward",
    repo_id="dataset_modified"
)
```

### 数据集合并

```python
from lerobot.datasets.dataset_tools import merge_datasets

# 合并多个数据集
merged = merge_datasets(
    [dataset1, dataset2, dataset3],
    output_repo_id="merged_dataset"
)

# 处理特征冲突
merged = merge_datasets(
    [dataset1, dataset2],
    output_repo_id="merged",
    conflict_resolution="union"  # 或 "intersection"
)
```

### Episode 删除

```python
from lerobot.datasets.dataset_tools import delete_episodes

# 删除指定 episodes
filtered = delete_episodes(
    dataset,
    episode_indices=[0, 2, 5, 10],
    repo_id="filtered_dataset"
)
```

### 数据增强

```python
from lerobot.datasets.transforms import ImageTransformsConfig, ImageTransforms

# 配置增强
config = ImageTransformsConfig(
    enable=True,
    max_num_transforms=3,
    random_order=False,
    tfs={
        "brightness": {
            "weight": 1.0,
            "type": "ColorJitter",
            "kwargs": {"brightness": (0.7, 1.3)}
        },
        "affine": {
            "weight": 0.5,
            "type": "RandomAffine",
            "kwargs": {"degrees": (-15, 15), "translate": (0.1, 0.1)}
        }
    }
)

transforms = ImageTransforms(config)

# 应用到数据集
dataset = LeRobotDataset(
    "lerobot/aloha",
    image_transforms=transforms
)
```

---

## 最佳实践

### 数据集设计

#### 1. 特征设计原则

```python
# ✅ 好的设计: 明确命名
features = {
    "observation.images.cam_front": {...},
    "observation.images.wrist": {...},
    "observation.state": {...},
    "action": {...},
    "next.done": {...}
}

# ❌ 不好的设计: 模糊的键名
features = {
    "img": {...},          # 不知道是哪个相机
    "state": {...},        # 不清楚是观测还是目标
    "cmd": {...}
}
```

#### 2. 帧率选择

```python
# 推荐帧率
fps = 30  # 标准机器人操作
fps = 60  # 快速操作/精确控制
fps = 10  # 慢速操作/低带宽场景

# 注意: fps 影响 coordination
delta_timestamps = {
    "action": [t / fps for t in range(50)]  # 50 帧动作块
}
```

#### 3. 视频 vs 图像选择

```python
# 视频模式: 大数据集
if total_size_gb > 10:
    use_videos = True
    data_files_size_in_mb = 100
    video_files_size_in_mb = 200

# 图像模式: 小数据集或需要频繁随机访问
else:
    use_videos = False
    data_files_size_in_mb = 50
```

### 数据采集

#### 1. Episode 长度

```python
# 推荐 episode 长度
min_episode_length = 30   # 至少 1 秒 @ 30fps
max_episode_length = 900  # 最多 30 秒 @ 30fps

# 过滤异常 episodes
dataset = filter_by_frames(dataset, min_frames=30, max_frames=900)
```

#### 2. 任务标注

```python
# 清晰的任务描述
tasks = [
    "pick up the red block",
    "place block in blue bin",
    "push object to target"
]

# 避免模糊描述
# ❌ "do something with the object"
```

#### 3. 时间戳质量

```python
# 检查时间戳间隔
def check_timestamp_quality(dataset, tolerance_s=1e-4):
    for i in range(len(dataset) - 1):
        ts_diff = dataset[i + 1]["timestamp"] - dataset[i]["timestamp"]
        expected = 1 / dataset.fps
        if abs(ts_diff - expected) > tolerance_s:
            logging.warning(f"Bad timestamp at index {i}")
```

### 性能优化

#### 1. 数据加载

```python
# 推荐 DataLoader 配置
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,          # 根据 CPU 核心数调整
    pin_memory=True,        # GPU 训练时加速
    persistent_workers=True, # 保持 workers 存活
    prefetch_factor=2       # 预取批次
)
```

#### 2. Episode 选择

```python
# 只加载需要的 episodes
# 比加载后过滤更高效
dataset = LeRobotDataset(
    "lerobot/aloha",
    episodes=list(range(0, 100, 2))  # 只加载偶数 episodes
)
```

#### 3. 视频后端选择

```python
# Linux/高性能服务器
video_backend = "torchcodec"  # 最快

# Mac/Windows 或特殊编解码器
video_backend = "pyav"      # 最兼容

# 旧版本 torchvision
video_backend = "video_reader"
```

#### 4. 统计预计算

```python
# 统计信息对训练至关重要
dataset.meta.stats = {
    "observation.state": {
        "mean": compute_mean(...),
        "std": compute_std(...),
        "min": compute_min(...),
        "max": compute_max(...),
        "q01": compute_quantile(..., 0.01),
        "q99": compute_quantile(..., 0.99)
    }
}

# 训练时使用
def normalize_state(state, stats):
    return (state - stats["mean"]) / (stats["std"] + 1e-6)
```

### 数据验证

#### 1. 特征验证

```python
from lerobot.datasets.utils import validate_frame

# 采集时验证
try:
    validate_frame(frame, dataset.features)
except ValueError as e:
    logging.error(f"Invalid frame: {e}")
    # 丢弃或修复
```

#### 2. Episode 验证

```python
from lerobot.datasets.utils import validate_episode_buffer

# 保存前验证
try:
    validate_episode_buffer(episode_buffer, dataset.meta.total_episodes, dataset.features)
except ValueError as e:
    logging.error(f"Invalid episode: {e}")
```

### 云端存储

#### 1. 推送到 Hub

```python
# 完整推送
dataset.push_to_hub(
    repo_id="username/my_dataset",
    private=False,
    push_videos=True  # 包含视频文件
)

# 部分推送(只推送小文件)
dataset.push_to_hub(
    repo_id="username/my_dataset",
    allow_patterns=["meta/*", "data/chunk-000/*"]
)
```

#### 2. 流式加载

```python
# 不下载完整数据集
dataset = LeRobotDataset(
    "username/my_dataset",
    download_videos=False,  # 只下载元数据
    episodes=[0, 1, 2]      # 只下载指定 episodes
)

# 按需下载
dataset.download(download_videos=True)  # 后续下载视频
```

---

## 版本兼容性

### 版本格式

LeRobot 使用语义化版本:

```
v{major}.{minor}.{patch}
```

- **major**: 破坏性变更,不向后兼容
- **minor**: 新功能,向后兼容
- **patch**: Bug 修复,完全兼容

### 兼容性检查

```python
from lerobot.datasets.utils import check_version_compatibility

# 检查数据集版本
check_version_compatibility(
    repo_id="lerobot/aloha",
    version_to_check="v3.0.0",
    current_version="v3.0.0"
)
```

**规则:**
- major 版本不同 → 错误(需要转换)
- minor 版本不同 → 警告(功能可能不完整)
- patch 版本不同 → 正常(兼容)

### 版本选择

```python
from lerobot.datasets.utils import get_safe_version

# 自动选择兼容版本
version = get_safe_version("lerobot/aloha", "v3.0.0")
# 如果 v3.0.0 不存在,选择最新的 v3.x.x < v3.0.0

dataset = LeRobotDataset(
    "lerobot/aloha",
    revision=version
)
```

### 版本转换

从 v2.1 升级到 v3.0:

```bash
python src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py \
    --input-dir ./dataset_v21 \
    --output-dir ./dataset_v30 \
    --fps 30
```

---

## 常见问题

### 1. 时间戳不匹配

**问题**: `ValueError: timestamps violate tolerance`

**原因**: delta_timestamps 不是 1/fps 的倍数

**解决**:
```python
# 检查 fps
fps = dataset.fps

# 调整 delta_timestamps
delta_timestamps = {
    "action": [round(t * fps) / fps for t in target_timestamps]
}
```

### 2. 视频解码慢

**问题**: 视频加载速度慢

**解决**:
```python
# 使用更快的后端
dataset = LeRobotDataset("...", video_backend="torchcodec")

# 增加关键帧
# 重新编码视频: keyframe_interval = fps (1秒)

# 转换到图像模式(小数据集)
dataset = LeRobotDataset("...", download_videos=False)
# 然后转换为图像模式...
```

### 3. 内存不足

**问题**: 数据集太大,无法全部加载

**解决**:
```python
# 1. Episode 选择
dataset = LeRobotDataset("...", episodes=range(0, 100))

# 2. 流式加载
dataset = LeRobotDataset("...", download_videos=False)

# 3. 降低批次大小
dataloader = DataLoader(dataset, batch_size=8)  # 而不是 32

# 4. 使用 IterableDataset
from lerobot.datasets.streaming_dataset import StreamingDataset
dataset = StreamingDataset("...")
```

### 4. 特征不匹配

**问题**: `ValueError: Feature mismatch`

**原因**: Episode 数据与特征定义不匹配

**解决**:
```python
# 验证帧
from lerobot.datasets.utils import validate_frame

validate_frame(frame, dataset.features)

# 检查 keys
expected = set(dataset.features)
actual = set(frame)
missing = expected - actual
extra = actual - expected
```

### 5. 上传/下载失败

**问题**: 网络错误或文件过大

**解决**:
```python
# 使用上传大文件夹
dataset.push_to_hub(
    repo_id="username/dataset",
    upload_large_folder=True  # 使用 git-lfs
)

# 分片上传
for chunk_idx in range(num_chunks):
    dataset.push_to_hub(
        repo_id="username/dataset",
        allow_patterns=[f"data/chunk-{chunk_idx:03d}/**"]
    )
```

---

## 示例代码

### 完整数据采集流程

```python
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 1. 创建数据集
features = {
    "observation.images.cam_front": {
        "dtype": "video",
        "shape": [240, 320, 3],
        "names": ["height", "width", "channels"]
    },
    "observation.state": {
        "dtype": "float32",
        "shape": [7],
        "names": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "gripper"]
    },
    "action": {
        "dtype": "float32",
        "shape": [7],
        "names": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "gripper"]
    }
}

dataset = LeRobotDataset.create(
    repo_id="my_robot_data",
    fps=30,
    features=features,
    robot_type="so100"
)

# 2. 采集 episodes
for episode_idx in range(100):
    # 开始新 episode
    dataset.episode_buffer = dataset.create_episode_buffer()

    # 采集帧
    for frame_idx in range(episode_length):
        # 获取传感器数据
        cam_image = robot.get_camera_image()
        joint_states = robot.get_joint_states()

        # 构建 frame
        frame = {
            "observation.images.cam_front": cam_image,
            "observation.state": joint_states,
            "action": np.zeros(7),  # 或实际动作
            "task": "pick and place"  # 任务标签
        }

        # 添加到 buffer
        dataset.add_frame(frame)

    # 保存 episode
    dataset.save_episode()

# 3. 计算统计
dataset.meta.stats = compute_stats(dataset)

# 4. 上传
dataset.push_to_hub("username/my_robot_data")
```

### 训练数据管道

```python
import torch
from torch.utils.data import DataLoader
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 1. 加载数据集
dataset = LeRobotDataset(
    "username/my_robot_data",
    delta_timestamps={
        "observation.images.cam_front": [-0.5, -0.2, 0.0],
        "observation.state": [-1.0, -0.5, -0.2, 0.0],
        "action": [t / 30 for t in range(16)]  # 未来 16 帧
    }
)

# 2. 数据增强
transforms = ImageTransforms(ImageTransformsConfig(
    enable=True,
    max_num_transforms=2,
    tfs={
        "brightness": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"brightness": (0.8, 1.2)}},
        "affine": {"weight": 0.5, "type": "RandomAffine", "kwargs": {"degrees": (-5, 5)}}
    }
))

# 重新应用变换
dataset.image_transforms = transforms

# 3. 创建 DataLoader
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# 4. 训练循环
for batch in train_loader:
    images = batch["observation.images.cam_front"]  # (B, 3, C, H, W)
    states = batch["observation.state"]              # (B, 4, state_dim)
    actions = batch["action"]                        # (B, 16, action_dim)

    # 模型前向传播
    pred_actions = policy(images, states)

    # 计算损失
    loss = criterion(pred_actions, actions)

    # 反向传播
    loss.backward()
    optimizer.step()
```

### 数据集验证脚本

```python
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import validate_frame

def validate_dataset(repo_id):
    """验证数据集的完整性和质量。"""
    print(f"加载数据集: {repo_id}")
    dataset = LeRobotDataset(repo_id)

    errors = []

    # 1. 检查 episode 数量
    if dataset.meta.total_episodes == 0:
        errors.append("没有 episodes")

    # 2. 检查帧数
    if dataset.meta.total_frames == 0:
        errors.append("没有帧")

    # 3. 检查时间戳
    print("检查时间戳...")
    for i in range(min(1000, len(dataset) - 1)):
        ts_diff = dataset[i + 1]["timestamp"] - dataset[i]["timestamp"]
        expected = 1 / dataset.fps
        if abs(ts_diff - expected) > 1e-4:
            errors.append(f"错误的时间戳间隔在索引 {i}")
            break

    # 4. 检查视频文件
    if dataset.meta.video_keys:
        print("检查视频文件...")
        for ep_idx in range(min(10, dataset.meta.total_episodes)):
            for vid_key in dataset.meta.video_keys:
                video_path = dataset.meta.get_video_file_path(ep_idx, vid_key)
                if not video_path.exists():
                    errors.append(f"缺失视频: episode {ep_idx}, {vid_key}")

    # 5. 检查特征完整性
    print("检查特征...")
    for i in range(min(100, len(dataset))):
        try:
            frame = dataset[i]
            validate_frame(frame, dataset.features)
        except ValueError as e:
            errors.append(f"特征验证失败在索引 {i}: {e}")
            break

    # 6. 检查统计信息
    print("检查统计信息...")
    if dataset.meta.stats is None:
        errors.append("缺少统计信息")

    # 报告
    if errors:
        print("\n发现错误:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("\n✅ 数据集验证通过!")
        return True

# 使用示例
if __name__ == "__main__":
    validate_dataset("lerobot/aloha_sim_transfer_cube_human")
```

---

## 相关资源和链接

### 官方资源

- **LeRobot GitHub**: https://github.com/huggingface/lerobot
- **Hugging Face Hub**: https://huggingface.co/lerobot
- **数据集示例**: https://huggingface.co/datasets?other=LeRobot

### 相关格式

- **Hugging Face Datasets**: https://huggingface.co/docs/datasets
- **Apache Parquet**: https://parquet.apache.org/
- **AV1 Codec**: https://aomedia.org/av1/

### 技术参考

- **Robotics Transformers**: https://arxiv.org/abs/2212.06824
- **RT-X Dataset**: https://arxiv.org/abs/2310.08864
- **Open X-Embodiment**: https://arxiv.org/abs/2310.08864

---

## 更新日志

### v3.0 (当前版本)

**主要改进:**
- 引入智能分块系统优化存储
- 支持视频和图像混合存储
- 增强的 delta_timestamps 系统
- 改进的统计信息(包含分位数)
- 异步图像写入支持
- 视频批量编码

**API 变更:**
- `LeRobotDataset.create()` 替代旧构造函数
- `LeRobotDatasetMetadata` 分离元数据操作
- `MultiLeRobotDataset` 支持多数据集合并
- 统一使用 Parquet 存储元数据

### v2.1

- Episode 元数据使用 JSONL
- 基础视频支持
- 简单时序查询

### v2.0

- 引入 Hugging Face Datasets 集成
- Parquet 作为主要存储格式
- 基础图像变换

---

## 贡献指南

### 实现新策略

当添加新策略时,需要:

1. 更新 `lerobot/__init__.py`:
```python
available_policies = [
    ...,
    "my_new_policy"
]
```

2. 在策略类中设置 `name` 属性:
```python
class MyNewPolicy:
    name = "my_new_policy"
    # ...
```

3. 更新测试:
```python
# tests/test_available.py
from lerobot.policies.my_new_policy import MyNewPolicy
```

### 添加新数据集

1. 更新 `lerobot/__init__.py`:
```python
available_datasets = [
    ...,
    "username/my_dataset"
]

available_datasets_per_env = {
    ...,
    "aloha": ["username/my_dataset", ...]
}
```

2. 推送数据集到 Hub:
```bash
huggingface-cli upload username/my_dataset ./dataset --repo-type=dataset
```

### 报告问题

在 GitHub 上报告问题时,请包含:

- LeRobot 版本
- 数据集 ID 和版本
- 最小复现代码
- 完整错误堆栈
- 环境信息(OS, PyTorch, Python 版本)

---

## 附录

### A. 配置文件示例

#### 训练配置 (Hydra/Draccus)

```yaml
# @package _global_
defaults:
  - override /policy: act
  - override /env: aloha

# 数据集
dataset_repo_id: lerobot/aloha_sim_transfer_cube_human
image_transforms:
  enable: true
  max_num_transforms: 2
  tfs:
    brightness:
      weight: 1.0
      type: ColorJitter
      kwargs:
        brightness: [0.8, 1.2]

delta_timestamps:
  observation.images.cam_front: [-0.5, -0.2, 0.0]
  action: [t/fps for t in range(50)]

# 数据加载
batch_size: 32
num_workers: 4
pin_memory: true
```

#### 数据集配置

```yaml
dataset:
  repo_id: my_robot_data
  fps: 30
  use_videos: true
  data_files_size_in_mb: 100
  video_files_size_in_mb: 200
  chunks_size: 1000

robot:
  type: so100
  cameras:
    cam_front:
      resolution: [240, 320]
      fps: 30
  state_dim: 7
  action_dim: 7
```

### B. 性能基准

#### 存储效率

| 数据类型 | 图像模式 | 视频模式 | 压缩率 |
|---------|---------|---------|--------|
| RGB 240x320 | 100% | 5-10% | 10-20x |
| RGB 480x640 | 100% | 5-8% | 12-20x |
| 深度图 | 100% | 10-15% | 6-10x |

#### 加载速度 (RTX 4090)

| 后端 | 分辨率 | 随机访问 | 顺序读取 | GPU 解码 |
|------|--------|---------|---------|---------|
| torchcodec | 224x224 | 5ms | 0.5ms | ✅ Yes |
| pyav | 224x224 | 20ms | 1ms | ❌ No |
| video_reader | 224x224 | 15ms | 1ms | ⚠️ Limited |

#### 内存使用

| 数据集大小 | Chunk 大小 | 内存使用 | 加载时间 |
|-----------|-----------|---------|---------|
| 1 GB | 32 | ~200 MB | 2s |
| 10 GB | 32 | ~500 MB | 10s |
| 100 GB | 32 | ~1 GB | 30s |
| 1 TB | 32 | ~2 GB | 60s |

### C. 术语表

| 术语 | 定义 |
|------|------|
| Episode | 一次完整的机器人任务执行 |
| Frame | 在特定时间点的完整数据快照 |
| Feature | 数据集的特定数据模态 |
| Chunk | 文件集合的目录分组 |
| FPS | Frames Per Second, 帧率 |
| Delta Timestamps | 相对时间戳偏移列表 |
| Padding | Episode 边界的填充标记 |
| Codec | 视频编解码器(AV1/H.264) |
| Keyframe | 视频关键帧(I-frame) |
| Parquet | 列式存储文件格式 |
| HWC | Height, Width, Channels 维度顺序 |
| CHW | Channels, Height, Width 维度顺序(PyTorch) |

---

## 许可证

此文档遵循 Apache License 2.0 协议。

**版权**: 2024 Hugging Face Inc. 团队。保留所有权利。

---

**文档版本**: v1.0
**最后更新**: 2025-01
**适用 LeRobot 版本**: v3.0+
