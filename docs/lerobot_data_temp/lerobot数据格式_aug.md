# LeRobot 数据格式技术文档 (v3.0)

> 本文档详细解析 LeRobot 数据集格式 v3.0 的完整技术规范，包括目录结构、元数据格式、数据存储、视频编解码、统计计算等核心内容。

## 目录

1. [版本概述](#1-版本概述)
2. [目录结构详解](#2-目录结构详解)
3. [元数据文件规范](#3-元数据文件规范)
4. [数据存储格式](#4-数据存储格式)
5. [视频处理机制](#5-视频处理机制)
6. [特征系统](#6-特征系统)
7. [统计计算](#7-统计计算)
8. [核心类与API](#8-核心类与api)
9. [时间戳与delta_timestamps机制](#9-时间戳与delta_timestamps机制)
10. [流式数据集](#10-流式数据集)

---

## 1. 版本概述

### 1.1 版本标识

```python
CODEBASE_VERSION = "v3.0"  # 定义于 lerobot_dataset.py
```

### 1.2 设计原则

v3.0 相比 v2.1 的核心改进：

| 特性 | v2.1 | v3.0 |
|------|------|------|
| 文件组织 | 每个episode一个文件 | 多episode合并到分片文件 |
| 存储效率 | 大量小文件 | 少量大文件，减少inode压力 |
| 流式支持 | 有限 | 原生支持 |
| 元数据 | 分散 | 集中在meta目录 |

### 1.3 默认配置常量

```python
# 来自 utils.py
DEFAULT_CHUNK_SIZE = 1000           # 每个chunk目录最大文件数
DEFAULT_DATA_FILE_SIZE_IN_MB = 100  # 数据文件最大大小(MB)
DEFAULT_VIDEO_FILE_SIZE_IN_MB = 200 # 视频文件最大大小(MB)
```

---

## 2. 目录结构详解

### 2.1 完整目录树

```
dataset_root/
├── meta/
│   ├── info.json                           # 数据集元信息
│   ├── stats.json                          # 全局统计信息
│   ├── tasks.parquet                       # 任务描述映射
│   └── episodes/
│       ├── chunk-000/
│       │   ├── file-000.parquet            # episode元数据分片
│       │   ├── file-001.parquet
│       │   └── ...
│       ├── chunk-001/
│       │   └── ...
│       └── ...
├── data/
│   ├── chunk-000/
│   │   ├── file-000.parquet                # 帧数据分片
│   │   ├── file-001.parquet
│   │   └── ...
│   ├── chunk-001/
│   │   └── ...
│   └── ...
└── videos/
    ├── observation.images.laptop/          # 按相机key组织
    │   ├── chunk-000/
    │   │   ├── file-000.mp4
    │   │   ├── file-001.mp4
    │   │   └── ...
    │   ├── chunk-001/
    │   │   └── ...
    │   └── ...
    ├── observation.images.phone/
    │   └── ...
    └── ...
```

### 2.2 路径模板常量

```python
# 来自 utils.py
INFO_PATH = "meta/info.json"
STATS_PATH = "meta/stats.json"
DEFAULT_TASKS_PATH = "meta/tasks.parquet"
DEFAULT_EPISODES_PATH = "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
DEFAULT_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
DEFAULT_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
DEFAULT_IMAGE_PATH = "images/{image_key}/episode_{episode_index:06d}/frame_{frame_index:06d}.png"

# 分片文件命名模式
CHUNK_FILE_PATTERN = "chunk-{chunk_index:03d}/file-{file_index:03d}"
```

---

## 3. 元数据文件规范

### 3.1 info.json 完整结构

```json
{
    "codebase_version": "v3.0",
    "robot_type": "koch",
    "total_episodes": 50,
    "total_frames": 20000,
    "total_tasks": 1,
    "fps": 30,
    "chunks_size": 1000,
    "data_files_size_in_mb": 100,
    "video_files_size_in_mb": 200,
    "splits": {
        "train": "0:50"
    },
    "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
    "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    "features": {
        "timestamp": {
            "dtype": "float32",
            "shape": [1],
            "names": null
        },
        "frame_index": {
            "dtype": "int64",
            "shape": [1],
            "names": null
        },
        "episode_index": {
            "dtype": "int64",
            "shape": [1],
            "names": null
        },
        "index": {
            "dtype": "int64",
            "shape": [1],
            "names": null
        },
        "task_index": {
            "dtype": "int64",
            "shape": [1],
            "names": null
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [6],
            "names": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
        },
        "action": {
            "dtype": "float32",
            "shape": [6],
            "names": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
        },
        "observation.images.laptop": {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.fps": 30.0,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": false,
                "has_audio": false
            }
        }
    }
}
```

### 3.2 info.json 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `codebase_version` | string | 数据格式版本，当前为 "v3.0" |
| `robot_type` | string | 机器人类型标识 |
| `total_episodes` | int | 总episode数量 |
| `total_frames` | int | 总帧数 |
| `total_tasks` | int | 任务种类数 |
| `fps` | int | 数据采集帧率 |
| `chunks_size` | int | 每个chunk目录最大文件数 |
| `data_files_size_in_mb` | int | 数据文件大小上限(MB) |
| `video_files_size_in_mb` | int | 视频文件大小上限(MB) |
| `splits` | dict | 数据集划分，格式为 "start:end" |
| `data_path` | string | 数据文件路径模板 |
| `video_path` | string | 视频文件路径模板 |
| `features` | dict | 特征定义字典 |

### 3.3 stats.json 结构

统计信息用于训练时的数据归一化：

```json
{
    "observation.state": {
        "min": [-1.5, -2.0, -1.8, -3.14, -1.57, -3.14],
        "max": [1.5, 2.0, 1.8, 3.14, 1.57, 3.14],
        "mean": [0.1, 0.2, 0.15, 0.0, 0.0, 0.0],
        "std": [0.5, 0.6, 0.4, 1.0, 0.8, 1.2],
        "count": 20000,
        "q01": [-1.2, -1.8, -1.5, -2.8, -1.4, -2.9],
        "q10": [-0.8, -1.2, -1.0, -2.0, -1.0, -2.0],
        "q50": [0.1, 0.2, 0.15, 0.0, 0.0, 0.0],
        "q90": [1.0, 1.4, 1.2, 2.0, 1.0, 2.0],
        "q99": [1.3, 1.9, 1.6, 2.9, 1.5, 3.0]
    },
    "action": {
        "min": [...],
        "max": [...],
        "mean": [...],
        "std": [...],
        "count": 20000,
        "q01": [...],
        "q10": [...],
        "q50": [...],
        "q90": [...],
        "q99": [...]
    }
}
```

**统计字段说明：**

| 字段 | 说明 |
|------|------|
| `min` | 最小值 |
| `max` | 最大值 |
| `mean` | 均值 |
| `std` | 标准差 |
| `count` | 样本数量 |
| `q01` | 1%分位数 |
| `q10` | 10%分位数 |
| `q50` | 50%分位数(中位数) |
| `q90` | 90%分位数 |
| `q99` | 99%分位数 |

### 3.4 tasks.parquet 结构

任务描述文件，使用Parquet格式存储：

| 列名 | 类型 | 说明 |
|------|------|------|
| `task_index` | int64 | 任务索引(0开始) |
| (index) | string | 任务描述文本(作为DataFrame索引) |

示例数据：
```
task_index | (index/task_description)
-----------+---------------------------
0          | "Pick up the red cube"
1          | "Place the cube in the box"
```

### 3.5 episodes/*.parquet 结构

每个episode的元数据记录：

| 列名 | 类型 | 说明 |
|------|------|------|
| `episode_index` | int64 | episode索引 |
| `tasks` | list[str] | 该episode包含的任务列表 |
| `length` | int64 | episode帧数 |
| `dataset_from_index` | int64 | 在全局数据集中的起始帧索引 |
| `dataset_to_index` | int64 | 在全局数据集中的结束帧索引 |
| `meta/episodes/chunk_index` | int64 | 元数据所在chunk索引 |
| `meta/episodes/file_index` | int64 | 元数据所在file索引 |
| `data/chunk_index` | int64 | 数据所在chunk索引 |
| `data/file_index` | int64 | 数据所在file索引 |
| `videos/{key}/chunk_index` | int64 | 视频所在chunk索引 |
| `videos/{key}/file_index` | int64 | 视频所在file索引 |
| `videos/{key}/from_timestamp` | float64 | 视频中的起始时间戳 |
| `videos/{key}/to_timestamp` | float64 | 视频中的结束时间戳 |
| `stats/{feature}/min` | list[float] | episode级别统计 |
| `stats/{feature}/max` | list[float] | episode级别统计 |
| `stats/{feature}/mean` | list[float] | episode级别统计 |
| `stats/{feature}/std` | list[float] | episode级别统计 |

---

## 4. 数据存储格式

### 4.1 Parquet数据文件结构

`data/chunk-XXX/file-XXX.parquet` 文件包含逐帧数据：

| 列名 | 类型 | 说明 |
|------|------|------|
| `timestamp` | float32 | 帧时间戳(秒) |
| `frame_index` | int64 | episode内帧索引 |
| `episode_index` | int64 | episode索引 |
| `index` | int64 | 全局帧索引 |
| `task_index` | int64 | 任务索引 |
| `observation.state` | list[float32] | 机器人状态观测 |
| `action` | list[float32] | 动作指令 |
| ... | ... | 其他自定义特征 |

### 4.2 分片策略

数据文件按以下规则分片：

1. **大小限制**: 单个文件不超过 `data_files_size_in_mb` (默认100MB)
2. **Chunk组织**: 每个chunk目录最多 `chunks_size` 个文件(默认1000)
3. **索引更新**: 当文件大小接近限制时，创建新文件并更新chunk/file索引

```python
def update_chunk_file_indices(chunk_idx: int, file_idx: int, chunks_size: int) -> tuple[int, int]:
    """更新chunk和file索引"""
    file_idx += 1
    if file_idx >= chunks_size:
        chunk_idx += 1
        file_idx = 0
    return chunk_idx, file_idx
```

---

## 5. 视频处理机制

### 5.1 支持的编解码器

```python
# 编码器优先级
ENCODING_CODECS = ["libsvtav1", "h264", "hevc"]

# 解码后端
VIDEO_BACKENDS = ["torchcodec", "pyav", "video_reader"]
```

### 5.2 视频编码参数

```python
def encode_video_frames(
    imgs_dir: Path,
    video_path: Path,
    fps: int,
    vcodec: str = "libsvtav1",
    pix_fmt: str = "yuv420p",
    g: int | None = 2,
    crf: int | None = 30,
    fast_decode: int = 0,
    log_level: str | None = "error",
    overwrite: bool = False,
) -> None:
    """
    将图像序列编码为视频

    Args:
        imgs_dir: 图像目录
        video_path: 输出视频路径
        fps: 帧率
        vcodec: 编码器 (libsvtav1/h264/hevc)
        pix_fmt: 像素格式
        g: GOP大小(关键帧间隔)
        crf: 质量参数(越小质量越高)
        fast_decode: 快速解码优化
    """
```

### 5.3 视频解码

```python
def decode_video_frames(
    video_path: Path,
    timestamps: list[float],
    tolerance_s: float,
    backend: str | None = None,
) -> torch.Tensor:
    """
    从视频中解码指定时间戳的帧

    Args:
        video_path: 视频文件路径
        timestamps: 要解码的时间戳列表
        tolerance_s: 时间戳容差(秒)
        backend: 解码后端

    Returns:
        torch.Tensor: 形状为 [T, C, H, W] 的帧张量
    """
```

### 5.4 VideoFrame数据类

```python
@dataclass
class VideoFrame:
    """视频帧的占位符，用于延迟加载"""
    path: str           # 视频文件路径
    timestamp: float    # 帧时间戳
```

### 5.5 视频信息获取

```python
def get_video_info(video_path: Path | str) -> dict:
    """
    获取视频元信息

    Returns:
        {
            "video.fps": 30.0,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False
        }
    """
```

---

## 6. 特征系统

### 6.1 默认特征定义

```python
DEFAULT_FEATURES = {
    "timestamp": {
        "dtype": "float32",
        "shape": (1,),
        "names": None,
    },
    "frame_index": {
        "dtype": "int64",
        "shape": (1,),
        "names": None,
    },
    "episode_index": {
        "dtype": "int64",
        "shape": (1,),
        "names": None,
    },
    "index": {
        "dtype": "int64",
        "shape": (1,),
        "names": None,
    },
    "task_index": {
        "dtype": "int64",
        "shape": (1,),
        "names": None,
    },
}
```

### 6.2 支持的数据类型

| dtype | 说明 | 存储格式 |
|-------|------|----------|
| `float32` | 32位浮点数 | Parquet |
| `float64` | 64位浮点数 | Parquet |
| `int64` | 64位整数 | Parquet |
| `int32` | 32位整数 | Parquet |
| `bool` | 布尔值 | Parquet |
| `string` | 字符串 | Parquet |
| `image` | 图像数据 | PNG文件 |
| `video` | 视频数据 | MP4文件 |

### 6.3 特征命名约定

```
observation.state           # 机器人状态
observation.images.<camera> # 相机图像
observation.effort          # 力矩/力反馈
action                      # 动作指令
action.gripper              # 夹爪动作
timestamp                   # 时间戳
frame_index                 # 帧索引
episode_index               # episode索引
task_index                  # 任务索引
```

### 6.4 HuggingFace Features映射

```python
def get_hf_features_from_features(features: dict) -> datasets.Features:
    """将LeRobot特征定义转换为HuggingFace Features"""
    hf_features = {}
    for key, ft in features.items():
        if ft["dtype"] == "video":
            hf_features[key] = VideoFrame  # 延迟加载
        elif ft["dtype"] == "image":
            hf_features[key] = datasets.Image()
        elif ft["dtype"] == "string":
            hf_features[key] = datasets.Value("string")
        else:
            # 数值类型
            if len(ft["shape"]) == 1 and ft["shape"][0] == 1:
                hf_features[key] = datasets.Value(ft["dtype"])
            else:
                hf_features[key] = datasets.Sequence(
                    datasets.Value(ft["dtype"]),
                    length=ft["shape"][0]
                )
    return datasets.Features(hf_features)
```

---

## 7. 统计计算

### 7.1 统计计算流程

```python
# 来自 compute_stats.py
DEFAULT_QUANTILES = [0.01, 0.10, 0.50, 0.90, 0.99]

def compute_episode_stats(episode_buffer: dict, features: dict) -> dict:
    """
    计算单个episode的统计信息

    Args:
        episode_buffer: episode数据缓冲区
        features: 特征定义

    Returns:
        {
            "feature_name": {
                "min": [...],
                "max": [...],
                "mean": [...],
                "std": [...],
                "count": int,
                "q01": [...],
                "q10": [...],
                "q50": [...],
                "q90": [...],
                "q99": [...]
            }
        }
    """
```

### 7.2 RunningQuantileStats类

用于增量计算分位数统计：

```python
class RunningQuantileStats:
    """
    增量分位数统计计算器
    使用t-digest算法近似计算分位数
    """
    def __init__(self, quantiles: list[float] = DEFAULT_QUANTILES):
        self.quantiles = quantiles
        self.digests = {}  # 每个特征维度一个t-digest

    def update(self, data: np.ndarray, feature_name: str):
        """更新统计"""

    def get_quantiles(self, feature_name: str) -> dict:
        """获取分位数结果"""
```

### 7.3 统计聚合

```python
def aggregate_stats(stats_list: list[dict]) -> dict:
    """
    聚合多个episode的统计信息

    使用加权平均合并mean/std，取全局min/max
    """
```

---

## 8. 核心类与API

### 8.1 LeRobotDatasetMetadata

元数据管理类，负责加载和管理数据集元信息：

```python
class LeRobotDatasetMetadata:
    """
    数据集元数据管理器

    Attributes:
        repo_id: 数据集仓库ID
        root: 本地存储路径
        info: info.json内容
        stats: stats.json内容
        tasks: tasks.parquet内容(DataFrame)
        episodes: episodes元数据列表
    """

    def __init__(
        self,
        repo_id: str,
        root: Path,
        revision: str | None = None,
        force_cache_sync: bool = False,
    ):
        """初始化并加载元数据"""

    # 核心属性
    @property
    def fps(self) -> int: ...
    @property
    def features(self) -> dict: ...
    @property
    def total_episodes(self) -> int: ...
    @property
    def total_frames(self) -> int: ...
    @property
    def video_keys(self) -> list[str]: ...
    @property
    def image_keys(self) -> list[str]: ...
    @property
    def camera_keys(self) -> list[str]: ...

    # 文件路径获取
    def get_data_file_path(self, ep_index: int) -> Path: ...
    def get_video_file_path(self, ep_index: int, vid_key: str) -> Path: ...

    # 数据保存
    def save_episode(self, episode_index, episode_length, episode_tasks, episode_stats, episode_metadata): ...
```

### 8.2 LeRobotDataset

主数据集类，继承自 `torch.utils.data.Dataset`：

```python
class LeRobotDataset(torch.utils.data.Dataset):
    """
    LeRobot数据集主类

    支持两种使用模式：
    1. 加载已有数据集（本地或Hub）
    2. 创建新数据集（使用create类方法）
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        batch_encoding_size: int = 1,
    ):
        """
        Args:
            repo_id: 数据集仓库ID
            root: 本地存储路径
            episodes: 要加载的episode索引列表
            image_transforms: 图像变换函数
            delta_timestamps: 时间窗口配置
            tolerance_s: 时间戳容差
            revision: Git版本
            force_cache_sync: 强制同步缓存
            download_videos: 是否下载视频
            video_backend: 视频解码后端
            batch_encoding_size: 批量编码大小
        """

    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        features: dict,
        robot_type: str | None = None,
        **kwargs,
    ) -> "LeRobotDataset":
        """创建新的空数据集"""

    def __len__(self) -> int:
        """返回数据集帧数"""
        return self.num_frames

    def __getitem__(self, idx: int) -> dict:
        """
        获取指定索引的数据

        Returns:
            {
                "timestamp": float,
                "frame_index": int,
                "episode_index": int,
                "index": int,
                "task_index": int,
                "task": str,  # 任务描述文本
                "observation.state": torch.Tensor,
                "action": torch.Tensor,
                "observation.images.xxx": torch.Tensor,  # [C, H, W]
                ...
            }
        """

    # 数据录制API
    def add_frame(self, frame: dict) -> None:
        """添加一帧数据到episode缓冲区"""

    def save_episode(self, episode_data: dict | None = None, parallel_encoding: bool = True) -> None:
        """保存当前episode到磁盘"""

    def finalize(self) -> None:
        """完成数据集写入，关闭所有writer"""

    def push_to_hub(self, **kwargs) -> None:
        """推送数据集到HuggingFace Hub"""
```

### 8.3 数据集创建示例

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 创建新数据集
dataset = LeRobotDataset.create(
    repo_id="user/my-dataset",
    fps=30,
    features={
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["j1", "j2", "j3", "j4", "j5", "j6"],
        },
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["j1", "j2", "j3", "j4", "j5", "j6"],
        },
        "observation.images.camera": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        },
    },
    robot_type="koch",
)

# 录制数据
for episode_idx in range(10):
    for frame_idx in range(100):
        frame = {
            "observation.state": np.random.randn(6).astype(np.float32),
            "action": np.random.randn(6).astype(np.float32),
            "observation.images.camera": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            "task": "Pick up the cube",
        }
        dataset.add_frame(frame)
    dataset.save_episode()

dataset.finalize()
dataset.push_to_hub()
```

---

## 9. 时间戳与delta_timestamps机制

### 9.1 delta_timestamps概念

`delta_timestamps` 允许在获取单帧数据时，同时获取相对于当前帧的历史或未来帧：

```python
delta_timestamps = {
    "observation.images.camera": [-0.1, 0.0],      # 当前帧和0.1秒前的帧
    "observation.state": [-0.2, -0.1, 0.0],        # 3帧历史状态
    "action": [0.0, 0.033, 0.066, 0.1],            # 当前和未来3帧动作
}
```

### 9.2 delta_indices计算

```python
def get_delta_indices(delta_timestamps: dict, fps: int) -> dict:
    """
    将时间偏移转换为帧索引偏移

    Args:
        delta_timestamps: {"feature": [t1, t2, ...]}
        fps: 帧率

    Returns:
        {"feature": [idx1, idx2, ...]}
    """
    delta_indices = {}
    for key, timestamps in delta_timestamps.items():
        delta_indices[key] = [round(t * fps) for t in timestamps]
    return delta_indices
```

### 9.3 查询机制

```python
def _get_query_indices(self, idx: int, ep_idx: int) -> tuple[dict, dict]:
    """
    计算需要查询的帧索引

    处理episode边界情况：
    - 如果请求的帧超出episode范围，使用边界帧填充
    - 返回padding信息用于标记哪些帧是填充的
    """
```

### 9.4 使用示例

```python
dataset = LeRobotDataset(
    repo_id="user/dataset",
    delta_timestamps={
        "observation.images.camera": [-0.1, 0.0],
        "action": [0.0, 0.033, 0.066],
    },
)

item = dataset[100]
# item["observation.images.camera"].shape = [2, C, H, W]  # 2帧
# item["action"].shape = [3, 6]  # 3帧动作
```

---

## 10. 流式数据集

### 10.1 StreamingLeRobotDataset

支持从Hub流式读取数据，无需完整下载：

```python
class StreamingLeRobotDataset(torch.utils.data.IterableDataset):
    """
    流式LeRobot数据集

    特点：
    - 无需下载完整数据集
    - 使用Backtrackable迭代器支持delta_timestamps
    - 适合大规模数据集或带宽受限场景
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        streaming: bool = True,
        buffer_size: int = 1000,
        max_num_shards: int = 16,
        seed: int = 42,
        shuffle: bool = True,
    ):
        """
        Args:
            streaming: 是否启用流式模式
            buffer_size: 缓冲区大小(用于shuffle和lookback)
            max_num_shards: 重分片数量
            shuffle: 是否打乱数据
        """
```

### 10.2 Backtrackable迭代器

```python
class Backtrackable:
    """
    支持回溯的迭代器包装器

    维护一个有界缓冲区，允许访问之前的元素
    用于支持delta_timestamps的负时间偏移
    """
    def __init__(self, iterator: Iterator, buffer_size: int):
        self.buffer = deque(maxlen=buffer_size)

    def lookback(self, n: int):
        """获取n步之前的元素"""

    def lookahead(self, n: int):
        """获取n步之后的元素(需要预取)"""
```

### 10.3 流式使用示例

```python
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset

dataset = StreamingLeRobotDataset(
    repo_id="lerobot/aloha_sim_insertion_human",
    delta_timestamps={
        "observation.images.top": [-0.1, 0.0],
        "action": [0.0, 0.033],
    },
    streaming=True,
    buffer_size=1000,
)

# 流式迭代
for i, item in enumerate(dataset):
    print(f"Frame {i}: episode={item['episode_index']}")
    if i >= 100:
        break
```

---

## 附录

### A. 常用工具函数

```python
# 加载元数据
from lerobot.datasets.utils import load_info, load_stats, load_tasks, load_episodes

info = load_info(root_path)
stats = load_stats(root_path)
tasks = load_tasks(root_path)
episodes = load_episodes(root_path)

# 创建空数据集信息
from lerobot.datasets.utils import create_empty_dataset_info

info = create_empty_dataset_info(
    codebase_version="v3.0",
    fps=30,
    robot_type="koch",
    features=features_dict,
)
```

### B. 数据验证

```python
from lerobot.datasets.utils import validate_frame, validate_episode_buffer

# 验证单帧数据
validate_frame(frame_dict, features)

# 验证episode缓冲区
validate_episode_buffer(episode_buffer, total_episodes, features)
```

### C. 版本兼容性检查

```python
from lerobot.datasets.utils import check_version_compatibility

check_version_compatibility(repo_id, dataset_version, codebase_version)
```

---

## 参考资料

- 官方文档: `docs/source/lerobot-dataset-v3.mdx`
- 迁移指南: `docs/source/porting_datasets_v3.mdx`
- 数据集工具: `docs/source/using_dataset_tools.mdx`
- 源代码: `src/lerobot/datasets/`

