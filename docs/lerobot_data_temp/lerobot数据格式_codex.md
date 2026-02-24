# LeRobot 数据格式指南（codex）

## 总览
- 当前标准为 **LeRobotDataset v3.0**（文件式存储，面向大规模数据）。核心思想：将多段 episode 拼接到少量 Parquet/MP4 文件中，通过元数据重建 episode 视图。
- 数据类型三类：1) **表格信号**（状态/动作/时间戳等，Parquet）；2) **视频**（每个相机一组 MP4 分片）；3) **元数据**（JSON/Parquet，描述 schema、offset、统计信息）。
- 主要代码与脚本位置：`src/lerobot/datasets/`（`LeRobotDataset`、`StreamingLeRobotDataset`、`dataset_tools`、图像增广），示例在 `examples/`，文档参考 `docs/source/lerobot-dataset-v3.mdx`、`docs/source/porting_datasets_v3.mdx`、`docs/source/using_dataset_tools.mdx`。

## 目录结构
```
dataset_root/
├─ meta/
│  ├─ info.json           # schema、FPS、路径模板、版本信息
│  ├─ stats.json          # 全局均值/方差/最值
│  ├─ tasks.jsonl         # 任务描述与 task_id 映射
│  └─ episodes/
│     └─ chunk-000/file-000.parquet  # 每行对应一个 episode，含 start/end offset 等
├─ data/
│  └─ chunk-000/file-000.parquet     # 多个 episode 的表格信号拼接
└─ videos/
   └─ <camera_name>/chunk-000/file-000.mp4  # 多个 episode 的该相机视频
```
关键约定：`file-000` 等分片包含多个 episode；切分逻辑依赖 `meta/episodes/*.parquet` 中的 offset/length，而非文件名。

## 元数据与特征
- **info.json**：`features`（名称、dtype、shape）、`fps`、`path_templates`（定位 data/video 分片）、代码版本。
- **stats.json**：每个特征的 mean/std/min/max，训练时可用于归一化。
- **episodes/*.parquet**：每个 episode 的 `episode_id`、`task_id`、`length`、`data_file`、`data_start`、`video_files` 与帧 offset 等。
- **特征命名**：`observation.state.*`、`observation.images.<camera>`、`action.*`、`timestamp`、`is_first`/`is_last` 等；保持 snake_case，新增特征需在 `features` 中声明 dtype/shape。

## 读取与索引示例
```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

repo_id = "yaak-ai/L2D-v3"
dataset = LeRobotDataset(repo_id)  # 缓存到本地后随机访问

sample = dataset[100]
# 常见键：'observation.state', 'action',
#        'observation.images.front_left', 'timestamp', ...

# 按时间窗取多帧（秒为单位，relative offsets）
delta = {"observation.images.front_left": [-0.2, -0.1, 0.0]}
dataset = LeRobotDataset(repo_id, delta_timestamps=delta)
stack = dataset[100]["observation.images.front_left"]  # shape [T, C, H, W]
```
- 批量训练：用 `torch.utils.data.DataLoader` 包装；视频帧为 `uint8` 张量，状态/动作多为 `float32`。
- 数据加载速度：表格数据通过 PyArrow memory-map，视频按需解码；同一分片多次访问具备缓存优势。

## 流式访问（无本地下载）
```python
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
dataset = StreamingLeRobotDataset("yaak-ai/L2D-v3")  # 直接从 Hub 流式读取
```
- 适用于磁盘/带宽受限的训练或快速预览；依赖 Hub 在线拉取，注意网络可用性。

## 图像增广
- API：`ImageTransforms`、`ImageTransformsConfig`、`ImageTransformConfig`（位于 `lerobot.datasets.transforms`）。
- 默认关闭，需传入 `image_transforms=...`。
- 常见配置：`enable`、`max_num_transforms`、`random_order`、以及各变换（brightness/contrast/saturation/sharpness 等）的 `weight` 与参数。
- 变换在训练时应用，原始数据不做离线增广，便于复用同一数据集尝试不同策略。

## 数据工具（编辑现有数据）
命令行：`lerobot-edit-dataset`（见 `lerobot.datasets.dataset_tools` 与 `examples/dataset/use_dataset_tools.py`）。
- 删除 episode：`--operation.type delete_episodes --operation.episode_indices "[0,2,5]"`
- 拆分数据集：`--operation.type split --operation.splits '{"train":0.8,"val":0.2}'`
- 合并分片：`--operation.type merge --operation.repo_ids "['a/train','a/val']"`
- 添加/移除特征：`--operation.type add_feature|remove_feature --operation.feature_names [...]`
- 追加 `--push_to_hub` 将结果上传；保持特征 schema 一致才能安全合并。

## 迁移与制作（v3 推荐流程）
- 结构差异：v2.1 为「每个 episode 一个文件」，v3 为「分片文件 + offset 元数据」，性能与可扩展性显著提升。
- 基本步骤：
  1) 按相机/表格生成合适大小的分片（可控文件大小，减少 inode 压力）。
  2) 生成 `meta/info.json`、`meta/stats.json` 和 `meta/episodes/*.parquet`（含 offset/长度/任务 ID）。
  3) 如需多任务，填充 `tasks.jsonl`。
  4) 本地验证读取（`LeRobotDataset`）、可选流式测试（`StreamingLeRobotDataset`），再推送 Hub。
- 大型数据（如 DROID）建议使用集群/SLURM 并行切分（参考 `docs/source/porting_datasets_v3.mdx` 和 `examples/port_datasets/`）。

## 实践建议
- 保持特征命名与 dtype/shape 的稳定性；新增特征时同步更新 `info.json` 与统计。
- 控制分片大小，确保视频与表格分片在数百 MB 级别，便于缓存与 Hub 传输。
- 更新数据后重新计算 `stats.json`，避免训练归一化失配。
- 在 PR 中附 `lerobot-edit-dataset` 或录制命令、生成的 repo_id，以及本地验证步骤，方便复现。
