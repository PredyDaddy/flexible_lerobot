# LeRobot v3.0 数据集完整检查清单

> 按重要性排序：致命错误 → 质量问题

---

## 一、目录结构与必需文件 [致命]

缺少任何一项都无法加载数据集。

```
dataset_root/
├── meta/
│   ├── info.json              ← 必须
│   ├── stats.json             ← 必须
│   ├── tasks.parquet          ← 必须
│   └── episodes/
│       └── chunk-*/file-*.parquet  ← 必须
├── data/
│   └── chunk-*/file-*.parquet      ← 必须
└── videos/                         ← 如果features有dtype=video则必须
    └── {video_key}/chunk-*/file-*.mp4
```

---

## 二、info.json 自洽性 [致命]

全局"账本"，所有其他文件都依赖它。

| 检查项 | 要求 |
|--------|------|
| codebase_version | 必须是 "v3.0" |
| total_episodes | 与实际episode数量一致 |
| total_frames | 与实际帧数一致 |
| total_tasks | 与tasks.parquet行数一致 |
| fps | 帧率（通常30） |
| splits | 范围必须在 [0, total_episodes] 内 |
| features | 与实际数据列/形状一致 |
| features.action.shape | 单臂[7]，双臂[14] |
| features.observation.state.shape | 单臂[7]，双臂[14] |
| features中的video_key | 必须有对应的videos目录 |

---

## 三、episodes parquet 一致性 [致命]

每条episode的"索引表"，删除数据后最容易出错。

| 检查项 | 要求 |
|--------|------|
| 行数 | == info.total_episodes |
| episode_index | 0 到 N-1 连续，无重复，无缺失 |
| length总和 | == info.total_frames |

### dataset_from/to_index 严格验证

```python
# 第一条从0开始
episodes[0]['dataset_from_index'] == 0

# 每条满足 to - from == length
episodes[i]['dataset_to_index'] - episodes[i]['dataset_from_index'] == episodes[i]['length']

# 前后首尾相接，无空洞无重叠
episodes[i+1]['dataset_from_index'] == episodes[i]['dataset_to_index']

# 最后一条等于total_frames
episodes[-1]['dataset_to_index'] == info['total_frames']
```

### 文件引用检查

| 检查项 | 要求 |
|--------|------|
| data/chunk_index + file_index | 指向的parquet文件必须存在 |
| videos/.../chunk_index + file_index | 指向的mp4文件必须存在 |

---

## 四、data parquet 帧数据正确性 [致命]

训练真正读取的内容。

| 检查项 | 要求 |
|--------|------|
| 总行数 | == info.total_frames |
| index (全局) | 0 到 total_frames-1 连续且唯一 |
| episode_index | 每个episode的行数 == episodes.length |
| frame_index | 每个episode内 0 到 length-1 连续 |
| task_index | 都在 [0, total_tasks-1] 范围内 |

### 数值合法性

| 检查项 | 要求 |
|--------|------|
| action | 无 NaN/Inf，维度与info一致 |
| observation.state | 无 NaN/Inf，维度与info一致 |
| timestamp | 单调递增，间隔 ≈ 1/fps |

---

## 五、videos 视频文件完整性 [致命/如果使用视频]

| 检查项 | 要求 |
|--------|------|
| 文件存在 | episodes引用的所有mp4都存在 |
| 可解码 | ffprobe能正常读取 |
| 帧数总和 | **每个相机key**的视频帧数之和 == total_frames |
| 分辨率 | 与info.json中定义一致 |

### 时间戳验证

```python
# to > from
episodes[i]['videos/.../to_timestamp'] > episodes[i]['videos/.../from_timestamp']

# 时间戳约束: (to - from) ≈ length / fps
(to_timestamp - from_timestamp) ≈ length / fps  # 允许0.1秒误差

# 同一视频文件内，各episode片段不重叠、首尾相接
# max(to_timestamp) 不能超过mp4实际时长
max(to_timestamp) <= mp4_duration
```

### 多相机同步（强烈建议）

同一episode的多路相机 from/to_timestamp 应一致或在可接受误差内。

---

## 六、tasks.parquet 与 task_index 对齐 [重要]

| 检查项 | 要求 |
|--------|------|
| 行数 | == info.total_tasks |
| task_index | 从0开始连续 |
| 任务描述 | 不能是占位符，应反映真实任务 |
| data.task_index | 都能在tasks.parquet中找到 |

---

## 七、stats.json 完整性 [重要]

用于训练时的数据归一化。

| 检查项 | 要求 |
|--------|------|
| 必需字段 | min, max, mean, std, count |
| 分位数字段 | q01, q10, q50, q90, q99 |
| count类型 | 可能是 int 或 list[int]，取值时需兼容 |
| 数值特征count | == total_frames |
| 图像特征 | shape为(3,1,1)，值在[0,1]范围 |
| 关节值范围 | 应在±π弧度内 |

---

## 八、数据质量检查 [建议]

不一定报错，但影响训练效果。

| 检查项 | 要求 |
|--------|------|
| 全零帧 | 不应有全零的action/state |
| 连续静止 | 最长连续静止帧数 < 阈值（如60帧） |
| 轨迹完整性 | episode结尾动作应趋于平稳 |
| 关节越界 | 无明显超出物理限制的值 |
| 画面质量 | 曝光正常、无遮挡、目标清晰 |
| 多相机同步 | 无明显漂移 |

---

## 九、手动删数据后最易踩坑

| 踩坑点 | 说明 |
|--------|------|
| info未更新 | total_episodes/total_frames/splits 忘了改 |
| episode_index不连续 | 删除后没有重新编号 |
| dataset_from/to有空洞 | 没有重排导致中间有gap |
| data.index不连续 | 训练时直接报错或隐性错乱 |
| 引用不存在的文件 | episodes还指向已删除的data/video |
| 视频时间戳越界 | to_timestamp超过mp4实际时长 |
| stats.count未更新 | 与实际帧数不一致 |

---

## 十、快速验证公式

```python
# 核心等式（必须全部成立）
info['total_frames'] == data总行数 == sum(episodes['length'])
info['total_episodes'] == episodes行数 == len(data['episode_index'].unique())
episodes[-1]['dataset_to_index'] == info['total_frames']

# 每个相机key的视频帧数之和 == total_frames
视频帧数总和(per camera key) == total_frames

# stats.count 取值需兼容 int 和 list
count = stats['action']['count']
count = count[0] if isinstance(count, list) else count
count == info['total_frames']
```

---

## 附录：检查脚本（推荐）

本目录提供 `check_lerobot_v3_dataset_final.py` 作为参考实现，覆盖本清单 1–8，并且实现时**不假设** parquet/data 行已经按 `episode_index/index/frame_index` 排好序（避免误报）。

**使用方式**:
```bash
# 支持任意路径
python3 check_lerobot_v3_dataset_final.py /path/to/dataset
python3 check_lerobot_v3_dataset_final.py ~/.cache/huggingface/lerobot/cqy/agilex_left_banana_final
```

---

**文档版本**: v1.2
**适用 LeRobot 版本**: v3.0+
**最后更新**: 2026-01
