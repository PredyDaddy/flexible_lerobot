# 数据清洗与检查：LeRobot 自带工具使用指南

本目录的目标是把 LeRobot 仓库里**自带的**数据集清洗/编辑能力（删除 episode、切分、合并、移除 feature）
整理成一份可直接执行的说明，便于你在本地离线数据集上做清洗与复核。

适用的数据集格式：`codebase_version = v3.0`（目录包含 `meta/ data/ videos/`）。

---

## 0. 环境准备（建议在本仓库 conda 环境）

```bash
conda activate my_lerobot
pip install -e .
lerobot-edit-dataset --help
```

## 1. 前置：本地数据集存放位置

LeRobot 默认把数据集放在：

- `HF_LEROBOT_HOME`（默认 `~/.cache/huggingface/lerobot`）
- 单个数据集路径：`${HF_LEROBOT_HOME}/${repo_id}`（例如 `cqy/agilex_vla_demo_ee_pinocchio_apple_grasp`）

如果你想把数据集放到别的盘，推荐方式是设置环境变量，而不是依赖各脚本的 `--root`：

```bash
export HF_LEROBOT_HOME=/path/to/lerobot_datasets
```

---

## 2. 快速检查（不改数据）

### 1.1 只看 schema（强烈推荐先看 names）

```bash
python - <<'PY'
import json
from pathlib import Path
from lerobot.utils.constants import HF_LEROBOT_HOME

repo_id = "cqy/agilex_vla_demo_ee_pinocchio_apple_grasp"
root = Path(HF_LEROBOT_HOME) / repo_id
info = json.loads((root / "meta/info.json").read_text())

print("root:", root)
print("codebase_version:", info.get("codebase_version"))
print("robot_type:", info.get("robot_type"))
print("fps:", info.get("fps"))
print("total_episodes:", info.get("total_episodes"))
print("total_frames:", info.get("total_frames"))

state = info["features"]["observation.state"]
action = info["features"]["action"]
print("observation.state shape:", state["shape"], "dim:", len(state["names"]))
print("action shape:", action["shape"], "dim:", len(action["names"]))

ee_names = [n for n in state["names"] if "ee." in n]
print("ee dims:", len(ee_names), ee_names[:6])
PY
```

### 1.2 用 `LeRobotDataset` 读一条样本（会解码视频，若不需要可关）

```bash
python - <<'PY'
from lerobot.datasets.lerobot_dataset import LeRobotDataset

repo_id = "cqy/agilex_vla_demo_ee_pinocchio_apple_grasp"
ds = LeRobotDataset(repo_id, download_videos=False)

print(ds)
print("state names tail:", ds.meta.features["observation.state"]["names"][-10:])
sample = ds[0]
print("keys:", sorted(sample.keys()))
print("observation.state:", sample["observation.state"].shape, sample["observation.state"].dtype)
print("action:", sample["action"].shape, sample["action"].dtype)
PY
```

---

## 3. 数据清洗/编辑（CLI：`lerobot-edit-dataset`）

LeRobot 自带一个命令行工具 `lerobot-edit-dataset`（实现：`src/lerobot/scripts/lerobot_edit_dataset.py`），
支持：

- 删除 episode（`delete_episodes`）
- 切分数据集（`split`）
- 合并数据集（`merge`）
- 移除 feature（`remove_feature`，例如删掉某一路相机）

### 2.0 重要说明：原地修改 vs 生成新数据集

- 推荐：始终使用 `--new_repo_id` 生成新数据集（最安全）。
- 如果不传 `--new_repo_id`，脚本会把原数据集目录移动成 `${repo_id}_old`，然后把输出写回原路径（有“原地替换”的效果）。

### 2.1 删除坏 episode（最常用的清洗）

示例：删除 episode `0, 2, 5`，输出到新数据集：

```bash
lerobot-edit-dataset \
  --repo_id "cqy/agilex_vla_demo_ee_pinocchio_apple_grasp" \
  --new_repo_id "cqy/agilex_vla_demo_ee_pinocchio_apple_grasp_after_deletion" \
  --operation.type delete_episodes \
  --operation.episode_indices "[0, 2, 5]"
```

### 2.2 切分数据集（train/val/test）

按比例切分（按 episode 顺序切片）：

```bash
lerobot-edit-dataset \
  --repo_id "cqy/agilex_vla_demo_ee_pinocchio_apple_grasp" \
  --operation.type split \
  --operation.splits '{"train": 0.8, "val": 0.2}'
```

按 episode 索引切分（更可控）：

```bash
lerobot-edit-dataset \
  --repo_id "cqy/agilex_vla_demo_ee_pinocchio_apple_grasp" \
  --operation.type split \
  --operation.splits '{"train": [0,1,2], "val": [3,4]}'
```

输出数据集的 `repo_id` 规则：在原 `repo_id` 后追加 split 名，例如 `..._train`、`..._val`。

### 2.3 合并多个数据集

注意：要合并的数据集 **features 必须完全一致**（包含 `observation.state` 的维度与 names）。

```bash
lerobot-edit-dataset \
  --repo_id "cqy/agilex_merged" \
  --operation.type merge \
  --operation.repo_ids "['cqy/ds_a', 'cqy/ds_b']"
```

### 2.4 移除 feature（例如删相机）

示例：移除 `camera_right` 这一路（会同时处理 `meta/episodes` 的视频索引，并删除对应视频文件）：

```bash
lerobot-edit-dataset \
  --repo_id "cqy/agilex_vla_demo_ee_pinocchio_apple_grasp" \
  --new_repo_id "cqy/agilex_vla_demo_ee_pinocchio_apple_grasp_no_right_cam" \
  --operation.type remove_feature \
  --operation.feature_names "['observation.images.camera_right']"
```

### 2.5 推送到 Hugging Face Hub（可选）

所有操作都可以追加 `--push_to_hub=true` 自动上传（默认 `false`）：

```bash
lerobot-edit-dataset \
  --repo_id "cqy/agilex_vla_demo_ee_pinocchio_apple_grasp" \
  --new_repo_id "cqy/agilex_vla_demo_ee_pinocchio_apple_grasp_after_deletion" \
  --operation.type delete_episodes \
  --operation.episode_indices "[0, 2, 5]" \
  --push_to_hub=true
```

---

## 4. 数据清洗/编辑（Python API：`lerobot.datasets.dataset_tools`）

当你需要在 Python 里做批处理/更复杂的管线时，用 API 更方便（实现：`src/lerobot/datasets/dataset_tools.py`）。

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import delete_episodes

ds = LeRobotDataset("cqy/agilex_vla_demo_ee_pinocchio_apple_grasp")
new_ds = delete_episodes(
    ds,
    episode_indices=[0, 2, 5],
    repo_id="cqy/agilex_after_deletion",
)
print(new_ds.root, new_ds.meta.total_episodes, new_ds.meta.total_frames)
```

> `dataset_tools` 里也提供 `add_features/modify_features`，但当前 `lerobot-edit-dataset` CLI 没覆盖“添加特征”的参数面。

---

## 5. 可视化检查（`lerobot-dataset-viz`）

用于“人工抽查”某个 episode 的动作/状态曲线 + 视频帧对齐（需要 `rerun` 环境支持）。

```bash
lerobot-dataset-viz \
  --repo-id "cqy/agilex_vla_demo_ee_pinocchio_apple_grasp" \
  --episode-index 0
```

脚本实现：`src/lerobot/scripts/lerobot_dataset_viz.py`。

---

## 6. 关于 EE 数据（与你当前场景强相关）

1. `record_ee_pose=true` 只会改变 `observation.state` 的维度与 names；`action` 仍是关节空间（AgileX 为 14 维）。
2. `lerobot-edit-dataset` 这类 episode/feature 级工具对 26 维 state **天然兼容**（不写死 14）。
3. 但如果你要“从 26 维里删除 EE 的 12 维，只保留 14 维关节”，这属于**向量内部维度编辑**：
   - 目前 `lerobot-edit-dataset`/`dataset_tools.remove_feature` 做不到（它只能移除整个 feature，比如 `observation.images.*`）。
   - 需要单独脚本按 `meta/info.json -> features["observation.state"]["names"]` 重建 `observation.state`、同步更新 `stats.json`。

EE 字段与录制语义的更完整说明见：`docs/ee_data_descriptions/EE_DATASET_SPEC.md`。
