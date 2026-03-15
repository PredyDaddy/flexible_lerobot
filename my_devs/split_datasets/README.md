# 双臂数据拆分使用说明

这份 README 只回答一件事:

- 以后你再有一份新的双臂 LeRobot 数据
- 想把它拆成左臂数据和右臂数据
- 你应该怎么做

当前工具的行为是固定的:

- 源数据不原地修改
- 从源 `LeRobotDataset` 读出数据
- 重建两个新的 `LeRobotDataset`
- 左臂输出保留 `action[:7]`、`observation.state[:7]`、`camera_front`、`camera_left`
- 右臂输出保留 `action[7:14]`、`observation.state[7:14]`、`camera_front`、`camera_right`
- 默认复制原始 `task`
- 默认保留原始 `timestamp`

## 你只需要改这 5 个量

以后换一份新数据时，最常改的是下面这 5 个参数：

- `--source-root`
  你的原始数据集所在根目录
- `--source-repo-id`
  你的原始数据集 repo id
- `--target-root`
  拆分后数据放到哪里
- `--left-repo-id`
  左臂数据集名字
- `--right-repo-id`
  右臂数据集名字

一个简单理解方式：

- `source-root + source-repo-id` 决定“原始数据在哪里”
- `target-root + left/right-repo-id` 决定“拆分结果写到哪里”

## 最常用流程

### 1. 进入环境

```bash
source /home/agilex/miniconda3/etc/profile.d/conda.sh
conda activate lerobot_flex
```

### 2. 运行拆分

把下面命令里的 5 个路径/名字换成你的新数据：

```bash
python -m my_devs.split_datasets.split_bimanual_lerobot_dataset \
  --source-root /path/to/source_root \
  --source-repo-id dummy/your_dataset \
  --target-root /path/to/target_root \
  --left-repo-id dummy/your_dataset_left \
  --right-repo-id dummy/your_dataset_right \
  --task-mode copy \
  --overwrite \
  --vcodec h264
```

### 3. 运行校验

```bash
python -m my_devs.split_datasets.validate_split_dataset \
  --source-root /path/to/source_root \
  --source-repo-id dummy/your_dataset \
  --target-root /path/to/target_root \
  --left-repo-id dummy/your_dataset_left \
  --right-repo-id dummy/your_dataset_right
```

### 4. 看结果

如果成功，终端会打印类似：

```text
[ok] wrote left dataset: ...
[ok] wrote right dataset: ...
[ok] dummy/your_dataset_left: arm=left episodes=... frames=... lag1_max=0.000000
[ok] dummy/your_dataset_right: arm=right episodes=... frames=... lag1_max=0.000000
```

只要左右两边都显示 `[ok]`，并且 `lag1_max=0.000000`，就说明这次拆分结果是正常的。

## 直接照抄的实际例子

这是当前 `replay_test4` 的实际拆分命令：

```bash
source /home/agilex/miniconda3/etc/profile.d/conda.sh
conda activate lerobot_flex

python -m my_devs.split_datasets.split_bimanual_lerobot_dataset \
  --source-root /home/agilex/cqy/flexible_lerobot/my_devs/add_robot/agilex/outputs \
  --source-repo-id dummy/replay_test4 \
  --target-root /home/agilex/cqy/flexible_lerobot/my_devs/split_datasets/outputs \
  --left-repo-id dummy/replay_test4_left \
  --right-repo-id dummy/replay_test4_right \
  --task-mode copy \
  --overwrite \
  --vcodec h264
```

对应校验命令：

```bash
source /home/agilex/miniconda3/etc/profile.d/conda.sh
conda activate lerobot_flex

python -m my_devs.split_datasets.validate_split_dataset \
  --source-root /home/agilex/cqy/flexible_lerobot/my_devs/add_robot/agilex/outputs \
  --source-repo-id dummy/replay_test4 \
  --target-root /home/agilex/cqy/flexible_lerobot/my_devs/split_datasets/outputs \
  --left-repo-id dummy/replay_test4_left \
  --right-repo-id dummy/replay_test4_right
```

## 如果你想改 task 文本

默认会直接复制原始 `task`。如果你后面想做 VLA，通常建议把语言改得更具体一点。

例如：

```bash
python -m my_devs.split_datasets.split_bimanual_lerobot_dataset \
  --source-root /path/to/source_root \
  --source-repo-id dummy/your_dataset \
  --target-root /path/to/target_root \
  --left-repo-id dummy/your_dataset_left \
  --right-repo-id dummy/your_dataset_right \
  --task-mode override \
  --left-task-text "Move the left arm toward the target object." \
  --right-task-text "Move the right arm toward the target object." \
  --overwrite
```

如果你用了 `override`，校验时建议加上：

```bash
python -m my_devs.split_datasets.validate_split_dataset \
  --source-root /path/to/source_root \
  --source-repo-id dummy/your_dataset \
  --target-root /path/to/target_root \
  --left-repo-id dummy/your_dataset_left \
  --right-repo-id dummy/your_dataset_right \
  --skip-task-copy-check
```

## 目前不建议优先用的参数

脚本里有一个 `--episode-indices`，表示只拆分部分 episode。

这个功能拆分脚本本身支持，但当前校验脚本还是按“目标数据必须和源全集 episode/frame 数量一致”来检查。所以：

- 如果你现在只是拆整份数据，正常使用即可
- 如果你想只拆一部分 episode，建议先不要作为主流程使用

## 目录说明

- [split_bimanual_lerobot_dataset.py](/home/agilex/cqy/flexible_lerobot/my_devs/split_datasets/split_bimanual_lerobot_dataset.py)
  拆分主脚本
- [validate_split_dataset.py](/home/agilex/cqy/flexible_lerobot/my_devs/split_datasets/validate_split_dataset.py)
  拆分结果校验脚本
- [outputs](/home/agilex/cqy/flexible_lerobot/my_devs/split_datasets/outputs)
  拆分输出目录

## 最小回归测试

```bash
source /home/agilex/miniconda3/etc/profile.d/conda.sh
conda activate lerobot_flex
pytest -q tests/my_devs/test_split_datasets.py
```
