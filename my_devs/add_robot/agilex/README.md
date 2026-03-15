# 我的简易使用
```bash
# bash my_devs/add_robot/agilex/record.sh <dataset_name> <episode_time_s> <num_episodes> <fps> <reset_time_s> <resume> <single_task_text>
bash my_devs/add_robot/agilex/record.sh my_demo 8 1 30 0 false 'Pick up the black cups and place them in the orange box.'

# 如果你确实还想录 master 作为 action，再显式打开：
ACTION_SOURCE=master bash my_devs/add_robot/agilex/record.sh my_demo 8 1 30 0 false 'Use master action'

#bash my_devs/add_robot/agilex/replay.sh <dataset_name> <episode_index>
HF_LEROBOT_HOME=/home/agilex/cqy/flexible_lerobot/my_devs/add_robot/agilex/outputs \
  DATASET_REPO_ID=dummy/replay_test2 \
  bash my_devs/add_robot/agilex/replay.sh replay_test2 1

# bash my_devs/add_robot/agilex/vis.sh <dataset_name> <episode_index>
HF_LEROBOT_HOME=/home/agilex/cqy/flexible_lerobot/my_devs/add_robot/agilex/outputs \
  DATASET_REPO_ID=dummy/replay_test2 \
  bash my_devs/add_robot/agilex/vis.sh replay_test2 1
```

# AgileX 脚本说明

这 3 个脚本默认使用 `lerobot_flex` 环境。

当前 `record.sh` 默认是 follower-only：

- `observation.state` 读 `/puppet/joint_left` 和 `/puppet/joint_right`
- `observation.state` 默认只保留 14 维 follower joint position，不再把 velocity / effort 写进数据集
- `action` 也默认读 `/puppet/joint_left` 和 `/puppet/joint_right`
- 默认不会再把 `/master/joint_left`、`/master/joint_right` 录进数据里的 `action`
- 三路相机默认写成 `camera_front`、`camera_left`、`camera_right`

如果你确实要恢复旧行为，用环境变量：

```bash
ACTION_SOURCE=master bash my_devs/add_robot/agilex/record.sh ...
```

## 1. 录制数据

```bash
bash my_devs/add_robot/agilex/record.sh <dataset_name> <episode_time_s> <num_episodes> <fps> <reset_time_s> <resume> <single_task_text>
```

例子：

```bash
bash my_devs/add_robot/agilex/record.sh my_demo 8 2 30 5 false
bash my_devs/add_robot/agilex/record.sh my_demo 8 1 30 0 true
bash my_devs/add_robot/agilex/record.sh my_demo 8 1 30 0 false 'Pick up the black cups and place them in the orange box.'
bash my_devs/add_robot/agilex/record.sh my_demo 8 1 30 0 false '把 "黑色杯子" 放进橙色箱子'
ACTION_SOURCE=master bash my_devs/add_robot/agilex/record.sh my_demo 8 1 30 0 false 'Use master action'
```

含义：

- `dataset_name`: 数据集名字
- `episode_time_s`: 每个 episode 录几秒
- `num_episodes`: 一共录几个 episode
- `fps`: 录制帧率
- `reset_time_s`: 两个 episode 之间留几秒做重置环境
- `resume`: 是否在已有数据集后面继续录，填 `true` 或 `false`
- `single_task_text`: 从第 7 个参数开始到命令结尾的所有文本，都会当成任务语义写进数据集，供 VLA / 语言条件训练使用

如果任务文本里有空格、中文、双引号，直接整体用单引号包起来即可。

额外环境变量：

- `ACTION_SOURCE`: 录制 `action` 的来源，默认 `follower`
- `ACTION_SOURCE=follower`: `action` 来自 `/puppet/joint_left`、`/puppet/joint_right`
- `ACTION_SOURCE=master`: `action` 来自 `/master/joint_left`、`/master/joint_right`
- 旧的相机 key `cam_high`、`cam_left_wrist`、`cam_right_wrist` 仍然可以传入；AgileX 代码会自动规范化成 `camera_front`、`camera_left`、`camera_right`

如果不传参数，默认等价于：

```bash
bash my_devs/add_robot/agilex/record.sh agilex_record_demo_video 8 1 10 0 false
```

录完后数据会在：

```text
my_devs/add_robot/agilex/outputs/dummy/<dataset_name>
```

例如：

```text
my_devs/add_robot/agilex/outputs/dummy/my_demo
```

语义文本会写到：

- 每一帧可读取的 `task`
- `meta/tasks.parquet`

## 2. 回放数据

```bash
bash my_devs/add_robot/agilex/replay.sh <dataset_name> <episode_index>
```

例子：

```bash
HF_LEROBOT_HOME=/home/agilex/cqy/flexible_lerobot/my_devs/add_robot/agilex/outputs \
  DATASET_REPO_ID=test_prefix/replay_test \
  bash my_devs/add_robot/agilex/replay.sh replay_test 0
```

含义：

- `dataset_name`: 数据集名字
- `episode_index`: 第几个 episode，从 `0` 开始

## 3. 可视化数据

```bash
bash my_devs/add_robot/agilex/vis.sh <dataset_name> <episode_index>
```

例子：

```bash
bash my_devs/add_robot/agilex/vis.sh my_demo 0
```

含义：

- `dataset_name`: 数据集名字
- `episode_index`: 第几个 episode，从 `0` 开始

## 4. 最常用的一条

第一次建议直接试这条：

```bash
bash my_devs/add_robot/agilex/record.sh test_001 8 1 30 0 false
```

这样会录 1 个 episode，时长 8 秒，数据集名字叫 `test_001`。

## 5. 你可以先忽略的东西

下面这些文件是脚本自动生成的，不用手动改：

- `my_devs/add_robot/agilex/outputs/record_config_<dataset_name>.json`
- `my_devs/add_robot/agilex/outputs/replay_config_<dataset_name>.json`
