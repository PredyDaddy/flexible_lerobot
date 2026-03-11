# AgileX 脚本说明

这 3 个脚本默认使用 `lerobot_flex` 环境。

## 1. 录制数据

```bash
bash my_devs/add_robot/agilex/record.sh <dataset_name> <episode_time_s> <num_episodes> <fps> <reset_time_s> <resume>
```

例子：

```bash
bash my_devs/add_robot/agilex/record.sh my_demo 8 2 30 5 false
bash my_devs/add_robot/agilex/record.sh my_demo 8 1 30 0 true
```

含义：

- `dataset_name`: 数据集名字
- `episode_time_s`: 每个 episode 录几秒
- `num_episodes`: 一共录几个 episode
- `fps`: 录制帧率
- `reset_time_s`: 两个 episode 之间留几秒做重置环境
- `resume`: 是否在已有数据集后面继续录，填 `true` 或 `false`

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

## 2. 回放数据

```bash
bash my_devs/add_robot/agilex/replay.sh <dataset_name> <episode_index>
```

例子：

```bash
bash my_devs/add_robot/agilex/replay.sh my_demo 0
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

- `my_devs/add_robot/agilex/outputs/record_config.json`
- `my_devs/add_robot/agilex/outputs/replay_config.json`
