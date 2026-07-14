# JZ Pin Timed raw18 / model16 兼容契约

本文只描述 X86 侧的数据与训练边界。Orin UDP wire protocol、机器人安全门限和 command executor
均不在本次改动范围内。以下检查和训练命令都是离线命令，不会发送机器人指令。

## Schema 标识

| 层级 | Schema ID | Version | 维度 |
| --- | --- | ---: | ---: |
| 原始 UDP scalar / 数据集边界 | `jz_pin_raw18_v1` | 1 | 18 |
| 模型 observation / action | `jz_pin_opening16_v1` | 1 | 16 |

manifest 格式是 `jz_pin_training_projection`，`schema_version` 为 `1`。新 timed 数据集的独立
sidecar 路径是 `meta/jz_pin_training_schema.json`；它不改变 `meta/info.json`、parquet 或 timing
sidecar。旧数据可以用数据集外部的显式 manifest，例如：

```text
my_devs/jz_robot_pin_timed/train/manifests/jz_robot_pin_timed_real_20260711_190502.json
```

## 精确 raw18 顺序

`observation.state` 和 `action` 的 raw18 字段顺序相同。映射先逐个校验字段名与顺序，再计算索引；
字段名或顺序不一致会失败，不会仅凭 18D shape 猜索引。

| raw 索引 | raw 字段 | model16 observation | model16 action | 处理 |
| ---: | --- | --- | --- | --- |
| 0 | `left_left_joint1.pos` | 同名 | 同名 | 保留 |
| 1 | `left_left_joint2.pos` | 同名 | 同名 | 保留 |
| 2 | `left_left_joint3.pos` | 同名 | 同名 | 保留 |
| 3 | `left_left_joint4.pos` | 同名 | 同名 | 保留 |
| 4 | `left_left_joint5.pos` | 同名 | 同名 | 保留 |
| 5 | `left_left_joint6.pos` | 同名 | 同名 | 保留 |
| 6 | `left_left_joint7.pos` | 同名 | 同名 | 保留 |
| 7 | `right_right_joint1.pos` | 同名 | 同名 | 保留 |
| 8 | `right_right_joint2.pos` | 同名 | 同名 | 保留 |
| 9 | `right_right_joint3.pos` | 同名 | 同名 | 保留 |
| 10 | `right_right_joint4.pos` | 同名 | 同名 | 保留 |
| 11 | `right_right_joint5.pos` | 同名 | 同名 | 保留 |
| 12 | `right_right_joint6.pos` | 同名 | 同名 | 保留 |
| 13 | `right_right_joint7.pos` | 同名 | 同名 | 保留 |
| 14 | `left_gripper.width` | `left_gripper.opening` | `left_gripper.target_opening` | 保留并按 manifest 映射方向 |
| 15 | `left_gripper.force` | 无 | 无 | 从模型 tensor 删除 |
| 16 | `right_gripper.width` | `right_gripper.opening` | `right_gripper.target_opening` | 保留并按 manifest 映射方向 |
| 17 | `right_gripper.force` | 无 | 无 | 从模型 tensor 删除 |

因此 raw18 到 model16 的保留索引严格为：

```text
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16]
```

model16 的索引 `0..13` 仍是双臂 14 个关节，索引 `14` 是左 opening，索引 `15` 是右 opening。
raw force 索引 `[15, 17]` 不参与训练 feature、统计量、normalization 或模型输出。

## Opening 语义与方向

模型统一使用 canonical opening：`0=closed`、`100=open`。每个 side 和 modality 都在 manifest
中分别声明 `raw_closed` 与 `raw_open`，换算为：

```text
canonical = (raw - raw_closed) / (raw_open - raw_closed) * 100
```

这使左右、state/action 的反向关系可见且可审计，不会静默反转。默认旧数据 manifest
`jz_robot_pin_timed_real_20260711_190502.json` 当前声明：

| 数据 | 来源语义 | raw closed | raw open | 到 canonical 的方向 |
| --- | --- | ---: | ---: | --- |
| 左 observation | `measured_opening` | 0 | 100 | 同向 |
| 右 observation | `commanded_opening` | 100 | 0 | 反向 |
| 左 action | `commanded_opening` | 100 | 0 | 反向 |
| 右 action | `commanded_opening` | 100 | 0 | 反向 |

例如该 manifest 下，raw observation 的左/右 width 为 `[25, 75]` 时，model opening 为
`[25, 25]`；model target opening `[20, 80]` 展开回 raw action width 则为 `[80, 20]`。

这份旧数据的来源结论来自离线部署源码审计：左侧 status 是 measured opening，右侧 status 是
command echo，因此绝不能把右侧字段标成 measured。方向端点仍需在任何 armed evaluation 前做
现场物理标定；manifest 中也保留了这项风险。源码审计依据见
[action/state 链路审计报告](../../../docs/jz_robot_action_state_link_report.md)。

全量 freshness 审计进一步发现这份历史数据的 894 帧中，左右 gripper generation 各复用了 6 次
（首次为 episode 0/frame 6）。因此它只保留作 raw18/legacy 兼容样本，**不具备 model16 训练批准**；
严格 checker 会按设计返回 `FAIL`，不能用 override 把这些缓存帧改称新反馈。

允许的 observation 来源只有：

- `measured_opening`：来自夹爪反馈；
- `commanded_opening`：来自已发送命令或 command echo；
- `unavailable`：无法可靠确认来源。

新录制配置默认是 `unavailable`。缺少 sidecar/外部 manifest 的旧 episode 也按 provenance
不可用处理：raw18 仍可读取，但严格训练会明确失败。不得用最近一次缓存、上一帧或 commanded
opening 伪造新的 measured opening。

一旦任一侧配置为 `measured_opening` 或 `commanded_opening`，X86 timed Robot 会在返回 observation
之前强制要求合法 `source_timing v1`，并要求该侧 gripper generation 相对上一 observation 严格推进；
停住的有限缓存值同样会失败。配置和已有 dataset manifest 的语义冲突也会在 `robot.connect()` 前预检失败。

### 当前 Orin 部署

Orin 已部署 gripper_node 调度修复（现场二进制 SHA256 前缀 `714c8bc7`）：左右各自使用唯一串口
I/O worker，command callback 仅写 latest-wins 命令槽；四类左右 command/status callback 分到独立
MutuallyExclusive group，并由四线程 executor 执行；两侧 status 使用独立 timer。只有收到合法
checksum/node-id 的硬件 RX 帧才推进 generation，RX 模式不再复用缓存值伪造新 state。

60 秒无 gripper command 的只读验收为：left `98.478 Hz`、P95 `11.104 ms`、max gap `21.731 ms`；
right `100.163 Hz`、P95 `10.969 ms`、max gap `20.849 ms`。18D UDP/state 以及 bridge 的 50 ms age、
20 ms skew 门限保持不变。

这解决的是 status 调度停顿，不改变右侧语义：当前 right `status_source=command` 仍是 ROS command
echo。因此新数据 manifest 必须继续记录 right observation 为 `commanded_opening`；它不能用于验证
右夹爪硬件真实状态。旧 `20260711_190502` 仍含历史 generation reuse，不能因新部署而追溯变成
训练批准数据，必须重新采集。

### Timed Bridge QoS 联调基线

后续 X86 timed 联调以 Orin 分支 `fix/jz-pin-timed-integration-v1` 的 commit
`188f64a8ef7c08615a1f5b30b2b9539f1a264ca7` 为状态 bridge 基线。修复前，bridge 对双夹爪 status
使用 `RELIABLE + depth=1`，实际接收约 `23 Hz`，可出现 `101-140 ms` 间隔；同一 publisher 以
BEST_EFFORT 订阅时左右可稳定约 `98.6/100 Hz`。修复后 bridge 显式使用：

```text
BEST_EFFORT + VOLATILE + KEEP_LAST(depth=1)
```

这不改变 UDP 18D/state wire protocol，也不放宽 state `30 Hz`、minimum ratio `0.9`、source age
`50 ms`、source skew `20 ms` 或 `require_all_sources_advanced`。Orin 离线测试 `168 passed`；正式
timed dry-run ready 已通过，首窗口 `28.996892 Hz`，后续稳定约 `30 Hz`，四源持续推进；稳定阶段
source age 约 `0.4-6.2 ms`、skew 约 `1.4-5.6 ms`，UDP payload `1366-1375 bytes`。该 dry-run 已停止，
未进入 armed，且没有 bridge/executor 残留。

下一次 X86 新录制前，应确认 Orin 运行的是上述基线，并由 X86 侧的 `source_timing` 检查验证四源
generation 严格推进、30 Hz 比例、50 ms age 和 20 ms skew。只有该检查和 raw18/model16 projection
checker 都通过，新 episode 才可用于训练。

## Force 边界

force 继续存在于 raw18 数据、UDP packet 和 command 边界，但不进入 model16。model16 action
展开到 raw18 时，索引 15/17 来自 manifest 的 `wire_force.value`，其来源必须是
`explicit_x86_boundary_config`。默认旧数据 manifest 的左右值均为 `80.0`。

这个值不是策略预测，不是 measured force，也不是从输入 tensor 或缓存中复制。它只是为了保持
现有 raw18 wire contract 的显式 X86 边界配置；后续 Orin 固定安全配置的职责与本 schema 分离。

## 兼容性

| 使用场景 | 对外维度 | 行为 |
| --- | ---: | --- |
| UDP state/command packet | 18 个命名 scalar | JSON map 无数组索引；字段和旧 decoder 均不变 |
| parquet、`meta/info.json` | 18 | 不原地改写，完整 force 无损保留 |
| timing sidecar | 18 | 格式不变，`action_key_count` 仍为 18 |
| 旧 episode raw 读取 | 18 | 继续可读；无显式 provenance 时不能严格训练 |
| 原始 replay | 18 | 继续按旧 raw action 回放，不走 model16 投影 |
| 训练 dataset view | 16 | 只在读取边界按 manifest 投影 |
| policy observation/action | 16 | 14 joints + 2 openings |
| policy postprocessor 输出 | 18 | 显式反向映射 width，并填入边界 force 配置 |

预处理步骤在 normalization 之前执行，后处理步骤在 unnormalization 之后执行，两者和完整 manifest
一起保存在 checkpoint。训练、离线评估和 checkpoint 推理因此共用同一套 16D 模型语义；原始数据
和 raw replay 不受影响。

本地 model16 checkpoint 进入通用 `lerobot_record --policy` 时，X86 loader 会从序列化
`policy_preprocessor.json` 识别该 schema，并仅用 model16 metadata view 构建 policy；recorder 自身的
dataset 和 `make_robot_action()` 仍使用 raw18。policy frame 的 timing action source 明确记为
`policy_output`，不会伪装成 target-action UDP packet。没有 projection step 的旧 raw18 checkpoint
继续按旧路径加载。

## 离线检查

审计默认旧数据（预期因 generation reuse 返回 `FAIL`，同时仍验证 raw18/model16 映射）：

```bash
cd /home/luzhuang/cqy/aaa/flexible_lerobot

conda run --no-capture-output -n lerobot_flex \
  python my_devs/jz_robot_pin_timed/data_check/check_training_projection.py \
  --dataset-root tests/outputs/jz_robot_pin_timed_real_20260711_190502 \
  --manifest my_devs/jz_robot_pin_timed/train/manifests/jz_robot_pin_timed_real_20260711_190502.json
```

对来源尚未确认的数据只做审计报告时，可显式允许 `unavailable`；这不会使其获得严格训练资格：

```bash
conda run --no-capture-output -n lerobot_flex \
  python my_devs/jz_robot_pin_timed/data_check/check_training_projection.py \
  --dataset-root /path/to/dataset \
  --manifest /path/to/jz_pin_training_schema.json \
  --allow-unavailable \
  --report-json /path/to/training_projection_report.json
```

checker 校验 raw18 的 shape、精确字段名和顺序、16D 映射、force 删除、opening 来源与方向，并扫描
全部 Parquet 数值行确认 opening 有效且 raw 数据未被修改。来源可用时，它还要求每个 dataset frame
都有 timing/source_timing，且对应 gripper generation 在 session 内严格推进。它不会解码视频，也不会
启动 bridge、recorder、replay 或 command executor。

## 离线训练

只对通过严格 checker 的新数据执行训练：

```bash
cd /home/luzhuang/cqy/aaa/flexible_lerobot
DATASET_NAME=<strict-check-passed-dataset> \
bash my_devs/jz_robot_pin_timed/train/train_act_20_epochs.sh
```

脚本要求显式 `DATASET_NAME`；优先使用数据集内 `meta/jz_pin_training_schema.json`，也可显式传
`TRAINING_SCHEMA`。它会先运行严格 checker，再把 raw18 dataset view 投影为 model16。输出 checkpoint
还会校验 16D policy config、序列化 projection/expansion、三路图像 resize 和 manifest 一致性。
该训练流程不连接机器人，也不会运行 armed 模式。
