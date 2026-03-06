# ACT 推理/验证脚本审核分析

目录：`my_devs/train/act/so101`

本次补充了 3 个脚本：

- `run_act_infer.py`：ACT 实机纯推理脚本，不录数据。
- `run_act_eval_record.sh`：ACT 实机验证录制脚本，走 `lerobot-record` CLI。
- `run_act_eval_record.py`：ACT 实机验证录制脚本，走 `RecordConfig + record(...)` Python API。

以下是我结合仓库现有实现做出的审核与设计结论。

## 1. 我实际核对过的仓库依据

### 1.1 训练日志里的 ACT 关键配置

来自 `logs/train_act_grasp_block_in_bin1_e15_20260305_190147.log`：

- 训练输出目录：`outputs/train/20260305_190147_act_grasp_block_in_bin1_e15`
- `policy.type=act`
- `chunk_size=100`
- `n_action_steps=100`
- `device=cuda`
- 数据集：`admin123/grasp_block_in_bin1`

这说明当前 ACT checkpoint 不是单步控制，而是“预测一个 100 步动作块，然后逐步消费”。

### 1.2 checkpoint 里的真实输入输出定义

来自 `outputs/train/20260305_190147_act_grasp_block_in_bin1_e15/checkpoints/last/pretrained_model/config.json`：

- 输入：`observation.state`，shape=`[6]`
- 输入：`observation.images.top`，shape=`[3, 480, 640]`
- 输入：`observation.images.wrist`，shape=`[3, 480, 640]`
- 输出：`action`，shape=`[6]`

所以推理脚本必须严格提供：

- 1 路 6 维状态
- 2 路相机输入：`top` 和 `wrist`
- 输出给 SO101 follower 的 6 维 action

### 1.3 ACT 在仓库里的真实推理行为

来自 `src/lerobot/policies/act/modeling_act.py`：

- `ACTPolicy.select_action()` 并不是每一步都完整前向。
- 当 `temporal_ensemble_coeff is None` 时，会维护内部 `_action_queue`。
- 当队列为空时，调用 `predict_action_chunk()` 预测一整个 chunk。
- 然后按照 `n_action_steps` 从队列里逐步 `popleft()` 执行。

这意味着：

- 你当前这个 checkpoint 的默认行为，是“一次推理，缓存 100 步动作”。
- 若现场更希望高频重规划，应该把 `n_action_steps` 调小。

因此我在 ACT 脚本里额外暴露了：

- `--policy-n-action-steps`
- `--policy-temporal-ensemble-coeff`
- `--policy-device`

这些不是瞎加，而是针对 ACT 的 chunk 推理特性补的部署开关。

## 2. 为什么这次不是自己手拼张量，而是复用仓库现成链路

### 2.1 推理链路直接复用 `predict_action`

推理脚本使用的是：

- `lerobot.utils.control_utils.predict_action`
- `PolicyProcessorPipeline.from_pretrained(...)`
- `build_dataset_frame(...)`
- `make_default_processors()`

原因是这条链路已经被仓库官方录制/控制流程使用，能自动保证：

- 图像转成 `[C, H, W]`
- 数值归一化与反归一化
- observation key 命名与 checkpoint 完全对齐
- 输出 action 回写到 robot action 结构时不出错

### 2.2 必须加载 checkpoint 自带的 pre/post processor

我没有重写 normalization，而是直接加载：

- `policy_preprocessor.json`
- `policy_postprocessor.json`

原因很明确：

- 训练时统计量、归一化器、特征处理流程都已经固化在 checkpoint 里
- 自己手抄一版最容易出错，尤其是 state/action 的 mean/std

这也是 `run_groot_infer.py` 和 `run_pi05_infer.py` 已经采用的做法。

## 3. 为什么验证脚本要走 `lerobot-record`

你给的参考是 `my_devs/train/groot/so101/run_groot_eval_record.sh`，而仓库官方已有完整记录入口：

- `src/lerobot/scripts/lerobot_record.py`

它内部已经负责：

- 构建 robot
- 构建 dataset
- 根据 `policy.path` 加载 policy
- 每个 episode/reset 时执行 `policy.reset()`
- 跑实时控制循环
- 记录观测与动作

所以 ACT 验证最稳妥的方式，不是再自己写一套“边控制边存数据”的轮子，而是直接复用 `lerobot-record`。

这也是为什么我同时给了：

- shell 版：方便你直接跑
- python 版：方便你以后改 ACT 特有参数

## 4. 我有意识规避掉的风险点

### 4.1 没复制 GROOT shell 里那条危险的 `rm -r`

你给的 `my_devs/train/groot/so101/run_groot_eval_record.sh` 文件最开头，在 shebang 前有一条：

- `rm -r /home/cqy/.cache/huggingface/lerobot/admin123/eval_run_*`

这有两个问题：

- 写在 shebang 前，非常不安全
- 会无提示删除本地 eval 数据缓存

因此 ACT 版我没有照抄这个行为。

我做的是：

- shell 版默认不删任何缓存
- Python 版提供 `--cleanup-eval-cache`，默认关闭

这样安全很多，也更适合你自己上机时掌控。

### 4.2 没引入 GROOT/TensorRT 特殊逻辑

`run_groot_infer.py` 里有一大段 TensorRT backend 分支，那是 GROOT 专属的。

ACT checkpoint 是标准 LeRobot ACT policy，不需要：

- TRT engine patch
- backbone/action head 拆分
- 特定 dtypes

因此 ACT 版我只保留了仓库通用推理主线，避免引入不必要复杂度。

### 4.3 没假设 ACT 一定使用文本 tokenizer

PI0.5 的脚本会处理本地 tokenizer 目录；ACT 没有这个依赖。

ACT 脚本仍保留 `--task` / `DATASET_TASK`：

- 对录制数据集来说，task 字段本来就需要
- 对推理来说，即使 ACT preprocessor 不消费文本，这个字段也不会破坏推理链路

## 5. 这三个脚本各自适合什么场景

### 5.1 `run_act_infer.py`

适合：

- 不录数据
- 只想让机械臂直接执行 checkpoint
- 想现场调 `n_action_steps` 看控制风格

建议优先 dry run：

```bash
conda run -n lerobot_flex python my_devs/train/act/so101/run_act_infer.py --dry-run true
```

### 5.2 `run_act_eval_record.sh`

适合：

- 快速上机
- 直接沿用 `lerobot-record` CLI
- 希望用环境变量覆盖参数

### 5.3 `run_act_eval_record.py`

适合：

- 后续你要继续定制 ACT 部署参数
- 想在 Python 层做更细的调试
- 想更安全地控制缓存清理、device override 等逻辑

## 6. 我建议你上机时重点确认的 5 件事

1. `POLICY_PATH` 是否真指向 `pretrained_model` 目录。
2. `top` / `wrist` 相机索引是否和当前机器一致。
3. `CALIB_DIR` 是否对应当前 SO101 follower 标定目录。
4. `DATASET_REPO_ID` 在验证录制时是否以 `eval_` 开头。
5. 当前 ACT 如果感觉重规划不够及时，先把 `POLICY_N_ACTION_STEPS` 从 100 下调到 16 或 8 再试。

## 7. 一个很关键的部署判断

当前 checkpoint 用的是：

- `chunk_size=100`
- `n_action_steps=100`

这更像“高吞吐 chunk 执行”配置，而不是“高频闭环重规划”配置。

如果你上机后发现：

- 动作计划更新太慢
- 对扰动不够敏感
- 起手预测明显，后续修正慢

优先不要先怀疑脚本，先试：

- `POLICY_N_ACTION_STEPS=16`
或者
- `POLICY_N_ACTION_STEPS=8`

如果你要进一步追求每步都重规划，可以试：

- `POLICY_N_ACTION_STEPS=1`
- 再配合 `POLICY_TEMPORAL_ENSEMBLE_COEFF=0.01`

但这一组是部署策略调整，不是训练配置复刻；上机时要你自己根据动作稳定性来观察。

## 8. 本次我没有做的事

- 没上机运行脚本
- 没连接机械臂或相机
- 没帮你改训练配置
- 没自动清理任何本地数据

这和你的要求一致：代码写好给你，你自己上机验证。
