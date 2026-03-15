# Agilex ACT Checkpoint 与 Runtime 约束脑暴文档

## 1. 文档目标

本文不是实现方案，而是站在架构约束视角，先把当前 Agilex ACT checkpoint 对实机推理链路提出的硬要求列清楚，避免后续上机脚本设计偏离训练事实。

本轮分析对象：

- checkpoint 路径：`outputs/train/20260313_194500_act_agilex_first_test_full/checkpoints/100000/pretrained_model`
- 参考脚本：`my_devs/train/act/so101/run_act_infer.py`

---

## 2. 结论先行

这个 checkpoint 不是一个“随便读图、随便拼状态、直接吐动作”的黑盒模型，而是一个对输入 schema、processor 流程、动作维度、设备迁移方式都有明确要求的标准 LeRobot ACT checkpoint。

因此最终 Agilex 上机推理脚本的核心原则应该是：

1. 不自己手搓归一化/反归一化。
2. 不自己猜 observation / action 的字段顺序。
3. 不绕开 checkpoint 保存下来的 `policy_preprocessor.json` 和 `policy_postprocessor.json`。
4. 不把训练期的 100 步 chunk 语义误当成单步控制策略。

---

## 3. Checkpoint 对实机推理的硬约束

### 3.1 输入特征是固定 schema，不允许推理时自由发挥

`config.json` 已经把策略输入写死为：

- `observation.state`: shape = `[14]`
- `observation.images.camera_front`: shape = `[3, 480, 640]`
- `observation.images.camera_left`: shape = `[3, 480, 640]`
- `observation.images.camera_right`: shape = `[3, 480, 640]`

这意味着实机端必须稳定提供：

1. 一份 14 维状态向量。
2. 三路命名严格一致的相机输入。
3. 图像分辨率与通道语义必须和训练期一致。

如果 Agilex 实机侧只接两路图像、交换了左右相机含义、或者状态维度不是这 14 维，那么不是“效果差一点”的问题，而是直接违反 checkpoint 输入假设。

### 3.2 输出动作固定为 14 维

`output_features.action.shape = [14]`，说明最终策略输出的是一个 14 维动作向量，而不是：

- 单臂 7 维
- 末端位姿
- delta action
- gripper-only action

因此最终控制链路必须把这 14 维动作无歧义地映射到 Agilex 双臂命令接口。

从当前 Agilex 机器人实现来看，最自然的目标语义就是：

- 左臂 7 维 joint position
- 右臂 7 维 joint position

也就是按 `left_joint{i}.pos` 和 `right_joint{i}.pos` 组织。

### 3.3 这是视觉 + 状态的多模态 ACT，不是纯状态策略

checkpoint 显式依赖三路视觉输入，说明部署方案不能设计成“先只接状态跑起来，后面再补图像”。

如果最终上机链路缺失任意一路图像，理论上就已经偏离训练分布。即便代码层面做了占位，也属于应急降级，不应视为正式部署方案。

### 3.4 当前训练设备信息默认是 `cuda`

checkpoint `config.json` 和 `policy_preprocessor.json` 内都记录了设备为 `cuda`。这不意味着部署必须永远在 CUDA 上执行，但意味着：

1. 推理脚本必须有显式的 device 解析逻辑。
2. 必须允许把 checkpoint 的设备声明迁移到当前机器可用设备。
3. 迁移后，policy、preprocessor 输入张量、相关 device processor 必须一致。

参考 `so101` 脚本的处理方式是对的：读取 checkpoint 配置后，再按 runtime override 解析真实执行设备。

---

## 4. 推理时必须对齐的 feature / shape / device / processor 行为

### 4.1 必须复用 checkpoint 自带 pre/post processor

`policy_preprocessor.json` 中已经保存了：

1. `rename_observations_processor`
2. `to_batch_processor`
3. `device_processor`
4. `normalizer_processor`

`policy_postprocessor.json` 中已经保存了：

1. `unnormalizer_processor`
2. `device_processor`

这意味着最终推理脚本必须做的不是“重新实现归一化逻辑”，而是：

1. 从 checkpoint 目录直接加载保存下来的 processor pipeline。
2. 把实机 observation 先转换成与 dataset frame 一致的结构。
3. 让 preprocessor 去做批处理、迁移设备、标准化。
4. 让 postprocessor 去做动作反标准化与设备回迁。

这是最关键的硬约束之一。如果绕过这条链路，动作尺度和特征分布都可能错。

### 4.2 观测命名必须和训练 schema 一致

对这个 checkpoint 来说，至少要保证下列键名严格匹配：

- `observation.state`
- `observation.images.camera_front`
- `observation.images.camera_left`
- `observation.images.camera_right`
- `action`

这里的重点不是 Python 字典表面长什么样，而是进入 `build_dataset_frame(...)` 和 `predict_action(...)` 之前，最终构造出的 dataset-style frame 要与训练输入完全一致。

### 4.3 状态维度不仅要对，还要保证顺序对

`observation.state` 的 shape 是 14，并不等于“任意 14 维都能喂进去”。  
对当前 Agilex 接入，合理假设是训练集使用了双臂 follower joint positions，顺序应与当前机器人特征定义一致。

因此最终脚本必须复用 robot observation/action processor 与 robot feature 定义，不能自己硬编码一个手写顺序表后直接拼 tensor。

### 4.4 图像不仅要同名，还要同尺寸、同通道语义

视觉输入固定为 `[3, 480, 640]`，这意味着：

1. 输入到策略前需要是 CHW。
2. 三路图像的宽高必须与训练一致。
3. RGB/BGR 语义必须一致。

Agilex ROS bridge 当前已经把 `bgr8` 转成 RGB，这一点应被视作部署链路的一部分，不能在最终方案里被省略。

### 4.5 最终动作发送前仍应走 robot action processor

参考 `so101` 脚本，策略输出经过 `make_robot_action(...)` 后，还会经过 `robot_action_processor((action_dict, obs))` 再发送给机器人。

这个步骤的架构意义是：

1. 把策略输出映射回 robot action schema。
2. 让机器人侧 processor 处理必要的动作格式对齐。

对于 Agilex，最终脚本也应保留这条路径，而不是直接把 policy 输出数组扔给 ROS publisher。

---

## 5. ACT Chunk 推理在部署时的风险

### 5.1 当前 checkpoint 的默认行为是“100 步动作块推理”

`chunk_size=100`，`n_action_steps=100`。这代表默认部署语义不是每一步都重新规划，而是：

1. 一次前向推理产生一个 100 步动作 chunk。
2. 策略内部逐步消费这个动作队列。
3. 消费完后再触发下一次完整前向。

这对离线训练是合理的，但对实机部署会带来明显风险。

### 5.2 风险一：重规划不够及时

如果真实环境出现偏差，例如：

- 目标位置变化
- 抓取失败
- 相机视角轻微偏移
- 机器人执行延迟或关节跟踪误差

那策略可能继续消费旧 chunk，导致动作滞后。

### 5.3 风险二：100 步 chunk 与实机控制周期耦合过重

如果部署频率是 30Hz，100 步就意味着一次计划覆盖约 3.3 秒；如果是 15Hz，就是约 6.7 秒。  
这会把“感知误差修正”和“动作更新节奏”拉得很长。

因此部署时应把 `n_action_steps` 视作最重要的 runtime override 之一。

### 5.4 风险三：并不是所有 override 都能随便组合

参考 `so101` 脚本的约束逻辑：

1. `policy_n_action_steps` 必须落在 `[1, chunk_size]`
2. `temporal_ensemble_coeff` 只有在 `n_action_steps == 1` 时才合理

这说明最终 Agilex 脚本也应保留同样的参数约束，而不是只暴露参数、不做校验。

### 5.5 风险四：低延迟需求与算力压力之间的折中

`n_action_steps` 越小，重规划越频繁，但前向调用越密集，对 GPU/CPU 和图像预处理的压力越大。  
因此部署参数不能只追求“越小越灵”，而要结合实机端到端循环时延。

---

## 6. 建议暴露给最终上机脚本的最小参数集合

这里强调“最小集合”，目标是先把正式上机链路做稳，不是把所有内部细节都暴露给用户。

### 6.1 必须暴露

1. `--policy-path`
   - 指向 `pretrained_model` 目录，属于最基础输入。
2. `--policy-device`
   - 允许 `cpu/cuda/auto`，解决部署机器差异。
3. `--policy-n-action-steps`
   - ACT 部署最关键调参项。
4. `--policy-temporal-ensemble-coeff`
   - 作为可选高级参数，但建议保留。
5. `--task`
   - 即使当前 ACT 不一定强依赖文本，也建议保留，与仓库其他推理脚本行为一致。
6. `--run-time-s`
   - 控制一次推理实验持续时间。
7. `--log-interval`
   - 便于观察实时状态。
8. `--dry-run`
   - 在不上机器人、不加载完整执行链路的前提下确认配置正确。

### 6.2 机器人侧建议暴露，但应尽量收敛

针对 Agilex 真实上机，建议把机器人配置参数收敛成少量高价值项，例如：

1. 机器人配置文件路径，或机器人 topic 配置入口
2. 控制模式
3. 是否要求等待三路图像 ready
4. 推理主循环频率或节拍控制参数

不建议第一版就把所有 topic 名称、所有 camera key、所有 bridge 细节都作为命令行参数暴露，否则接口会迅速失控。

### 6.3 不建议第一版暴露的内容

1. 各种训练期 optimizer / dropout / backbone 参数
2. 手工 feature rename_map
3. 手工 normalizer 开关
4. 动作维度/状态维度手工覆盖

这些都属于 checkpoint 已经定义好的静态事实，不应交给部署用户随意改。

---

## 7. 对最终架构方案的建议

### 7.1 推理主线要尽量贴近 `so101` 的标准实现

推荐保留下面这条主链路：

1. 读取 checkpoint config
2. 做 runtime device / ACT override 解析
3. 加载 policy
4. 加载 checkpoint 自带 pre/post processor
5. 通过 robot processors 把 Agilex observation/action 对齐成 dataset schema
6. 调用 `predict_action(...)`
7. 通过 robot action processor 后下发到机器人

这条路线的优点是：

1. 最贴近训练与推理的一致性要求
2. 最少重复实现
3. 最容易定位问题到底出在 robot、processor、policy 还是 runtime 参数

### 7.2 第一版目标应是“正确推理”，不是“功能大全”

第一版最终上机脚本应该优先保证：

1. feature 对齐正确
2. processor 对齐正确
3. chunk 参数可控
4. dry-run 可用
5. 日志足够排障

不建议第一版同时追求：

1. 自动录评测数据
2. UI 面板
3. 在线热切换模型
4. 自动 topic 探测

这些属于第二阶段增强能力。

---

## 8. 架构判断

如果后续要把 Agilex 上机 ACT 推理做稳，最重要的不是“把 `so101` 脚本改成 Agilex 版”这么简单，而是把下面四件事锁死：

1. checkpoint schema 与 Agilex 机器人特征完全一致
2. pre/post processor 直接复用 checkpoint 原件
3. ACT chunk 行为通过 runtime override 受控
4. 最终脚本暴露少量但关键的部署参数

在这个前提下，Agilex 版推理脚本才会是一个“受 checkpoint 约束的标准部署入口”，而不是一份临时试验脚本。
