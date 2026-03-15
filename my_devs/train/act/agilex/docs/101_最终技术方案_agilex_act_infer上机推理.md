# Agilex ACT 上机推理最终技术方案

## 1. 目标与范围

### 1.1 目标

本方案的目标是：基于已经训练完成的 Agilex ACT checkpoint，规划一套可交给实现团队直接落地的实机上机推理方案。

本方案解决的是：

1. 如何把当前训练好的 ACT 模型安全地部署到 Agilex 双臂实机
2. 如何在不破坏训练-推理一致性的前提下复用现有 LeRobot 推理主线
3. 如何把上机流程拆成可执行、可验证、可分工的工作包

### 1.2 范围

本方案覆盖：

1. 推理架构
2. 交付形态
3. CLI / 配置建议
4. 上线分级策略
5. 风险与待确认项
6. 实施工作包

本方案不覆盖：

1. 具体代码实现
2. 代码评审
3. teleoperator 混控
4. 复杂在线可视化
5. 自动化统计平台

### 1.3 参考输入

本方案基于以下事实：

1. 训练权重位于：
   - `outputs/train/20260313_194500_act_agilex_first_test_full/checkpoints/100000/pretrained_model`
2. Agilex 当前已注册进仓库并具备：
   - `AgileXRobot`
   - `AgileXRosBridge`
   - `AgileXTeleoperator`
3. 现有参考推理脚本位于：
   - `my_devs/train/act/so101/run_act_infer.py`
4. 发散式脑暴文档已经完成，并形成了较稳定的架构共识

---

## 2. 训练产物与部署约束

### 2.1 当前 checkpoint 的硬约束

从 `config.json` 可以冻结出以下部署契约：

1. `policy.type = act`
2. `observation.state.shape = [14]`
3. 图像输入固定为三路：
   - `observation.images.camera_front`
   - `observation.images.camera_left`
   - `observation.images.camera_right`
4. 每路图像 shape 固定为 `3 x 480 x 640`
5. `output_features.action.shape = [14]`
6. `chunk_size = 100`
7. `n_action_steps = 100`

结论：

实现团队不得改变输入输出 schema，不得自行增减图像路数，不得把 14 维动作替换为其他动作语义。

### 2.2 当前 Agilex 机器人侧的现实约束

Agilex 当前接入层已经规定：

1. 状态来自：
   - `/puppet/joint_left`
   - `/puppet/joint_right`
2. 图像来自：
   - `/camera_f/color/image_raw`
   - `/camera_l/color/image_raw`
   - `/camera_r/color/image_raw`
3. 命令默认发布到：
   - `/master/joint_left`
   - `/master/joint_right`
4. `AgileXRobotConfig.control_mode` 只有两种：
   - `passive_follow`
   - `command_master`

### 2.3 Processor 约束

checkpoint 已保存：

1. `policy_preprocessor.json`
2. `policy_postprocessor.json`

因此部署必须遵循以下原则：

1. 必须从 checkpoint 直接加载 pre/post processor
2. 不允许手写 normalization / unnormalization 替代
3. 不允许手写 state / action 顺序并直接喂 tensor
4. 必须继续通过 LeRobot 的 robot feature / processor 桥接特征

### 2.4 最关键的部署前提

部署前必须确认：

1. 训练数据中的 14 维 `action` 语义，与现场 `/master/joint_*` 控制链路期望接收的动作语义完全一致
2. 双臂 joint order 在训练数据、processor、ROS bridge、命令发布四处保持一致
3. 三路图像在实机上稳定可用且语义不交换

其中第 1 条是上线前最高优先级前提。

---

## 3. 推荐的最终交付形态

### 3.1 核心交付物

建议实现团队交付以下四类产物：

1. Python 主脚本
   - 建议路径：`my_devs/train/act/agilex/run_act_infer.py`
2. Shell 包装脚本
   - 建议路径：`my_devs/train/act/agilex/run_act_infer.sh`
3. 运行文档
   - 建议更新 `my_devs/train/act/agilex/README.md` 或补充专门运行文档
4. 日志输出规范
   - 建议日志根目录继续使用仓库根目录 `logs/`

### 3.2 为什么采用双层交付

#### Python 主脚本负责

1. 参数解析
2. checkpoint 校验
3. 机器人构造
4. policy / processor 加载
5. 推理主循环
6. readiness 检查
7. 日志与异常处理

#### Shell 包装脚本负责

1. 固定使用 `lerobot_flex`
2. 固定默认 checkpoint 路径
3. 固定常用 topic 默认值
4. 统一日志文件命名
5. 简化现场执行命令

### 3.3 日志建议

建议把上机推理日志命名为：

1. `logs/infer_act_agilex_<job_name>_<timestamp>.log`

日志内容至少覆盖：

1. 启动配置摘要
2. readiness 结果
3. 周期运行摘要
4. 异常上下文
5. 结束原因

---

## 4. 推荐的推理脚本架构与主流程

### 4.1 总体架构决策

推荐架构如下：

1. 复用 `so101/run_act_infer.py` 的标准 LeRobot 推理主线
2. 只替换机器人接入层为 Agilex
3. 第一版只依赖 `AgileXRobot`
4. 第一版不接入 `AgileXTeleoperator`

### 4.2 推理主流程

建议实现团队按以下顺序组织主流程：

1. 解析命令行参数与环境变量
2. 解析并校验 `policy_path`
3. 加载 checkpoint config
4. 校验 checkpoint 是否满足：
   - 14 维 state
   - 3 路 `camera_front/camera_left/camera_right`
   - 14 维 action
5. 解析真实运行设备
6. 应用 ACT 运行期 override
7. 构造 `AgileXRobotConfig`
8. 加载 policy、preprocessor、postprocessor
9. 构造 robot observation / action processors 和 dataset feature bridge
10. 在 dry-run 下输出完整配置并退出
11. 连接 Agilex，等待 state + 三路图像 ready
12. 进入主循环：
   - 获取 observation
   - 走 observation processor
   - 构造 dataset frame
   - `predict_action(...)`
   - `make_robot_action(...)`
   - 走 robot action processor
   - 在 `command_master` 下发送动作
   - 在 `passive_follow` 下只做影子推理
13. 周期打印运行摘要
14. 异常时立即中止闭环并断开 robot
15. 正常退出时打印结束原因并断开 robot

### 4.3 机器人对象与控制模式

#### 第一版固定使用 `AgileXRobot`

原因：

1. 它已经同时具备观测采集和动作发布能力
2. 它的 feature 定义与当前数据链路一致
3. 它可以天然承载只读和闭环两种模式

#### 第一版默认 `control_mode=passive_follow`

这是一个架构级安全决策，理由如下：

1. 上机首次执行更适合从只读模式开始
2. `passive_follow` 下可以完成 observation check 和 shadow inference
3. 只有显式切到 `command_master` 才应允许真实闭环

### 4.4 关于 `AgileXTeleoperator`

第一版不纳入最小闭环的原因：

1. teleoperator 的职责是读取主臂动作，不是执行 policy 输出
2. 一旦把 teleoperator 混入第一版闭环，会模糊动作来源边界
3. 这会增加排障复杂度和上线风险

保留意见：

后续若要做人机接管、示教对比或策略切换，可单独立项扩展。

---

## 5. CLI / 配置项设计建议

### 5.1 必须暴露的参数

建议 Python 主脚本至少暴露以下参数：

1. `--policy-path`
2. `--policy-device`
3. `--policy-n-action-steps`
4. `--policy-temporal-ensemble-coeff`
5. `--task`
6. `--fps`
7. `--run-time-s`
8. `--log-interval`
9. `--dry-run`
10. `--control-mode`
11. `--state-left-topic`
12. `--state-right-topic`
13. `--command-left-topic`
14. `--command-right-topic`
15. `--front-camera-topic`
16. `--left-camera-topic`
17. `--right-camera-topic`
18. `--observation-timeout-s`

### 5.2 默认策略

建议默认值遵循以下原则：

1. checkpoint 默认指向当前已训练完成的 `pretrained_model`
2. `control_mode` 默认 `passive_follow`
3. topic 默认使用当前 Agilex 注册时的标准 topic
4. `policy_device` 默认 `auto`
5. `run_time_s` 默认允许短时试跑或由包装脚本设置

### 5.3 不建议第一版暴露的参数

不建议把以下内容作为用户可调参数开放：

1. 手工 feature rename
2. 手工 image key alias
3. 手工 state / action shape override
4. 手工 normalizer 开关
5. 训练期 optimizer / backbone 等配置

这些都属于静态契约，不属于部署用户应修改的内容。

### 5.4 Dry-run 输出要求

`--dry-run` 必须至少打印：

1. policy path
2. policy type
3. input / output feature 摘要
4. `chunk_size`
5. `n_action_steps`
6. `temporal_ensemble_coeff`
7. `control_mode`
8. state / command / image topic
9. 解析后的运行设备

---

## 6. 分阶段上线与验证策略

本方案明确采用分级上线策略，而不是单一脚本直接闭环。

### 6.1 阶段 0：静态资产核对

目标：

1. 核对 checkpoint、processor、设备环境
2. 冻结本次上线使用的 checkpoint

通过标准：

1. checkpoint 路径、processor 文件、设备策略全部明确
2. checkpoint schema 与 Agilex 当前契约一致

退出条件：

1. processor 缺失
2. checkpoint 不匹配当前 schema
3. 设备不可用且无替代策略

### 6.2 阶段 1：脚本级 dry-run

目标：

1. 不连 robot
2. 确认参数、设备、mode、topic 绑定

通过标准：

1. dry-run 输出可让现场工程师准确判断“这次要跑什么”

退出条件：

1. 解析结果与预期不符
2. 默认进入闭环模式
3. ACT override 未生效

### 6.3 阶段 2：只读观测检查

目标：

1. 使用 `passive_follow`
2. 连接 Agilex
3. 只检查 state 和 image readiness

通过标准：

1. 双臂 state 稳定刷新
2. 三路图像全部在线
3. 图像尺寸和编码正确

退出条件：

1. 任一 topic 缺失
2. 任一路图像不刷新
3. 图像 shape / 编码不符

### 6.4 阶段 3：影子推理

目标：

1. 保持 `passive_follow`
2. 跑完整 policy 前向
3. 不发布动作

通过标准：

1. 连续推理稳定
2. 动作 shape 始终为 14
3. 无 NaN / Inf
4. 动作统计值处于可接受范围

退出条件：

1. 推理报错
2. 动作维度错误
3. 动作统计明显异常
4. 实际循环频率无法接受

### 6.5 阶段 4：受限闭环

目标：

1. 切换到 `command_master`
2. 在短时间、低风险工况下测试真实动作发布

建议：

1. 首次闭环缩短 `run_time_s`
2. 首次闭环不要默认 `n_action_steps=100`
3. 优先使用更保守的 `n_action_steps=8` 或 `16` 起步
4. 操作员全程在场

通过标准：

1. 动作方向正确
2. 未出现明显震荡或发散
3. 系统可稳定停机

退出条件：

1. 动作异常
2. topic 丢失
3. loop 超时严重
4. 现场判断存在碰撞风险

### 6.6 阶段 5：正式任务闭环

目标：

1. 在受限闭环通过后，再进入更完整任务验证

放开顺序建议：

1. 先放开运行时长
2. 再放开起始位姿复杂度
3. 最后再评估是否恢复更长 action chunk

---

## 7. 实施工作包与优先级

### 7.1 P0：上线前提确认

必须优先完成：

1. 动作语义确认
2. 双臂 joint order 确认
3. 三路图像稳定性确认
4. 现场控制链路确认 `/master/joint_*` 的真实含义

没有这一步，不允许进入实现完成后的闭环测试。

### 7.2 P1：最小可用交付

实现团队第一阶段应完成：

1. `run_act_infer.py`
2. `run_act_infer.sh`
3. dry-run
4. 只读观测检查
5. 影子推理
6. 受限闭环
7. 日志落盘

### 7.3 P2：文档与运行手册

应完成：

1. 命令示例
2. 故障排查入口
3. 上线步骤清单
4. 退出条件清单

### 7.4 P3：后续增强项

可以后置：

1. teleoperator 接管
2. 动作保护层增强
3. 可视化面板
4. 自动记录验证数据
5. 更细粒度性能分析

---

## 8. 风险、依赖与待确认项

### 8.1 最高优先级风险

1. 动作语义与命令 topic 不一致
2. 14 维 state / action 顺序存在静默错位
3. 三路图像语义交换或缺失
4. 训练期 `n_action_steps=100` 直接带到首次闭环导致重规划过慢

### 8.2 关键依赖

1. `lerobot_flex` conda 环境
2. Agilex ROS topic 在线且稳定
3. 当前 `AgileXRobot` 接入层可正常连接
4. checkpoint 与 processor 文件完整

### 8.3 待确认项

1. 是否需要在第一版就加入动作限幅 / 首帧保持
2. 是否需要单独的 observation-only 子命令
3. 首次闭环的默认 `n_action_steps` 应设置为多少
4. 是否在 shell 包装层开放全部 topic override，还是只保留常用项

---

## 9. 最终建议

本方案建议实现团队按以下原则执行：

1. 贴着 `so101/run_act_infer.py` 的 LeRobot 标准推理路径迁移，不重写推理主线。
2. 第一版只使用 `AgileXRobot`，不把 teleoperator 带入纯推理闭环。
3. 把 `passive_follow` 作为默认安全模式，把 `command_master` 作为显式开启的闭环模式。
4. 把 dry-run、只读观测、影子推理、受限闭环做成第一版的内建能力。
5. 在动作语义确认完成前，不允许把当前 checkpoint 直接用于正式闭环任务。

如果实现团队只执行一条总原则，那就是：

先确保训练契约和控制语义完全一致，再谈模型效果和任务成功率。
