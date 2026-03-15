# Agilex ACT 推理脚本迁移策略脑暴

## 1. 文档目的

本文件不是最终方案，而是给架构师收敛用的一份发散式临时分析。

目标问题只有一个：

- 如何把 `my_devs/train/act/so101/run_act_infer.py` 这条已经成型的 ACT 实机推理链路，迁移成适用于 Agilex 双臂机器人的上机推理方案。

## 2. 我的核心结论

我不建议从零写一套新的推理主线。

更稳妥的路线是：

1. 保留 `run_act_infer.py` 的主干推理架构。
2. 把 `SOFollowerRobotConfig + OpenCVCameraConfig` 这一层替换成 `AgileXRobotConfig + ROS topics`。
3. 保留 checkpoint 自带的 pre/post processor 加载方式。
4. 保留 `predict_action -> make_robot_action -> robot_action_processor -> robot.send_action` 这一条标准出动作路径。
5. 在交付形态上，建议同时提供：
   - 一个纯 Python 主脚本，承载真实推理逻辑。
   - 一个薄 shell 包装脚本，承载上机时常改的环境变量和默认参数。

原因很直接：

- Python 主脚本负责结构正确性和可维护性。
- shell 包装脚本负责上机便利性和参数标准化。
- 只给 shell，不利于后续调试。
- 只给 Python，不利于现场快速执行和重复调用。

## 3. 哪些模块可以原样复用

以下部分我认为可以基本原样复用，不应该重写：

### 3.1 推理主循环结构

`run_act_infer.py` 的核心循环结构是健康的：

1. 读取 robot observation
2. 通过 `predict_action(...)` 进行推理
3. 把动作转换成 robot action
4. 下发给机器人
5. 按固定频率打印日志

这条主线对 Agilex 同样成立，因为 Agilex 也已经在仓库里实现了标准 `Robot` 接口。

### 3.2 checkpoint 加载方式

以下机制应原样保留：

- `PreTrainedConfig.from_pretrained(...)`
- `policy_class.from_pretrained(...)`
- `PolicyProcessorPipeline.from_pretrained(...)`
- `load_pre_post_processors(...)`

理由：

- 训练期的 normalizer/unnormalizer 已经固化在 checkpoint 中。
- 重新手写 normalization 是高风险行为。
- 迁移对象变的是机器人接入层，不是 policy/checkpoint 语义。

### 3.3 ACT 运行期覆盖逻辑

SO101 脚本里的这类能力应继续保留：

- `--policy-device`
- `--policy-n-action-steps`
- `--policy-temporal-ensemble-coeff`
- `--dry-run`

这些开关与机器人品牌无关，而是 ACT 部署本身的刚需。

### 3.4 默认 processor 与特征桥接

以下组合应继续沿用：

- `make_default_processors()`
- `aggregate_pipeline_dataset_features(...)`
- `create_initial_features(...)`
- `combine_feature_dicts(...)`
- `build_dataset_frame(...)`
- `make_robot_action(...)`

这是整个迁移里最应该保留的部分，因为它能保证：

- observation/action key 命名对齐
- 图像转置和张量组织对齐
- state/action 归一化统计对齐
- 输出 action 回写到机器人接口时不丢维度

## 4. 哪些模块必须替换

### 4.1 机器人配置构造必须替换

SO101 脚本当前依赖：

- `SOFollowerRobotConfig`
- 串口 `--robot-port`
- OpenCV 相机索引 `--top-cam-index`、`--wrist-cam-index`

Agilex 不能继续走这一层，因为 Agilex 当前接入是 ROS topic 架构，不是串口加 USB 相机架构。

因此迁移时必须替换为：

- `AgileXRobotConfig`
- 左右从臂状态 topic
- 左右命令 topic
- 三路图像 topic
- `control_mode`

### 4.2 相机输入模型必须替换

SO101：

- 相机来自 `OpenCVCameraConfig`
- 常见 key 是 `top` 和 `wrist`
- 脚本直接控制 camera index

Agilex：

- 图像来自 ROS `sensor_msgs/Image`
- 当前 key 是 `camera_front`、`camera_left`、`camera_right`
- 解码逻辑在 `AgileXRosBridge._decode_image(...)`

这不是简单改参数名，而是整套输入源不同：

- SO101 是本地设备索引
- Agilex 是 ROS 消息订阅

### 4.3 动作下发模型必须替换

SO101 的 `send_action` 最终走的是 follower 机器人本地控制链路。

Agilex 的 `send_action` 则依赖：

- `AgileXRosBridge.publish_action(...)`
- 发布两路 `JointState`
- 左右臂各 7 维，共 14 维

此外还有一个关键差异：

- `AgileXRobotConfig.control_mode == "command_master"` 时才真的发命令
- 默认值却是 `passive_follow`

这意味着如果迁移脚本没有显式设置 control mode，上机时可能“脚本看起来在跑，但机器人不动”。

这是 Agilex 版本必须重点处理的部署陷阱。

### 4.4 机器人类型与运行前校验必须替换

SO101 脚本会校验：

- `robot_type in {"so100_follower", "so101_follower"}`

Agilex 版本需要改成自己的机器人标识，并在运行前校验：

- checkpoint 输入是否是 14 维 state
- 是否存在 `camera_front/camera_left/camera_right`
- output action 是否是 14 维
- 机器人 config 是否处于可下发命令的 control mode

## 5. Agilex 与 SO101 的结构差异

### 5.1 机器人实例化差异

SO101：

- 机器人是单机直连设备
- 需要 `robot_port`
- 需要标定目录
- 相机配置直接挂在 robot config 上

Agilex：

- 机器人是 ROS bridge 驱动
- 不依赖串口
- 当前实现里 `calibrate()` 和 `configure()` 基本为空
- 关键是 topic 绑定正确和启动阶段能等到 topic ready

架构意义上，Agilex 更像“topic 接入型 robot”，不是“硬件直连型 robot”。

### 5.2 观测输入差异

SO101：

- 典型是 1 路状态 + 2 路图像
- key 通常是 `top` / `wrist`

Agilex：

- 当前实现固定是 14 维 follower 关节位置
- 图像 key 固定是 `camera_front` / `camera_left` / `camera_right`
- `wait_for_ready(..., require_images=True)` 会在启动阶段阻塞，直到三路图像和双臂状态都到齐

这意味着 Agilex 版本比 SO101 更依赖启动前 topic 健康度。

### 5.3 动作语义差异

SO101：

- 单臂 action 维度较小
- 机器人侧动作语义和本地 follower 电机组织高度绑定

Agilex：

- 双臂 14 维 action
- key 命名是 `left_joint{i}.pos` 与 `right_joint{i}.pos`
- 下发时拆成左右两条 `JointState` 消息

因此 Agilex 版本更需要在脚本启动前做 shape 和 feature key 的显式打印，否则排错成本会高很多。

## 6. 推荐交付形态

我建议最终交付两层：

### 6.1 第一层：纯 Python 主脚本

建议形态：

- `my_devs/train/act/agilex/run_act_infer.py`

职责：

- 解析参数
- 构造 `AgileXRobotConfig`
- 加载 checkpoint / processors / policy
- 执行纯推理主循环
- 打印结构化运行日志

原因：

- 真正的架构复杂度都在 Python 里。
- 只有 Python 脚本，后续才容易继续扩展 dry-run、health-check、metrics、异常处理。

### 6.2 第二层：薄 shell 包装脚本

建议形态：

- `my_devs/train/act/agilex/run_act_infer.sh`

职责：

- 固定使用 `lerobot_flex`
- 填默认 checkpoint 路径
- 提供现场最常改的 topic / device / task / runtime 参数入口
- 统一日志落盘路径

原因：

- 现场上机时更少打长命令
- 更方便形成团队统一调用方式
- 也能与现有 Agilex 训练脚本风格保持一致

### 6.3 不建议只交付 shell

原因：

- shell 很难优雅处理 shape 校验、processor 加载、异常分类和 dry-run 输出。
- 一旦后面要加“topic 健康检查”或“只连机器人不发命令”的模式，shell 会很快变脆。

## 7. 最终脚本应保留的能力

### 7.1 dry-run 能力

必须保留：

- `--dry-run`

dry-run 至少应输出：

- 解析后的 `policy_path`
- 解析后的 `policy.type`
- 输入输出 feature 摘要
- `chunk_size`、`n_action_steps`
- `control_mode`
- 左右状态 topic
- 左右命令 topic
- 三路图像 topic

Agilex 上机时，这个能力比 SO101 更重要，因为 ROS topic 错一个名字，整个链路就不成立。

### 7.2 runtime override 能力

建议保留的最小集合：

- `--policy-path`
- `--policy-device`
- `--policy-n-action-steps`
- `--policy-temporal-ensemble-coeff`
- `--task`
- `--run-time-s`
- `--log-interval`
- `--control-mode`
- `--state-left-topic`
- `--state-right-topic`
- `--command-left-topic`
- `--command-right-topic`
- `--front-camera-topic`
- `--left-camera-topic`
- `--right-camera-topic`

如果只保留很少参数，会导致脚本强依赖现场 ROS 拓扑不变化，这在实机环境里不现实。

### 7.3 日志能力

建议保留三层日志：

1. 启动日志
   - 打印所有解析后的关键配置。
2. 周期日志
   - 每 N 步打印 step、耗时、平均频率。
3. 异常日志
   - 启动期区分“checkpoint 无效”和“ROS topic 未就绪”和“图像编码不支持”和“control mode 不可发送”。

Agilex 版本还建议额外打印：

- 首帧 observation 的 key 列表
- 首次 action 的维度摘要

这类日志对双臂系统排错价值很高。

## 8. 我认为最终方案最容易踩坑的地方

### 8.1 `control_mode` 默认值不适合纯推理上机

`AgileXRobotConfig` 默认 `control_mode="passive_follow"`。

如果最终脚本直接吃默认值：

- observation 能读到
- policy 能推理
- send_action 也可能返回
- 但实际上不会发布命令

因此最终方案必须显式约束：

- 纯推理执行机器人动作时，默认使用 `command_master`
- 或者在 dry-run/启动日志中非常明确地警告当前模式不会发命令

### 8.2 checkpoint key 与 robot key 必须严格对齐

Agilex 当前 checkpoint 预期的是：

- `observation.images.camera_front`
- `observation.images.camera_left`
- `observation.images.camera_right`
- `observation.state` shape = 14
- `action` shape = 14

所以最终脚本绝不能把相机 key 写成：

- `front`
- `left`
- `right`

也不能继续沿用 SO101 的：

- `top`
- `wrist`

### 8.3 ROS 启动健康度会成为首要故障源

SO101 的首要问题通常是：

- 串口
- 相机索引

Agilex 的首要问题会变成：

- topic 不在线
- 图像编码不匹配
- 某一路图像没有数据
- 命令 topic 配置错但状态 topic 配对成功

因此 Agilex 版本应比 SO101 更强调启动前检查与错误分类。

## 9. 对最终方案的建议

如果让我给架构层做决策，我建议最终收敛成下面这个结论：

1. 不重写推理主线，直接继承 `run_act_infer.py` 的 LeRobot 标准路径。
2. 只替换 robot/config/input/output 适配层，让 checkpoint、processor、predict_action 逻辑保持不变。
3. 最终交付采用“Python 主脚本 + shell 包装脚本”的双层形态。
4. 把 Agilex 特有风险收敛到启动参数和 dry-run 检查里，尤其是：
   - `control_mode`
   - topic 绑定
   - 三路图像 key
   - 双臂 14 维 action/state 对齐

这条路线最省风险，也最符合当前仓库已经成型的抽象边界。
