# PI05 实时推理工程化技术方案

本稿已根据 `reviews/review_01.md`、`reviews/review_02.md`、`reviews/review_03.md` 的独立评审意见完成收敛修订。

## 1. 结论先行

本轮方案的最终结论不是“直接把当前 `run_pi05_infer.py` 改成通用 `async_inference`”，也不是“继续在同步单循环上打补丁”，而是采用一条分阶段、但目标明确的路线：

1. 第一阶段先在 `my_devs/` 内落一个 **PI05 本地 chunk runner**。
2. 这个 runner 不再走 `select_action()`，而是切换到 `predict_action_chunk()`。
3. 它在本地进程内引入 **显式 ActionQueue、双线程执行、延迟统计、chunk 级日志**。
4. RTC 不作为第一步直接打开，而是在 chunk runner 稳定后，以可开关方式接入。
5. `src/lerobot/async_inference/` 仍然保留为目标架构的一部分，但它更适合作为 **第二阶段后端化/服务化能力**，而不是第一版落地路径。

一句话概括：

> 先把当前已经能跑起来的 PI05 实机推理，升级成“本地 chunk + 可观测 + 可切换 RTC”的稳定运行时；再决定是否把这套运行时语义映射到通用 `async_inference` 的 server/client 模式。

## 2. 当前事实与问题边界

### 2.1 当前脚本已经“能跑”，但还不是目标形态

用户当前运行的是：

- `my_devs/train/pi/so101/run_pi05_infer.py`

从日志看，它已经可以：

- 正常加载 checkpoint
- 正常进入持续推理
- 正常在机器人上连续发送动作

这说明一件重要的事：

- **模型本身和当前 checkpoint 基本可用**

但它并没有证明以下这些更重要的实时控制问题已经解决：

- 是否真正稳定达到目标控制频率
- 是否存在 action starvation
- chunk 切换时是否平滑
- 推理延迟波动对动作连续性有多大影响
- 是否已经具备 RTC 所需的运行时状态

所以现在不是“模型能不能用”的问题，而是“运行时怎么工程化”的问题。

### 2.2 当前脚本是同步单循环，且走的是单步 action 路径

`run_pi05_infer.py` 的主循环是典型同步结构：

1. `robot.get_observation()`
2. 预处理 observation
3. `predict_action(...)`
4. `robot.send_action(...)`
5. `precise_sleep(...)`

而 `src/lerobot/utils/control_utils.py` 中的 `predict_action(...)` 最终调用的是：

- `policy.select_action(...)`

这意味着当前运行路径是：

- 单步 action 推理
- policy 内部 action queue
- 调用方对 chunk 调度没有控制权

### 2.3 PI05 的 RTC 根本不支持当前这条路径

在 `src/lerobot/policies/pi05/modeling_pi05.py` 里，`PI05Policy.select_action()` 明确断言：

- RTC 不支持 `select_action()`
- RTC 只能通过 `predict_action_chunk()` 使用

也就是说，只要当前脚本还维持：

- `predict_action(...) -> policy.select_action(...)`

那么 RTC 就不可能真正接入。

### 2.4 当前 checkpoint 的运行语义需要被保留

当前工作脚本通过：

- `policy_class.from_pretrained(str(policy_path), strict=False)`

来加载模型，并且日志中已经出现：

- missing key
- vision embedding warning

但整体仍能正常跑起来。

这说明当前 checkpoint 的“已知可用加载语义”是：

- 非严格加载
- 使用 checkpoint 内保存的 pre/post processor
- 使用本地 tokenizer 目录兜底

任何新的运行时方案都必须保留这些行为，否则非常容易出现：

- 方案在架构上更“漂亮”
- 但实际把当前可用脚本跑挂了

这个风险在直接切换到通用 `async_inference` 时尤其明显。

### 2.5 checkpoint 当前配置表明，它本来就是 chunk policy

当前 checkpoint 的 `config.json` 显示：

- `type = pi05`
- `chunk_size = 50`
- `n_action_steps = 50`
- `num_inference_steps = 10`
- `rtc_config = null`

这表明：

1. 模型本身天然支持 chunk 推理。
2. 当前没有启用 RTC，只是在用 PI05 的普通 chunk policy 能力。
3. 从同步单步路径迁移到 chunk runner，并不是“换模型”，而是“把模型原生能力真正用起来”。
4. 第一阶段最好把“模型生成 chunk 大小”和“运行时实际执行 horizon”解耦，否则默认一次执行 50 步在 30 FPS 下对应约 1.67 秒 horizon，过长。
5. 如果未来要真正打开 RTC，不能只传 runtime kwargs，还必须显式启用 `policy.config.rtc_config` 并确保 RTC processor 已正确初始化，否则仍会走非 RTC 分支。

## 3. 本次最终推荐路线

## 3.1 第一阶段目标

第一阶段不追求分布式、不追求 gRPC、不追求 server/client 解耦，而是追求：

- 保留当前脚本的已知可用行为
- 切到 `predict_action_chunk()`
- 把动作执行与推理解耦
- 建立运行时观测能力
- 为 RTC 做好状态机准备

因此第一阶段的最优解是：

- 在 `my_devs/` 里实现一个 **PI05 本地实时 chunk runner**

这个 runner 应该是新的主入口，而不是继续往当前同步脚本里叠逻辑。

## 3.2 第二阶段目标

当第一阶段稳定后，再考虑：

- 让推理进程和机器人控制进程解耦
- 用通用 `async_inference` 做更标准的后端化
- 把本地 runner 的语义迁移为 server/client 运行模式

因此：

- `src/lerobot/async_inference/` 是 **第二阶段目标后端**
- 不是第一阶段直接替换现有脚本的入口

## 3.3 为什么不是直接上 `async_inference`

虽然 `async_inference` 已经支持：

- `pi05` policy type
- chunk 推理
- client/server 解耦
- action queue refill 机制

但它目前并不满足 PI05 这次落地的关键要求：

### 第一，它不是 RTC-aware 的

现成 `PolicyServer` 调的是：

- `policy.predict_action_chunk(observation)`

但没有向 PI05 传入运行期 RTC 所需参数：

- `prev_chunk_left_over`
- `inference_delay`
- `execution_horizon`

所以它目前只能做到：

- chunk 异步执行

还做不到：

- RTC 连续性增强

### 第二，它的 queue 语义不是 RTC 的 queue 语义

`src/lerobot/async_inference/robot_client.py` 使用的是：

- Python `Queue`
- overlapping timestep aggregation
- 侧重“重叠区融合”

而 `src/lerobot/policies/rtc/action_queue.py` 的语义是：

- 同时维护 original actions 和 processed actions
- 通过 `last_index` 跟踪消费位置
- RTC enabled 时按 `real_delay` 替换队列
- RTC disabled 时追加队列

两者关注点不同。

对 RTC 来说，queue 必须是：

- 显式的
- 有 action consumption index 的
- 能给出 `prev_chunk_left_over` 的
- 能基于 `real_delay` 做替换的

现成 `async_inference` 不是不能改，但它不是当前最短路径。

### 第三，它不天然保留当前脚本的“已知可用加载语义”

当前 working script 的几个关键点：

- `strict=False`
- 本地 tokenizer 兜底
- checkpoint-local processor
- 当前 so101 摄像头/机器人 wiring

如果直接切换到 generic async stack，很容易在模型加载或 processor 细节上偏离当前已知可用路径。

### 第四，它会放大早期排障复杂度

当前用户只需要：

- 一个脚本
- 一个日志流
- 一个进程内状态面

而 generic async 方案意味着：

1. 起 server
2. 起 client
3. 处理 gRPC 依赖
4. 处理两边日志
5. 处理更多时间戳与队列问题

这对第二阶段是合理成本，但对第一阶段是不必要的排障负担。

## 4. 第一阶段的推荐架构

本阶段的核心参考不是空想，而是两部分的组合：

1. 用户当前的 working script：
   - `my_devs/train/pi/so101/run_pi05_infer.py`
2. 仓库内已有的 RTC 实机参考：
   - `examples/rtc/eval_with_real_robot.py`

这两个合在一起，已经足够构成第一阶段方案。

### 4.1 运行时线程模型

推荐采用双线程本地运行时：

1. `ChunkProducer` 线程
2. `ActorExecutor` 线程

职责如下：

### `ChunkProducer`

负责：

- 读取 observation
- 走 robot observation processor
- 构建 dataset frame
- 执行 preprocessor
- 调用 `policy.predict_action_chunk(...)`
- 执行 postprocessor
- 更新 `ActionQueue`
- 记录 latency / delay / leftover 等指标

### `ActorExecutor`

负责：

- 从 `ActionQueue` 取下一个动作
- 转成 robot action dict
- 下发给机器人
- 以固定 FPS 执行
- 更新 action consumption index

### 4.2 运行时数据流

建议固定为下面这条链路：

1. `robot.get_observation()`
2. `robot_observation_processor(obs)`
3. `build_dataset_frame(...)`
4. 转 tensor / image 格式调整 / 补 batch 维度
5. `preprocessor(...)`
6. `policy.predict_action_chunk(...)`
7. `original_actions = actions.squeeze(0).clone()`
8. `postprocessor(actions)`
9. `ActionQueue.merge(original_actions, postprocessed_actions, ...)`
10. actor thread 周期性 `action_queue.get()`
11. `robot.send_action(...)`

这里要明确一条硬边界：

- `original_actions` 始终保留在模型输出动作空间中，只服务于 leftover、delay 对齐和 RTC merge。
- `postprocessed_actions` 只服务于机器人执行。
- 运行时绝不能从 `postprocessed_actions` 反推 `original_actions`。

这个流程几乎可以直接对齐 `examples/rtc/eval_with_real_robot.py`，但要保留当前 PI05 working script 的以下细节：

- checkpoint-local processor 加载方式
- `strict=False` 加载策略
- `ensure_local_tokenizer_dir(...)`
- 当前 so101 相机和机器人配置

这里还要明确一条实现红线：

- `original_actions` 只用于 RTC leftover、delay 对齐和 queue merge，必须保留在模型动作空间。
- `processed_actions` 只用于机器人 rollout，不能再反推成 `original_actions` 参与 RTC 逻辑。

### 4.3 queue 统一采用 RTC 的 queue 语义

第一阶段即使 RTC 默认先不开，也建议直接采用：

- `src/lerobot/policies/rtc/action_queue.py`

而不是自己再写一个新 queue。

原因是：

1. 非 RTC 模式下它也能工作
2. RTC 开启后不需要重写 queue
3. 它已经把最关键的语义抽象好了：
   - processed queue
   - original queue
   - `last_index`
   - `get_left_over()`
   - `merge()`

这能避免第一阶段做完以后，第二阶段为 RTC 再推翻一次。

此外，plain chunk 模式也不能假设“所有新 chunk 都天然新鲜”：

- 如果推理完成时前缀已经过期，运行时应丢弃已过期前缀，只把仍然有效的后缀写入队列。
- 一旦 plain 模式下的过期前缀比例超过阈值，应把事件记入日志并视为需要调参的信号。

### 4.4 延迟估计采用 `LatencyTracker + ceil(latency / step_dt)`

推荐直接继承 `examples/rtc/eval_with_real_robot.py` 的做法：

1. 用 `LatencyTracker` 记录近期 chunk 推理耗时
2. 用：
   - `time_per_chunk = 1.0 / fps`
   - `inference_delay = ceil(inference_latency / time_per_chunk)`
3. 在当前次推理完成后，用新测得的 `new_latency` 更新：
   - `new_delay = ceil(new_latency / time_per_chunk)`
4. 用 `new_delay` 和 `action_index_before_inference` 做 queue merge

这个机制虽然简单，但已经足够形成第一阶段稳定运行语义。

但这里需要明确一个优先级：

- wall-clock latency 主要用于统计与近似估计
- action consumption index delta 才是 queue merge 时更可信的真值信号

如果两者不一致，运行时应该优先以 action index 的真实消费差作为 merge 依据，并把偏差写进日志。

## 5. RTC 的接入策略

## 5.1 RTC 不是第一阶段默认开启项

推荐顺序必须是：

1. 先把 chunk runner 跑稳
2. 再把 RTC state plumbing 接上
3. 最后再打开 RTC guidance

不要在同步脚本改成 chunk runner 的同时，再一口气把 RTC 也打开。

这样做的原因很明确：

- 你无法区分故障到底来自线程/队列重构，还是来自 RTC guidance 参数

## 5.2 RTC 开启前必须具备的运行时状态

要让 RTC 真正生效，运行时必须拥有以下状态：

- `prev_chunk_left_over`
- `action_index_before_inference`
- `real_delay`
- `execution_horizon`
- `original_actions`
- `processed_actions`

缺一不可。

其中最关键的是：

- `prev_chunk_left_over = action_queue.get_left_over()`
- `inference_delay` 或 `real_delay`

另外还有一个必须显式处理的实现事实：

- 当前 checkpoint 配置里 `rtc_config = null`

这意味着 RTC 不是仅靠运行时传 kwargs 就能打开的。运行时必须在 policy 初始化前或初始化后以受控方式显式启用 `rtc_config`，否则 `predict_action_chunk()` 仍会走非 RTC 分支。

推荐的受控做法是：

1. 构造 `RTCConfig(...)`
2. 设置：
   - `policy.config.rtc_config = rtc_cfg`
3. 调用：
   - `policy.init_rtc_processor()`

如果这些状态不可靠，RTC 的 guidance 反而会制造抖动。

## 5.3 RTC 打开后的策略

推荐默认参数从保守值开始：

- `rtc.enabled = true`
- `rtc.execution_horizon = 8~12`
- `rtc.max_guidance_weight = 10.0`
- `rtc.prefix_attention_schedule = EXP`

同时保留一个运行期开关，使其可以：

- 一键退回 plain chunk 模式

这样一旦发现：

- chunk 连续性异常
- 抖动加剧
- queue merge 不稳定

就可以立刻把问题缩小到“RTC 打开后才出现”。

## 5.4 第一阶段必须内建安全门槛

这次是实机运行时改造，不是离线推理脚本改造。

因此第一阶段必须把以下安全门槛写进实现要求，而不是留给编码时临场决定：

1. **首个 chunk 未就绪时的行为**
   - actor thread 不能在首个 chunk 未 ready 时随意发送动作
   - 推荐显式 barrier：首个 chunk ready 后 actor 再进入稳态执行
2. **空队列时的行为**
   - `ActionQueue.get()` 返回空时，必须有固定策略
   - 推荐优先级：
     1. `hold-last-action`
     2. 若实现困难，则 `skip-send + 记录 starvation`
   - 不允许留成隐式未定义行为
3. **异常动作保护**
   - 运行时应预留 action delta / joint delta sanity check
   - 若单步动作超出安全阈值，应拒绝发送并打 warning
4. **异常时降级路径**
   - plain chunk 与 RTC 两种模式都必须支持：
     - stop
     - hold
     - fallback 到 baseline runner
     三者至少具备前两者
5. **优雅停机**
   - Ctrl+C、线程异常、queue 语义异常时，必须有一致的 robot shutdown / disconnect 路径

第一阶段不要求做复杂的安全控制框架，但必须把这些门槛纳入 runtime contract。

## 6. 代码改造边界

## 6.1 新增代码应主要放在 `my_devs/`

按用户仓库约定，第一阶段代码开发默认放在 `my_devs/`。

因此建议新增一个 PI05 实时运行模块，形态可以是：

- `my_devs/train/pi/so101/pi05_realtime_runtime.py`
- `my_devs/train/pi/so101/run_pi05_realtime.py`

其中：

- 一个负责 runtime/library
- 一个负责 CLI 入口

### 6.2 当前 `run_pi05_infer.py` 的定位

不建议直接把它堆成复杂双线程大脚本。

更好的做法是：

1. 保留 `run_pi05_infer.py` 作为 baseline / fallback
2. 新增实时版本入口
3. 后续如果新方案稳定，再决定是否收敛入口

这样可以保证：

- 现有“能跑”的路径不会被立即破坏

### 6.3 第一阶段尽量不改 `src/lerobot/async_inference/`

第一阶段不推荐直接改通用 async stack。

因为当前最重要的是先解决：

- PI05 当前本地运行时可用性
- chunk scheduling
- RTC queue 语义
- checkpoint 加载一致性

这些事情在 `my_devs/` 内就能完成。

### 6.4 `src/` 内可复用的现成模块

第一阶段建议复用，而不是重写：

- `src/lerobot/policies/rtc/action_queue.py`
- `src/lerobot/policies/rtc/latency_tracker.py`
- `examples/rtc/eval_with_real_robot.py` 中的线程职责与 RTC 调用模式
- `run_pi05_infer.py` 中的：
  - repo root 解析
  - 本地 tokenizer 检查
  - checkpoint-local processor 加载
  - robot config / camera config 组装

### 6.5 执行 horizon 要与模型 chunk size 区分

当前 checkpoint 的：

- `chunk_size = 50`
- `n_action_steps = 50`

并不意味着第一阶段必须把 50 个动作全部压入执行队列。

推荐运行时显式区分：

1. 模型生成长度
2. 实际执行 horizon

也就是：

- 模型可以一次生成完整 chunk
- 但运行时只把前 `N` 步压入执行队列
- 通过类似 `--actions-per-chunk` 的参数控制

这样在第一阶段实机验证时，可以先从更短 horizon 起步。

## 7. 第二阶段如何接到 `async_inference`

当第一阶段稳定后，再进入第二阶段。

推荐不是直接让用户手写 generic draccus 命令，而是新增：

- PI05 专用 server 启动脚本
- PI05 专用 client 启动脚本

并且把第一阶段已经验证过的语义映射过去。

第二阶段的目标不是重新设计运行时，而是把第一阶段已经成立的运行时语义后端化。

### 第二阶段可能要补的事情

#### `policy_server.py`

需要考虑：

- 支持 `strict=False` 的加载兼容
- 保留当前 processor 行为
- 若要支持 RTC，则需要接受并转发：
  - `prev_chunk_left_over`
  - `inference_delay`
  - `execution_horizon`

#### `robot_client.py`

需要考虑：

- 现有 aggregation queue 是否换成 RTC queue
- 何处维护 `action_index_before_inference`
- 如何定义 PI05 RTC 模式下的 chunk merge 语义

#### `configs.py`

需要考虑新增 PI05/RTC 专用参数，例如：

- rtc enabled
- execution horizon
- strict load behavior
- queue strategy

## 8. 实施风险

## 8.1 最大风险不是模型，而是运行时状态错位

当前最危险的问题不是“模型推不出来”，而是：

- delay 算错
- queue merge 算错
- leftover 取错
- actor thread 和 producer thread 状态不同步

这些问题不会立刻报错，但会体现在：

- 机器人动作不连续
- 偶发变向
- chunk 切换抖动
- 明明 FPS 没掉，但动作品质变差
- plain chunk 模式下 stale chunk 的过期前缀被错误执行

## 8.2 checkpoint 加载一致性风险

由于当前脚本依赖 `strict=False` 跑通：

- 新 runtime 必须先复刻当前加载语义

否则即便线程架构是对的，也可能在模型初始化阶段偏离当前基线。

## 8.3 早期切入 gRPC 的风险

如果第一阶段就进入 generic async client/server：

- 排障成本明显升高
- 配置面更大
- 状态面被拆散
- 更难证明问题到底来自：
  - 模型加载
  - processor
  - queue
  - RTC
  - 网络/序列化

所以不建议。

## 9. 验证标准

这个方案要成立，验证不能只看“机器人动了”。

至少要补上下面这些运行指标：

- chunk 推理耗时
- latency max / p95
- queue depth
- queue starvation 次数
- `inference_delay`
- `real_delay`
- leftover 长度
- actor loop 实际频率
- chunk 替换/追加次数

此外还必须单独观测启动期，不要把 warmup 和稳态混在一起：

- 首个 chunk ready 时间
- 首次推理 latency
- actor thread 首次开始执行时间
- 首次 starvation 是否发生

## 9.1 建议的第一阶段量化门槛

第一阶段不需要追求非常激进的性能，但要避免只靠主观观察判断“差不多可以”。

建议把下面这些值写进实现验收 checklist，作为初始门槛：

1. plain chunk 稳定运行时长：
   - 至少连续 60 秒无致命错误
2. 启动期：
   - 首个 chunk 未 ready 前 actor 不发送未定义动作
3. queue starvation：
   - 稳态阶段应接近 0
   - 若持续出现，则默认视为调参失败
4. actor loop 频率：
   - 稳态平均频率应落在目标 FPS 的 `90%~105%` 区间内
5. RTC 开启前门槛：
   - plain chunk 模式已满足上面条件
   - 且 queue/latency 日志已经稳定可解释

这些数值不是永久不变的标准，但应该作为第一轮落地的最低硬门槛。
- postprocessor 独立耗时
- stale-prefix trimming 次数

并且建议把验证分成三档：

### 档位 A：plain chunk

- 关闭 RTC
- 只验证 chunk queue 是否稳定
- 重点看 warmup、starvation、steady-state FPS

### 档位 B：RTC enabled

- 打开 RTC
- 看动作连续性是否优于 plain chunk
- 同时确认异常时可一键回退 plain chunk

### 档位 C：async phase 2

- 在第一阶段稳定后，再把相同语义移植到 `async_inference`

## 10. 最终方案定稿

本次最终定稿如下：

### 方案名称

**PI05 本地 Chunk Runtime 优先，RTC 分阶段接入，Async Server/Client 作为第二阶段后端化方案**

### 具体决策

1. 当前 `run_pi05_infer.py` 不作为最终实时方案继续叠改。
2. 在 `my_devs/` 下新增 PI05 实时运行时入口与 runtime 模块。
3. 第一阶段直接复用：
   - `ActionQueue`
   - `LatencyTracker`
   - `examples/rtc/eval_with_real_robot.py` 的线程职责
4. 第一阶段先上 plain chunk，不默认开 RTC。
5. RTC 在 chunk runner 稳定后，以开关形式接入。
6. `src/lerobot/async_inference/` 保留为第二阶段目标，不作为第一阶段替代入口。
7. 所有新方案必须保留当前 working script 的模型加载与 processor 行为一致性。

### 为什么这是最合适的方案

因为它同时满足四件事：

1. 不破坏当前已知能跑的 PI05 路径。
2. 不把第一版实现锁死在同步单步控制上。
3. 不过早把问题复杂化为 generic async server/client 迁移。
4. 最终仍然与仓库已有 async/RTC 能力收敛，而不是另起炉灶。

## 11. 本次脑暴工作收敛结果

本方案综合了以下三份独立脑暴报告：

- `reports/agent_01_runtime_architecture.md`
- `reports/agent_02_rtc_integration.md`
- `reports/agent_03_async_server_client.md`

并额外结合了仓库已有参考实现：

- `examples/rtc/eval_with_real_robot.py`
- `docs/source/rtc.mdx`
- `docs/source/async.mdx`

因此这不是纯理论方案，而是：

- 建立在当前工作脚本事实之上
- 建立在现有仓库模块能力之上
- 建立在实际可复用参考实现之上
