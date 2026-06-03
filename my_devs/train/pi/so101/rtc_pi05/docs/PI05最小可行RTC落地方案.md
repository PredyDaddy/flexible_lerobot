# PI05 最小可行 RTC 落地方案

本文档用于在 `my_devs/train/pi/so101/rtc_pi05` 目录下重新规划一条 **最小可行** 的 PI0.5 RTC 落地路线。

这份方案刻意不继承 `my_devs/pi05_engineering` 的实现思路，因为这个方案是极其失败的,而是回到仓库里真正可信的三类来源：

- 官方文档：`docs/source/rtc.mdx`
- 官方实机参考：`examples/rtc/eval_with_real_robot.py`
- 当前已经验证可运行的基线：`my_devs/train/pi/so101/run_pi05_infer.py`

目标不是一上来做“完整工程化版本”，而是先做一条 **最小、可验证、可回退** 的 RTC 路线。

---

## 1. 先说结论

### 1.1 RTC 对 PI0.5 是可以做的

从仓库源码看，PI0.5 原生支持 RTC：

- `src/lerobot/policies/pi05/configuration_pi05.py`
  - `PI05Config` 里有 `rtc_config: RTCConfig | None = None`
- `src/lerobot/policies/pi05/modeling_pi05.py`
  - `init_rtc_processor()` 会创建 `RTCProcessor`
  - `predict_action_chunk()` 会把 RTC 相关参数透传到底层采样循环

所以这不是“要不要魔改模型”的问题，而是“要不要正确地接入模型已经支持的 RTC 推理路径”的问题。

### 1.2 RTC 不是训练范式，而是推理增强

官方 `docs/source/rtc.mdx` 已经写得很清楚：

- RTC 是 `inference-time method`
- 适用于 flow-matching policy，例如 `pi0`、`pi05`、`smolvla`

这意味着：

- **训练主线仍然是正常训练 `pi05`**
- **RTC 是推理时开启的能力**

不要把“训练一个 PI0.5 policy”和“把 RTC 接到 PI0.5 推理里”混成一件事。

### 1.3 之前那套失败，不等于 RTC 本身失败

`my_devs/pi05_engineering` 的失败，说明的是：

- 那套 runtime 结构不稳定
- 它的 queue 语义、refill 时机、chunk 裁剪策略存在风险
- 它没有成为评估 RTC 的可信平台

它不能直接推出：

- “RTC 不适合你的任务”
- “PI0.5 + RTC 一定做不成”

真正合理的结论应该是：

- 需要重新搭一个 **最小可行 RTC 入口**
- 先复现基线，再开启 RTC

---

## 2. 必须先搞清楚的源码事实

这一节只讲仓库里可以直接确认的事实。

### 2.1 `select_action()` 不能用于 RTC

在 `src/lerobot/policies/pi05/modeling_pi05.py` 里，`select_action()` 明确断言：

```python
assert not self._rtc_enabled(), (
    "RTC is not supported for select_action, use it with predict_action_chunk"
)
```

这句话非常关键。

它说明：

- 你现在的 `run_pi05_infer.py` 这条同步单步路径，不能直接“加个 RTC 开关”就变成 RTC
- 如果要做 RTC，必须切到 `predict_action_chunk()`

### 2.2 RTC 的真正入口是 `predict_action_chunk()`

`PI05Policy.predict_action_chunk()` 会调用：

```python
actions = self.model.sample_actions(images, img_masks, tokens, masks, **kwargs)
```

而在底层采样循环里，如果 RTC 开启，会读取这些运行时参数：

- `inference_delay`
- `prev_chunk_left_over`
- `execution_horizon`

然后走：

```python
self.rtc_processor.denoise_step(...)
```

也就是说，RTC 不是“外面套一个平滑器”，而是直接进入模型的 denoising / flow matching 采样过程。

### 2.3 官方参考实现是“后台补货 + 前台执行”

官方 `examples/rtc/eval_with_real_robot.py` 的结构很清晰：

1. 一个线程持续执行动作
2. 一个线程在后台请求新的 action chunk
3. 两个线程之间通过 `ActionQueue` 交互
4. 机器人 I/O 通过 `RobotWrapper + Lock()` 做串行化

这是最值得对齐的参考结构。

### 2.4 `prev_chunk_left_over` 的语义非常严格

官方文档和 `ActionQueue` 的语义一致：

- `prev_chunk_left_over` 指的是“旧 chunk 里还没被执行完的那一段”

它不是：

- 某个历史缓存
- 任意保留下来的未来动作
- 和实际 rollout 队列不同步的另一段轨迹

如果这里语义错了，RTC guidance 就会对着错误目标做对齐，效果会非常差。

### 2.5 你当前有效的基线并不是“单步模型”

虽然 `my_devs/train/pi/so101/run_pi05_infer.py` 的主循环是一步一步发动作，但底层走的是：

- `predict_action(...)`
- `policy.select_action(...)`

而 `PI05Policy.select_action()` 在内部其实会：

1. 先调用 `predict_action_chunk()`
2. 再把 chunk 放进内部 action queue
3. 逐步消费这个 chunk

所以你现在能达到 60% 到 70% 成功率的基线，本质上已经是：

- **PI0.5 原生 chunk 推理**
- 只是它的 queue 是藏在 policy 内部

这件事很重要，因为它告诉我们：

- 新的 RTC runner 第一目标不是“证明 chunk 能不能工作”
- 而是“证明显式 chunk runtime 不会比当前隐式 chunk 基线更差”

---

## 3. 这次最小可行方案的总目标

`my_devs/train/pi/so101/rtc_pi05` 下面的新方案，目标应该非常克制。

### 3.1 不是要做什么

第一版不追求：

- 大而全的 runtime 框架
- 复杂的 metrics 系统
- 多层封装
- 自定义 queue 语义
- 一次性把 plain chunk、RTC、debug visualization、async server 全做完

### 3.2 第一版真正要做到什么

第一版只追求四件事：

1. 保持和 `run_pi05_infer.py` 一致的模型加载、processor 加载、robot 配置方式
2. 把推理入口从 `select_action()` 明确切到 `predict_action_chunk()`
3. 用官方 `ActionQueue` 和官方 `RTCConfig` 跑起来一个最小双线程闭环
4. 在 `RTC off` 和 `RTC on` 两种模式下都能稳定运行

### 3.3 第一版的验收标准

只有满足下面这些条件，才算第一版过关：

1. `RTC off` 时，显式 chunk runtime 的表现不能明显差于当前 `run_pi05_infer.py`
2. 没有机器人串口并发访问问题
3. 队列不会频繁见底
4. `RTC on` 后，行为变化是可解释的，而不是系统性崩盘

---

## 4. 这次明确不要再走的思路

这一节是为了避免重蹈覆辙。

### 4.1 不要在 MVP 阶段发明新的 queue 语义

最小可行版必须优先复用：

- `src/lerobot/policies/rtc/action_queue.py`

原因很简单：

- 它和官方 `rtc.mdx` 的语义是一致的
- 它至少代表了仓库作者对 RTC queue 的原意

MVP 阶段不建议自己重新定义：

- “剩余动作”的语义
- “过期前缀”的裁剪规则
- “plain 模式”和“RTC 模式”两套完全不同的自定义队列逻辑

### 4.2 不要在 MVP 阶段把执行 chunk 改得比模型原生 chunk 更短

你的 checkpoint 当前配置是：

- `chunk_size = 50`
- `n_action_steps = 50`
- `rtc_config = null`

第一版最稳的做法是：

- **先按 50 step chunk 跑**

不要一开始就把执行窗口改成：

- `actions_per_chunk = 8`

因为这会引入一个额外复杂度：

- 模型输出 chunk 长度
- 实际可执行 chunk 长度
- leftover 语义

三者会出现耦合问题。

而 RTC 恰恰最怕时间语义错位。

### 4.3 不要把“queue starvation”当成正常现象

如果 actor 经常拿不到动作，只能：

- hold last action
- skip send

那说明 runtime 还没有稳定。

这种情况下，不应该继续评估：

- 成功率
- RTC 是否有提升

因为系统已经先被 runtime 时序问题污染了。

### 4.4 不要在 MVP 阶段一开始就上 30 FPS

你当前同步基线能跑，不代表显式 chunk + 双线程 + RTC 在 30 FPS 下也一定稳。

MVP 阶段建议从更保守的控制频率开始，例如：

- `10 FPS`
- `12 FPS`
- `15 FPS`

先把结构跑稳，再往上推。

### 4.5 不要为了 RTC 先去重训模型

这一步最容易浪费时间。

如果现在的问题还没有分清是：

- policy 本体能力不足
- 还是 runtime 结构不稳定

那先重训只会把变量搅在一起。

RTC MVP 阶段建议：

- **固定同一个 checkpoint**
- 只比较推理 runtime 的变化

---

## 5. 建议的目录结构

推荐在 `my_devs/train/pi/so101/rtc_pi05` 下先保持一个非常小的结构。

```text
my_devs/train/pi/so101/rtc_pi05/
├── docs/
│   └── PI05最小可行RTC落地方案.md
├── run_pi05_rtc_infer.py
└── runtime/
    ├── __init__.py
    ├── config.py
    ├── robot_wrapper.py
    ├── producer.py
    ├── actor.py
    └── metrics.py
```

其中：

- `run_pi05_rtc_infer.py`
  - 唯一入口
- `runtime/config.py`
  - 只放最少量 runtime 参数
- `runtime/robot_wrapper.py`
  - 只做 `get_observation()` / `send_action()` 的共享锁串行化
- `runtime/producer.py`
  - 后台补货线程
- `runtime/actor.py`
  - 定时执行动作线程
- `runtime/metrics.py`
  - 只记录最必要指标，不做大而全系统

MVP 阶段不要拆得太碎。

---

## 6. 最小实现路线

建议严格分成三步做。

### 6.1 第一步：先做 Plain Chunk MVP，不开 RTC

第一步只做：

- 显式 `predict_action_chunk()`
- 显式 `ActionQueue`
- 双线程结构
- 机器人 I/O 锁保护

但是：

- `policy.config.rtc_config = None`
- 不传 `prev_chunk_left_over`
- 不传 `inference_delay`

或者即使调用 `predict_action_chunk()`，也只把它当普通 chunk policy 使用。

这一步的目标是验证：

- 显式 chunk runtime 本身不会破坏当前基线成功率

#### 这一步的关键约束

1. 加载 policy、preprocessor、postprocessor 的方式尽量照抄 `run_pi05_infer.py`
2. 机器人配置、camera 配置也尽量照抄 `run_pi05_infer.py`
3. 线程模型参考 `examples/rtc/eval_with_real_robot.py`
4. 机器人读写必须共用同一把锁
5. 先完整执行模型原生 chunk，不要先裁成 8 步

### 6.2 第二步：让 Plain Chunk 稳定，不饿队列

第二步还不开 RTC，只调时序。

要重点关注：

- 单次推理总延迟 `latency`
- `queue_depth`
- 队列是否见底
- actor 是否经常重复上一条动作

建议做法：

- 使用历史 latency 的最大值或高分位值估计补货阈值
- `queue_low_watermark` 不要设成 0
- 在 plain chunk 稳定前，不开启 RTC

这一步的目标是：

- 证明你的显式 chunk runtime 是健康的

### 6.3 第三步：再正式开启 RTC

在 plain chunk 已经稳定后，再做 RTC 接入。

#### 这一阶段必须做的事

1. 构造 `RTCConfig`

```python
rtc_cfg = RTCConfig(
    enabled=True,
    execution_horizon=8,
    max_guidance_weight=10.0,
    prefix_attention_schedule=RTCAttentionSchedule.EXP,
)
```

2. 把它挂到 policy 上

```python
policy.config.rtc_config = rtc_cfg
policy.init_rtc_processor()
```

3. producer 每次补货前拿到当前 leftover

```python
prev_chunk_left_over = action_queue.get_left_over()
```

4. 根据历史推理延迟估计 `inference_delay`

```python
inference_delay = ceil(latency_estimate / dt)
```

5. 生成新 chunk

```python
actions = policy.predict_action_chunk(
    batch,
    inference_delay=inference_delay,
    prev_chunk_left_over=prev_chunk_left_over,
)
```

6. 对本次真实推理时长再测一次 `real_delay`

```python
real_delay = ceil(real_latency / dt)
```

7. 用官方 `ActionQueue.merge(...)` 做接管

```python
action_queue.merge(original_actions, processed_actions, real_delay, action_index_before_inference)
```

这才是仓库语义上真正的 RTC。

---

## 7. 最小可行版本的推荐参数

下面给的是 **MVP 起步参数**，不是最终最优参数。

### 7.1 Plain Chunk MVP 参数

- 控制频率：`fps = 10` 或 `12`
- 模型 chunk：先保持 checkpoint 原生 `50`
- queue low watermark：先从 `20` 到 `30` 试
- `RTC = false`
- 不开额外动作裁剪
- 不开复杂安全限幅

### 7.2 RTC MVP 参数

在 plain chunk 稳定后，再尝试：

- `RTC = true`
- `execution_horizon = 8`
- `max_guidance_weight = 10.0`
- `prefix_attention_schedule = EXP`
- `fps = 10` 或 `12`

为什么先从 `8` 开始：

- 官方文档给的典型范围是 `8` 到 `12`
- 你的任务成功率本来就不是很高
- 更保守的 horizon 更容易定位问题

### 7.3 暂时不要用的高风险组合

MVP 阶段不建议直接上：

- `fps = 30`
- `queue_low_watermark = 0`
- `actions_per_chunk = 8`
- `execution_horizon = 10+` 同时又有较高推理延迟

这些参数组合非常容易把：

- 队列补货过晚
- 剩余动作语义
- RTC guidance 对齐位置

全部搅乱。

---

## 8. 为什么这条最小路线更可信

这条路线的核心思想是：

- **尽量不改变你已经验证过的东西**
- **只引入 RTC 必须引入的最少结构**

具体来说：

### 8.1 保留当前有效基线的三件事

我们尽量保留：

1. `run_pi05_infer.py` 的 policy 加载方式
2. `run_pi05_infer.py` 的 preprocessor / postprocessor 加载方式
3. `run_pi05_infer.py` 的 robot 与 camera 配置方式

### 8.2 只新增 RTC 必须新增的三件事

只新增：

1. 显式 chunk queue
2. producer / actor 双线程
3. `prev_chunk_left_over + inference_delay`

这样做的好处是：

- 如果效果变差，问题范围很小
- 如果效果变好，也更能说明真的是 RTC 在起作用

---

## 9. 训练应该怎么理解

这部分很重要，因为很容易想偏。

### 9.1 最小可行 RTC 不需要重新训练

MVP 阶段建议：

- 使用你当前已经有的 PI0.5 checkpoint
- 不改训练命令
- 不改模型结构

目的是先回答这个问题：

> 同一个 checkpoint，在正确的 RTC runtime 下，是否比当前基线更平滑、更稳、更少 chunk 边界抖动？

### 9.2 真正需要重训的情况

只有当你确认下面这种情况时，才应该把重点转回训练：

- 当前同步基线 `run_pi05_infer.py` 就已经因为语义理解、抓取定位、夹爪时机等问题表现很差
- 而不是因为动作 chunk 边界停顿、抖动、延迟导致差

换句话说：

- 如果问题是 policy 本体不会抓，RTC 没法救它
- 如果问题是 policy 会抓，但大模型推理太慢、chunk 边界不连续，RTC 才可能明显有帮助

### 9.3 训练主线仍然按普通 PI0.5 走

如果后面你真的要继续训练，仍然建议沿用：

- `src/lerobot/scripts/lerobot_train.py`
- `--policy.type=pi05`

并尽量保持和当前 checkpoint 一致的归一化口径。

你当前 checkpoint 配置显示：

- `normalization_mapping = {"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}`

所以后续训练如果继续和它对比，最好保持这个口径不变。

---

## 10. 第一版代码应该怎么写

下面给出一个最小伪代码骨架，作为 `run_pi05_rtc_infer.py` 的目标结构。

```python
load robot config
load policy config
load policy
load preprocessor/postprocessor

if enable_rtc:
    policy.config.rtc_config = rtc_cfg
    policy.init_rtc_processor()
else:
    policy.config.rtc_config = None

robot = make_robot_from_config(...)
robot = RobotWrapperWithLock(robot)
robot.connect()

action_queue = ActionQueue(policy.config.rtc_config or RTCConfig(enabled=False))
latency_tracker = LatencyTracker()

start producer thread:
    while running:
        if action_queue.qsize() <= queue_low_watermark:
            obs = robot.get_observation()
            batch = build_batch_like_run_pi05_infer(obs)

            prev_chunk_left_over = action_queue.get_left_over() if enable_rtc else None
            inference_delay = ceil(latency_tracker.max() / dt) if enable_rtc else 0

            t0 = now()
            actions = policy.predict_action_chunk(
                batch,
                inference_delay=inference_delay,
                prev_chunk_left_over=prev_chunk_left_over,
            )
            processed_actions = postprocessor(actions)
            real_delay = ceil((now() - t0) / dt)
            latency_tracker.add(now() - t0)

            action_queue.merge(
                actions.squeeze(0),
                processed_actions.squeeze(0),
                real_delay,
            )
        else:
            sleep(short_interval)

start actor thread:
    while running:
        action = action_queue.get()
        if action is not None:
            robot.send_action(...)
        sleep(dt)
```

这份骨架的重点是：

- 尽量贴近官方 `examples/rtc/eval_with_real_robot.py`
- 尽量贴近你当前 `run_pi05_infer.py` 的数据准备方式
- 不额外发明新的 runtime 抽象

---

## 11. 第一版开发顺序

建议按下面顺序写代码，而不是一次性写完。

### Step 1

先写一个只支持：

- `RTC off`
- `predict_action_chunk`
- 双线程
- `ActionQueue`

的版本。

### Step 2

确认这个版本：

- 能加载模型
- 能连接机器人
- 不会串口冲突
- 队列不会频繁饿死

### Step 3

在不改整体结构的前提下，再加：

- `--enable-rtc`
- `RTCConfig`
- `prev_chunk_left_over`
- `inference_delay`

### Step 4

最后再加日志和少量诊断指标，例如：

- 最新推理延迟
- 最新 `inference_delay`
- 最新 `real_delay`
- 当前 queue depth
- 是否出现 queue empty

---

## 12. 这份方案对应的下一步行动

如果按这份文档往下做，下一步最合理的是：

1. 在 `my_devs/train/pi/so101/rtc_pi05/` 下新建最小入口 `run_pi05_rtc_infer.py`
2. 第一版先只支持 plain chunk，不开 RTC
3. 在实机上把 plain chunk 跑到接近当前 `run_pi05_infer.py` 的成功率
4. 再开启 RTC

换句话说，这次不应该先问：

- “RTC 参数怎么调最优”

而应该先问：

- “我们有没有搭出一条可信的 RTC 评估入口”

---

## 13. 参考文件

本方案主要参考以下文件：

- `docs/source/rtc.mdx`
- `docs/source/pi05.mdx`
- `examples/rtc/eval_with_real_robot.py`
- `src/lerobot/policies/pi05/configuration_pi05.py`
- `src/lerobot/policies/pi05/modeling_pi05.py`
- `src/lerobot/policies/rtc/action_queue.py`
- `src/lerobot/policies/rtc/modeling_rtc.py`
- `my_devs/train/pi/so101/run_pi05_infer.py`
- `my_devs/docs/principle/rtc/RTC原理详解_从零理解Real-Time-Chunking.md`

---

## 14. 最后一句话

对你这个任务来说，**最小可行 RTC** 的关键不是“先把 RTC 打开”，而是：

> 先搭出一条不会破坏当前基线的显式 chunk runtime，再在这条可信 runtime 上开启 RTC。

只有这样，后面看到的好坏变化才是可信的。
