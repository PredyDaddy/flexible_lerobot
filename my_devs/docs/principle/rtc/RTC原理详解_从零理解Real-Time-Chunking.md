# RTC 原理详解：从零理解 Real-Time Chunking

## 1. 这份文档是写给谁的

这份文档是写给“几乎完全不了解 RTC 是什么”的读者的。

我会假设你可能：

- 不了解机器人策略模型；
- 不了解 action chunking；
- 不了解 diffusion / flow matching 这类生成式策略；
- 也不了解这个仓库里 `src/lerobot/policies/rtc/` 这一套代码到底在做什么。

这份文档的目标不是“快速扫一遍 API”，而是把下面几件事讲透：

1. RTC 到底是什么，不是什么。
2. 它为什么会出现，它试图解决什么真实问题。
3. 它背后的直觉和算法原理是什么。
4. 它在这个仓库里是怎么落地的。
5. 你在看 `pi0 / pi0.5 / smolvla` 这些策略代码时，RTC 具体插在什么位置。
6. 如果你以后要调 RTC 参数，应该如何理解这些参数。

## 2. 先说结论：这里的 RTC 不是“实时时钟”

很多工程里 `RTC` 常常代表 `Real-Time Clock`，也就是硬件上的“实时时钟芯片”。

但这个仓库里的 `RTC` 完全不是这个意思。

这里的 `RTC` 指的是：

`Real-Time Chunking`

也就是：

“实时地生成并平滑衔接动作块（action chunks）的一种推理时方法”。

它不是一个新的机器人。
它不是一个新的 policy 类型。
它不是训练框架。
它也不是底层控制器。

它本质上是一个“推理时增强机制”，专门给这类模型使用：

- 一次会预测一整段未来动作，而不是只预测下一步；
- 模型推理比较慢；
- 模型通常基于 flow matching 或相近的生成式动作建模方法；
- 比如这个仓库里的 `Pi0`、`Pi0.5`、`SmolVLA`。

仓库里官方文档也明确说了这一点：

- `docs/source/rtc.mdx`
- `src/lerobot/policies/rtc/README.md`

## 3. 为什么会需要 RTC

要理解 RTC，先要理解一个更基础的问题：

机器人控制是“按时间推进”的，但大模型推理不是免费的。

### 3.1 机器人控制循环是什么

假设你的机器人以 10Hz 运行。

这意味着每 0.1 秒，系统都要给机器人一个新的动作。

比如：

- 第 1 个时刻给 1 个关节目标；
- 0.1 秒后给第 2 个；
- 再过 0.1 秒给第 3 个；
- 如此不断循环。

如果在该给动作的时候，你还没算出来，那么机器人就会：

- 停一下；
- 维持上一帧动作；
- 或者出现顿挫和迟滞。

对真实机器人来说，这种“没动作可发”的情况非常常见，也非常致命。

### 3.2 传统单步策略 vs 动作块策略

最朴素的策略是：

- 输入当前观察；
- 输出下一步动作；
- 执行；
- 再循环。

但近些年很多机器人大模型不再一次只出 1 步，而是一次输出一整个未来动作序列。

例如：

- 未来 10 步；
- 未来 16 步；
- 未来 50 步。

这就是 `action chunking`，也可以理解为：

“一次不是给机器人 1 个动作，而是给它一段动作计划。”

这么做有两个好处：

1. 模型可以在时间上做更强的规划。
2. 执行时可以先吃着这一段动作，不必每一步都重新推理。

这也是 ACT 这类工作里强调过的思路。`Action Chunking with Transformers` 本身就把“分块输出动作”作为核心设计之一。

### 3.3 光有 action chunking 还不够

问题来了。

如果模型一次输出 50 步动作，但是它推这 50 步花了 0.4 秒，而你的机器人 10Hz 执行，那么在这 0.4 秒里，机器人已经执行了：

`0.4 / 0.1 = 4` 步

也就是说：

- 你刚算出来的这个“新 chunk”，它前 4 步其实已经过时了；
- 机器人在你算的时候，已经按旧 chunk 往前走了。

如果你此时粗暴地把新 chunk 从第 1 步开始接上，就会出现：

- 边界处突然拐一下；
- 新 chunk 开头和机器人当前真实动作趋势不连续；
- 视觉上就是“抖”“顿”“抽一下”。

这就是 RTC 要解决的核心问题。

## 4. RTC 解决的到底是什么问题

一句话：

RTC 解决的是“当机器人正在执行旧动作块时，后台生成出来的新动作块，怎样和旧动作块平滑接上”的问题。

这个问题里有两个子问题：

1. **延迟问题**
   在生成新 chunk 的过程中，机器人没有停下，它还在执行旧 chunk。

2. **边界平滑问题**
   新 chunk 生成出来以后，如何让它的开头和机器人当前正在执行的轨迹顺滑连接，而不是硬拼。

注意，这两个子问题不是一回事。

- “后台生成，不要让机器人干等着”解决的是延迟问题。
- “新旧 chunk 边界要平滑”解决的是过渡问题。

RTC 主要针对的是第二件事，但它通常和异步执行一起使用。

## 5. 先用一个生活化比喻理解 RTC

想象你在开车，导航每隔一段时间会重新规划一小段未来路线。

如果导航特别慢，会出现这样一种情况：

- 你已经沿着旧路线开出了一段；
- 新路线这时才算出来。

如果新路线完全不考虑你刚刚已经开的那段轨迹，它可能会突然告诉你：

- 立刻往另一个方向打；
- 或者突然修正到另一条轨迹上。

这就会非常生硬。

RTC 做的事情就像：

- 新路线在生成时，会参考“你现在已经开到了哪、刚刚的行驶趋势是什么”；
- 让新路线开头先顺着你当前的方向接一下；
- 再慢慢转向新的、更优的路线。

所以 RTC 不是“不让模型改主意”，而是：

“允许模型改主意，但不能在 chunk 边界处猛地改。”

## 6. 理解 RTC 之前，你必须先懂 action chunking

### 6.1 什么叫 chunk

假设策略每次输出 10 个动作：

```text
[a1, a2, a3, a4, a5, a6, a7, a8, a9, a10]
```

这一整段就叫一个 `chunk`。

机器人执行时，并不是把 10 个动作一口气同时发出去，而是按时间顺序一个一个发：

```text
t0 -> a1
t1 -> a2
t2 -> a3
...
t9 -> a10
```

### 6.2 为什么模型喜欢输出 chunk

因为很多高维机器人任务不是单步反应，而是需要时间上的连续结构。

例如：

- 手臂先伸过去；
- 再闭合夹爪；
- 再抬起来；
- 再移动到目标位置；
- 再放下。

如果一次只出一步，模型每一步都重新想，很可能会缺乏稳定的中期规划。

而一次输出一整个 chunk，模型可以在短时间窗内保持一致性。

### 6.3 chunk 带来的新问题

chunk 的好处是“看得更远”，但代价是：

- 你每次推理更重；
- 推理时间可能超过控制步长；
- 新旧 chunk 之间会出现“衔接问题”。

RTC 就是在 action chunking 已经存在的前提下，专门用来修这个衔接问题的。

## 7. RTC 的核心直觉：把新 chunk 的开头当成“补全问题”

仓库文档和原始研究都把 RTC 解释成一种 `inpainting` 风格的问题。

这句话非常关键。

什么叫 inpainting？

最直白的理解就是：

“我不是从完全空白开始自由生成，而是有一段前缀必须和已有内容自然接起来。”

在 RTC 里：

- 已有内容 = 上一个 chunk 尚未执行完、或者已经开始执行的那条动作趋势；
- 需要补全的内容 = 新 chunk 的前缀；
- 目标 = 让新 chunk 的开头和旧 chunk 的尾巴平滑衔接。

所以 RTC 的重点不是“把整个新 chunk 固定成旧 chunk”。

它只是对 **新 chunk 的前半部分** 加一个平滑约束。

后半部分仍然允许模型根据最新观测进行调整。

## 8. 用时间线看 RTC 在做什么

假设：

- 控制频率 `fps = 10Hz`
- 每步时长 `dt = 0.1s`
- chunk 长度 `chunk_size = 10`
- 推理延迟 `latency = 0.35s`

那么：

`inference_delay = ceil(0.35 / 0.1) = 4`

也就是在你生成新 chunk 的这段时间里，机器人已经继续执行了 4 步旧动作。

### 8.1 没有 RTC 时

旧 chunk：

```text
old = [o1, o2, o3, o4, o5, o6, o7, o8, o9, o10]
```

机器人在推理新 chunk 的时候继续执行，已经执行到了：

```text
执行完 o1, o2, o3, o4
```

这时新 chunk 才出来：

```text
new = [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10]
```

如果直接从 `n1` 开始接：

- `n1 ~ n4` 本质上已经过时；
- `n5` 开始也未必和当前动作趋势平滑；
- 于是边界就可能出现跳变。

### 8.2 有 RTC 时

RTC 会做两件事：

1. **在生成新 chunk 的时候**，让新 chunk 的前缀朝着旧 chunk 的剩余轨迹靠拢。
2. **在入队执行的时候**，跳过因为推理延迟已经来不及执行的前 `real_delay` 步。

也就是说最终入队的不是完整 `new`，而更像：

```text
skip(new[1:4])
execute(new[5:10])
```

但更关键的是，`new[5:10]` 不是“裸生成”的，它在生成过程中已经被 RTC 引导过了。

所以即使从中间接，也会更平滑。

## 9. RTC 的算法层直觉

这一节开始稍微技术一点，但我会尽量不用过于抽象的数学符号。

### 9.1 生成式动作策略在做什么

对于 `Pi0 / Pi0.5 / SmolVLA` 这类 flow-matching 风格策略，可以粗略理解成：

- 模型先从一个噪声动作序列开始；
- 然后经过多步去噪 / 流动更新；
- 最后得到一个干净的动作 chunk。

仓库里 `pi0 / pi0.5 / smolvla` 的采样过程都能看到类似这样的结构：

- 初始化 `x_t = noise`
- 多步循环
- 每一步算一个 denoise step / velocity `v_t`
- 再更新 `x_t`

在 `PI05` 里能直接看到这条链路：

- `src/lerobot/policies/pi05/modeling_pi05.py`

### 9.2 RTC 没有重写整个模型，它只是改“每一步去噪的方向”

RTC 最关键的工程思想是：

它不要求你重训一个新模型。

它是在 **推理时** 包住原来的 denoiser，然后给原有去噪方向加一个修正项。

仓库里的实现就在：

- `src/lerobot/policies/rtc/modeling_rtc.py`

核心接口是：

`RTCProcessor.denoise_step(...)`

它接收：

- 当前的 `x_t`
- 上一个 chunk 的 leftover：`prev_chunk_left_over`
- 当前估计的推理延迟：`inference_delay`
- 原始 denoise step 的调用函数

然后它会：

1. 先跑原始 denoiser，得到基础方向 `v_t`
2. 计算当前“干净动作预测” `x1_t`
3. 比较 `x1_t` 和 `prev_chunk_left_over` 的差异
4. 基于差异求一个 correction
5. 用一个 guidance weight 把 correction 加回去，修正 `v_t`

最终返回“更适合 chunk 平滑衔接”的 denoise 方向。

### 9.3 为什么叫 guidance

因为 RTC 并不是把模型的输出硬改成旧 chunk。

它只是给模型一个额外的优化方向：

“你生成的新 chunk 前缀，最好靠近上一段 chunk 剩余的轨迹。”

所以这个东西本质上是一个“引导项”，不是硬约束。

### 9.4 为什么只改前缀，不改整个 chunk

因为你既想要平滑，又想要反应性。

如果把整个新 chunk 都强行贴近旧 chunk，那么模型就无法根据新观测快速修正策略。

所以 RTC 会使用一个前缀权重 mask。

其思想是：

- chunk 开头的若干步：权重大，强约束
- 再往后：权重逐渐衰减
- 到更远的未来：权重接近 0，让模型自由发挥

这就是参数 `execution_horizon` 和 `prefix_attention_schedule` 的意义。

## 10. 代码里 RTC 的核心公式应该怎样读

看 `src/lerobot/policies/rtc/modeling_rtc.py`，可以把它读成下面这个逻辑。

### 10.1 第一步：得到基础去噪方向

代码里先调用原始 denoiser：

```python
v_t = original_denoise_step_partial(x_t)
```

这表示：

“如果完全不用 RTC，模型本来会朝哪个方向继续去噪。”

### 10.2 第二步：算当前的干净动作预测

代码里有：

```python
x1_t = x_t - time * v_t
```

注意这里的正负号和具体时间参数，是由这个仓库的时间方向实现决定的。

你可以先不要执着于符号，抓住直觉就够了：

`x1_t` 表示“当前这一步去噪后，对最终动作 chunk 的一个预测”。

### 10.3 第三步：看它和旧 chunk leftover 差多远

代码里：

```python
err = (prev_chunk_left_over - x1_t) * weights
```

这句话非常关键。

意思是：

- 如果新 chunk 前缀和旧 chunk leftover 差得很大，误差就大；
- 但不是每个时间步都同样重要，要乘上 `weights`；
- 越靠前的时间步通常越重要。

### 10.4 第四步：把误差变成 correction

代码通过 autograd 求导，得到一个 correction：

```python
correction = torch.autograd.grad(...)
```

它的意义是：

“如果想让新 chunk 更靠近旧 chunk leftover，当前 denoise 方向应该往哪里改。”

### 10.5 第五步：用 guidance weight 控制修正强度

代码又算了一个 `guidance_weight`，并用 `max_guidance_weight` 截断。

最终输出：

```python
result = v_t - guidance_weight * correction
```

这可以直接读成：

- `v_t`：原模型本来想走的方向
- `correction`：为了更平滑地接上旧 chunk，应该补的方向
- `guidance_weight`：补多少

如果 `guidance_weight` 太小：

- 修正不明显；
- chunk 边界可能还是生硬。

如果太大：

- 新 chunk 会过于执着于延续旧轨迹；
- 导致不够 reactive，不容易快速转向新策略。

这就是平滑性和反应性之间的经典权衡。

## 11. `prefix_attention_schedule` 到底在做什么

这个参数名字里虽然有 `attention`，但在这里更应该把它理解成：

“前缀约束权重随时间怎么分配。”

仓库支持 4 种 schedule：

- `ZEROS`
- `ONES`
- `LINEAR`
- `EXP`

定义在：

- `src/lerobot/configs/types.py`

具体实现逻辑在：

- `src/lerobot/policies/rtc/modeling_rtc.py`

### 11.1 `ZEROS`

可以粗略理解成：

- 前 `inference_delay` 步保持 1
- 后面快速归零

它相对更“硬边界”。

### 11.2 `ONES`

可以理解成：

- 在 `execution_horizon` 范围内全都保持强约束

这种最强硬，也最容易牺牲灵活性。

### 11.3 `LINEAR`

最容易理解：

- 前面强；
- 后面线性减弱。

### 11.4 `EXP`

前期更强，后期衰减更快。

仓库文档和官方示例里都比较推荐从 `EXP` 开始尝试，因为它通常更适合作为“平滑开头、但后面放手”的折中方案。

## 12. `execution_horizon` 到底是什么意思

这是最关键的 RTC 参数之一。

可以把它理解成：

“我希望新 chunk 有多少步需要和旧 chunk 保持连续性。”

如果 `execution_horizon = 10`，那么直觉上就是：

- 新 chunk 最前面的 10 步，会更强地被要求和旧 chunk leftover 接得上；
- 10 步之后，约束会快速衰减或结束。

### 12.1 它大一点会怎样

优点：

- 过渡更平滑；
- 看起来更“丝滑”；
- 边界跳变更少。

缺点：

- 模型会更恋旧；
- 对新观测的响应可能更慢。

### 12.2 它小一点会怎样

优点：

- 更灵活；
- 更容易快速变向。

缺点：

- 新旧 chunk 的边界更容易显得不连续。

所以 `execution_horizon` 本质上是在控制：

“平滑优先，还是反应优先。”

## 13. `inference_delay` 为什么是 RTC 成败的关键

很多人第一次看 RTC，只盯着 guidance，却忽略了 `inference_delay`。

其实 `inference_delay` 非常关键。

因为 RTC 要知道：

“当我把新 chunk 算出来时，旧 chunk 已经往前执行了多少步？”

这个步数估不准，后面全都会偏。

### 13.1 代码里怎么估

在 `examples/rtc/eval_with_real_robot.py` 里：

- 先记录历史推理延迟；
- 再用 `ceil(latency / dt)` 换算成离散步数；
- 这个值就是 `inference_delay`。

其中：

- `dt = 1 / fps`
- `latency` 来自 `LatencyTracker`

### 13.2 为什么通常取 ceil

因为你宁可把延迟估得稍微保守一点，也不要低估。

低估意味着：

- 你以为旧 chunk 只执行了 3 步；
- 实际上已经执行了 4 步；
- 那么新 chunk 的对齐位置就错了。

### 13.3 为什么建议看 max 或 p95，而不是平均值

平均值会掩盖长尾。

真实系统里最危险的往往不是平均慢，而是偶尔非常慢。

一次长尾就可能把队列吃空，或者让 chunk 边界严重错位。

所以工程上更稳健的做法是：

- 用最近窗口的 max
- 或者 p95

而不是简单平均。

## 14. RTC 不只是算法，还包含运行时队列语义

这一点在很多讲解里容易被忽略。

很多人以为 RTC 只是一个 guidance 公式，但在真实系统里，仅有 guidance 还不够。

你还必须处理：

- 新旧 chunk 怎么交接；
- 推理晚了多少步；
- 当前队列里哪些动作已经过期；
- 哪些 leftover 要留给下一次 RTC 使用。

这部分在这个仓库里主要由：

- `src/lerobot/policies/rtc/action_queue.py`

负责。

## 15. `ActionQueue` 在 RTC 模式下到底干了什么

`ActionQueue` 维护两套队列：

1. `original_queue`
2. `queue`

### 15.1 为什么要两套

因为：

- `original_queue` 是原始动作，用来给 RTC 计算 leftover；
- `queue` 是经过后处理后真正要发给机器人的动作。

这两者在一些系统里可能一样，但设计上它们不是一回事。

### 15.2 `get_left_over()` 为什么重要

这个方法会返回当前 chunk 里还没被消费掉的那一段原始动作。

这就是下一个 chunk 在做 RTC guidance 时的参考轨迹。

也就是说：

- RTC 不是凭空参考“过去”
- 它参考的是“当前队列里尚未执行完的旧动作块剩余部分”

### 15.3 RTC 开启时为什么不是 append，而是 replace

这是整个运行时设计中最重要的一点之一。

在 RTC enabled 模式下，`merge()` 会走 replace 语义：

- 把新 chunk 替换成当前队列内容；
- 但会先丢掉前 `real_delay` 步。

原因很简单：

当新 chunk 终于生成出来时，它前面那几步已经晚了，没必要再执行。

所以 RTC 模式下的语义是：

“新 chunk 接管未来，但已经过期的那一小段直接剪掉。”

### 15.4 RTC disabled 时为什么是 append

如果 RTC 关闭，`ActionQueue` 就只是一个更普通的动作缓冲区：

- 把新动作块接到后面；
- 删掉已经消费过的部分；
- 维持连续输出。

所以同一个 `ActionQueue` 在仓库里承担了两种不同语义：

- plain chunk queue
- RTC queue

## 16. RTC 在这个仓库里的代码结构

如果你以后要读源码，可以按下面顺序看。

### 16.1 配置层

`src/lerobot/policies/rtc/configuration_rtc.py`

这里定义：

- `enabled`
- `execution_horizon`
- `max_guidance_weight`
- `prefix_attention_schedule`
- `debug`
- `debug_maxlen`

### 16.2 算法层

`src/lerobot/policies/rtc/modeling_rtc.py`

这里定义：

- `RTCProcessor`
- prefix weights 的构造
- guidance correction 的计算
- debug tracker 的记录

### 16.3 队列层

`src/lerobot/policies/rtc/action_queue.py`

这里定义：

- 动作队列的 get / merge / leftover
- RTC enabled 和 disabled 两种 merge 语义

### 16.4 延迟统计层

`src/lerobot/policies/rtc/latency_tracker.py`

这里定义：

- 最近延迟窗口
- `max()`
- `percentile()`
- `p95()`

### 16.5 示例入口

`examples/rtc/eval_with_real_robot.py`

这里是最接近“真实运行时”的参考实现。

## 17. 以 `Pi0.5` 为例，RTC 是怎么接进模型采样循环的

这是理解仓库实现最重要的一节。

### 17.1 `PI05Config` 里有一个 `rtc_config`

在：

- `src/lerobot/policies/pi05/configuration_pi05.py`

里可以看到：

```python
rtc_config: RTCConfig | None = None
```

也就是说：

- `pi0.5` 原生支持挂 RTC 配置；
- 如果没配，默认不启用。

### 17.2 `PI05Policy` 初始化时会尝试构造 `RTCProcessor`

在：

- `src/lerobot/policies/pi05/modeling_pi05.py`

里，`init_rtc_processor()` 会在 `config.rtc_config is not None` 时创建 `RTCProcessor`。

但注意：

- “存在 rtc_processor”不等于“RTC guidance 真的开启了”
- 真正决定是否走 RTC 路径的是：

```python
self.config.rtc_config is not None and self.config.rtc_config.enabled
```

### 17.3 为什么 `select_action()` 明确禁止 RTC

因为 RTC 的前提是：

- 你得有一个完整 chunk
- 你得知道旧 chunk leftover
- 你得在 chunk 级别做前缀对齐

而 `select_action()` 只是单步接口。

单步接口没有“未来 chunk 前缀”这个概念，所以仓库直接在 `select_action()` 里断言：

```python
RTC is not supported for select_action, use it with predict_action_chunk
```

这个设计非常合理，也非常重要。

如果你看见有人说“我在单步 select_action 上开 RTC”，那基本上是概念混了。

### 17.4 `predict_action_chunk()` 才是 RTC 的真正入口

`predict_action_chunk()` 会把额外的 RTC 相关参数透传给底层 `sample_actions(...)`：

- `inference_delay`
- `prev_chunk_left_over`
- `execution_horizon`

然后在底层采样循环里，如果 `_rtc_enabled()` 为真，就走：

```python
self.rtc_processor.denoise_step(...)
```

否则就走普通 denoise。

这说明 RTC 的插入点不是策略外部套壳，而是直接插在模型采样循环内部。

## 18. 用一条完整调用链理解 RTC

下面把整个链条从头到尾串起来。

### 18.1 Actor 线程持续消费动作

机器人执行线程会：

1. 定时从 `ActionQueue` 里取一个动作；
2. 发给机器人；
3. 重复。

### 18.2 Get-actions 线程在后台补货

后台线程会不断检查：

- 如果队列快空了，就开始推一个新 chunk。

### 18.3 推新 chunk 之前，先拿旧 chunk leftover

后台线程先做：

```python
prev_actions = action_queue.get_left_over()
```

这一步拿到的是：

“旧 chunk 目前还没执行完的那一段”

### 18.4 再估计推理延迟

后台线程会根据历史 latency 算出：

```python
inference_delay = ceil(inference_latency / dt)
```

### 18.5 再把这两个量喂给 policy

然后调用：

```python
policy.predict_action_chunk(
    obs,
    inference_delay=inference_delay,
    prev_chunk_left_over=prev_actions,
)
```

此时 RTC guidance 就真正生效了。

### 18.6 新 chunk 生成出来后，再测一次真实延迟

因为预测用的是历史估计值，但 merge 阶段最好用本次真实延迟。

所以代码里还会测：

```python
new_delay = ceil(new_latency / dt)
```

### 18.7 最后用 RTC queue 语义接管未来动作

再调用：

```python
action_queue.merge(original_actions, postprocessed_actions, new_delay, ...)
```

如果 RTC enabled：

- 队列会被替换；
- 过期前缀会被跳过；
- 新 chunk 从“仍然来得及执行”的位置开始接管。

这就是一条完整的 RTC 运行时闭环。

## 19. 为什么 RTC 不是“简单拼接 chunk”

这是最常见误解之一。

很多人第一次听 RTC，会以为它只是：

- “把旧 chunk 剩余部分和新 chunk 拼起来”

这理解是不够的。

RTC 至少包含两层：

1. **采样时 guidance**
   在新 chunk 生成阶段，就让它更接近旧 chunk 的剩余轨迹。

2. **队列时语义处理**
   生成完之后，再根据实际延迟剪掉过期前缀，并替换未来动作队列。

如果你只做第 2 层，不做第 1 层，那只是一个更聪明的 queue，不是真正的 RTC。

如果你只做第 1 层，不做第 2 层，运行时还是会因为过期动作和队列对接问题出错。

真正可用的 RTC，必须两层一起成立。

## 20. RTC 和 async inference 的区别

这是另一个非常容易混淆的点。

### 20.1 async inference 解决什么

异步推理主要解决：

“不要让机器人在等模型推理结果时空转。”

也就是把：

- 动作执行
- 模型推理

拆成并行流程。

### 20.2 RTC 解决什么

RTC 主要解决：

“即使已经并行了，新旧 chunk 的边界仍然可能不平滑，怎么让它顺起来。”

### 20.3 两者的关系

它们不是互斥关系，而是互补关系。

可以这么理解：

- async inference：解决“有没有动作”的问题
- RTC：解决“动作接得顺不顺”的问题

所以工程上最好的状态通常是：

- 异步补货
- RTC 平滑衔接

仓库里的 `examples/rtc/eval_with_real_robot.py` 本质上就是这两件事一起做。

## 21. RTC 和 temporal ensembling 的区别

如果你之前接触过 ACT、Diffusion Policy 或一些机器人推理工程，还可能见过 `temporal ensembling`。

它和 RTC 不是一回事。

### 21.1 temporal ensembling 是什么

temporal ensembling 更像：

- 每一时刻重新预测一段动作；
- 然后对多个时间窗里对同一步动作的预测做加权平均。

它的重点是：

- 多个预测互相投票；
- 减少单次预测的噪声。

### 21.2 RTC 是什么

RTC 的重点不是“多次平均同一步”，而是：

- 在新 chunk 生成时，让其前缀朝着旧 chunk leftover 靠拢；
- 让 chunk 边界连续。

### 21.3 一个非常粗糙但够用的对比

- temporal ensembling：更像“多个人投票后取平均”
- RTC：更像“新计划必须和正在执行的旧计划顺滑交接”

## 22. RTC 为什么特别适合 flow-matching policy

仓库文档和源码都强调了一个前提：

RTC 主要是给 `flow-matching based policies` 用的。

原因在于：

- 这类模型本身就是多步迭代地产生动作 chunk；
- 去噪 / 流动过程天然给了你一个“在推理时插 guidance”的位置；
- RTC 就是往这个位置加一项修正。

对于没有这种迭代采样结构的模型，RTC 并不是天然可插的。

也正因为如此：

- `pi0`
- `pi0.5`
- `smolvla`

都可以比较自然地接 RTC；

而别的 policy 是否能接，取决于它有没有类似的 chunk 级生成采样路径。

## 23. 调参数时应该怎么理解每个 knob

这一节尽量只讲“原理上的调参含义”，不讲过细经验数值。

### 23.1 `enabled`

这是总开关。

需要注意：

- `rtc_config` 对象存在，不等于 `enabled=True`
- 有时系统会保留 config 但关闭 guidance，用于 debug 或统一接口

### 23.2 `execution_horizon`

决定需要和旧 chunk 保持连续的时间长度。

可以把它理解成“平滑衔接带宽”。

### 23.3 `max_guidance_weight`

决定“为了平滑而修正原始采样方向”的最大力度。

它越大：

- 越强调连续性；
- 越可能牺牲反应性。

### 23.4 `prefix_attention_schedule`

决定这段连续性约束在时间上怎么分布。

它不是问“要不要连续”，而是问：

“连续性要求在前缀不同位置上，应该有多强。”

### 23.5 `inference_delay`

决定 RTC 认为旧 chunk 已经被执行掉多少步。

这个值偏差太大，会直接破坏对齐位置。

### 23.6 `debug`

打开后可以记录更多中间量，帮助你看：

- `x_t`
- `v_t`
- `x1_t`
- `correction`
- `weights`
- `guidance_weight`

这对于理解 RTC 非常有帮助，但会有额外开销。

## 24. 什么情况下 RTC 特别有价值

RTC 特别适合下面几类场景：

### 24.1 模型比较大，推理延迟明显

例如：

- 视觉语言动作模型
- 多相机输入
- flow-matching 迭代采样步数较多

### 24.2 policy 一次输出 chunk，而不是单步

RTC 的基础前提就是 chunk。

没有 chunk，就没有“前缀衔接”这个问题。

### 24.3 真实机器人对动作连续性很敏感

例如：

- 抓取前对位
- 倒液体
- 插孔
- 门把手旋转
- 柔性接触任务

这些任务一旦边界抽一下，成功率就可能明显下降。

## 25. 什么情况下 RTC 的收益不明显

也要反过来看，RTC 并不是任何地方都值。

### 25.1 模型本来就很快

如果推理快到几乎不形成实际 `inference_delay`，那么 RTC 的收益会变小。

### 25.2 任务本身对动作连续性不敏感

如果任务非常粗糙、离散、容错高，chunk 边界的小抖动可能无所谓。

### 25.3 根本不是 chunk policy

如果策略是单步输出，RTC 不成立。

### 25.4 系统瓶颈根本不在 policy 端

比如：

- 机器人驱动本身慢；
- 相机采集卡住；
- 总线阻塞；
- 控制线程调度有问题；

那你先该修的是系统时序，不是 RTC。

## 26. 这个仓库里关于 RTC 的几个关键事实

为了避免误解，我把几个最重要的结论单独列出来。

### 26.1 RTC 不是独立 policy

它是 `pi0 / pi0.5 / smolvla` 等 policy 的推理增强模块。

### 26.2 RTC 只在 chunk 推理路径上工作

也就是：

- `predict_action_chunk()` 可以
- `select_action()` 不可以

### 26.3 RTC 依赖上一段 chunk 的 leftover

没有 `prev_chunk_left_over`，RTC 的核心指导信息就不存在。

### 26.4 RTC 依赖对推理延迟的估计

没有 `inference_delay`，新 chunk 就不知道该和旧 chunk 的哪一段对齐。

### 26.5 RTC 不只是 guidance，还依赖 queue merge 语义

这点很关键。

### 26.6 这个仓库已经提供了完整参考实现

最值得看的入口就是：

- `examples/rtc/eval_with_real_robot.py`

## 27. 如果你要自己向别人解释 RTC，可以只记住这段话

你可以这样解释：

> 许多机器人大模型一次不是输出 1 个动作，而是输出未来一段动作 chunk。  
> 但模型推理常常有延迟，所以当新 chunk 生成出来时，机器人已经执行了旧 chunk 的一部分。  
> 如果直接把新 chunk 硬接上，边界就会抖。  
> RTC 的做法是在生成新 chunk 时，把它的前缀当成“补全问题”，让它朝着旧 chunk 剩余轨迹靠拢；  
> 然后在运行时再把因为延迟已经过期的前几步剪掉。  
> 这样机器人一边运行，一边补未来动作，而且新旧 chunk 的过渡更平滑。  

如果对方能理解上面这段话，他就已经抓住 RTC 的 80% 了。

## 28. 你在读代码时的推荐顺序

如果你准备继续深入源码，推荐按这个顺序读：

1. `docs/source/rtc.mdx`
   先建立大图景。
2. `examples/rtc/eval_with_real_robot.py`
   理解真实运行时怎么调 RTC。
3. `src/lerobot/policies/rtc/configuration_rtc.py`
   先认识参数。
4. `src/lerobot/policies/rtc/action_queue.py`
   理解队列语义。
5. `src/lerobot/policies/rtc/latency_tracker.py`
   理解 delay 怎么估。
6. `src/lerobot/policies/rtc/modeling_rtc.py`
   再进入 guidance 算法。
7. `src/lerobot/policies/pi05/modeling_pi05.py`
   最后看 RTC 是怎么插进具体 policy 采样循环的。

这样读，理解负担最小。

## 29. 和网上资料对照后，我认为最值得保留的三个认知

结合原始研究、实现仓库、技术博客和本仓库代码，我认为下面三个认知最重要。

### 29.1 RTC 的本质不是“让模型更快”

它的本质是：

**让慢模型在 chunk 边界处看起来仍然连续。**

### 29.2 RTC 的本质不是“强行复制旧轨迹”

它的本质是：

**只对新 chunk 的前缀施加连续性引导，后面仍允许策略根据新观察改变计划。**

### 29.3 真正能跑起来的 RTC 必须同时考虑算法和运行时

也就是：

- guidance
- latency
- leftover
- queue merge

缺一块都不完整。

## 30. 参考资料

下面这些资料是我在整理本说明时实际参考过的，优先保留了一手资料和官方实现。

### 30.1 原始研究与官方实现

1. Physical Intelligence 研究页  
   `https://www.physicalintelligence.company/research/real_time_chunking`

2. 论文：`Real-Time Execution of Action Chunking Flow Policies`  
   `https://arxiv.org/abs/2506.07339`

3. Physical Intelligence 的参考实现仓库  
   `https://github.com/Physical-Intelligence/real-time-chunking-kinetix`

4. OpenPI 仓库  
   `https://github.com/Physical-Intelligence/openpi`

### 30.2 背景资料

5. ACT 论文：`Action Chunking with Transformers`  
   `https://arxiv.org/abs/2304.13705`

6. LeRobot 官方 RTC 文档  
   `https://huggingface.co/docs/lerobot/main/en/rtc`

### 30.3 技术讲解

7. Alexander Soare 的技术博客：`Smooth-As-Butter Robot Policies`  
   `https://alexander-soare.github.io/robotics/2025/08/05/smooth-as-butter-robot-policies.html`

## 31. 本仓库对应源码位置索引

为了方便以后继续深入，这里把最关键的仓库位置列一下。

- `docs/source/rtc.mdx`
- `src/lerobot/policies/rtc/README.md`
- `src/lerobot/policies/rtc/configuration_rtc.py`
- `src/lerobot/policies/rtc/modeling_rtc.py`
- `src/lerobot/policies/rtc/action_queue.py`
- `src/lerobot/policies/rtc/latency_tracker.py`
- `src/lerobot/policies/pi0/modeling_pi0.py`
- `src/lerobot/policies/pi05/modeling_pi05.py`
- `src/lerobot/policies/smolvla/modeling_smolvla.py`
- `examples/rtc/eval_with_real_robot.py`

## 32. 最后一段总结

RTC 可以用一句最精炼的话概括：

它是一种面向 action-chunking 生成式机器人策略的推理时衔接机制。

当机器人正在执行旧 chunk，而新 chunk 又因为推理延迟晚到时，RTC 会在新 chunk 生成过程中，把它的前缀引导得更贴近旧 chunk 的剩余轨迹；随后在运行时再跳过已经过期的前几步，让新 chunk 从一个仍然来得及、而且更平滑的位置接管执行。

因此，RTC 的价值不在于“让模型本身更聪明”，而在于：

- 让大模型在真实时间系统里更可用；
- 让 chunk policy 不至于在边界处抽动；
- 让“高延迟模型 + 连续控制”这对天然冲突的组合，更接近工程可落地。
