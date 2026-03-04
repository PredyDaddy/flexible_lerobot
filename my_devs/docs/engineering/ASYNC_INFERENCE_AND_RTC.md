# 在 LeRobot 中加速推理：`async_inference` 与 `RTC`（结合 `run_groot_infer.py`）

本文是一个面向“要把实时机器人推理跑顺”的使用与源码解读文档，重点回答两件事：

1. `src/lerobot/async_inference/` 这套异步推理（PolicyServer/RobotClient）怎么用，适合解决什么问题？
2. `src/lerobot/policies/rtc/` 这套 Real-Time Chunking（RTC）怎么用，适合解决什么问题？

并且会结合你的脚本 `my_devs/train/groot/run_groot_infer.py` 来解释：它为什么是同步/单线程的，怎么迁移/替换为异步方案。

注意：本文偏“工程落地 + 读代码”，会比官方文档更啰嗦一些。官方对应文档也在：

- `docs/source/async.mdx`
- `docs/source/rtc.mdx`

## 0. 背景：`run_groot_infer.py` 为什么会“看起来是单线程”

你的推理主循环（简化版）是这样的：

```text
while True:
  obs = robot.get_observation()        # 采集
  obs = preprocess(obs)               # 预处理
  action = policy_infer(obs)          # 模型推理
  action = postprocess(action)        # 后处理/反归一化
  robot.send_action(action)           # 执行
  sleep_to_keep_fps()
```

这就是“同步推理”的典型形式：一次循环里同时做采集、推理、执行，推理的耗时会直接阻塞下一步执行。

当 `policy_infer()` 的耗时接近或超过 `1/fps` 时，就会出现：

- 机器人会出现“空转/停顿帧”（idle frames）：队列里没有动作可执行，只能等模型算完。
- 即使平均耗时够快，只要抖动大（比如偶发卡顿），也会造成动作执行的节奏不稳定。

这不是“多线程能不能算更快”的问题，而是“推理和执行耦合导致必须等推理”的问题。

因此仓库里有两种不同方向的“让实时控制更丝滑”的方案：

- `async_inference`：解决“等推理导致机器人空转”的问题（解耦预测与执行）。
- `RTC`：解决“动作 chunk 之间衔接不连续/抖动”的问题（对 flow-matching 生成过程加 guidance，让新 chunk 接上旧 chunk）。

两者关注点不同，可以组合，但不是同一件事。

## 1. 两套方案分别解决什么问题（先把边界说清楚）

下面这张表是最重要的定位：

| 方案 | 主要解决的问题 | 典型表现 | 适用策略类型 | 关键代价 |
| --- | --- | --- | --- | --- |
| `async_inference`（PolicyServer/RobotClient） | 推理阻塞执行导致的 idle frames | 机器人每隔一段时间“顿一下”等模型 | 只要 policy 支持 `predict_action_chunk()` 就行（包括 Groot） | 需要额外依赖（grpc），要跑 server/client 两个进程（或至少两线程） |
| `RTC`（Real-Time Chunking） | chunk 拼接处的 discontinuity（新 chunk 和已执行轨迹不一致导致跳变） | chunk 切换时动作突然改变、抖动、策略“变卦” | **只针对 flow-matching 类**：`pi0`/`pi05`/`smolvla`（代码里已集成） | 需要你用 `predict_action_chunk()` + 自己维护 action queue，并计算 `inference_delay` |

对你当前的 Groot 推理脚本来说：

- `async_inference` 是直接可用、最符合“加速实时推理体验”的（它减少等待）。
- `RTC` **目前并没有接入 Groot 的推理代码路径**（GrootPolicy 里没有 RTCProcessor 的调用），所以不能“开个开关就用”。RTC 更像是 pi0/pi05/smolvla 的专项增强。

补一条你关心的结论（非常重要）：

- `pi0.5` 可以同时使用“异步执行 + RTC”。
  - 异步执行：通过双线程/队列，让“执行动作”和“生成下一段 chunk”并行。
  - RTC：在 chunk 重叠区域做 guidance，让衔接更平滑。
- `Groot` 目前只能直接使用异步执行；RTC 需要额外改造模型推理路径。

## 2. `src/lerobot/async_inference/`：怎么用（非常详细）

### 2.1 先说核心思想：把“算动作 chunk”提前做

同步方案的问题是：动作队列空了才去算下一段 chunk，于是必然会出现等待。

异步方案把它拆开成两个并行循环：

1. 执行循环（高优先级）：只管按固定频率把 action queue 里的动作发给机器人。
2. 预测循环（后台）：当 queue 变少时（低于阈值），马上采集 observation 发给 policy server，请它开始算下一段 chunk。

这样推理可以“重叠”到执行过程中，大幅减少 idle。

### 2.2 架构与线程模型（对照源码更容易理解）

这套模块是 gRPC client/server：

- PolicyServer：`src/lerobot/async_inference/policy_server.py`
- RobotClient：`src/lerobot/async_inference/robot_client.py`

你可以把它想象成这样：

```text
RobotClient 进程
  - Thread A: receive_actions()  (不停 GetActions 拉取动作 chunk)
  - Thread B: control_loop()     (执行动作 + 发送 observation)

PolicyServer 进程
  - gRPC worker threads
  - 内部只有一个 observation_queue(maxsize=1)
  - GetActions() 从 queue 取最新 observation，跑 policy.predict_action_chunk()
```

关键细节：

- **PolicyServer 的 observation_queue maxsize=1**：它永远只保留“最新的一帧 obs”，不会排长队。
  - 如果你发送 obs 太快，旧的会被丢掉，只保留最新的。
  - 这对实时控制是对的：你更关心“现在”而不是“积压的过去”。
- client 只有当 action queue 变少时才发 obs（`chunk_size_threshold` 控制），目的是让 server 提前算下一段。

### 2.3 依赖安装（你现在环境里 `grpc` 很可能缺）

我在当前 `lerobot_flex` 环境里直接 `import lerobot.async_inference.policy_server` 会报：

```text
ModuleNotFoundError: No module named 'grpc'
```

这是预期的：`grpcio` 属于可选依赖，需要安装 extra。

在仓库 `pyproject.toml` 里定义了：

- `.[async]` 会带上 `grpcio`、`protobuf`、`matplotlib` 等
- `grpcio` 版本被 pin 到 `grpcio==1.73.1`

在你们仓库规定的 conda 环境 `lerobot_flex` 中安装建议如下：

```bash
conda activate lerobot_flex

# 在 repo 根目录
python -m pip install -e ".[async]"
```

如果你只想装最小的 gRPC 依赖（不装 matplotlib），也可以：

```bash
python -m pip install -e ".[grpcio-dep]"
```

### 2.4 运行方式（命令行最直观）

#### 2.4.1 启动 PolicyServer

开一个终端：

```bash
conda activate lerobot_flex
cd /data/cqy_workspace/flexible_lerobot

python -m lerobot.async_inference.policy_server \
  --host=127.0.0.1 \
  --port=8080 \
  --fps=30 \
  --inference_latency=0.033 \
  --obs_queue_timeout=2
```

参数含义（来自 `PolicyServerConfig`）：

- `fps`：用于 server 端把 action chunk 打时间戳时的 dt（`environment_dt = 1/fps`）。
- `inference_latency`：一个“节流”参数。server 每次生成 action chunk 后会 `sleep`，保证不会比这个更快地产出（避免 server 忙等）。
- `obs_queue_timeout`：GetActions 等 observation 的超时时间，超时则返回空（client 会继续下一轮请求）。

#### 2.4.2 启动 RobotClient（以 so101_follower + Groot checkpoint 为例）

再开一个终端（或同机另一个 tmux pane）：

```bash
conda activate lerobot_flex
cd /data/cqy_workspace/flexible_lerobot

python -m lerobot.async_inference.robot_client \
  --server_address=127.0.0.1:8080 \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_so101 \
  --robot.calibration_dir=/home/cqy/.cache/huggingface/lerobot/calibration/robots/so101_follower \
  --robot.cameras="{ top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30} }" \
  --task="Put the block in the bin" \
  --policy_type=groot \
  --pretrained_name_or_path=/data/cqy_workspace/flexible_lerobot/outputs/train/.../checkpoints/last/pretrained_model \
  --policy_device=cuda \
  --client_device=cpu \
  --actions_per_chunk=16 \
  --chunk_size_threshold=0.5 \
  --aggregate_fn_name=weighted_average \
  --debug_visualize_queue_size=False
```

这条命令对应你 `run_groot_infer.py` 里几个参数：

- `--robot.port` 对应 `--robot-port`
- `--robot.cameras` 里 `top/wrist` 的 `index_or_path` 对应 `--top-cam-index/--wrist-cam-index`
- `--task` 对应脚本 `--task`
- `--pretrained_name_or_path` 对应脚本 `--policy-path`

关于 `actions_per_chunk=16`：

- 对 Groot 来说很关键：`src/lerobot/policies/groot/processor_groot.py` 里明确写了
  - `action_horizon = min(config.chunk_size, 16)`
  - 也就是 **GR00T N1.5 的动作 horizon 在当前实现里上限是 16**。
- 因此在 async inference 里建议你把 `actions_per_chunk` 设为 `<=16`，否则 server 端也会截断/浪费带宽。

### 2.5 参数怎么调（把关键参数讲透）

#### 2.5.1 `actions_per_chunk`（动作 chunk 长度）

它控制 server 每次推理返回多少个动作，直接影响：

- 推理频率：chunk 越短，越频繁推理；chunk 越长，推理频率越低。
- 对环境反应速度：chunk 越短越“reactive”（但算得更频繁）。

经验建议：

- Groot：从 `16` 开始（上限）。
- pi0/pi05/smolvla：常见 `10~50`，看你模型延迟。

#### 2.5.2 `chunk_size_threshold`（什么时候发新 observation）

源码在 `RobotClient._ready_to_send_observation()`：

```python
return self.action_queue.qsize() / self.action_chunk_size <= self._chunk_size_threshold
```

解释：

- `action_chunk_size` 是目前见过的最大 chunk 长度（初始为 -1，收到第一段动作后变为正数）。
- `action_queue.qsize()` 是当前剩余动作数。
- 当 “剩余比例 <= threshold” 时，就发新 observation 给 server，让它开始算下一段 chunk。

建议值：

- `0.5`：queue 用掉一半就开始算下一段，通常比较稳。
- 太小（比如 0.1）：等快用完才算，容易来不及。
- 太大（比如 0.9）：算得过于频繁，server 压力大，且网络/序列化开销更大。

#### 2.5.3 `aggregate_fn_name`（chunk 重叠段怎么融合）

当 server 回来的 action chunk 与 client 本地 queue 有时间步重叠时，会做融合（避免突变）。

可用函数定义在 `src/lerobot/async_inference/configs.py` 的 `AGGREGATE_FUNCTIONS`：

- `weighted_average`: `0.3 * old + 0.7 * new`（默认推荐）
- `latest_only`: 直接用新的
- `average`: `0.5/0.5`
- `conservative`: `0.7 * old + 0.3 * new`（更保守）

如果你想加自定义融合函数：

1. 在 `AGGREGATE_FUNCTIONS` 里加一个名字。
2. client 用 `--aggregate_fn_name=your_name`。

#### 2.5.4 `PolicyServerConfig.inference_latency`（server 端“节流”）

server 每次 GetActions() 里会：

1. 拿到 obs
2. 推理得到 chunk
3. pickle 序列化
4. `sleep(max(0, inference_latency - elapsed))`

这个参数的作用不是“让模型更快”，而是“控制 server 不要无上限地跑推理循环”，避免 CPU/GPU 被打满导致系统整体抖动更大。

### 2.6 和你当前 Groot TensorRT 推理的关系

你现在的 `run_groot_infer.py` 已经支持：

- `--backend=tensorrt`
- `setup_tensorrt_engines(policy._groot_model, ...)`

而 `async_inference` 的 `PolicyServer` 目前 **不包含** 这段 TensorRT patch 逻辑。

这意味着：

- 你可以先把 async inference 跑起来解决 idle frames。
- 如果还想继续榨性能，建议做法是写一个“自定义 PolicyServer 启动脚本”，在加载 policy 后调用 `setup_tensorrt_engines()`。
  - 入口可以参考 `src/lerobot/async_inference/policy_server.py:serve()`
  - Groot 的 TensorRT patch 入口在 `lerobot.policies.groot.trt_runtime.patch:setup_tensorrt_engines`

### 2.7 常见坑与排查清单

1. `ModuleNotFoundError: grpc`
   - 没装 `.[async]` 依赖。按 2.3 安装。
2. client 能连上 server，但 server 报 KeyError/feature mismatch
   - 相机 key 不一致。
   - 解决：让 `--robot.cameras` 的 key（例如 `top/wrist`）与训练时 policy 的 `config.json -> input_features` 对齐。
3. 机器人仍然会偶发停顿
   - 推理时间仍然比“chunk 覆盖时间”长，队列会被耗尽。
   - 解决：减小 `fps`、增加 `actions_per_chunk`（不超过模型上限）、换更快 device、启用 TensorRT、减少图像分辨率等。
4. 安全性
   - server/client 使用 `pickle` 传输（见 `pickle.loads`），只适合在可信网络内使用。
   - 不要把端口直接暴露到公网。

## 3. `src/lerobot/policies/rtc/`：怎么用（非常详细）

### 3.1 RTC 是什么，和 async_inference 有什么不同

RTC（Real-Time Chunking）解决的是：**你在执行上一个 chunk 的同时生成下一个 chunk** 时，新 chunk 的前半段和旧 chunk 的后半段会重叠。

如果直接拼接，常见问题是：

- 新 chunk 的前几步和“机器人当前真实状态”不一致
- chunk 边界处动作跳变，视觉上是抖动/顿挫

RTC 的做法是：

- 把生成下一个 chunk 的过程当成“inpainting”
- 在 flow-matching 的 denoising/flow step 里增加 guidance
- 让新 chunk 的重叠前缀更贴近旧 chunk 剩余的轨迹（`prev_chunk_left_over`）

对应实现的关键类是：

- `RTCConfig`：`src/lerobot/policies/rtc/configuration_rtc.py`
- `RTCProcessor`：`src/lerobot/policies/rtc/modeling_rtc.py`
- `ActionQueue`：`src/lerobot/policies/rtc/action_queue.py`

### 3.2 先说最关键限制：RTC 不能配合 `select_action()`

以 `pi0` 为例（`src/lerobot/policies/pi0/modeling_pi0.py`）：

```python
def select_action(...):
  assert not self._rtc_enabled(), "RTC is not supported for select_action, use it with predict_action_chunk"
```

也就是说：

- 你如果只是调用 `policy.select_action()`（同步单步推理接口），RTC 是不会工作的。
- 你必须使用 `policy.predict_action_chunk(...)`，并且传入 RTC 需要的参数：
  - `inference_delay`
  - `prev_chunk_left_over`
  - `execution_horizon`（可选）

### 3.3 推荐直接参考的“真正在跑”的例子

仓库里有一个完整的、可运行的 RTC + 真实机器人 demo：

- `examples/rtc/eval_with_real_robot.py`

它做了这些关键工程点：

- 两线程：
  - `get_actions` 线程：负责在 queue 变少时，采 obs，跑 `predict_action_chunk`，merge 进 queue
  - `actor_control` 线程：负责按固定频率从 queue 取 action 发给机器人
- 用 `LatencyTracker` 测 inference latency，转换成离散的 `inference_delay = ceil(latency / dt)`
- 用 `ActionQueue.merge()` 在 RTC enabled 时做“替换队列并跳过 real_delay 步”的逻辑

如果你想把 RTC 用到自己的机器人推理 pipeline，最靠谱的方式就是从这个脚本裁剪。

### 3.4 RTC 的最小工作流程（抽象出来）

你需要 4 个状态量：

1. `ActionQueue`（跨线程共享）
2. `prev_chunk_left_over = action_queue.get_left_over()`（上一段 chunk 未执行部分）
3. `inference_delay`（推理耗时折算成环境步数）
4. `RTCConfig`（execution_horizon / guidance_weight / schedule）

推理线程每次生成新 chunk 的伪代码是：

```python
prev = action_queue.get_left_over()
delay = ceil(inference_latency / dt)

chunk = policy.predict_action_chunk(
  obs,
  inference_delay=delay,
  prev_chunk_left_over=prev,
)

action_queue.merge(
  original_actions=chunk_original,   # 未 postprocess 的 actions，用于 RTC 计算 leftover
  processed_actions=chunk_processed, # postprocess 后的 actions，用于真实执行
  real_delay=delay,
)
```

执行线程是：

```python
while True:
  a = action_queue.get()
  if a is not None:
    robot.send_action(a)
  sleep(dt)
```

### 3.5 `RTCConfig` 参数怎么理解（对照源码）

`RTCConfig` 在 `src/lerobot/policies/rtc/configuration_rtc.py`：

- `enabled`
  - 是否启用 RTC guidance。
- `execution_horizon`
  - 你希望“对齐/平滑”的重叠区域长度（单位：步）。
  - 越大越平滑，但可能更不 reactive。
- `max_guidance_weight`
  - guidance 强度上限。
  - 太小可能不够平滑，太大可能过拟合旧轨迹导致“转不过弯”。
- `prefix_attention_schedule`
  - 对重叠前缀的权重 mask 怎么衰减。
  - 见 `RTCProcessor.get_prefix_weights()`：
    - `LINEAR`
    - `EXP`
    - `ONES`
    - `ZEROS`
- `debug` / `debug_maxlen`
  - 打开后会记录 denoise_step 的一些中间量（注意这会有额外开销）。

### 3.6 `inference_delay` 怎么算才靠谱

RTC 里 `inference_delay` 是“推理期间机器人执行了多少步”的估计。

在 `examples/rtc/eval_with_real_robot.py` 里他们用：

- `time_per_chunk = 1/fps`
- `inference_delay = ceil(max_latency / time_per_chunk)`

建议你用“最近一段时间的 max 或 p95 latency”，不要只用平均值，因为偶发长尾最容易把队列耗尽。

如果你愿意更工程化：

- 在推理线程里测 `t0 = perf_counter()`
- 推理完成后 `latency = perf_counter() - t0`
- `delay = ceil(latency / dt)`
- `LatencyTracker.add(latency)`，用 `max()` 或 `p95()` 作为下一次的估计。

### 3.7 RTC 能不能和 `async_inference` 结合？

理论上可以，思路是：

- 让 PolicyServer 在 server 端也维护一个类似 `ActionQueue` 的结构
- 每次收到 observation 时，带上（或 server 自己推断）`prev_chunk_left_over` 与 `inference_delay`
- 然后调用 `policy.predict_action_chunk(..., inference_delay=..., prev_chunk_left_over=...)`

但请注意当前代码状态：

- `src/lerobot/async_inference/policy_server.py` 现在只是简单地：
  - `action_tensor = policy.predict_action_chunk(observation)`
  - **没有传 RTC kwargs**
  - 也没有维护“上一段 chunk leftover”

因此“开箱即用”层面：

- 要用 RTC，就先用 `examples/rtc/eval_with_real_robot.py` 这种“两线程同进程”的模式。
- 要把 RTC 融进 gRPC 的 async_inference，需要你做一些 server 端改造。

### 3.8 `pi0.5` 和 `Groot` 的差异（为什么 `pi0.5` 可以两个一起用）

先看最核心的代码差异：

1. `pi0.5` 的 policy config 原生带 `rtc_config` 字段（可直接开 RTC）。
2. `pi0.5` 的 `predict_action_chunk(..., **kwargs)` 会把 RTC 参数传给模型采样路径。
3. `pi0.5` 的 `select_action` 明确禁止 RTC，要求必须走 chunk 接口。
4. `Groot` 的 `predict_action_chunk` 目前没有接 `inference_delay/prev_chunk_left_over` 这类 RTC 参数入口。

所以结论是：

- `pi0.5`：可以“异步 + RTC”一起用（仓库已有可运行样例）。
- `Groot`：当前只能直接做异步；RTC 不是开关项，需要额外实现。

### 3.9 `pi0.5` 双方案联合（异步 + RTC）怎么用

这里的“异步 + RTC”指的是：

- 异步：`get_actions` 线程在后台生成 chunk；
- 执行：`actor_control` 线程按固定频率消费队列；
- RTC：生成新 chunk 时传 `inference_delay + prev_chunk_left_over`；
- 合并：用 `ActionQueue.merge()` 按 real delay 对齐并替换队列。

也就是你要的“两个一起用”，在 `examples/rtc/eval_with_real_robot.py` 里已经是这样实现的。

#### 3.9.1 直接可跑命令（`pi0.5`）

```bash
conda activate lerobot_flex
cd /data/cqy_workspace/flexible_lerobot

python examples/rtc/eval_with_real_robot.py \
  --policy.path=<你的pi05模型路径或HF repo> \
  --policy.device=cuda \
  --rtc.enabled=true \
  --rtc.execution_horizon=10 \
  --rtc.max_guidance_weight=10.0 \
  --rtc.prefix_attention_schedule=EXP \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_so101 \
  --robot.cameras="{ top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30} }" \
  --task="Put the block in the bin" \
  --fps=10 \
  --duration=120
```

上面这条命令里，`--rtc.enabled=true` 就是打开 RTC；而脚本本身已经是异步线程架构，所以天然是“两个一起用”。

#### 3.9.2 参数起步建议（先稳再快）

1. `fps=10` 起步，先看是否有队列耗尽。
2. `rtc.execution_horizon=8~12` 起步。
3. `rtc.max_guidance_weight=10.0` 起步。
4. `rtc.prefix_attention_schedule=EXP` 起步。

观察指标：

1. 是否还出现“动作队列经常清空导致停顿”。
2. chunk 切换时是否还有明显抖动。
3. 平均延迟与 p95 延迟是否在可控范围。

#### 3.9.3 关键实现点（对照代码）

`examples/rtc/eval_with_real_robot.py` 里最关键的四步：

1. 计算 `inference_delay`（由 latency 推导成步数）。
2. 调用 `policy.predict_action_chunk(..., inference_delay=..., prev_chunk_left_over=...)`。
3. 保存 `original_actions`（用于 RTC leftover 对齐）。
4. `action_queue.merge(original_actions, postprocessed_actions, new_delay, ...)`。

如果你要把这个逻辑迁移到自己的 `run_pi05_*.py`，就围绕这四步搬就够了。

#### 3.9.4 关于 gRPC `async_inference` 的说明（避免概念混淆）

你提的“异步”有两种语义：

1. “控制环异步”：本地双线程 + 队列（`examples/rtc/eval_with_real_robot.py`）；
2. “进程/机器异步”：gRPC PolicyServer/RobotClient（`src/lerobot/async_inference`）。

`pi0.5` 的“RTC + 异步”开箱即用是第一种（本地双线程）。

第二种（gRPC）当前默认 `PolicyServer` 还没传 RTC kwargs，所以要改造 server 才能实现“gRPC 异步 + RTC 同时开启”。

## 4. 结合你的 Groot 推理：建议怎么选

如果你的目标是“Groot 推理更丝滑”：

1. 第一优先：用 `async_inference` 去掉 idle frames（解耦推理与执行）。
2. 第二优先：你脚本里已经有 TensorRT backend（算得更快），可以考虑把它挪到 PolicyServer 里（自定义 server 启动脚本）。
3. RTC：目前不适用于 Groot（除非你愿意自己把 RTCProcessor 接入 Groot 的生成过程）。

如果你未来会跑 `pi0/pi05/smolvla` 这种 flow-matching 策略，并且模型延迟较大：

1. `async_inference`：解决“等推理”。
2. `RTC`：解决“chunk 拼接抖动”。
3. 对 `pi0.5` 来说，两者在 `examples/rtc/eval_with_real_robot.py` 已经是“同进程两线程”的可行实现（可直接跑）。
4. 如果你想用 gRPC 版本做“两者结合”，则需要改造 `PolicyServer`，给 `predict_action_chunk` 传 RTC 参数并维护 leftover。

## 5. 一句话速记

`pi0.5` 可以同时用 RTC 和异步；`Groot` 目前只能直接用异步，RTC 需要额外实现。

## 6. 代码索引（从哪里开始读最划算）

Async inference：

- `src/lerobot/async_inference/policy_server.py`
- `src/lerobot/async_inference/robot_client.py`
- `src/lerobot/async_inference/configs.py`
- `src/lerobot/async_inference/helpers.py`

RTC：

- `src/lerobot/policies/rtc/configuration_rtc.py`
- `src/lerobot/policies/rtc/modeling_rtc.py`
- `src/lerobot/policies/rtc/action_queue.py`
- `examples/rtc/eval_with_real_robot.py`

结合 Groot：

- `my_devs/train/groot/run_groot_infer.py`
- `src/lerobot/policies/groot/modeling_groot.py`
- `src/lerobot/policies/groot/trt_runtime/patch.py`
