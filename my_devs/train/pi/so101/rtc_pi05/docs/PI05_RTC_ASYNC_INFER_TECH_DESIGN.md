# PI0.5 SO101 异步推理与 Real-Time Chunking 技术方案

日期: 2026-07-07

本文档从当前可工作的同步脚本 `my_devs/train/pi/so101/run_pi05_infer.py` 出发，重新设计一套干净的 PI0.5 本地异步推理 runtime。目标开发位置是:

```text
my_devs/train/pi/so101/rtc_pi05
```

本方案明确不以 `my_devs/pi05_engineering` 为迁移基础。后续如果需要参考其中的个别测试思想，也只能按新架构重写，不能复制旧 runtime 结构。

## 1. 背景

当前可工作的命令是同步闭环:

```bash
conda run --no-capture-output -n lerobot_flex \
python my_devs/train/pi/so101/run_pi05_infer.py \
    --policy-path outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
    --robot-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00 \
    --top-cam /dev/video4 \
    --wrist-cam /dev/video6 \
    --task "Put the eraser into the small box" \
    --run-time-s 120
```

它的主循环是:

```text
robot.get_observation()
  -> robot_observation_processor
  -> build_dataset_frame
  -> prepare_observation_for_inference
  -> policy preprocessor
  -> policy.select_action()
  -> policy postprocessor
  -> make_robot_action
  -> robot_action_processor
  -> robot.send_action()
  -> sleep 到下一帧
```

问题是 PI0.5 单次推理较慢，控制循环会被模型推理阻塞，无法稳定保持机器人控制频率。PI0.5 本身是 action chunking policy，一次能生成一段动作，所以更合理的结构是:

```text
控制线程: 固定频率发送动作
推理线程: 低水位时异步生成下一段 action chunk
队列: 保存可执行动作，同时保存未 postprocess 的原始 chunk 供 RTC 使用
```

## 2. 目标

1. 保持当前 checkpoint 直接推理能力:
   - full checkpoint: `model.safetensors`
   - LoRA checkpoint: `adapter_model.safetensors` + base policy
   - 继续复用 checkpoint 内的 `policy_preprocessor.json` 和 `policy_postprocessor.json`

2. 做成本地异步推理:
   - 控制线程不等待 PI0.5 模型推理
   - 推理线程按队列低水位补充 action chunk
   - actor loop 尽量稳定在 `fps`，比如 30 Hz

3. 接入 `src/lerobot/policies/rtc` 中已有的 Real-Time Chunking 能力:
   - `PI05Policy.predict_action_chunk(..., inference_delay, prev_chunk_left_over, execution_horizon)`
   - `PI05Pytorch.sample_actions()` 内部已经有 `RTCProcessor.denoise_step(...)` hook
   - runtime 需要正确维护 `prev_chunk_left_over`
   - runtime 需要正确估计和记录 inference delay

4. 所有新开发默认放在:

```text
my_devs/train/pi/so101/rtc_pi05
```

5. 所有运行、测试、格式化、lint 命令必须使用:

```bash
conda run --no-capture-output -n lerobot_flex ...
```

## 3. 非目标

第一阶段不做这些事情:

1. 不重写 `src/lerobot/policies/pi05/modeling_pi05.py` 的模型主体。
2. 不改训练流程。
3. 不先做远程 server/client。先做本机双线程 runtime，稳定后再考虑网络化。
4. 不把新代码塞进 `src/lerobot/async_inference`。
5. 不依赖 `policy.select_action()` 做 RTC，因为当前 PI0.5 在 RTC enabled 时明确要求走 `predict_action_chunk()`。
6. 不用旧的 `my_devs/pi05_engineering` runtime 作为骨架。

## 4. 当前代码事实

### 4.1 `run_pi05_infer.py`

现有同步脚本已经解决了这些关键问题:

1. 离线 tokenizer 路径检查:

```text
google/paligemma-3b-pt-224
```

2. checkpoint artifact 校验:

```text
config.json
model.safetensors 或 adapter_model.safetensors
policy_preprocessor.json
policy_postprocessor.json
normalizer / unnormalizer safetensors
train_config.json
```

3. full checkpoint 和 LoRA checkpoint 加载。
4. SO101/SO100 follower robot config 构建。
5. top/wrist OpenCV camera config 构建。
6. LeRobot dataset feature mapping 构建。
7. policy preprocessor / postprocessor 与 robot processor 的连接。

新 runtime 应该复用这些“加载与适配逻辑”，但不能复用它的同步主循环。

### 4.2 `src/lerobot/policies/pi05`

PI0.5 的关键接口是:

```python
policy.predict_action_chunk(batch, **kwargs)
```

当 `policy.config.rtc_config.enabled == True` 时，`select_action()` 会 assert:

```text
RTC is not supported for select_action, use it with predict_action_chunk
```

因此异步 runtime 必须直接调用 `predict_action_chunk()`。

`predict_action_chunk()` 返回的是 policy action space 的 raw chunk，尚未经过 checkpoint postprocessor。runtime 必须保留这份 raw chunk，因为 RTC 的 `prev_chunk_left_over` 必须和模型 denoise/action chunk 空间一致，不能用已经 unnormalize 后的机器人动作反推。

### 4.3 `src/lerobot/policies/rtc`

已有 RTC 模块提供:

1. `RTCConfig`
   - `enabled`
   - `prefix_attention_schedule`: `ZEROS` / `ONES` / `LINEAR` / `EXP`
   - `max_guidance_weight`
   - `execution_horizon`
   - `debug`

2. `RTCProcessor`
   - 在 denoise step 内使用上一段 chunk 的 leftover 做 prefix guidance。

3. `ActionQueue`
   - 低层 action queue primitive。
   - 支持 RTC replacement merge 与 plain append merge。

注意: `src/lerobot/policies/rtc` 不是完整机器人 runtime。它只解决“模型生成 chunk 时如何用 leftover 做 RTC”和“动作队列如何 merge”的一部分问题。机器人控制频率、观测时间戳、线程生命周期、安全策略、日志、验收测试，都需要在 `rtc_pi05` runtime 中设计。

## 5. 头脑风暴后的核心判断

### 5.1 不把“异步”做进 policy

policy 应保持纯计算接口:

```text
observation batch -> action chunk
```

异步线程、队列、水位、机器人 IO 都不应该塞进 `PI05Policy`。否则 policy 会同时负责模型、调度、硬件，后续难测也难复用。

### 5.2 不用 `predict_action()` 公共 helper 跑 RTC

`lerobot.utils.control_utils.predict_action()` 内部用了:

```python
torch.inference_mode()
```

RTCProcessor 内部需要 `torch.enable_grad()` 对 `x_t` 求 correction。`torch.no_grad()` 可以在局部被 `enable_grad()` 覆盖，但 `torch.inference_mode()` 更强，会让 RTC 的 autograd correction 有失败风险。

因此 RTC path 要自己写推理调用逻辑:

```text
prepare observation
-> preprocessor
-> policy.predict_action_chunk(... rtc kwargs ...)
-> postprocessor
```

外层不要包 `torch.inference_mode()`。非 RTC path 可以使用 `torch.no_grad()`，RTC path 依赖 PI0.5 内部的 no_grad 与 RTCProcessor 的局部 enable_grad。

### 5.3 runtime 队列必须保存两套动作

每个 chunk 有两种表示:

```text
raw_actions:
  policy.predict_action_chunk() 的输出
  模型 action space
  用于下一次 RTC 的 prev_chunk_left_over

processed_actions:
  postprocessor(raw_actions) 的输出
  机器人可执行 action space
  用于 make_robot_action / robot_action_processor / robot.send_action
```

这两套动作不能混用。

### 5.4 actor loop 和 inference worker 的职责分开

推荐两线程结构:

```text
Actor / Sensor thread:
  - 固定 fps tick
  - 读取 robot observation
  - 更新 latest observation buffer
  - 从 action queue pop 一个 processed action
  - 转成 robot action 并发送
  - 队列空时按策略 hold / skip / stop

Inference worker thread:
  - 等待 latest observation 可用
  - 发现 action queue 低水位时复制 latest observation snapshot
  - 读取 raw leftover
  - 调用 policy.predict_action_chunk()
  - postprocess 得到 processed chunk
  - 按 latency/drop_steps merge 回 queue
```

这样模型推理不会阻塞机器人控制发送。

### 5.5 observation 时间戳比“函数耗时”更重要

一次 chunk 的语义是:

```text
在 t_obs 看到 observation，预测从 t_obs 开始的未来动作序列
```

如果 chunk 在 `t_ready` 才可执行，那么前面一部分动作已经过期，应丢弃:

```text
drop_steps = ceil((t_ready - t_obs) / control_dt)
```

只看模型耗时是不够的，因为 observation 可能在 worker 拿到前就已经有一些年龄。runtime 必须给 observation snapshot 带时间戳。

## 6. 总体架构

建议目录:

```text
my_devs/train/pi/so101/rtc_pi05/
  run_rtc_pi05_infer.py
  rtc_pi05/
    __init__.py
    config.py
    checkpoint_loader.py
    robot_builder.py
    observation_buffer.py
    action_chunk_queue.py
    inference_worker.py
    actor_loop.py
    runtime_state.py
    metrics.py
    safety.py
  docs/
    PI05_RTC_ASYNC_INFER_TECH_DESIGN.md
```

后续测试建议放在:

```text
tests/my_devs/rtc_pi05/
```

如果想完全避免包名和脚本目录混淆，也可以使用:

```text
my_devs/train/pi/so101/rtc_pi05/runtime/
```

但要保证入口脚本可直接运行，并且 `REPO_ROOT/src` 和 `REPO_ROOT` 都能正确进入 `sys.path`。

## 7. 核心数据结构

### 7.1 `RuntimeConfig`

建议字段:

```python
@dataclass
class RuntimeConfig:
    policy_path: Path
    task: str
    fps: int = 30
    run_time_s: float = 0.0

    queue_low_watermark: int = 8
    queue_target_size: int = 24
    max_queue_size: int = 50
    first_chunk_timeout_s: float = 30.0

    enable_rtc: bool = True
    rtc_execution_horizon: int = 10
    rtc_max_guidance_weight: float = 10.0
    rtc_prefix_attention_schedule: str = "LINEAR"
    rtc_debug: bool = False

    empty_queue_strategy: str = "hold-last-action"
    max_action_delta: float | None = None
    warmup_chunks: int = 1
```

默认建议:

```text
fps = 30
queue_low_watermark = 8
queue_target_size = 24
max_queue_size = 50
rtc_execution_horizon = 10
rtc_prefix_attention_schedule = LINEAR
empty_queue_strategy = hold-last-action
```

### 7.2 `ObservationSnapshot`

```python
@dataclass(frozen=True)
class ObservationSnapshot:
    observation: dict[str, Any]
    timestamp_s: float
    sequence_id: int
```

`timestamp_s` 应该在 `robot.get_observation()` 成功返回后立即记录，或者更严格地记录读取开始和结束:

```python
read_started_s: float
read_finished_s: float
```

第一版可以用 `read_finished_s` 做 `timestamp_s`，后续再细化。

### 7.3 `ActionChunk`

```python
@dataclass
class ActionChunk:
    raw_actions: torch.Tensor          # shape: (T, action_dim), policy action space
    processed_actions: torch.Tensor    # shape: (T, action_dim), robot action space
    obs_timestamp_s: float
    ready_timestamp_s: float
    drop_steps: int
    rtc_inference_delay: int
    source_observation_seq: int
```

### 7.4 `ActionChunkQueue`

队列内部必须维护:

```text
processed_queue:
  actor loop 每 tick pop 一个动作

raw_queue:
  与 processed_queue 对齐的 raw action backlog
  供下一次 inference worker 读取 leftover

cursor:
  当前已经消耗的动作数量

last_sent_action:
  队列空时 hold-last-action 使用
```

对外接口:

```python
class ActionChunkQueue:
    def pop_processed_action(self) -> torch.Tensor | None: ...
    def get_raw_leftover(self) -> torch.Tensor | None: ...
    def depth(self) -> int: ...
    def action_cursor(self) -> int: ...

    def merge_plain(self, chunk: ActionChunk) -> MergeResult: ...
    def merge_rtc(self, chunk: ActionChunk) -> MergeResult: ...
```

plain mode 和 RTC mode 的 merge 策略不同。

## 8. Plain Async Chunking 策略

不开 RTC 时，推理线程只负责补 chunk:

```text
如果 queue depth <= low_watermark:
  拿 latest observation
  推理 raw chunk
  postprocess processed chunk
  drop stale prefix
  append 到当前 queue 后面
```

plain mode 不知道新旧 chunk 如何平滑拼接，所以要保守:

1. 只在低水位补队列，不频繁覆盖。
2. 优先 append，不 replace。
3. `queue_target_size` 不宜过大，否则动作会越来越旧。
4. 如果 `drop_steps >= chunk_len`，本次 chunk 全部过期，直接丢弃并记录 starvation。

plain mode 的作用:

```text
先验证异步控制 loop、processor、robot IO、安全策略和日志。
```

它不是最终效果最好的模式。

## 9. RTC Async Chunking 策略

开启 RTC 时，推理线程每次生成 chunk 都会利用上一段未执行完的 raw leftover:

```python
raw_leftover = action_queue.get_raw_leftover()

raw_chunk = policy.predict_action_chunk(
    batch,
    inference_delay=predicted_delay_steps,
    prev_chunk_left_over=raw_leftover,
    execution_horizon=rtc_execution_horizon,
)
```

PI0.5 内部会在 denoise step 调用:

```text
RTCProcessor.denoise_step(...)
```

它会把 `prev_chunk_left_over` 当作 prefix guidance，让新 chunk 的前缀与旧 chunk 未执行部分保持一致。

RTC merge 推荐 replace 语义:

```text
new_queue = processed_chunk[drop_steps:]
new_raw_backlog = raw_chunk[drop_steps:]
```

原因:

1. 新 chunk 是基于最新 observation 重新规划的。
2. RTC 已经通过 leftover guidance 处理新旧 chunk 前缀连续性。
3. append 会让最新 observation 的动作延迟执行，反而削弱 RTC 意义。

## 10. delay 的定义

这里必须区分三个 delay。

### 10.1 `predicted_delay_steps`

推理开始前还不知道这次真实耗时，只能估计:

```text
predicted_delay_steps = ceil(latency_estimate_s / control_dt)
```

`latency_estimate_s` 可以来自:

1. 上一次 inference latency
2. 最近窗口 p95
3. EWMA

第一版建议:

```text
没有历史: 0
有历史: ceil(p95_total_latency_s / control_dt)
```

这个值传给 RTC:

```python
inference_delay=predicted_delay_steps
```

### 10.2 `drop_steps`

chunk 生成完成后，用 observation 时间戳计算真正过期了多少步:

```text
drop_steps = ceil((ready_timestamp_s - obs_timestamp_s) / control_dt)
```

它用于丢弃 stale prefix:

```text
raw_chunk = raw_chunk[drop_steps:]
processed_chunk = processed_chunk[drop_steps:]
```

### 10.3 `cursor_delta_steps`

推理开始到结束期间，actor 实际 pop 了多少动作:

```text
cursor_delta_steps = queue.action_cursor_after - queue.action_cursor_before
```

它用于诊断:

```text
如果 abs(cursor_delta_steps - drop_steps) 很大:
  说明控制 loop jitter、队列饥饿、或者 observation timestamp 口径有问题
```

第一版 merge 以 `drop_steps` 为准，因为 action chunk 的语义是基于 observation time，而不是基于函数返回时间。

## 11. 线程模型

### 11.1 主线程

职责:

1. parse args
2. 设置 repo root 与 tokenizer
3. 加载 policy config
4. 如果开启 RTC:

```python
policy.config.rtc_config = RTCConfig(...)
policy.init_rtc_processor()
```

5. 加载 preprocessor / postprocessor
6. 构建 robot 和 dataset features
7. 启动 actor thread 和 inference thread
8. 捕获 `KeyboardInterrupt`
9. 通知 stop event
10. join threads
11. disconnect robot

LoRA checkpoint 需要特别处理。当前同步脚本加载 LoRA 时会返回 `PeftModel` wrapper，并把 `policy.config` 指到 base policy config。RTC 注入时不能只改 wrapper 上的 config，必须确认底层真实 `PI05Policy` 也重新执行了 `init_rtc_processor()`。建议实现一个小工具函数:

```python
def enable_policy_rtc(policy: Any, rtc_config: RTCConfig) -> Any:
    policy.config.rtc_config = rtc_config
    if hasattr(policy, "init_rtc_processor"):
        policy.init_rtc_processor()
        return policy

    base = getattr(policy, "base_model", None)
    if base is not None and hasattr(base, "model") and hasattr(base.model, "init_rtc_processor"):
        base.model.config.rtc_config = rtc_config
        base.model.init_rtc_processor()
        return policy

    raise TypeError("Cannot enable RTC: underlying PI05Policy was not found.")
```

实际实现时需要按 PEFT 当前对象结构验证一次，不能假设 wrapper 一定会把自定义方法完整 forward 出来。

### 11.2 Actor / Sensor Thread

伪代码:

```python
next_tick = perf_counter()

while state.running:
    tick_start = perf_counter()

    obs = robot_io.get_observation()
    observation_buffer.update(obs, timestamp_s=perf_counter())

    action = queue.pop_processed_action()
    if action is None:
        handle_empty_queue()
    else:
        action_dict = make_robot_action(action, dataset_features)
        robot_action = robot_action_processor((action_dict, obs))
        safety.check(robot_action)
        robot_io.send_action(robot_action)

    metrics.record_actor_tick(...)

    next_tick += control_dt
    precise_sleep(max(next_tick - perf_counter(), 0.0))
```

注意:

1. 第一版建议 `robot_io.get_observation()` 和 `robot_io.send_action()` 使用同一个锁，避免 serial bus 并发访问。
2. 摄像头底层已有 `async_read()`，因此每 tick 调 `get_observation()` 不一定会完整阻塞摄像头采集。
3. 如果实测锁导致控制抖动，再拆成 camera observation buffer 和 motor state buffer。

### 11.3 Inference Worker Thread

伪代码:

```python
while state.running:
    if queue.depth() > queue_low_watermark:
        wait_or_sleep()
        continue

    obs_snapshot = observation_buffer.latest(timeout=first_chunk_timeout_s)
    if obs_snapshot is None:
        continue

    cursor_before = queue.action_cursor()
    raw_leftover = queue.get_raw_leftover()
    predicted_delay_steps = latency_estimator.predicted_steps()

    batch = build_policy_batch(obs_snapshot.observation)

    raw_chunk = policy.predict_action_chunk(
        batch,
        inference_delay=predicted_delay_steps if enable_rtc else None,
        prev_chunk_left_over=raw_leftover if enable_rtc else None,
        execution_horizon=rtc_execution_horizon if enable_rtc else None,
    )

    processed_chunk = postprocessor(raw_chunk)
    ready_s = perf_counter()
    drop_steps = ceil((ready_s - obs_snapshot.timestamp_s) / control_dt)

    chunk = ActionChunk(
        raw_actions=raw_chunk.squeeze(0),
        processed_actions=processed_chunk.squeeze(0),
        obs_timestamp_s=obs_snapshot.timestamp_s,
        ready_timestamp_s=ready_s,
        drop_steps=drop_steps,
        rtc_inference_delay=predicted_delay_steps,
        source_observation_seq=obs_snapshot.sequence_id,
    )

    if enable_rtc:
        queue.merge_rtc(chunk)
    else:
        queue.merge_plain(chunk)

    cursor_after = queue.action_cursor()
    metrics.record_inference(...)
```

## 12. Policy 输入输出路径

新 runtime 不走 `predict_action()` helper，而是显式拆开。

### 12.1 observation 到 batch

复用同步脚本逻辑:

```text
raw obs
  -> robot_observation_processor(obs)
  -> build_dataset_frame(dataset_features, obs_processed, prefix="observation")
  -> prepare_observation_for_inference(...)
  -> policy preprocessor
```

### 12.2 batch 到 raw chunk

```python
raw_chunk = policy.predict_action_chunk(batch, **rtc_kwargs)
```

输出形状:

```text
(B, T, action_dim)
```

实际 runtime 只支持 `B=1`。

### 12.3 raw chunk 到 processed chunk

```python
processed_chunk = postprocessor(raw_chunk)
```

需要离线测试确认 postprocessor 对 `(B, T, D)` chunk 保持形状。如果发现当前 processor 只支持 `(B, D)`，则实现一个 chunk wrapper:

```text
把 (B, T, D) reshape 成 (B*T, D)
postprocess
再 reshape 回 (B, T, D)
```

这个 wrapper 必须只放在 runtime 层，不改 checkpoint processor 文件。

## 13. 安全策略

真实机器人上第一版必须有安全约束:

1. `max_relative_target`
   - 继续使用 SO follower config 自带限制。

2. `max_action_delta`
   - 相邻发送 action 的最大关节变化量。
   - 超过则 stop。

3. `empty_queue_strategy`
   - `hold-last-action`: 默认，队列空时重复上一帧动作。
   - `skip-send`: 队列空时不发送。
   - `stop`: 队列空直接停止。

4. `first_chunk_timeout_s`
   - 启动后长期没有 action chunk，停止而不是盲动。

5. `drop_all_chunk_limit`
   - 连续多次 chunk 全部过期，说明模型太慢或 fps 太高，应停止。

6. `KeyboardInterrupt`
   - 主线程必须能可靠通知两个 worker 停止并 disconnect robot。

## 14. CLI 草案

入口:

```text
my_devs/train/pi/so101/rtc_pi05/run_rtc_pi05_infer.py
```

命令:

```bash
conda run --no-capture-output -n lerobot_flex \
python my_devs/train/pi/so101/rtc_pi05/run_rtc_pi05_infer.py \
    --policy-path outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
    --robot-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00 \
    --top-cam /dev/video4 \
    --wrist-cam /dev/video6 \
    --task "Put the eraser into the small box" \
    --fps 30 \
    --run-time-s 120 \
    --enable-rtc true \
    --queue-low-watermark 8 \
    --queue-target-size 24 \
    --rtc-execution-horizon 10 \
    --rtc-prefix-attention-schedule LINEAR \
    --rtc-max-guidance-weight 10.0 \
    --empty-queue-strategy hold-last-action
```

调试命令:

```bash
conda run --no-capture-output -n lerobot_flex \
python my_devs/train/pi/so101/rtc_pi05/run_rtc_pi05_infer.py \
    --policy-path outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
    --dry-run true
```

```bash
conda run --no-capture-output -n lerobot_flex \
python my_devs/train/pi/so101/rtc_pi05/run_rtc_pi05_infer.py \
    --policy-path outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
    --check-policy-load true
```

## 15. metrics

每隔 `metrics_log_interval_s` 打印:

```text
actor:
  actor_hz
  actor_tick_ms_p50/p95/max
  send_action_ms_p50/p95/max
  observation_ms_p50/p95/max

queue:
  depth
  low_watermark_hits
  empty_events
  hold_last_events
  dropped_chunks

inference:
  total_latency_ms_p50/p95/max
  preprocess_ms
  model_ms
  postprocess_ms
  predicted_delay_steps
  drop_steps
  cursor_delta_steps
  delay_mismatch_count

rtc:
  enabled
  execution_horizon
  prefix_attention_schedule
  leftover_len
  guidance debug count, if rtc_debug enabled
```

建议每次结束时写一份 JSON:

```text
my_devs/train/pi/so101/rtc_pi05/outputs/<timestamp>/runtime_summary.json
```

## 16. 分阶段实施计划

### 阶段 0: 文档与接口冻结

产物:

```text
docs/PI05_RTC_ASYNC_INFER_TECH_DESIGN.md
```

验收:

1. 明确线程模型。
2. 明确 raw/processed 双队列。
3. 明确 delay 口径。
4. 明确 RTC 调用方式。

### 阶段 1: 离线 runtime 骨架

实现:

```text
config.py
observation_buffer.py
action_chunk_queue.py
runtime_state.py
metrics.py
```

测试:

```bash
conda run --no-capture-output -n lerobot_flex \
pytest -q tests/my_devs/rtc_pi05/test_observation_buffer.py

conda run --no-capture-output -n lerobot_flex \
pytest -q tests/my_devs/rtc_pi05/test_action_chunk_queue.py
```

验收:

1. 多线程读写不会破坏 queue。
2. raw leftover 与 processed queue cursor 对齐。
3. plain append 与 RTC replace 语义清晰。
4. drop_steps 边界正确。

### 阶段 2: fake policy 集成

实现:

```text
inference_worker.py
actor_loop.py
```

用 fake policy 返回确定性 chunk，不连机器人。

测试:

```bash
conda run --no-capture-output -n lerobot_flex \
pytest -q tests/my_devs/rtc_pi05/test_inference_worker_fake_policy.py

conda run --no-capture-output -n lerobot_flex \
pytest -q tests/my_devs/rtc_pi05/test_actor_loop_fake_robot.py
```

验收:

1. actor loop 不等待 inference。
2. worker 只在低水位触发。
3. RTC kwargs 中的 `prev_chunk_left_over` 来自 raw queue。
4. `drop_steps` 按 observation timestamp 计算。

### 阶段 3: 真实 PI0.5 checkpoint 离线加载

实现:

```text
checkpoint_loader.py
```

测试:

```bash
conda run --no-capture-output -n lerobot_flex \
python my_devs/train/pi/so101/rtc_pi05/run_rtc_pi05_infer.py \
    --policy-path outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
    --check-policy-load true
```

验收:

1. full checkpoint 可加载。
2. LoRA checkpoint 可加载。
3. preprocessor / postprocessor 可加载。
4. 开启 RTC 后 `policy.config.rtc_config.enabled == True`。
5. 调用 `policy.predict_action_chunk(... rtc kwargs ...)` 不使用 `torch.inference_mode()`。

### 阶段 4: 真实机器人小步验收

端口安全说明:

```text
SO101 follower 推理输出端口: /dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00
不要把 /dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123582-if00 用作 follower 推理输出端口；
该端口在当前机器上对应主臂/leader，误用会导致主臂动作。
```

先跑 plain async:

```bash
conda run --no-capture-output -n lerobot_flex \
python my_devs/train/pi/so101/rtc_pi05/run_rtc_pi05_infer.py \
    --policy-path outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
    --robot-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00 \
    --top-cam /dev/video4 \
    --wrist-cam /dev/video6 \
    --task "Put the eraser into the small box" \
    --run-time-s 30 \
    --enable-rtc false
```

再跑 RTC:

```bash
conda run --no-capture-output -n lerobot_flex \
python my_devs/train/pi/so101/rtc_pi05/run_rtc_pi05_infer.py \
    --policy-path outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
    --robot-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00 \
    --top-cam /dev/video4 \
    --wrist-cam /dev/video6 \
    --task "Put the eraser into the small box" \
    --run-time-s 30 \
    --enable-rtc true \
    --rtc-execution-horizon 10
```

验收:

1. actor loop 不被模型推理阻塞。
2. 队列不会长期空。
3. 真实发送频率接近目标 fps。
4. RTC 模式下 raw leftover 长度和 drop_steps 合理。
5. 没有异常大 action delta。
6. Ctrl+C 能可靠断开机器人。

## 17. 关键风险与规避

### 17.1 `torch.inference_mode()` 破坏 RTC autograd

规避:

1. RTC path 不使用 `predict_action()` helper。
2. RTC path 不在外层包 `torch.inference_mode()`。
3. 单独写测试确认 RTC enabled 时能完成一次 `predict_action_chunk()`。

### 17.2 postprocessor 不支持 chunk 维度

规避:

1. 阶段 3 单独测试 `(1, T, D)` 输入。
2. 不支持时写 `ChunkPostprocessorAdapter`，reshape 后再恢复。

### 17.3 observation 太旧导致 chunk 大量过期

规避:

1. observation snapshot 带 timestamp。
2. `drop_steps` 用 `ready_s - obs_timestamp_s`。
3. 连续 drop all 时停止并提示降低 fps 或优化模型。

### 17.4 robot IO 线程安全

规避:

1. 第一版 `RobotIO` 用单锁包住 `get_observation()` 和 `send_action()`。
2. 如果 actor jitter 过大，再拆 camera/motor 锁。

### 17.5 RTC 参数不稳定

规避:

1. 先 plain async 验证 runtime。
2. RTC 从保守参数开始:

```text
execution_horizon = 10
max_guidance_weight = 10
schedule = LINEAR
```

3. 记录 guidance debug，但默认不打开，避免显存和日志膨胀。

## 18. 推荐第一版实现顺序

1. `config.py`
2. `observation_buffer.py`
3. `action_chunk_queue.py`
4. `checkpoint_loader.py`
5. `inference_worker.py`
6. `actor_loop.py`
7. `run_rtc_pi05_infer.py`
8. fake tests
9. checkpoint load smoke
10. real robot short run

第一版最重要的是架构边界正确，不追求一开始就把控制效果调到最好。

## 19. 最终形态

理想最终数据流:

```text
SO101 robot
  -> actor thread get_observation
  -> ObservationBuffer(latest obs + timestamp)
  -> inference worker reads snapshot
  -> policy preprocessor
  -> PI05Policy.predict_action_chunk(
         inference_delay=predicted_delay_steps,
         prev_chunk_left_over=raw_leftover,
         execution_horizon=rtc_execution_horizon,
     )
  -> policy postprocessor
  -> ActionChunkQueue.merge_rtc(drop stale prefix, replace executable queue)
  -> actor thread pop processed action
  -> make_robot_action
  -> robot_action_processor
  -> robot.send_action
```

这套结构里，`src/lerobot/policies/rtc` 负责模型层 realtime chunking，`rtc_pi05` 负责机器人实时运行时。两个边界清楚后，后续无论是接 TensorRT、远程 server，还是把 runtime 上移到通用 LeRobot，都有清晰的迁移点。
