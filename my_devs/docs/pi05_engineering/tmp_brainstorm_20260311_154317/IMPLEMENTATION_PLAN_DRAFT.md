# PI05 实时推理工程化实施 Plan

本稿已根据 `reviews/review_01.md`、`reviews/review_02.md`、`reviews/review_03.md` 的独立评审意见完成收敛修订。

## 1. Plan 目标

本 Plan 只解决一件事：

- 把当前已经能跑通的 `run_pi05_infer.py`，升级出一条 **第一阶段可用的 PI05 本地 chunk 实时运行时**

并且满足以下约束：

1. 保留当前 checkpoint 的已知可用加载路径。
2. 第一阶段默认开发放在 `my_devs/`。
3. 第一阶段先落地本地 chunk runtime，不直接改通用 `async_inference`。
4. 第一阶段先把 plain chunk 跑稳，再接 RTC。
5. 保留原同步脚本作为 fallback，不在第一阶段破坏它。

## 2. 第一阶段明确交付物

第一阶段结束时，应当交付：

1. 一个新的 PI05 本地实时入口脚本。
2. 一套位于 `my_devs/` 下的 runtime 模块。
3. 明确的 queue / latency / starvation 运行指标。
4. 可关闭的 RTC plumbing，但默认先不开 RTC。
5. 原 `run_pi05_infer.py` 仍然可运行，作为基线与回退路径。

第一阶段不要求交付：

1. gRPC server/client 版本。
2. 通用多 policy runtime。
3. 对 `src/lerobot/async_inference/` 的正式改造。
4. 对 `src/lerobot/policies/pi05/` 的模型结构修改。

## 3. 推荐文件布局

推荐在 `my_devs/train/pi/so101/` 下新增如下结构：

```text
my_devs/train/pi/so101/
├── run_pi05_infer.py                     # 保留现有同步 baseline
├── run_pi05_chunk_infer.py               # 新增：第一阶段主入口
└── pi05_runtime/
    ├── __init__.py
    ├── common.py                         # 共享初始化与加载逻辑
    ├── runtime_config.py                 # runtime 专用参数与默认值
    ├── runtime_state.py                  # 运行态 dataclass
    ├── queue_controller.py               # ActionQueue 封装与模式切换
    ├── chunk_inference.py                # chunk 推理链路
    ├── actor_loop.py                     # 动作执行线程
    ├── producer_loop.py                  # observation + chunk refill 线程
    ├── metrics.py                        # 运行指标与日志聚合
    └── rtc_debug.py                      # RTC debug 数据导出（可选）
```

说明：

- 第一阶段所有新增代码都放在 `my_devs/`。
- `src/` 下优先复用现有实现，不主动新增通用层。
- 如果第一阶段验证稳定，再决定第二阶段是否把部分模块抽回 `src/`。
- 这个布局是推荐上限，不要求第一版就拆到最细。
- 如果实现初期为了尽快跑通，需要先合并 `actor_loop`、`producer_loop`、`queue_controller` 到较少文件，可以接受；前提是接口边界和职责不要丢。

## 4. 第一阶段明确不改的文件

以下内容在第一阶段应明确不动，避免把问题面放大：

### 不改 `src/lerobot/async_inference/`

原因：

- 当前目标是先把本地 chunk runtime 语义跑稳。
- 过早切进 gRPC / server-client，只会增加排障复杂度。

### 不改 `src/lerobot/policies/pi05/modeling_pi05.py`

原因：

- 当前模型已经支持 `predict_action_chunk()` 与 RTC kwargs。
- 现在缺的是运行时，不是模型结构。

### 不重写 checkpoint processor

不能在第一阶段自己重写：

- preprocessor
- postprocessor

必须沿用 checkpoint 自带的：

- `policy_preprocessor.json`
- `policy_postprocessor.json`

### 不删除或强改 `run_pi05_infer.py`

原因：

- 它是当前唯一已知可用的 baseline。
- 后续验证必须拿它做对照。

## 5. 第一阶段具体改造顺序

## Phase 0: 基线固化

### 目标

先把当前 baseline 固定住，作为后续对照。

### 动作

1. 不改 `my_devs/train/pi/so101/run_pi05_infer.py` 的主行为。
2. 记录当前默认 checkpoint 路径、机器人端口、相机 index、task。
3. 记录当前脚本日志格式，作为对比基线。

### 验证

在 `lerobot_flex` 环境下执行：

```bash
conda run -n lerobot_flex python my_devs/train/pi/so101/run_pi05_infer.py --dry-run true
```

通过条件：

- repo root 可解析
- policy path 存在
- tokenizer 本地目录存在
- robot/camera 配置可以完成 dry-run 解析

## Phase 1: 提取共享初始化逻辑

### 目标

把当前同步脚本里“脆弱但已知可用”的部分提到共享模块，避免新脚本复制时走样。

### 新增文件

- `my_devs/train/pi/so101/pi05_runtime/common.py`

### 建议迁移/封装内容

1. `resolve_repo_root(...)`
2. `parse_bool(...)`
3. `env_bool(...)`
4. `maybe_path(...)`
5. `ensure_local_tokenizer_dir(...)`
6. `load_pre_post_processors(...)`
7. robot config 构建逻辑
8. policy 加载逻辑
9. dataset feature 构建逻辑

### 明确要求

policy 加载必须保留：

```python
policy_class.from_pretrained(str(policy_path), strict=False)
```

不能在第一阶段把它改回严格加载。

### 验证

使用 baseline 和新模块做 dry-run 比对：

```bash
conda run -n lerobot_flex python my_devs/train/pi/so101/run_pi05_infer.py --dry-run true
```

如果后续把 baseline 也切到共享模块，再补：

```bash
conda run -n lerobot_flex python my_devs/train/pi/so101/run_pi05_infer.py --dry-run true
```

通过条件：

- 新旧 dry-run 输出关键路径一致
- checkpoint processor 路径一致
- tokenizer 兜底逻辑一致

## Phase 2: 搭建 plain chunk runtime 骨架

### 目标

先让 PI05 从单步路径切到 chunk 路径，但先不开 RTC。

### 新增文件

- `my_devs/train/pi/so101/run_pi05_chunk_infer.py`
- `my_devs/train/pi/so101/pi05_runtime/runtime_config.py`
- `my_devs/train/pi/so101/pi05_runtime/runtime_state.py`
- `my_devs/train/pi/so101/pi05_runtime/chunk_inference.py`
- `my_devs/train/pi/so101/pi05_runtime/actor_loop.py`
- `my_devs/train/pi/so101/pi05_runtime/producer_loop.py`

### 具体实现要求

#### `run_pi05_chunk_infer.py`

负责：

- CLI 入口
- 初始化 runtime config
- 创建 robot / policy / processors
- 启动线程
- 统一 shutdown

建议新增参数：

- `--queue-low-watermark`
- `--actions-per-chunk`
- `--enable-rtc`
- `--metrics-log-interval`
- `--dry-run`
- `--run-time-s`
- `--startup-wait-for-first-chunk`
- `--empty-queue-strategy`
- `--max-action-delta`

其中 `--actions-per-chunk` 的语义要写死：

- 它用于限制“实际投入执行队列的动作前缀长度”
- 不要求修改 checkpoint 的 `chunk_size`
- 默认允许模型继续生成完整 chunk，但第一阶段 rollout 可以只执行更短前缀

#### `chunk_inference.py`

负责：

1. `robot.get_observation()`
2. robot observation processor
3. `build_dataset_frame(...)`
4. tensor 转换与 batch 维度处理
5. `preprocessor(...)`
6. `policy.predict_action_chunk(...)`
7. `postprocessor(...)`
8. 返回：
   - `original_actions`
   - `processed_actions`

这里要保持一条严格边界：

- RTC leftover 相关逻辑只使用 `original_actions`
- 机器人执行只使用 `processed_actions`
- 不能从 `processed_actions` 反推 `original_actions`

并且第一阶段建议允许：

- 模型生成完整 chunk
- 但只把前 `actions-per-chunk` 步压入执行队列

这样 rollout horizon 可控，不需要修改 checkpoint 的 `chunk_size=50`。

### 验证

先只做 dry-run 和无机器人逻辑检查，再做短时间实跑。

建议最低验证命令：

```bash
conda run -n lerobot_flex python my_devs/train/pi/so101/run_pi05_chunk_infer.py --dry-run true
```

通过条件：

- 新入口可解析
- 共享加载逻辑可复用
- `predict_action_chunk()` 路径接通

## Phase 3: 引入显式 queue controller

### 目标

把 action queue 从 policy 内部隐式队列，迁到 runtime 外部显式控制。

### 新增文件

- `my_devs/train/pi/so101/pi05_runtime/queue_controller.py`

### 实现原则

第一阶段不建议自写一套完全新 queue，而是优先封装：

- `src/lerobot/policies/rtc/action_queue.py`

建议做一个轻包装，统一暴露：

1. `qsize()`
2. `empty()`
3. `get_action_index()`
4. `get_left_over_original_actions()`
5. `merge_plain(...)`
6. `merge_rtc(...)`
7. `pop_next_action()`

### 模式语义

#### plain chunk 模式

- RTC disabled
- 新 chunk 进入后按 append 语义合入队列
- 如果推理完成太晚，已经过期的 prefix 动作必须被修剪并记录日志
- `pop_next_action()` 返回空时，行为必须配置化且默认安全

#### RTC 模式

- RTC enabled
- 按 `real_delay` 做 replace 语义

### 验证

这一阶段至少要补本地单元级验证：

- 空队列行为
- append 行为
- replace 行为
- `last_index` 前进后 leftover 是否正确

如果当前仓库里没有对应 test 文件，可新增：

- `tests/test_pi05_runtime_queue_controller.py`

注意：

- 若测试依赖真实 robot 或 checkpoint，则用 mock / fake tensor 做快速测试

## Phase 3.5: 启动期与安全门槛落地

### 目标

把实机运行时最基本的安全 contract 写死，避免后续实现出现隐式行为。

### 必须落实的规则

1. 首个 chunk 未 ready 前，actor thread 不允许发送未定义动作。
2. `empty_queue_strategy` 必须明确，推荐默认：
   - `hold-last-action`
   - 若没有 last action，则 `skip-send + warning`
3. 每步动作发送前应预留简单 sanity check：
   - `max_action_delta`
   - 越界则拒绝发送并报警
4. Ctrl+C、线程异常、queue 异常时，必须统一走：
   - stop loop
   - robot disconnect
   - 明确日志

### 建议新增/修改文件

- `my_devs/train/pi/so101/pi05_runtime/actor_loop.py`
- `my_devs/train/pi/so101/pi05_runtime/runtime_state.py`
- `my_devs/train/pi/so101/pi05_runtime/runtime_config.py`

### 验证

至少验证：

1. 首个 chunk ready 前不会误发动作
2. 空队列时行为符合配置
3. 异常中断时 robot 能安全退出

## Phase 4: 指标与日志落地

### 目标

把“能跑”变成“可证明在正确地跑”。

### 新增文件

- `my_devs/train/pi/so101/pi05_runtime/metrics.py`

### 必须记录的指标

1. control loop step
2. actor loop 实际频率
3. chunk inference latency
4. preprocess / model / postprocess 分段耗时
5. queue depth
6. queue starvation 次数
7. queue refill 次数
8. 当次 chunk 长度
9. `action_index_before_inference`
10. `real_delay`
11. leftover 长度
12. stale-prefix trimming 次数
13. 首个 chunk ready 时间
14. 首次 warmup latency

其中第 4 项必须保持为强制项，不是可选优化项：

- preprocess
- model forward / denoising
- postprocess

特别是 postprocess 耗时必须单独记录，不能只并入总 latency。

### 日志要求

建议两种输出：

1. 终端简洁日志
2. 可选结构化日志文件（jsonl 或普通文本均可）

### 验证

在实机短跑中确认：

- queue depth 会动态变化
- starvation 能被准确统计
- latency 非负且数值合理
- actor loop 频率接近目标 FPS
- 启动期指标与稳态指标可区分

## Phase 5: plain chunk 实机稳定性验证

### 目标

在不开 RTC 的前提下，验证 chunk runtime 是否已经优于同步 baseline。

### 建议执行

先 baseline：

```bash
conda run -n lerobot_flex python my_devs/train/pi/so101/run_pi05_infer.py
```

再 plain chunk：

```bash
conda run -n lerobot_flex python my_devs/train/pi/so101/run_pi05_chunk_infer.py --enable-rtc false
```

### 对比项

1. 是否持续稳定输出动作
2. queue 是否出现频繁清空
3. actor loop 频率是否更稳定
4. chunk latency 是否可接受
5. 实机动作是否有明显停顿
6. `actions-per-chunk` 调小后是否更稳定

### 通过条件

- plain chunk 模式下没有明显退化
- starvation 显著少于同步模式
- 能够稳定跑至少 60 秒
- 稳态 actor loop 平均频率达到目标 FPS 的 90% 以上
- 首个 chunk 未 ready 前没有误发动作

## Phase 6: 接 RTC plumbing，但默认不开

### 目标

先把 RTC 运行时所需状态链路接完整，但先不默认开启 guidance。

### 新增文件

- `my_devs/train/pi/so101/pi05_runtime/rtc_debug.py`

### 要接入的状态

1. `RTCConfig`
2. `action_index_before_inference`
3. `prev_chunk_left_over`
4. `inference_delay`
5. `real_delay`
6. `execution_horizon`

### 运行时逻辑

在 producer loop 中：

1. 记录开始推理前的 `action_index_before_inference`
2. 从 queue 拿：
   - `prev_chunk_left_over`
3. 用最近 latency 估算：
   - `inference_delay = ceil(latency / dt)`
4. 调：

```python
policy.predict_action_chunk(
    preprocessed_obs,
    inference_delay=inference_delay,
    prev_chunk_left_over=prev_chunk_left_over,
)
```

5. 结束后计算：
   - `real_delay = ceil(new_latency / dt)`
6. 调 queue 的 RTC merge 路径

这里建议把实现优先级写死：

1. `inference_delay = ceil(latency / dt)` 只用于推理开始前的预测
2. queue merge 时，优先使用 `current_action_index - action_index_before_inference` 作为 authoritative `real_delay`
3. `ceil(new_latency / dt)` 作为交叉校验值
4. 若 index-based delay 与 wall-clock delay 偏差过大，必须 warning

这里再补一条判定优先级：

- 用于 queue merge 的权威 delay 信号，优先取 `action_index_before_inference` 到推理完成后的 action index 差值
- `ceil(latency / dt)` 推导出的 delay 主要用于估计、预热和监控
- 如果两者不一致，优先按 action index 真值 merge，并把偏差写日志

另外这里必须补一条实现要求：

- 如果要真正让 RTC 生效，必须显式启用 `RTCConfig`
- 不能只在 `predict_action_chunk(...)` 时传 runtime kwargs
- 若 checkpoint 原始配置 `rtc_config = null`，则需要在 policy 初始化前或加载后受控地启用 RTC 配置并重新完成对应初始化

### 这里明确不要做的事

- 不要同时改模型逻辑
- 不要同时切到 async server/client
- 不要把 RTC 和 plain chunk 混成一个不透明代码路径

### 验证

不开 RTC 时：

- 相关状态字段存在
- 但不会影响 plain chunk 结果

开 RTC 前：

- 所有日志可见
- `real_delay` 和 `action_index_before_inference` 可以对齐核对
- queue index delta 与 wall-clock latency 两套延迟指标同时可见

## Phase 7: RTC 保守启用

### 目标

在 plain chunk 已经稳定的前提下，保守地打开 RTC。

### 推荐初始参数

- `--enable-rtc true`
- `--actions-per-chunk 8`
- `--rtc-execution-horizon 8`
- `--rtc-max-guidance-weight 10.0`
- `--rtc-prefix-attention-schedule EXP`
- `--rtc-debug false`

### 要求

必须保留一键回退：

- 只要把 `--enable-rtc false` 切回去，就能退回 plain chunk 模式

### 验证

关注以下现象：

1. chunk 切换时是否更平滑
2. 是否出现突然跳变
3. queue 是否被错误替换
4. 是否出现 runaway motion
5. RTC on/off 的差异是否可解释

### 通过条件

- 开启 RTC 后运行稳定
- 能看到连续性改善，或至少不退化
- 出问题时可以立即回退 plain chunk

## 6. 第二阶段预留，不在当前实现

第二阶段才考虑这些事情：

1. 基于当前本地 runtime 语义，包装成：
   - PI05 专用 `policy_server` 启动器
   - PI05 专用 `robot_client` 启动器
2. 再决定是否扩展：
   - `src/lerobot/async_inference/policy_server.py`
   - `src/lerobot/async_inference/robot_client.py`
3. 再决定是否把 queue / delay / RTC state 抽成通用层

当前实现阶段不要做这些修改。

## 7. 推荐修改清单

### 第一阶段新增

必须新增：

- `my_devs/train/pi/so101/run_pi05_chunk_infer.py`
- `my_devs/train/pi/so101/pi05_runtime/__init__.py`
- `my_devs/train/pi/so101/pi05_runtime/common.py`
- `my_devs/train/pi/so101/pi05_runtime/runtime_config.py`
- `my_devs/train/pi/so101/pi05_runtime/runtime_state.py`
- `my_devs/train/pi/so101/pi05_runtime/queue_controller.py`
- `my_devs/train/pi/so101/pi05_runtime/chunk_inference.py`
- `my_devs/train/pi/so101/pi05_runtime/actor_loop.py`
- `my_devs/train/pi/so101/pi05_runtime/producer_loop.py`
- `my_devs/train/pi/so101/pi05_runtime/metrics.py`

建议新增：

- `my_devs/train/pi/so101/pi05_runtime/rtc_debug.py`

可选新增测试：

- `tests/test_pi05_runtime_queue_controller.py`
- `tests/test_pi05_runtime_common.py`

### 第一阶段允许小改

可以小改：

- `my_devs/train/pi/so101/run_pi05_infer.py`

但仅限：

- 引入共享 helper
- 补注释
- 不改变 baseline 主行为

### 第一阶段禁止大改

不应大改：

- `src/lerobot/async_inference/*`
- `src/lerobot/policies/pi05/*`
- `src/lerobot/utils/control_utils.py`

## 8. 验证命令建议

所有命令都应在 `lerobot_flex` 环境下运行。

### Dry-run

```bash
conda run -n lerobot_flex python my_devs/train/pi/so101/run_pi05_chunk_infer.py --dry-run true
```

### Plain chunk

```bash
conda run -n lerobot_flex python my_devs/train/pi/so101/run_pi05_chunk_infer.py --enable-rtc false
```

### RTC on

```bash
conda run -n lerobot_flex python my_devs/train/pi/so101/run_pi05_chunk_infer.py \
  --enable-rtc true \
  --rtc-execution-horizon 8 \
  --rtc-max-guidance-weight 10.0 \
  --rtc-prefix-attention-schedule EXP
```

### 快速测试

如果新增了本地测试：

```bash
conda run -n lerobot_flex pytest -q tests/ -k pi05_runtime
```

## 9. Phase 1 完成定义

Phase 1 完成的标准不是“代码写完”，而是以下条件全部成立：

1. `run_pi05_chunk_infer.py` 可以在 `lerobot_flex` 里启动。
2. 它使用 `predict_action_chunk()`，而不是 `select_action()`。
3. 它有显式 queue，而不是依赖 policy 内部队列。
4. queue / latency / starvation 指标可见。
5. plain chunk 模式已经实机稳定。
6. RTC 所需状态链路已经接通。
7. 原 `run_pi05_infer.py` 仍然可作为 fallback 跑通。
8. 当前技术方案与实施 Plan 文档都与代码实现方向一致。

## 10. 一句话执行顺序

实际执行时，严格按下面顺序推进，不要跳步：

1. 固化 baseline
2. 抽共享初始化
3. 新建 plain chunk runtime
4. 引入显式 queue controller
5. 落指标与日志
6. 验 plain chunk
7. 接 RTC plumbing
8. 保守开启 RTC
9. 通过后再考虑第二阶段 async 化
