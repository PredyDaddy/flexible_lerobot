# vlash_iner 隔离推理工程改造方案

日期：2026-06-03

目标目录：

- `/data/cqy_workspace/flexible_lerobot/my_devs/vla_engineering/vlash_iner`

相关参考：

- 训练脚本：`/data/cqy_workspace/flexible_lerobot/my_devs/train/pi/so101/easy_train.sh`
- 当前 PI0.5 上机推理脚本：`/data/cqy_workspace/flexible_lerobot/my_devs/train/pi/so101/run_pi05_infer.py`
- VLASH 异步推理参考：`/data/cqy_workspace/flexible_lerobot/my_devs/vla_engineering/vlash-main/vlash/run.py`
- VLASH 示例推理配置：`/data/cqy_workspace/flexible_lerobot/my_devs/vla_engineering/vlash-main/examples/inference/async.yaml`
- 当前数据集：`/data/cqy_workspace/flexible_lerobot/datasets/desk_cleanup_v1/eraser_cup_multi_task`

## 结论

可以按你的想法做，而且这是当前最推荐的工程路线：

> **训练继续使用 `my_devs/train/pi/so101/easy_train.sh` 产出当前 LeRobot PI0.5 checkpoint；推理优化和上机 runtime 单独放到 `my_devs/vla_engineering/vlash_iner/`；只借鉴 VLASH 的 runtime 思路，不改 `src/lerobot`。**

这条路线比“直接把 VLASH 训练/模型代码迁入当前仓库”更稳，也比“VLASH 独立训练再转换 checkpoint”少一个 checkpoint 兼容风险。

核心架构：

```text
easy_train.sh
  -> 当前 LeRobot PI0.5 checkpoint
  -> vlash_iner 同步推理
  -> vlash_iner 异步 chunk runtime
  -> vlash_iner 服务化推理 / 上机客户端
  -> SO101 机械臂
```

## 为什么这条路线更适合当前阶段

### 1. 训练产物和当前 PI0.5 loader 天然匹配

`easy_train.sh` 使用的是当前主仓库的 `lerobot-train --policy.type=pi05`：

```bash
lerobot-train \
  --dataset.repo_id=desk_cleanup_v1/eraser_cup_multi_task \
  --dataset.root=/data/cqy_workspace/flexible_lerobot/datasets/desk_cleanup_v1/eraser_cup_multi_task \
  --policy.type=pi05 \
  --policy.pretrained_path=/data/cqy_workspace/flexible_lerobot/assets/modelscope/lerobot/pi05_base \
  --policy.dtype=bfloat16 \
  --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}'
```

因此它输出的 checkpoint 应该优先由当前仓库的 PI0.5 loader 加载，而不是改用 VLASH 的 PI0.5 model class。

### 2. 不破坏 `src/lerobot`

所有新代码都放在：

```text
my_devs/vla_engineering/vlash_iner/
```

这样不会污染：

```text
src/lerobot/policies/pi05/
src/lerobot/robots/
src/lerobot/processor/
```

如果后续实验失败，可以直接回滚 `vlash_iner`，不会影响主框架。

### 3. 可复用现有上机代码

当前已经有一个可用参考：

```text
my_devs/train/pi/so101/run_pi05_infer.py
```

它已经包含：

- repo root 自动解析。
- 本地 tokenizer 检查。
- `PreTrainedConfig.from_pretrained(...)`。
- `get_policy_class(policy_cfg.type)`。
- `policy_class.from_pretrained(...)`。
- checkpoint 自带 preprocessor/postprocessor 加载。
- SO101 / SO100 follower robot config。
- camera config。
- 真实 robot observation -> policy inference -> action send。

`vlash_iner` 第一阶段应该复制并拆分这个脚本，而不是从 VLASH 重新开始。

### 4. 可复用 VLASH runtime 思想

VLASH 的价值主要在 runtime：

- current chunk / next chunk。
- `inference_overlap_steps`。
- `action_quant_ratio`。
- compile warmup。
- latency logging。
- chunk 切换。
- 未来可选 future-state-aware。

这些可以在 `vlash_iner` 内独立实现，不需要搬 VLASH policy/model。

## 命名说明

当前目录名是：

```text
my_devs/vla_engineering/vlash_iner
```

这里可能少了一个 `f`。如果这是误拼，建议后续改成：

```text
my_devs/vla_engineering/vlash_infer
```

但如果你已经决定用 `vlash_iner`，本方案先沿用现有目录名，避免无谓改名。

## 总体改造原则

1. **不改 `src/lerobot`。**
2. **不直接搬 VLASH policy/model。**
3. **优先复制当前稳定推理脚本，再拆模块。**
4. **先同步，后异步。**
5. **先本地推理，后服务化。**
6. **先低速上机，后 action quantization / future-state-aware。**
7. **每一步都保留 dry-run 和日志。**

## 推荐目录结构

第一版建议不要太复杂：

```text
my_devs/vla_engineering/vlash_iner/
  README.md
  run_pi05_sync.py
  run_pi05_async.py
  async_manager.py
  policy_loader.py
  robot_runtime.py
  safety.py
  logging_utils.py
  configs/
    so101_sync.yaml
    so101_async.yaml
  docs/
    上机流程.md
    风险清单.md
```

服务化阶段再增加：

```text
my_devs/vla_engineering/vlash_iner/
  run_infer_server.py
  run_robot_client.py
  service_protocol.py
  server_adapter.py
```

如果第一版想更快落地，可以先只建：

```text
my_devs/vla_engineering/vlash_iner/
  README.md
  run_pi05_sync.py
  async_manager.py
  run_pi05_async.py
```

## 分阶段改造方案

### 阶段 0：固定训练输出约定

目标：让 `vlash_iner` 能稳定找到 `easy_train.sh` 训练出来的 checkpoint。

当前 `easy_train.sh` 输出：

```text
/data/cqy_workspace/flexible_lerobot/outputs/pi05_eraser_cup_multi_task_runs/${RUN_ID}
```

建议训练完成后明确记录：

```text
OUTPUT_DIR/checkpoints/last/pretrained_model
```

或某个具体 step：

```text
OUTPUT_DIR/checkpoints/010000/pretrained_model
```

建议增加一个软链接约定：

```text
/data/cqy_workspace/flexible_lerobot/outputs/pi05_eraser_cup_multi_task_runs/latest/pretrained_model
```

如果暂时不改训练脚本，也可以在 `vlash_iner` 运行参数中显式传：

```bash
--policy-path /data/cqy_workspace/flexible_lerobot/outputs/pi05_eraser_cup_multi_task_runs/<RUN_ID>/checkpoints/last/pretrained_model
```

验收：

- `vlash_iner` 脚本通过 `--policy-path` 能定位 checkpoint。
- checkpoint 中存在 `config.json`、`model.safetensors`、`policy_preprocessor.json`、`policy_postprocessor.json`。

### 阶段 1：复制同步推理基线

目标：先在 `vlash_iner` 中复刻当前稳定上机推理能力。

建议直接复制：

```text
my_devs/train/pi/so101/run_pi05_infer.py
```

到：

```text
my_devs/vla_engineering/vlash_iner/run_pi05_sync.py
```

第一版复制是合理的，因为该脚本已经包含很多上机细节。复制后再逐步整理，不要一开始过度抽象。

需要保留的能力：

- `--dry-run`
- `--policy-path`
- `--robot-port`
- `--top-cam-index`
- `--wrist-cam-index`
- `--task`
- `--run-time-s`
- 本地 tokenizer 检查。
- checkpoint 自带 processor 加载。
- SO101/SO100 follower 支持。
- 低速安全运行。

需要新增的能力：

- `--max-action-delta`
- `--max-action-abs`
- `--warmup-steps`
- `--save-log-dir`
- latency 统计。
- action NaN/Inf 检查。

验收：

- `python .../run_pi05_sync.py --dry-run` 能打印完整配置。
- 能加载 `easy_train.sh` 产物 checkpoint。
- 能在不接机械臂时完成 policy load smoke test。
- 能低速真实上机执行短时间 smoke test。

### 阶段 2：拆出 policy_loader

目标：减少 `run_pi05_sync.py` 的复杂度，让 async/server 都复用同一套加载逻辑。

新增：

```text
my_devs/vla_engineering/vlash_iner/policy_loader.py
```

职责：

- resolve repo root。
- ensure local tokenizer。
- load config。
- load policy class。
- load checkpoint。
- load preprocessor/postprocessor。
- warmup policy。
- 设置 `eval()`。
- 设置 device。

建议接口：

```python
load_pi05_bundle(policy_path: Path, device: str | None = None) -> Pi05PolicyBundle
```

`Pi05PolicyBundle` 包含：

```text
policy
policy_cfg
preprocessor
postprocessor
device
policy_path
```

注意：

- 第一版不要改 `src/lerobot`。
- 不要自定义 PI0.5 model。
- 直接复用当前仓库 factory 和 checkpoint processor。

### 阶段 3：拆出 robot_runtime

目标：把 SO101 机器人、相机、observation frame、action send 逻辑隔离出来。

新增：

```text
my_devs/vla_engineering/vlash_iner/robot_runtime.py
```

职责：

- 构造 `SOFollowerRobotConfig`。
- 构造 `OpenCVCameraConfig`。
- connect/disconnect robot。
- get observation。
- build dataset frame。
- convert policy action to robot action。
- send action。

保留当前 `run_pi05_infer.py` 的实现细节，不要一开始换成 VLASH robot config。

验收：

- 同步脚本只负责流程，robot 细节在 `robot_runtime.py`。
- 后续 async/server client 可以复用。

### 阶段 4：增加 safety

目标：真实机械臂上机必须有动作安全层。

新增：

```text
my_devs/vla_engineering/vlash_iner/safety.py
```

检查：

- action 是否为空。
- action 是否包含 NaN/Inf。
- action shape 是否匹配 robot action dim。
- 单步 action delta 是否过大。
- action 绝对值是否超范围。
- 连续推理失败是否停机。
- 服务化阶段是否请求超时。

建议第一版策略：

- 超限直接拒绝发送。
- 连续 N 次异常退出主循环。
- dry-run 时只打印不发送。

### 阶段 5：仿照 VLASH 增加 async_manager

目标：迁移 VLASH 最有价值的 action chunk runtime 思想。

参考：

```text
vlash-main/vlash/run.py
```

主要借鉴：

- `VLASHAsyncManager.current_chunk`
- `VLASHAsyncManager.next_chunk`
- `VLASHAsyncManager.chunk_index`
- `should_launch_next_inference()`
- `should_switch_chunk()`
- `should_fetch_observation()`
- `get_action()`

新增：

```text
my_devs/vla_engineering/vlash_iner/async_manager.py
```

第一版建议：

- 只做 chunk overlap。
- 默认关闭 future-state-aware。
- 默认关闭 action quantization。
- 不引入线程，先在单线程主循环里验证。

接口建议：

```python
class AsyncChunkManager:
    def __init__(self, predict_chunk_fn, n_action_steps, overlap_steps)
    def should_fetch_observation(self) -> bool
    def get_action(self, observation_frame) -> np.ndarray
```

为什么先不打开 future-state-aware：

- 当前 PI0.5 processor 会把 state 离散后拼进 language prompt。
- 用当前 chunk 末尾 action 替换 state，会改变 prefix tokens。
- 这可能有效，但必须单独验证，不应作为第一版默认行为。

验收：

- 同步和异步输出 action shape 一致。
- chunk 切换无空洞。
- inference latency 有日志。
- overlap 设置为 0 时行为接近同步 chunk 执行。

### 阶段 6：run_pi05_async.py

目标：把 async manager 接到真实上机循环。

新增：

```text
my_devs/vla_engineering/vlash_iner/run_pi05_async.py
```

参数：

```text
--policy-path
--task
--robot-port
--top-cam-index
--wrist-cam-index
--fps
--n-action-steps
--inference-overlap-steps
--action-quant-ratio
--future-state-aware false
--dry-run
--run-time-s
--log-dir
```

第一版建议：

- `future-state-aware=false`
- `action_quant_ratio=1`
- `inference_overlap_steps=0` 先跑通。
- 再逐步调大 overlap。

验收：

- 低速真实机械臂 smoke test。
- 每个 action 发送时间有日志。
- 每次 inference 起止时间有日志。
- 退出时 robot disconnect。

### 阶段 7：服务化

目标：让 `vlash_iner` 既可以本机推理，也可以服务化推理。

新增：

```text
my_devs/vla_engineering/vlash_iner/run_infer_server.py
my_devs/vla_engineering/vlash_iner/run_robot_client.py
my_devs/vla_engineering/vlash_iner/service_protocol.py
```

服务端职责：

- 加载当前 LeRobot PI0.5 checkpoint。
- 接收 image/state/task。
- 构造 observation frame。
- 调用 policy 预测 action chunk。
- 返回 action list 和 latency。

客户端职责：

- 连接机械臂。
- 采集 camera/state。
- 请求服务端。
- 安全检查。
- 执行动作。

第一版协议可以参考 Realtime-VLA V2 的 pickle/FastAPI，但建议封装清楚：

```text
request:
  request_id
  timestamp
  task
  state
  images
  pending_actions

response:
  request_id
  action_list
  raw_action_list
  infer_time
  preprocess_time
  model_time
  postprocess_time
```

注意：

- 内网实验可以用 pickle。
- 长期建议改成更明确的协议，例如 msgpack 或 JSON metadata + binary image。

## 哪些代码可以直接复制

### 可以直接复制的代码

从 `run_pi05_infer.py` 复制：

- `resolve_repo_root`
- `parse_bool`
- `env_bool`
- `maybe_path`
- `build_parser` 的基础参数。
- `load_pre_post_processors`
- `ensure_local_tokenizer_dir`
- policy config / policy load 流程。
- SO follower robot config 构造。

从 `vlash-main/vlash/run.py` 复制或改写：

- `VLASHAsyncManager` 的状态机思路。
- `validate_robot_cameras`
- `warmup_compiled_policy` 的思路。
- `action_quant_ratio` 和 `inference_overlap_steps` 参数。

从 Realtime-VLA V2 复制或改写：

- `server/infer_server.py` 的 FastAPI 外壳。
- `client/local_client.py` 的请求记录思路。
- request/response latency logging 思路。

### 不建议直接复制的代码

不建议复制：

- `vlash-main/vlash/policies/pi05/modeling_pi05.py`
- `vlash-main/vlash/policies/pi05/configuration_pi05.py`
- Realtime-VLA V2 的 `server/pi05_infer.py`
- Realtime-VLA V2 的 Triton kernels
- Realtime-VLA V2 的 AIRBOT 专用 action reorder

原因：

- 你的训练 checkpoint 来自当前 LeRobot PI0.5。
- 直接换模型实现会引入 checkpoint/key/normalization/state_cond 差异。
- Triton backend 是后期性能专项，不适合第一版。

## 风险清单

### 风险 1：`vlash_iner` 和训练 checkpoint 的 processor 不一致

应对：

- 必须从 checkpoint 加载 `policy_preprocessor.json` 和 `policy_postprocessor.json`。
- 不要手写 normalization。

### 风险 2：camera 名称不一致

`easy_train.sh` 使用的数据集 camera schema 必须和上机 camera schema 对齐。

应对：

- 在 `run_pi05_sync.py --dry-run` 中打印 policy image features 和 robot cameras。
- 复用或改写 VLASH 的 `validate_robot_cameras`。

### 风险 3：异步调度引入旧 observation

应对：

- 第一版 overlap=0。
- 再逐步增加 overlap。
- 记录 observation timestamp 和 action send timestamp。

### 风险 4：future-state-aware 改变 PI0.5 prompt

应对：

- 默认关闭。
- 单独做 A/B 测试。
- 只有任务成功率和稳定性都更好时才启用。

### 风险 5：action quantization 降低精细控制能力

应对：

- 默认 `action_quant_ratio=1`。
- 只在执行端成为瓶颈时启用。

### 风险 6：服务化网络延迟

应对：

- 先本机服务。
- 记录 `roundtrip_latency` 和 `server_infer_time`。
- 超时停机或保持安全动作。

## 推荐里程碑

### M1：同步推理脚本落地

完成条件：

- `run_pi05_sync.py --dry-run` 成功。
- 能加载 `easy_train.sh` checkpoint。
- 能打印 policy features、robot cameras、action dim。

### M2：同步低速上机

完成条件：

- SO101 低速执行。
- action safety 检查生效。
- 日志完整。

### M3：异步状态机离线验证

完成条件：

- mock observation 下 chunk 切换正常。
- overlap=0 等价于同步 chunk。
- overlap>0 无 action 空洞。

### M4：异步低速上机

完成条件：

- `run_pi05_async.py` 可运行。
- inference latency 和 action send timestamp 可记录。
- 机械臂动作稳定。

### M5：服务化本机验证

完成条件：

- `run_infer_server.py` 加载 checkpoint。
- `run_robot_client.py` 能请求 action。
- mock robot / dry-run 可稳定运行。

### M6：服务化真实上机

完成条件：

- 真实 SO101 通过服务拿 action。
- 服务超时保护有效。
- 日志可回放。

## 优先级

P0：复制 `run_pi05_infer.py` 到 `vlash_iner/run_pi05_sync.py`，先跑 dry-run。

P1：补充 checkpoint processor 加载、tokenizer 检查、camera schema 打印。

P2：加入 safety 和 latency logging。

P3：拆 `policy_loader.py`、`robot_runtime.py`。

P4：加入 `async_manager.py`，仿 VLASH 做 chunk overlap。

P5：写 `run_pi05_async.py`。

P6：服务化 `run_infer_server.py` / `run_robot_client.py`。

P7：再考虑 compile warmup、action quantization、future-state-aware。

P8：最后再考虑 QKV/MLP fusion 或 Triton。

## 第一版建议命令形态

同步 dry-run：

```bash
conda run -n lerobot_flex python \
  my_devs/vla_engineering/vlash_iner/run_pi05_sync.py \
  --policy-path /data/cqy_workspace/flexible_lerobot/outputs/pi05_eraser_cup_multi_task_runs/<RUN_ID>/checkpoints/last/pretrained_model \
  --task "clean the desk" \
  --dry-run true
```

同步上机：

```bash
conda run -n lerobot_flex python \
  my_devs/vla_engineering/vlash_iner/run_pi05_sync.py \
  --policy-path /data/cqy_workspace/flexible_lerobot/outputs/pi05_eraser_cup_multi_task_runs/<RUN_ID>/checkpoints/last/pretrained_model \
  --robot-port /dev/ttyACM0 \
  --top-cam-index 4 \
  --wrist-cam-index 6 \
  --task "clean the desk" \
  --run-time-s 60
```

异步上机：

```bash
conda run -n lerobot_flex python \
  my_devs/vla_engineering/vlash_iner/run_pi05_async.py \
  --policy-path /data/cqy_workspace/flexible_lerobot/outputs/pi05_eraser_cup_multi_task_runs/<RUN_ID>/checkpoints/last/pretrained_model \
  --robot-port /dev/ttyACM0 \
  --top-cam-index 4 \
  --wrist-cam-index 6 \
  --task "clean the desk" \
  --inference-overlap-steps 4 \
  --future-state-aware false \
  --run-time-s 60
```

## 最终建议

当前不要先动 `src/lerobot`，也不要先把 VLASH policy 搬过来。

最优路径是：

1. `easy_train.sh` 继续负责训练。
2. `vlash_iner/run_pi05_sync.py` 复制当前稳定上机脚本，先跑通同步推理。
3. `vlash_iner/async_manager.py` 借鉴 VLASH 的 chunk overlap。
4. `vlash_iner/run_pi05_async.py` 做隔离异步上机。
5. 后续再做服务化。

一句话总结：

> **`vlash_iner` 应该成为“当前 LeRobot PI0.5 checkpoint 的隔离推理工程”，不是 VLASH 模型代码的搬运目录。能直接复制的，是当前稳定上机脚本和 VLASH runtime 状态机；不要复制 VLASH PI0.5 model，也不要改 `src/lerobot`。**
