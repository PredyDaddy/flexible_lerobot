# VLASH 训练与服务化上机方案

日期：2026-06-02

目标：

- 在 `vlash-main` 中独立训练 `datasets/desk_cleanup_v1` 对应的 PI0.5 / VLASH 模型。
- 训练完成后不急着把 VLASH 代码迁入当前主仓库。
- 通过服务化方式把 VLASH 推理结果发送给机械臂。
- 当前主仓库主要承担上机、机器人 I/O、安全控制、日志与客户端运行时。

相关路径：

- 当前主仓库：`/data/cqy_workspace/flexible_lerobot`
- VLASH：`/data/cqy_workspace/flexible_lerobot/my_devs/vla_engineering/vlash-main`
- Realtime-VLA V2 服务参考：`/data/cqy_workspace/flexible_lerobot/my_devs/vla_engineering/realtime-vla-v2-main`
- 数据集根目录：`/data/cqy_workspace/flexible_lerobot/datasets/desk_cleanup_v1`
- 已发现子数据集：`/data/cqy_workspace/flexible_lerobot/datasets/desk_cleanup_v1/eraser_cup_multi_task`

## 总体判断

这条路线是可行的，而且比“先把 VLASH 异步/训练/融合代码迁入当前仓库”更适合当前阶段。

推荐架构：

```text
datasets/desk_cleanup_v1
        |
        v
VLASH 独立训练环境
        |
        v
VLASH checkpoint / merged checkpoint
        |
        v
推理服务端：加载 VLASH checkpoint，接收 observation，返回 action chunk
        |
        v
当前主仓库上机客户端：采集相机和状态，调用推理服务，发送动作给机械臂
```

核心边界：

- **VLASH 训练端**：负责数据读取、训练、LoRA/async/shared observation、checkpoint 产出。
- **推理服务端**：负责加载训练好的 checkpoint，执行 `predict_action_chunk`，返回 action list。
- **当前主仓库上机端**：负责机械臂、相机、状态采集、请求服务、动作执行、安全和日志。

这让训练、推理服务、上机控制三件事保持相对隔离。

## 两个系统分别怎么做

### 系统 1：VLASH 训练系统

VLASH 训练系统只回答一个问题：

> 用 `desk_cleanup_v1` 能不能训练出一个对任务有效的 PI0.5 checkpoint？

它应该独立于当前主仓库上机逻辑运行。

建议工作内容：

1. 建立 VLASH 独立环境。
2. 检查 `desk_cleanup_v1/eraser_cup_multi_task` 是否能被 VLASH dataset loader 读取。
3. 基于 `vlash-main/examples/train/pi05/sync.yaml` 或 `async.yaml` 改一份任务配置。
4. 先跑最小训练 smoke test。
5. 再跑正式训练。
6. 在 VLASH 内做离线推理验证。
7. 产出 checkpoint 和训练时 normalization/config 元数据。

第一阶段不要关心机械臂服务调用。

### 系统 2：服务化上机系统

服务化上机系统只回答一个问题：

> 机械臂端能不能通过网络请求拿到 action chunk，并稳定、安全地执行？

它不负责训练。

建议工作内容：

1. 服务端加载 VLASH checkpoint。
2. 客户端从当前机器人框架采集 observation。
3. 客户端把 observation 编码成请求。
4. 服务端返回 action chunk。
5. 客户端执行 action chunk，并记录延迟、队列长度、动作轨迹。

服务端可以参考 Realtime-VLA V2：

- `realtime-vla-v2-main/server/infer_server.py`
- `realtime-vla-v2-main/server/model.py`
- `realtime-vla-v2-main/client/local_client.py`

但第一版不建议直接使用 Realtime-VLA V2 的 Triton backend。更稳的第一版是：

> **服务端先用 PyTorch/VLASH checkpoint 做推理，等服务链路稳定后再考虑 Triton/CUDA Graph。**

## 推荐阶段路线

### 阶段 0：数据集与 schema 检查

目标：确认 `desk_cleanup_v1/eraser_cup_multi_task` 能作为 VLASH 训练输入。

需要检查：

- `meta/info.json`
- `meta/stats.json`
- `meta/tasks.parquet`
- episode 数量
- video/image 路径
- `observation.images.*` 字段
- `observation.state` 维度
- `action` 维度
- task 文本
- fps
- camera 名称是否和上机一致

验收标准：

- VLASH dataset loader 能构造 batch。
- batch 中 image/state/action/task 字段都存在。
- state/action 维度与机械臂真实接口一致。
- camera 命名有明确映射。

### 阶段 1：VLASH 独立环境与最小训练

目标：让 VLASH 先自己训练起来。

建议环境：

```bash
conda create -n vlash_desk_cleanup python=3.10 -y
conda activate vlash_desk_cleanup
cd /data/cqy_workspace/flexible_lerobot/my_devs/vla_engineering/vlash-main
pip install -e .
pip install -U torch torchvision torchcodec
```

说明：

- 当前主仓库开发和脚本执行仍使用 `lerobot_flex`。
- VLASH 训练环境可以单独建，因为它是隔离训练系统。
- 训练环境版本要记录下来，尤其是 torch、transformers、safetensors、tokenizers、peft、bitsandbytes、CUDA。

建议先改一份配置：

```text
vlash-main/examples/train/pi05/desk_cleanup_async.yaml
```

参考：

- `vlash-main/examples/train/pi05/sync.yaml`
- `vlash-main/examples/train/pi05/async.yaml`
- `vlash-main/examples/train/pi05/async_lora.yaml`

第一版建议：

- 先不要开太多功能。
- 先跑几十步 smoke test。
- smoke test 成功后再开 LoRA / async / shared observation。

建议功能顺序：

1. `sync.yaml` 风格最小训练。
2. LoRA 训练。
3. async 训练。
4. shared observation。
5. QLoRA。

不建议第一版同时打开所有能力。

### 阶段 2：VLASH 离线推理验证

目标：服务化之前，先确认 checkpoint 本身能推理。

验证内容：

- 加载训练 checkpoint。
- 从 `desk_cleanup_v1` 抽取若干 observation。
- 调用 `policy.predict_action_chunk(...)`。
- 检查 action shape。
- 检查 action 数值范围。
- 检查 unnormalize 后 action 是否接近真实机器人动作范围。
- 记录单次推理 latency。
- 记录 warmup 后 p50/p95/p99。

验收标准：

- action chunk shape 正确。
- action 数值没有明显爆炸。
- latency 可记录、可复现。
- checkpoint 可以稳定重复加载。

如果这一步失败，不进入服务化。

### 阶段 3：服务端设计

目标：把训练好的 VLASH checkpoint 包成一个推理服务。

第一版服务端建议：

```text
POST /infer

request:
{
  "images": {
    "wrist": <jpeg bytes 或 base64>,
    "front": <jpeg bytes 或 base64>
  },
  "state": [float, ...],
  "task": "clean the desk ...",
  "timestamp": float,
  "request_id": string,
  "pending_actions": int
}

response:
{
  "action_list": [[float, ...], ...],
  "raw_action_list": [[float, ...], ...],
  "infer_time": float,
  "model_time": float,
  "preprocess_time": float,
  "postprocess_time": float,
  "request_id": string
}
```

是否使用 pickle：

- Realtime-VLA V2 现在使用 pickle RPC，见 `server/infer_server.py` 和 `client/local_client.py`。
- 内网实验阶段可以用 pickle，开发最快。
- 长期建议换成更明确的协议，比如 msgpack / JSON metadata + binary image。

第一版可以参考 Realtime-VLA V2 的 FastAPI 结构：

```text
server/infer_server.py
InferPipeline.__call__()
model_adapter.infer_actions(request)
optimizer.optimize(raw_actions)
```

但 model adapter 应该换成 VLASH PyTorch adapter：

```text
VLASHPi05Adapter
  - load checkpoint
  - warmup
  - decode images
  - prepare observation
  - policy.predict_action_chunk
  - return action_list
```

第一版不建议做：

- Triton backend
- CUDA Graph
- MPC optimizer
- complex action prefill

这些都后置。

### 阶段 4：客户端设计

目标：当前主仓库作为机械臂客户端，采集 observation，调用服务，执行动作。

客户端职责：

- 读取相机。
- 读取机械臂 state。
- 编码图片。
- 请求 `/infer`。
- 接收 action chunk。
- 做安全检查。
- 按控制频率发送 action。
- 记录日志。

可以参考 Realtime-VLA V2：

- `client/local_client.py`
- `client/robot_io.py`
- `client/executor.py`

建议第一版客户端逻辑：

```text
主循环：
  1. 获取最新 image/state
  2. 请求推理服务
  3. 拿到 action_list
  4. 检查 action shape / range / NaN
  5. 低速执行 action chunk
  6. 记录 request_time / response_time / send_action_time
```

第一版不要马上启用异步多线程。

建议演进：

1. 同步请求 + 低速执行。
2. 请求线程和控制线程分离。
3. action queue。
4. inference overlap。
5. pending action / future-state-aware。
6. action quantization。

## 服务化架构选择

### 方案 A：VLASH PyTorch 服务

这是第一推荐方案。

特点：

- 服务端直接加载 VLASH checkpoint。
- 使用 PyTorch 推理。
- 使用 VLASH 自己的 processor / normalization / config。
- 对 checkpoint 兼容性最好。

优点：

- 最快打通。
- 不需要把 checkpoint 转成当前仓库格式。
- 不需要一开始做 Triton 权重转换。
- 训练和推理逻辑一致。

缺点：

- latency 可能不如 Triton/CUDA Graph。
- 服务端依赖 VLASH 环境。
- 需要写一个适配当前机械臂请求格式的 adapter。

适合当前阶段。

### 方案 B：当前仓库加载 VLASH checkpoint，本地推理

特点：

- 不做推理服务。
- 当前主仓库直接加载 checkpoint。

优点：

- 上机链路简单。
- 无网络延迟。

缺点：

- VLASH checkpoint 可能不能被当前 PI0.5 loader 直接加载。
- 当前仓库会逐渐混入 VLASH policy/processor 逻辑。
- 不符合你现在“隔离训练”的思路。

适合作为后续优化，不作为第一版。

### 方案 C：Realtime-VLA V2 Triton 服务

特点：

- 服务端使用 Triton kernels + CUDA Graph。
- latency 上限最好。

优点：

- 性能强。
- 服务化结构已有参考。

缺点：

- 需要权重转换。
- 需要固定 shape。
- 需要 norm_stats 对齐。
- 需要 Torch vs Triton 数值验证。
- 工程成本高。

不建议第一版使用。

### 当前推荐

当前推荐：

> **先做方案 A：VLASH PyTorch 服务。**

等它跑通后再决定：

- 是否把服务端换成 Triton。
- 是否把 checkpoint 转到当前主仓库。
- 是否迁移 async runtime。

## 请求/响应协议建议

### 请求字段

```text
request_id: string
timestamp: float
task: string
state: list[float]
images: dict[str, bytes]
pending_actions: int
metadata:
  robot_type: string
  fps: float
  camera_names: list[str]
```

### 响应字段

```text
request_id: string
action_list: list[list[float]]
raw_action_list: list[list[float]]
infer_time: float
preprocess_time: float
model_time: float
postprocess_time: float
server_timestamp: float
model_info:
  checkpoint: string
  action_dim: int
  chunk_size: int
```

### 必须记录的日志

客户端：

- request_id
- image_timestamp
- state_timestamp
- request_send_time
- response_recv_time
- roundtrip_latency
- action_send_time
- action_index
- safety_check_result

服务端：

- request_id
- preprocess_time
- model_time
- postprocess_time
- total_infer_time
- action shape
- NaN/Inf check
- checkpoint path

## 安全策略

服务化上机必须加安全保护。

客户端执行前检查：

- action 是否为空。
- action shape 是否等于机械臂 action dim。
- action 是否包含 NaN/Inf。
- action 是否超出关节限位。
- action delta 是否过大。
- action velocity 是否过大。
- 服务超时是否触发 hold / stop。
- 连续失败请求是否停机。

第一版建议：

- 服务超时：停止发送新动作，保持或缓慢回安全姿态。
- action 超限：拒绝执行并记录。
- 连续 N 次服务失败：进入 emergency stop 或人工确认。

## 工程目录建议

建议在文档之外，后续真正开发时按下面的目录组织。

VLASH 侧：

```text
vlash-main/
  examples/train/pi05/desk_cleanup_async.yaml
  examples/train/pi05/desk_cleanup_lora.yaml
  outputs/train/desk_cleanup_pi05/
  outputs/eval/desk_cleanup_pi05/
```

当前主仓库侧：

```text
my_devs/vla_engineering/
  doc/pi05_inference_optimization_from_vla_refs/
  service_bridge/
    docs/
    server_adapter_design.md
    client_runtime_design.md
```

如果后面写代码，建议服务桥接代码放在 `my_devs/vla_engineering/service_bridge/`，不要直接污染 `src/lerobot/`。

## 关键验证点

### 验证点 1：VLASH 训练是否成功

证据：

- training loss 正常下降。
- checkpoint 正常保存。
- 离线 action chunk shape 正确。
- 离线 action 数值合理。

### 验证点 2：服务端推理是否稳定

证据：

- 连续 100 次 mock 请求无异常。
- p50/p95/p99 latency 可接受。
- action 无 NaN/Inf。
- GPU 显存稳定。

### 验证点 3：客户端执行是否安全

证据：

- mock robot 能执行 action chunk。
- action limit check 生效。
- 服务超时时客户端不会继续执行旧危险动作。
- 日志能对齐请求、响应、动作发送时间。

### 验证点 4：真实机械臂低速 smoke test

证据：

- 低速执行不抖动。
- action 与任务方向一致。
- 急停和超限保护有效。
- 失败时能回放日志定位。

## 主要风险与应对

### 风险 1：VLASH 训练出的 checkpoint 只能被 VLASH 加载

应对：

- 第一版服务端直接跑 VLASH checkpoint，不强求当前仓库加载。
- 等任务有效后，再评估是否转换 checkpoint。

### 风险 2：服务化增加网络延迟

应对：

- 记录 `roundtrip_latency` 和 `server_infer_time`。
- 先同机服务测试，再远端服务。
- 如果网络延迟不可接受，再考虑本地加载或异步 action queue。

### 风险 3：图片编码/解码耗时太大

应对：

- 第一版用 JPEG 简化。
- 记录 encode/decode time。
- 后续可换 shared memory、原始 bytes、硬件压缩或 gRPC streaming。

### 风险 4：训练和上机 camera 命名不一致

应对：

- 做 camera mapping 表。
- 服务端 adapter 内明确从请求 camera name 映射到 VLASH policy image feature。

### 风险 5：normalization / state prompt 不一致

应对：

- 服务端使用 VLASH 自己的 processor 和训练 checkpoint 绑定的 stats。
- 当前客户端只发送原始 state/image，不自行做模型 normalization。

## 推荐里程碑

### M1：数据集可训练

完成条件：

- VLASH 能读取 `desk_cleanup_v1/eraser_cup_multi_task`。
- 最小训练 smoke test 成功。

### M2：checkpoint 可离线推理

完成条件：

- VLASH checkpoint 能加载。
- 离线 `predict_action_chunk` 正常。
- latency 和 action 数值有记录。

### M3：VLASH PyTorch 推理服务可用

完成条件：

- 服务端可加载 checkpoint。
- mock request 返回 action chunk。
- 连续请求稳定。

### M4：当前主仓库客户端可调用服务

完成条件：

- 当前上机客户端能发送 image/state/task。
- 能接收 action chunk。
- mock robot 或 dry-run 执行正常。

### M5：真实机械臂低速 smoke test

完成条件：

- 真实机械臂低速执行。
- 安全检查有效。
- 日志可回放。

### M6：性能优化

完成条件：

- 根据 latency 决定是否加入：
  - bf16
  - compile warmup
  - async action queue
  - inference overlap
  - action quantization
  - Triton/CUDA Graph

## 最终建议

当前最建议的主路线是：

> **VLASH 负责训练和服务端推理，当前主仓库负责机械臂客户端和安全执行。**

第一版不要急着迁移 VLASH 到当前仓库，也不要急着上 Triton。

最小闭环应该是：

```text
VLASH train checkpoint
  -> VLASH PyTorch inference server
  -> 当前主仓库 client 发送 observation
  -> server 返回 action chunk
  -> 当前主仓库低速安全执行
```

这样做的好处是：

- 训练环境隔离。
- 上机代码稳定。
- checkpoint 兼容性压力最小。
- 服务接口可以逐步演进。
- 后续是否迁移 async / fusion / Triton，会由真实 latency 和任务效果决定。

一句话总结：

> **先做服务化闭环，不先做代码迁移；先证明 VLASH 训练出的模型能通过服务控制机械臂，再决定要不要把能力迁回当前仓库。**
