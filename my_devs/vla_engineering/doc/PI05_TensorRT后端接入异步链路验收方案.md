# PI0.5 TensorRT 后端接入服务化异步链路验收方案

日期：2026-06-05

目标：

```text
将 my_devs/openpi_trt 已有 split TensorRT 后端接入 vlash_iner 服务化异步推理链路，
并在真实上机前完成分层验收，证明它能用、可控、可调参。
```

相关入口：

- TensorRT 技术报告：`my_devs/openpi_trt/docs/lerobot_pi05_tensorrt优化完整技术报告.md`
- TensorRT runtime：`my_devs/openpi_trt/runtime/pi05_trt_split.py`
- 服务端：`my_devs/vla_engineering/vlash_iner/server/run_pi05_async_server.py`
- 客户端：`my_devs/vla_engineering/vlash_iner/server/run_pi05_async_client.py`
- 后端验收：`my_devs/vla_engineering/vlash_iner/server/run_pi05_backend_acceptance.py`

## 1. 接入边界

TensorRT 后端只替换服务端内部的 PI0.5 `sample_actions(...)` 后端，不改客户端机器人控制逻辑。

正确分层：

```text
客户端机器人层：
  robot.get_observation
  HTTP /infer
  AsyncChunkManager
  safety check
  robot.send_action

服务端协议层：
  observation -> action_chunk

模型后端层：
  torch
  torch_compile
  tensorrt_split
```

TensorRT 接入方式：

```text
加载 LeRobot PI0.5 policy
  -> 加载 split TensorRT prefix_cache / denoise_step engines
  -> patch_sample_actions_with_split_trt(policy, ...)
  -> 继续通过 policy.predict_action_chunk(...) 产生 action chunk
```

只要服务端继续提供：

```text
POST /infer: observation -> action_chunk
```

客户端的 `control-fps`、`overlap`、`blend`、`background-inference` 都可以复用。

## 2. 当前后端能力

现有 TensorRT split 后端：

```text
prefix_cache engine
  + denoise_step engine
  + Python 10-step denoise loop
```

当前 engine：

```text
my_devs/openpi_trt/artifacts/pi05_so101_prefix_cache_b1_fp32.engine
my_devs/openpi_trt/artifacts/pi05_so101_denoise_step_b1_fp32.engine
```

技术报告中的基准：

```text
PyTorch sample_actions 平均耗时:      173.47 ms
split TensorRT sample_actions 平均耗时: 84.81 ms
sample_actions 提速:                 2.05x
真实 camera smoke action max_abs_diff: 0.001039
真实上机: 已验证能抓取
```

注意：

```text
当前 split TensorRT engine 是 FP32。
服务端 tensorrt_split backend 默认用 --trt-model-dtype float32 加载 policy。
```

## 3. 后端启动命令

TensorRT split 服务端：

```bash
cd /data/cqy_workspace/flexible_lerobot/my_devs/vla_engineering

conda run --no-capture-output -n lerobot_flex python -u -m vlash_iner.server.run_pi05_async_server \
  --host 127.0.0.1 \
  --port 8008 \
  --endpoint /infer \
  --backend tensorrt_split \
  --policy-path /data/cqy_workspace/flexible_lerobot/outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
  --task "Put the eraser into the small box" \
  --robot-type so101_follower \
  --prefix-engine-path /data/cqy_workspace/flexible_lerobot/my_devs/openpi_trt/artifacts/pi05_so101_prefix_cache_b1_fp32.engine \
  --denoise-engine-path /data/cqy_workspace/flexible_lerobot/my_devs/openpi_trt/artifacts/pi05_so101_denoise_step_b1_fp32.engine \
  --trt-model-dtype float32
```

健康检查：

```bash
curl http://127.0.0.1:8008/health
```

期望 `/health` 至少包含：

```text
ready: true
backend: tensorrt_split
chunk_size: 50
n_action_steps: 50
backend_info.prefix_engine_path
backend_info.denoise_engine_path
backend_info.model_dtype: float32
```

## 4. L1：服务端连续推理验收

目标：

```text
证明 tensorrt_split 服务端能够稳定完成 observation -> action_chunk。
```

命令：

```bash
cd /data/cqy_workspace/flexible_lerobot/my_devs/vla_engineering

conda run --no-capture-output -n lerobot_flex python -u -m vlash_iner.server.run_pi05_backend_acceptance \
  --server-url http://127.0.0.1:8008 \
  --endpoint /infer \
  --mode infer \
  --requests 100 \
  --task "Put the eraser into the small box" \
  --robot-type so101_follower \
  --img-width 640 \
  --img-height 480 \
  --state-dim 6 \
  --action-dim 6 \
  --output-json /data/cqy_workspace/flexible_lerobot/my_devs/vla_engineering/vlash_iner/server/reports/trt_infer_acceptance.json
```

通过标准：

```text
连续 100 次 /infer 无 crash
action_chunk shape == [50, 6]
无 NaN
无 Inf
server health backend == tensorrt_split
```

建议指标：

```text
request_latency_s.p95 稳定
server_infer_s.p95 稳定
```

当前 FP32 split TRT 初始验收不强制要求优于 torch_compile，因为 benchmark 口径不同。第一阶段先证明它作为服务端后端稳定可替换。

## 5. L1：异步 mock 验收

目标：

```text
不连接机器人，只用 AsyncChunkManager 模拟客户端 45Hz 执行，
验证 tensorrt_split 服务端能跟上 overlap 调度。
```

命令：

```bash
cd /data/cqy_workspace/flexible_lerobot/my_devs/vla_engineering

conda run --no-capture-output -n lerobot_flex python -u -m vlash_iner.server.run_pi05_backend_acceptance \
  --server-url http://127.0.0.1:8008 \
  --endpoint /infer \
  --mode async \
  --run-time-s 120 \
  --control-fps 45 \
  --inference-overlap-steps 8 \
  --background-inference true \
  --chunk-blend-steps 2 \
  --future-state-aware false \
  --task "Put the eraser into the small box" \
  --robot-type so101_follower \
  --img-width 640 \
  --img-height 480 \
  --state-dim 6 \
  --action-dim 6 \
  --max-wait-count 0 \
  --output-json /data/cqy_workspace/flexible_lerobot/my_devs/vla_engineering/vlash_iner/server/reports/trt_async_acceptance.json
```

通过标准：

```text
连续模拟 120 秒
observed_hz 接近 45
wait_count == 0
request_latency_s.p95 < 8 / 45 = 0.178s
action safety 通过
无请求超时
无服务端 crash
```

如果失败：

```text
wait_count > 0:
  尝试 overlap 10，或检查 request_latency/server_infer。

request_latency_s.p95 > 0.178s:
  说明当前 45Hz/overlap8 预算不够。
```

## 6. L2：真实相机只读验收

目标：

```text
连接真实机器人和相机，采集真实 observation，调用 TensorRT 服务端，
但不下发 action。
```

命令：

```bash
cd /data/cqy_workspace/flexible_lerobot/my_devs/vla_engineering

conda run --no-capture-output -n lerobot_flex python -u -m vlash_iner.server.run_pi05_async_client \
  --server-url http://127.0.0.1:8008 \
  --endpoint /infer \
  --task "Put the eraser into the small box" \
  --robot-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00 \
  --top-cam /dev/video4 \
  --wrist-cam /dev/video6 \
  --run-time-s 60 \
  --fps 30 \
  --control-fps 45 \
  --reuse-observation-within-chunk true \
  --inference-overlap-steps 8 \
  --background-inference true \
  --future-state-aware false \
  --chunk-blend-steps 2 \
  --log-interval 10 \
  --action-quant-ratio 1 \
  --connect-retries 3 \
  --connect-retry-s 1.0 \
  --no-send-action true \
  --output-json /data/cqy_workspace/flexible_lerobot/my_devs/vla_engineering/vlash_iner/server/reports/trt_readonly_60s.json
```

通过标准：

```text
真实相机读取正常
真实 observation 能被服务端接收
action_chunk shape 正常
action safety 不报错
wait_count == 0
机器人不执行动作
```

这一步是上机发送动作前的硬 gate。

安全 gate：

```text
--no-send-action true
  连接机器人和相机，运行远程推理和 safety check，但不调用 robot.send_action(...)。

--confirm-control true
  允许真实 robot.send_action(...)。只有低速真机和主力真机阶段才允许使用。
```

客户端默认拒绝真实发送 action。如果没有显式传入：

```bash
--confirm-control true
```

并且没有使用：

```bash
--no-send-action true
```

客户端会直接退出，避免误触发机械臂动作。

结构化报告：

```text
--output-json <path>
```

会记录：

```text
finished / error
steps / observed_hz
request_count / manager_inference_count
wait_count / pending_inference
request_latency_s mean/p50/p95/min/max
server_infer_s mean/p50/p95/min/max
no_send_action / confirm_control
backend health
```

后续 L2/L3 是否通过，应优先以 JSON 报告和真实任务视频/观察结果为准，而不是只看终端滚动日志。

## 7. L2：低速真机验收

目标：

```text
开始真实发送 action，但使用保守参数确认安全。
```

命令：

```bash
cd /data/cqy_workspace/flexible_lerobot/my_devs/vla_engineering

conda run --no-capture-output -n lerobot_flex python -u -m vlash_iner.server.run_pi05_async_client \
  --server-url http://127.0.0.1:8008 \
  --endpoint /infer \
  --task "Put the eraser into the small box" \
  --robot-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00 \
  --top-cam /dev/video4 \
  --wrist-cam /dev/video6 \
  --run-time-s 20 \
  --fps 30 \
  --control-fps 30 \
  --reuse-observation-within-chunk false \
  --inference-overlap-steps 0 \
  --background-inference false \
  --future-state-aware false \
  --chunk-blend-steps 0 \
  --log-interval 5 \
  --action-quant-ratio 1 \
  --connect-retries 3 \
  --connect-retry-s 1.0 \
  --confirm-control true \
  --output-json /data/cqy_workspace/flexible_lerobot/my_devs/vla_engineering/vlash_iner/server/reports/trt_lowspeed_robot_20s.json
```

通过标准：

```text
机器人动作方向合理
无明显突然跳动
无异常大动作
无 safety 异常
无服务端请求失败
Ctrl+C 能正常退出
```

## 8. L3：主力真机验收

目标：

```text
使用主力异步参数验证 TensorRT 后端是否达到可用/好用标准。
```

命令：

```bash
cd /data/cqy_workspace/flexible_lerobot/my_devs/vla_engineering

conda run --no-capture-output -n lerobot_flex python -u -m vlash_iner.server.run_pi05_async_client \
  --server-url http://127.0.0.1:8008 \
  --endpoint /infer \
  --task "Put the eraser into the small box" \
  --robot-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00 \
  --top-cam /dev/video4 \
  --wrist-cam /dev/video6 \
  --run-time-s 120 \
  --fps 30 \
  --control-fps 45 \
  --reuse-observation-within-chunk true \
  --inference-overlap-steps 8 \
  --background-inference true \
  --future-state-aware false \
  --chunk-blend-steps 2 \
  --log-interval 10 \
  --action-quant-ratio 1 \
  --connect-retries 3 \
  --connect-retry-s 1.0 \
  --confirm-control true \
  --output-json /data/cqy_workspace/flexible_lerobot/my_devs/vla_engineering/vlash_iner/server/reports/trt_main_robot_120s.json
```

通过标准：

```text
连续 120 秒无异常退出
wait_count == 0
pending 不长期卡住
request_latency P95 < 0.178s
action safety 不触发
chunk 切换无明显顿挫
机械臂无突然跳变
```

任务标准：

```text
能用：5 次中至少 3 次成功，且无安全问题。
好用：10 次中至少 8 次成功，动作平滑，不需要频繁人工干预。
```

## 9. A/B 对比验收

最终要比较：

```text
torch_compile 服务端
tensorrt_split 服务端
```

客户端参数保持一致：

```text
control-fps 45
overlap 8
blend 2
reuse-observation-within-chunk true
future-state-aware false
```

记录表：

```text
backend          server_p50  server_p95  req_p50  req_p95  wait_count  success  note
torch_compile
tensorrt_split
```

TensorRT 作为正式后端至少要满足：

```text
wait_count 不高于 torch_compile
安全异常不高于 torch_compile
任务成功率不明显低于 torch_compile
动作质量不明显变差
```

## 10. 验收等级

### L1：能接入

```text
服务端 /health backend=tensorrt_split
连续 /infer 100 次通过
async mock 120 秒 wait_count=0
```

### L2：能上机

```text
真实相机只读 60 秒通过
低速真机 20 秒安全执行
```

### L3：好用

```text
主力参数 45/8/2 稳定 120 秒
wait_count=0
任务成功率 >= 80%
动作平滑
服务端可连续多轮运行
```

## 11. 当前实现状态

已完成：

```text
1. 服务端支持 --backend torch / torch_compile / tensorrt_split。
2. tensorrt_split 后端接入 patch_sample_actions_with_split_trt(...)。
3. 服务端 /health 返回 backend 和 backend_info。
4. 新增 run_pi05_backend_acceptance.py 支持 infer/async 两类无硬件验收。
5. 客户端新增 --no-send-action true，用于真实相机只读验收。
6. 客户端新增 --confirm-control true，真实 robot.send_action 必须显式确认。
7. 客户端新增 --output-json，用于真实只读/真机阶段输出结构化验收报告。
```

待真实运行验证：

```text
1. tensorrt_split 服务端 engine 加载。
2. L1 infer 100 次验收。
3. L1 async mock 120 秒验收。
4. L2 真实相机只读。
5. L2/L3 真机任务。
```

## 12. 最低上机前 gate

在真实发送 action 前，必须完成：

```text
1. TensorRT 服务端正常启动，/health 显示 backend=tensorrt_split。
2. L1 infer 100 次通过。
3. L1 async mock 120 秒通过，wait_count=0。
4. L2 真实相机只读 60 秒通过，--no-send-action true。
```

这四条没有完成，不建议直接真实发送 action。

## 13. 2026-06-05 L0/L1 实测结果

### 环境和 engine 状态

检查项：

```text
GPU: NVIDIA GeForce RTX 4090
memory total: 49140 MiB
server loaded memory used: 约 27874 MiB
server loaded memory free: 约 20646 MiB
```

engine 文件：

```text
prefix_cache engine: 约 11 GB
denoise_step engine: 约 1.7 GB
```

依赖：

```text
tensorrt: 可导入
torch: 可导入
fastapi: 可导入
uvicorn: 可导入
requests: 可导入
```

### L0：服务端 health

TensorRT split 服务端已成功启动：

```text
backend: tensorrt_split
policy_path: .../checkpoints/020000/pretrained_model
device: cuda
chunk_size: 50
n_action_steps: 50
model_dtype: float32
```

`/health` 返回了 prefix_cache 和 denoise_step engine 的 I/O 信息，说明 engine 已经成功反序列化并挂到服务端。

### L1：infer smoke

报告：

```text
my_devs/vla_engineering/vlash_iner/server/reports/trt_infer_smoke.json
```

结果：

```text
requests: 10
passed: true
action_chunk shape: [50, 6]
request_latency mean: 98.2 ms
request_latency p50:  90.1 ms
request_latency p95: 164.1 ms
server_infer mean:    96.0 ms
server_infer p50:     88.2 ms
server_infer p95:    160.8 ms
```

首个请求偏慢，后续稳定在约 90 ms request latency。

### L1：infer 100 次正式验收

报告：

```text
my_devs/vla_engineering/vlash_iner/server/reports/trt_infer_acceptance.json
my_devs/vla_engineering/vlash_iner/server/reports/trt_infer_acceptance.log
```

结果：

```text
passed: true
backend: tensorrt_split
requests: 100
action_chunk shape: [50, 6]
failures: []
```

延迟：

```text
request_latency mean: 90.8 ms
request_latency p50:  90.0 ms
request_latency p95:  94.5 ms
request_latency max: 109.1 ms

server_infer mean: 88.7 ms
server_infer p50:  88.0 ms
server_infer p95:  92.5 ms
server_infer max: 106.2 ms
```

action 范围：

```text
action_min: -13.1067
action_max:  74.4197
```

结论：

```text
L1 infer 100 次通过。
TensorRT split 服务端可稳定完成 observation -> action_chunk。
```

### L1：AsyncChunkManager mock 120 秒验收

报告：

```text
my_devs/vla_engineering/vlash_iner/server/reports/trt_async_acceptance.json
my_devs/vla_engineering/vlash_iner/server/reports/trt_async_acceptance.log
```

参数：

```text
control-fps: 45
overlap: 8
background-inference: true
blend: 2
future-state-aware: false
run_time_s: 120
```

结果：

```text
passed: true
steps: 5376
observed_hz: 44.80
request_count: 108
manager_inference_count: 108
wait_count: 0
failures: []
```

延迟：

```text
request_latency mean: 100.4 ms
request_latency p50:  100.0 ms
request_latency p95:  106.3 ms
request_latency max:  112.4 ms

server_infer mean: 94.8 ms
server_infer p50:  94.0 ms
server_infer p95:  100.0 ms
server_infer max:  106.1 ms
```

overlap 预算：

```text
overlap 8 @ 45Hz = 8 / 45 = 177.8 ms
request_latency p95 = 106.3 ms
余量 = 71.5 ms
```

结论：

```text
L1 async mock 120 秒通过。
TensorRT split 服务端能跟上 45Hz / overlap8 / blend2 的异步 chunk 调度。
wait_count=0，说明 chunk 边界没有等待。
```

### 当前验收状态

已完成：

```text
L0 服务端启动和 /health：通过
L1 infer 100 次：通过
L1 async mock 120 秒：通过
L2 真实相机只读 60 秒：通过
```

仍需完成：

```text
L2 低速真机 20 秒
L3 主力真机 120 秒
A/B 对比 torch_compile vs tensorrt_split
```

当前结论：

```text
TensorRT split 后端已经达到“能接入”的 L1 标准。
真实相机只读链路已经通过 L2 的第一道硬 gate。
下一步可以进入 L2 低速真机 20 秒验收。
```

### L2：真实相机只读 smoke

报告：

```text
my_devs/vla_engineering/vlash_iner/server/reports/trt_readonly_smoke.log
```

参数：

```text
run_time_s: 10
control-fps: 45
overlap: 8
background-inference: true
blend: 2
no-send-action: true
```

结果：

```text
finished: true
contains ERROR: false
contains WARN: false
no_send_action: true
max wait_count: 0
```

### L2：真实相机只读 60 秒正式验收

报告：

```text
my_devs/vla_engineering/vlash_iner/server/reports/trt_readonly_60s.log
```

参数：

```text
run_time_s: 60
control-fps: 45
overlap: 8
background-inference: true
blend: 2
future-state-aware: false
no-send-action: true
```

结果：

```text
finished: true
contains ERROR: false
contains WARN: false
no_send_action: true
last_step: 2570
last_elapsed: 59.89s
observed_hz: 42.91
request_count: 52
max wait_count: 0
```

延迟：

```text
request_latency mean: 100.5 ms
request_latency p50:  100.0 ms
request_latency max:  107.0 ms

server_infer mean: 94.8 ms
server_infer p50:  94.0 ms
server_infer max: 104.0 ms
```

结论：

```text
真实机器人连接、真实 top/wrist 相机采集、真实 observation -> TensorRT 服务端 -> action chunk 链路通过。
由于 --no-send-action true，本阶段没有向机器人下发动作。
真实发送 action 需要额外显式传入 --confirm-control true。
```
