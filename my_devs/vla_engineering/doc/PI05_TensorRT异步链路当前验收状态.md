# PI0.5 TensorRT 异步链路当前验收状态

日期：2026-06-05

## 当前结论

`tensorrt_split` 后端已经完成服务化接入，并已通过真实发送动作前的自动/只读 gate。

已通过：

```text
L1 infer 100        PASS
L1 async mock 120s  PASS
L2 readonly 60s     PASS
```

未完成：

```text
L2 lowspeed robot 20s  缺 trt_lowspeed_robot_20s.json
L3 main robot 120s     缺 trt_main_robot_120s.json
```

因此当前状态是：

```text
可以进入低速真机动作验收，但整体验收尚未完成。
```

## 已确认的工程状态

- 服务端推荐入口：`vlash_iner.server.run_pi05_async_server`
- 兼容旧入口：`vlash_iner.server.run_pi05_async_erver`
- 服务端后端：`torch` / `torch_compile` / `tensorrt_split`
- TensorRT engine：
  - `my_devs/openpi_trt/artifacts/pi05_so101_prefix_cache_b1_fp32.engine`
  - `my_devs/openpi_trt/artifacts/pi05_so101_denoise_step_b1_fp32.engine`
- 当前 engine dtype：FP32
- 当前 policy 加载 dtype：`--trt-model-dtype float32`
- 客户端默认拒绝真实发送 action，必须显式传入 `--confirm-control true`
- 真实相机只读必须传入 `--no-send-action true`

## 当前机器在线验证

已用 `lerobot_flex` 环境验证：

```bash
conda run --no-capture-output -n lerobot_flex python -m py_compile \
  my_devs/vla_engineering/vlash_iner/server/run_pi05_async_server.py \
  my_devs/vla_engineering/vlash_iner/server/run_pi05_async_erver.py \
  my_devs/vla_engineering/vlash_iner/server/run_pi05_async_client.py \
  my_devs/vla_engineering/vlash_iner/server/run_pi05_backend_acceptance.py \
  my_devs/vla_engineering/vlash_iner/server/run_pi05_acceptance_audit.py
```

已确认硬件节点存在：

```text
/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00
/dev/video4
/dev/video6
```

已确认服务端可以用新入口启动，并且 `/health` 返回：

```text
ready: true
backend: tensorrt_split
chunk_size: 50
n_action_steps: 50
backend_info.model_dtype: float32
```

已完成最新在线 smoke：

```bash
conda run --no-capture-output -n lerobot_flex python -u -m vlash_iner.server.run_pi05_backend_acceptance \
  --server-url http://127.0.0.1:8008 \
  --endpoint /infer \
  --mode infer \
  --requests 10 \
  --task "Put the eraser into the small box" \
  --robot-type so101_follower \
  --img-width 640 \
  --img-height 480 \
  --state-dim 6 \
  --action-dim 6 \
  --output-json /data/cqy_workspace/flexible_lerobot/my_devs/vla_engineering/vlash_iner/server/reports/trt_infer_smoke_latest.json
```

结果：

```text
passed: true
shape: [50, 6]
request_latency_s.p50: about 0.090s
server_infer_s.p50: about 0.088s
```

## 继续验收步骤

### 1. 启动 TensorRT split 服务端

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

### 2. L2 低速真机 20s

这一项会真实发送 action。运行前需要确认机械臂周围安全、任务物体摆放正确、有人能立即接管。

推荐使用受控 runner，避免手敲长命令：

```bash
cd /data/cqy_workspace/flexible_lerobot/my_devs/vla_engineering

conda run --no-capture-output -n lerobot_flex python -u -m vlash_iner.server.run_pi05_robot_acceptance \
  --server-url http://127.0.0.1:8008 \
  --run-lowspeed \
  --confirm-control true
```

等价的底层客户端命令：

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
finished == true
error == null
health.backend == tensorrt_split
no_send_action == false
confirm_control == true
elapsed_s >= 18
机器人动作方向合理，无明显突然跳动，无异常大动作
```

### 3. L3 主力真机 120s

L2 低速通过后再执行。

推荐使用受控 runner：

```bash
cd /data/cqy_workspace/flexible_lerobot/my_devs/vla_engineering

conda run --no-capture-output -n lerobot_flex python -u -m vlash_iner.server.run_pi05_robot_acceptance \
  --server-url http://127.0.0.1:8008 \
  --run-main \
  --confirm-control true
```

如果现场准备充分，也可以 L2 + L3 连续执行：

```bash
cd /data/cqy_workspace/flexible_lerobot/my_devs/vla_engineering

conda run --no-capture-output -n lerobot_flex python -u -m vlash_iner.server.run_pi05_robot_acceptance \
  --server-url http://127.0.0.1:8008 \
  --run-lowspeed \
  --run-main \
  --confirm-control true
```

等价的底层客户端命令：

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
finished == true
error == null
health.backend == tensorrt_split
no_send_action == false
confirm_control == true
elapsed_s >= 115
wait_count == 0
request_latency_s.p95 < 8 / 45 = 0.178s
动作平滑，无明显顿挫或突然跳变
```

### 4. 汇总审计

```bash
cd /data/cqy_workspace/flexible_lerobot/my_devs/vla_engineering

conda run --no-capture-output -n lerobot_flex python -m vlash_iner.server.run_pi05_acceptance_audit \
  --report-dir /data/cqy_workspace/flexible_lerobot/my_devs/vla_engineering/vlash_iner/server/reports \
  --output-json /data/cqy_workspace/flexible_lerobot/my_devs/vla_engineering/vlash_iner/server/reports/trt_acceptance_audit.json \
  --strict
```

只有审计输出全部 gate 通过，才能认为整体验收完成。
