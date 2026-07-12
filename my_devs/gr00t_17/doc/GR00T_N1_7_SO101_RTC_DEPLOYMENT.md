# GR00T N1.7 SO101 RTC Server/Client Deployment

## 1. 目标与边界

本实现为正式 `checkpoint-63600` 增加 GR00T N1.7 原生 Real-Time Chunking (RTC) 推理，所有新增代码、日志和产物均位于 `my_devs/gr00t_17`。实现不会写入原始数据、参考仓库或正式权重。

当前 policy server 默认使用 TensorRT full-pipeline 后端，构建与验证细节见 `GR00T_N1_7_SO101_TENSORRT.md`。

RTC 不是简单地在客户端拼接两个动作块。服务端会把上一条 16 步物理动作块连同当前状态重新送入 checkpoint processor，得到与当前状态一致的归一化相对动作；N1.7 action head 随后对重叠段执行冻结和渐进重采样。

## 2. 时间轴

- 模型对外动作块：16 步。
- 新请求间隔：8 步。
- RTC 重叠：8 步。
- 控制频率：30 Hz。
- 稳态本机推理：约 50 ms 模型时间、57--64 ms端到端，即约 2 个控制步。
- 推理线程运行时，主控制线程继续执行旧块中的动作。
- 新块返回后丢弃已经过去的约 2 步，从当前全局控制步开始替换未来动作。
- 若整个 16 步块在返回前已经过期，客户端停止并报告 queue underrun，不会重复旧动作。

启动阶段先执行一次不进入动作队列的 CUDA warmup，随后 reset RTC 状态。机械臂开始运动前取得的第一条正式动作块因此使用稳态推理路径。

## 3. 运动配置

正式脚本固定沿用已经上机验证可运动的配置：

```text
bounds_mode=physical
max_command_delta=200
max_relative_target=200
control_hz=30
run_time_s=120
```

`200/200` 对 SO101 的标定范围而言等价于关闭额外的逐步相对限幅。客户端仍拒绝 NaN/Inf，并保留机械臂物理标定域 `[-100, 100]`、夹爪 `[0, 100]`，避免向电机发送域外数值；这不是先前导致机械臂不动的 dataset min/max 或 1 度相对限幅。

## 4. 文件

- `scripts/so101_rtc_policy_server.py`：带会话状态的 N1.7 原生 RTC policy。
- `scripts/so101_rtc_robot_client.py`：30 Hz 控制线程、异步请求和动作时间轴合并。
- `scripts/run_so101_rtc_policy_server.sh`：独立 RTC server 入口。
- `scripts/run_so101_rtc_client.sh`：独立 SO101 RTC client 正式上机入口。
- `scripts/run_so101_rtc_smoke.sh`：真实相机/关节状态、真实 checkpoint、零动作 dry-run。
- `scripts/run_so101_rtc_infer.sh`：120 秒正式上机入口。
- `tests/test_so101_rtc.py`：RTC 参数、时间轴和无限幅默认值测试。

## 5. Smoke 结果

已完成真实设备零动作异步 smoke：

```text
outputs/inference/rtc_smoke/rtc_final_dry_run_20260712_110138/reports/summary.json
```

结果：`status=passed`、90/90 控制步、30.0 Hz、11 次正式请求、稳态端到端推理 58--68 ms、RTC 实际启用、0 次队列欠载、0 条电机动作。另有一次 317 ms 冷启动 warmup，其输出在 reset 前被丢弃。

## 6. 正式上机

### 6.1 独立 Server/Client（推荐）

终端一启动本地 policy server，保持该终端运行：

```bash
cd /data/cqy_workspace/flexible_lerobot

SERVER_HOST=127.0.0.1 SERVER_PORT=5556 \
  /home/cqy/miniconda3/bin/conda run --no-capture-output -n lerobot_flex \
  bash my_devs/gr00t_17/scripts/run_so101_rtc_policy_server.sh
```

看到 `[RTC SERVER] ready on 127.0.0.1:5556 backend=tensorrt` 后，在终端二启动机械臂 client：

```bash
cd /data/cqy_workspace/flexible_lerobot

RUN_ID=so101_n17_rtc_unrestricted_$(date +%Y%m%d_%H%M%S) \
SERVER_HOST=127.0.0.1 SERVER_PORT=5556 \
  /home/cqy/miniconda3/bin/conda run --no-capture-output -n lerobot_flex \
  bash my_devs/gr00t_17/scripts/run_so101_rtc_client.sh \
    --robot-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00 \
    --top-cam /dev/video4 \
    --wrist-cam /dev/video6 \
    --task "Put the eraser into the small box"
```

client 退出后 server 会继续运行，可以再次启动 client。每次 client 开始推理前都会 reset 服务端 RTC 会话。结束 server 使用 `Ctrl+C`。

### 6.2 单脚本组合入口

从仓库根目录运行：

```bash
cd /data/cqy_workspace/flexible_lerobot

RUN_ID=so101_n17_rtc_unrestricted_$(date +%Y%m%d_%H%M%S) \
  /home/cqy/miniconda3/bin/conda run --no-capture-output -n lerobot_flex \
  bash my_devs/gr00t_17/scripts/run_so101_rtc_infer.sh \
    --robot-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00 \
    --top-cam /dev/video4 \
    --wrist-cam /dev/video6 \
    --task "Put the eraser into the small box"
```

脚本内部已固定 `120 s / 30 Hz / horizon 8 / physical / 200 / 200` 以及双重上机确认，不需要重复传入。

## 7. 验收日志

正式结果写入：

```text
outputs/inference/rtc_formal/<RUN_ID>/reports/inference.json
outputs/inference/rtc_formal/<RUN_ID>/logs/policy_server.log
outputs/inference/rtc_formal/<RUN_ID>/logs/robot_client.log
outputs/inference/rtc_formal/<RUN_ID>/logs/gpu_usage.log
```

重点检查 `status=passed`、`queue_underruns=0`、`achieved_control_hz` 接近 30，以及除第一条外所有 request 的 `rtc_applied=true`。
