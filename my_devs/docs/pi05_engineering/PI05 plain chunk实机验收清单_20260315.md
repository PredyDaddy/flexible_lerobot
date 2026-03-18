# PI05 plain chunk 实机验收清单

文档日期：2026-03-15

文档位置：`my_devs/docs/pi05_engineering/PI05 plain chunk实机验收清单_20260315.md`

适用对象：

- `my_devs/pi05_engineering/run_pi05_chunk_infer.py`
- 当前 Phase-1.5 的 plain chunk 运行时验收

## 1. 目标与边界

这份清单只解决一件事：

- 确认当前 PI05 plain chunk runtime 是否已经达到“可重复验收”的实机运行水平

这里的 plain chunk 明确指：

- `--enable-rtc false`

本清单不覆盖：

- RTC on/off 对照实验
- 第二阶段 server/client 化
- 通用 `async_inference` 改造

本清单的判定口径也要说清楚：

1. 通过不等于“已经完成所有后续工作”。
2. 通过只说明 plain chunk 在当前机器人、当前相机、当前 checkpoint、当前参数组合下已经达到可重复的稳定运行门槛。
3. 任一步失败，都优先先定位故障域，再决定是否继续下一步，不允许直接跳步骤。

## 2. 验收前准备

### 2.1 环境与前提

必须满足：

- 使用 `lerobot_flex` conda 环境
- 当前工作目录在 repo root
- 机器人已上电并处于可连接状态
- 相机索引与标定目录使用当前已知可用配置
- 当前 checkpoint 路径保持为默认 PI05 路径，避免引入新的变量

建议先确认：

1. 当前没有其他进程占用机器人串口或相机设备。
2. 机器人连接线、供电、USB 口状态稳定。
3. 执行前记录当前日期、机器人端口、相机 index、运行命令。

### 2.2 当前建议基线参数

在 plain chunk 验收阶段，固定使用下面这组保守参数作为基线：

```bash
--enable-rtc false
--fps 10
--camera-fps 30
--actions-per-chunk 12
--queue-low-watermark 5
```

说明：

- `--fps 10` 是控制频率
- `--camera-fps 30` 是相机配置频率
- 这两个参数不能再混为一个概念

### 2.3 建议记录的基础信息

每次验收建议至少记录：

1. 执行命令全文
2. 开始时间与结束时间
3. 是否一次通过
4. 首次失败的报错原文
5. 最终 metrics snapshot

## 3. 分层验收步骤

## Step 1: `connect_smoke`

### 执行命令

```bash
conda run -n lerobot_flex python my_devs/pi05_engineering/run_pi05_chunk_infer.py \
  --connect-smoke true \
  --fps 10 \
  --camera-fps 30
```

### 这一步验证什么

只验证：

- `robot.connect()`
- `robot.disconnect()`

不验证：

- policy 权重加载
- pre/post processor 加载
- producer / actor 双线程
- 实际动作执行

### 通过标准

满足以下条件才算通过：

1. 命令正常退出。
2. 没有出现 `OpenCVCamera(... failed to set fps=...)`。
3. 没有出现 `Torque_Enable ... Incorrect status packet!`。
4. 没有出现 Python traceback。

### 失败归因

如果失败，优先按下面分类：

1. 相机 FPS 相关报错：
   - 归因到相机配置与设备返回档位不一致
   - 先检查 `--camera-fps`，不要误判成 policy/runtime 故障
2. `Torque_Enable` / `Incorrect status packet`：
   - 归因到 connect/configure 生命周期风险
   - 优先检查总线、供电、接触、串口瞬态
3. 其他 connect traceback：
   - 先归到硬件 connect 域，不进入 live 运行阶段排查

### 回退动作

失败后不要直接跑 full live，先做：

1. 断开并重新连接机器人电源或串口
2. 确认无其他进程占用 `/dev/ttyACM0`
3. 重新执行一次同样的 `connect_smoke`
4. 如果连续失败，停止继续后续步骤，先处理 connect 生命周期问题

## Step 2: `load_connect_smoke`

### 执行命令

```bash
conda run -n lerobot_flex python my_devs/pi05_engineering/run_pi05_chunk_infer.py \
  --load-connect-smoke true \
  --enable-rtc false \
  --fps 10 \
  --camera-fps 30
```

### 这一步验证什么

验证：

- policy/processors 加载
- `robot.connect()/disconnect()`

不验证：

- producer / actor 双线程 live
- queue refill 行为
- 实际长时间动作执行

### 通过标准

满足以下条件才算通过：

1. policy 权重加载成功
2. pre/post processors 加载成功
3. connect/disconnect 正常完成
4. 没有出现 traceback

### 失败归因

优先分成两类：

1. 加载失败：
   - policy path
   - tokenizer
   - processor 文件
2. connect 失败：
   - 相机 connect/configure
   - 电机 torque enable

### 回退动作

1. 若是加载失败，回退到 `--offline-load-smoke true`
2. 若是 connect 失败，回退到 `connect_smoke`
3. 两种 smoke 都不稳定时，不进入 plain chunk live

## Step 3: plain chunk 30 秒

### 执行命令

```bash
conda run -n lerobot_flex python my_devs/pi05_engineering/run_pi05_chunk_infer.py \
  --enable-rtc false \
  --fps 10 \
  --camera-fps 30 \
  --actions-per-chunk 12 \
  --queue-low-watermark 5 \
  --run-time-s 30
```

### 这一步验证什么

验证：

- producer / actor 双线程 live 主链路
- queue refill
- first chunk ready
- 实际动作持续发送
- 30 秒内的基础稳定性

### 通过标准

满足以下条件才算通过：

1. 命令在 30 秒左右按时结束，而不是提前异常退出。
2. `first_chunk_ready=True`。
3. `queue_actions_popped` 持续增长且大于 0。
4. `queue_empty_events=0`。
5. `queue_held_last_action_events=0`。
6. `actor_rate_hz` 接近 10Hz，可接受小幅波动。
7. 没有出现：
   - `Port is in use!`
   - `Incorrect status packet!`
   - 其他 traceback

### 重点观察项

1. `producer_rate_hz`
2. `latency_total_p95_s`
3. `delay_mismatch_count`
4. `latest_real_delay`
5. `latest_leftover_length`

这里要注意：

- `delay_mismatch_count` 在 plain 模式下不能单独判死刑
- `latest_leftover_length` 当前更像账本指标，不等于真实待执行 backlog

### 失败归因

1. 若提前在 connect 阶段失败：
   - 回到 Step 1/2，归因到 connect 生命周期
2. 若运行中出现 `Port is in use!`：
   - 优先怀疑 robot I/O ownership 仍有回归
3. 若 `queue_empty_events` 明显大于 0：
   - 优先怀疑补货余量不足或延迟抖动超出当前参数组合承受范围
4. 若 actor rate 明显低于 10Hz：
   - 优先怀疑发送侧阻塞、相机/总线抖动、或主循环异常

### 回退动作

1. 不要直接跳到 RTC
2. 先保留这次日志与 metrics
3. 若失败发生在运行中，优先复跑一次同样命令确认是否可复现
4. 若连续两次都失败，再进入代码与硬件分域排查

## Step 4: plain chunk 120 秒

### 执行命令

```bash
conda run -n lerobot_flex python my_devs/pi05_engineering/run_pi05_chunk_infer.py \
  --enable-rtc false \
  --fps 10 \
  --camera-fps 30 \
  --actions-per-chunk 12 \
  --queue-low-watermark 5 \
  --run-time-s 120
```

### 这一步验证什么

验证：

- 中时长稳定性
- queue 是否随时间退化
- shutdown 是否稳定
- 是否存在慢性积累类问题

### 通过标准

满足以下条件才算通过：

1. 120 秒内无 traceback。
2. 无 `Port is in use!`。
3. 无 `Incorrect status packet!`。
4. `queue_empty_events` 仍保持为 0，或至少没有持续增长趋势。
5. `actor_rate_hz` 仍接近目标值。
6. worker 能正常退出，最终日志里不应显示线程卡死或 teardown 冲突。

### 失败归因

1. 若 connect 阶段偶发失败：
   - 仍归 connect 生命周期问题
2. 若运行前半段正常、后半段开始退化：
   - 优先怀疑慢性 backlog、总线抖动、或观测/执行节奏漂移
3. 若只剩 `delay_mismatch_count` 增长，但 queue 不空、actor rate 正常：
   - 更像观测口径待收敛，不应直接判定为控制失败

### 回退动作

1. 若 120 秒失败，但 30 秒稳定：
   - 不进入 RTC
   - 先把问题归类为“中时长稳定性不足”
2. 保留全量终端输出与最终 metrics snapshot
3. 重新从 Step 1 开始做一次完整矩阵，而不是只重跑 Step 4

## 4. 推荐记录的关键指标

每一步至少建议记录以下指标：

1. `first_chunk_ready_at_s`
2. `first_warmup_latency_s`
3. `actor_rate_hz`
4. `producer_rate_hz`
5. `queue_empty_events`
6. `queue_held_last_action_events`
7. `queue_actions_popped`
8. `delay_mismatch_count`
9. `delay_mismatch_ratio`
10. `latest_delay_error`
11. `delay_error_mean`
12. `delay_error_max_abs`
13. `latest_real_delay`
14. `latest_original_backlog_length`
15. `latest_executable_backlog_length`
16. `latest_backlog_gap`
17. `latest_inference_overrun_ratio`
18. `latency_total_p95_s`
19. `worker_shutdown_clean`
20. `alive_worker_names`

此外，当前建议额外人工备注：

1. 是否出现机器人动作卡顿、抖动、停顿
2. 是否出现明显视觉帧率异常
3. 是否存在 shutdown 时间异常偏长

解释建议：

1. `delay_mismatch_count` 不再单独判定稳定性，优先配合 `delay_mismatch_ratio` 和 `latest_delay_error` 看。
2. `latest_leftover_length` 仍可保留作兼容字段，但诊断时优先使用：
   - `latest_original_backlog_length`
   - `latest_executable_backlog_length`
   - `latest_backlog_gap`

## 5. 验收结论判定

## 5.1 判定为“通过”

满足以下条件，可认为 plain chunk 已通过当前 Phase-1.5 的实机验收门槛：

1. `connect_smoke` 稳定通过
2. `load_connect_smoke` 稳定通过
3. plain chunk 30 秒通过
4. plain chunk 120 秒通过
5. 无串口并发错误
6. 无持续 starvation 迹象

## 5.2 判定为“部分通过”

满足以下条件，可判为“主链路可运行，但未完成验收”：

1. smoke 都通过
2. 30 秒通过
3. 120 秒失败或指标语义仍不足以支撑判断

这种情况下，不应直接进 RTC，只能继续做 plain chunk 稳定性治理。

## 5.3 判定为“未通过”

出现以下任一情况，判为未通过：

1. connect_smoke 不稳定
2. load_connect_smoke 不稳定
3. plain chunk 30 秒内频繁 queue starvation
4. 出现 `Port is in use!`
5. worker shutdown 不干净

## 6. 当前阶段建议

在当前项目节奏下，这份清单的用途不是“为了多做几次命令”，而是为了把后续工作分成两个阶段：

1. 先把 plain chunk 做成可以稳定验收的运行时
2. 再进入 RTC 对照与第二阶段 server/client 化

更直白地说：

> 在这份清单没有稳定跑通前，不建议直接把 RTC 打开后投入正式工作。
