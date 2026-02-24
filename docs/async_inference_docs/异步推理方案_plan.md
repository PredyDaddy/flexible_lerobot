# 异步推理方案（Plan）：把 `agilex_infer_single_cc_vertical.py` 改成异步（方案3）

## 0. 目标

把你当前的同步单臂推理：

```bash
python agilex_infer_single_cc_vertical.py \
  --checkpoint outputs/act_agilex_left_box/checkpoints/030000/pretrained_model \
  --arm left --fps 30 --binary-gripper
```

升级为 **异步推理**：
- 控制环（取观测/下发动作）持续 30Hz 跑，不再被推理阻塞
- 推理端按需计算 **action chunk**（一次输出 N 步动作），降低推理调用频率
- 保持你现有行为：**单臂控制 + 另一臂保持当前位置**、支持 `--binary-gripper`

## 1. 方案选型：直接复用仓库内置 `lerobot.async_inference`

本仓库已内置异步推理（gRPC）实现：
- `src/lerobot/async_inference/policy_server.py`：加载 policy，在 server 端推理，输出 action chunk
- `src/lerobot/async_inference/robot_client.py`：robot 端维护 action queue + 控制环 + 发送 observation

核心机制（关键点）：
- server 端 observation 队列 `maxsize=1`，只处理最新帧，避免堆积带来的“延迟越来越大”
- client 端 action queue：队列还在时继续执行；当队列比例低于 `chunk_size_threshold` 时提前补货
- `actions_per_chunk`：一次推理输出多步动作，显著降低推理频率与网络往返

参考：
- 官方 async 教程：`docs/source/async.mdx`
- 你当前同步脚本：`agilex_infer_single_cc_vertical.py`

## 2. AgileX 单臂模型的关键适配点（必须做，否则一定维度不匹配）

你的 checkpoint 是**单臂**（7 维关节 + 2 相机）：
- `observation.state` 期望 7 维（left_* 或 right_*）
- `action` 期望 7 维

但 `AgileXRobot` 默认是**双臂**（14 维）：
- `observation.state` 是 14 维
- `action` 是 14 维

异步推理 server 端会根据 client 提供的 `lerobot_features` 组装 `observation.state`，所以要在 client 侧把 features **裁剪成单臂**：
- state：只保留 `left_*.pos` 或 `right_*.pos`（按 `JOINT_NAMES` 顺序）
- images：只保留该臂对应相机 + front（left: `camera_left`+`camera_front`；right: `camera_right`+`camera_front`）

## 3. 代码落地（已提供）

新增脚本：
- `agilex_scripts/agilex_async_infer_single_cc_vertical.py`（垂直初始位置）
- `agilex_scripts/agilex_async_infer_single_cc_horizal.py`（水平初始位置）

它做了以下“仓库内对齐”的实现选择：
- 仍使用 `AgileXRobot` + `RosCameraConfig`（与同步脚本一致）
- client 侧构造 **单臂 lerobot_features**（7 维 + 2 相机），发给 PolicyServer
- 执行动作时，把单臂输出补齐成 14 维：另一臂从 `robot.ros_bridge.get_puppet_state()` 读取当前值填回去
- 下发动作走 `robot.send_action()`（沿用 AgileXRobot 的安全裁剪/限幅逻辑）
- 复用 `actions_per_chunk / chunk_size_threshold / aggregate_fn_name` 的队列策略（与 `src/lerobot/async_inference/robot_client.py` 同源逻辑）

## 4. 运行方式（推荐）

### 4.1 环境准备

在 `(my_lerobot)` conda 环境中安装 async 依赖（grpcio 等）：

```bash
(my_lerobot) pip install -e ".[async]"
```

### 4.2 两终端运行（最稳定、最符合方案3）

终端 1：启动 policy server

```bash
(my_lerobot) python agilex_scripts/agilex_async_infer_single_cc_vertical.py server \
  --host 127.0.0.1 --port 8080 --fps 30
```

终端 2：启动 robot client（左臂示例）

```bash
(my_lerobot) python agilex_scripts/agilex_async_infer_single_cc_vertical.py client \
  --checkpoint outputs/act_agilex_left_box/checkpoints/030000/pretrained_model \
  --arm left --fps 30 --binary-gripper \
  --server-address 127.0.0.1:8080 \
  --actions-per-chunk 50 --chunk-size-threshold 0.8
```

右臂把 `--arm right` + checkpoint 换掉即可。

如果你的数据/策略是“水平初始位置”，把 client 脚本换成：

```bash
(my_lerobot) python agilex_scripts/agilex_async_infer_single_cc_horizal.py client \
  --checkpoint outputs/act_agilex_left_yellow_bottle/checkpoints/last/pretrained_model \
  --arm left --fps 30 --binary-gripper \
  --server-address 127.0.0.1:8080 \
  --actions-per-chunk 50 --chunk-size-threshold 0.8
```

### 4.3 推荐把 server 放到 GPU 机器（如果你有两台机器）

- 机器人侧（client）机器：ROS + 相机 + 控制环
- 推理侧（server）机器：GPU 推理

这通常能显著降低“偶发卡顿”（推理侧负载抖动不再影响控制环节）。

## 5. 参数建议（从可跑通到“更稳/更快”）

### 5.1 `actions_per_chunk`

30Hz 下建议从 **30~60** 起步（1~2 秒动作缓存）：
- 越大：推理调用越少，更稳、更省带宽，但响应新观测更慢
- 越小：更“跟手”，但更依赖推理吞吐，卡顿风险更高

### 5.2 `chunk_size_threshold`

建议从 **0.8** 起步：
- 0.8：提前补货，队列不易断粮（推荐）
- 0.5：更省观测发送，但对 server 性能要求更高

### 5.3 `aggregate_fn_name`

默认 `weighted_average` 更平滑；如果你想更“跟手”可以试：
- `latest_only`

## 6. 验证清单（建议按顺序排查）

1) server 能加载 checkpoint（日志里能看到 `SendPolicyInstructions` 成功）
2) client 能收到 action chunk（action receiving thread 有输出）
3) `observation.state` 维度是 7（否则就是 features 没裁剪对）
4) 相机 key 与训练一致（左臂：`camera_left`+`camera_front`；右臂：`camera_right`+`camera_front`）
5) `actions_per_chunk` 不要设置得远超 policy 最大 chunk（虽不会直接报错，但无意义）

## 7. 下一步扩展（可选）

1) 双臂各自单臂模型：开两个 server（不同端口），client 侧融合两边动作下发（参考 `docs/async_inference_docs/temp_docs/异步推理方案2.md`）
2) 降低网络/序列化开销：进一步裁剪 obs 字段、降低分辨率、压缩图像（JPEG/PNG），或减少发送频率
3) 真正“端到端更快”：把推理 server 放到 GPU、并配合 TensorRT/ONNXRuntime（见 `trt_act/`）
