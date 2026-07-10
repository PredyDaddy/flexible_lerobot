# RealtimeVLA Backend 技术方案

## 1. 背景与目标

当前 `my_devs/openpi_train/openpi_so101` 这一套 OpenPI SO101 原版链路已经完成基础验证，现状如下：

- 服务端脚本：`easy_use/serve_robot_policy_10epoch_bs48.sh`
- 模型来源：`outputs/checkpoints/pi05_so101_eraser_cup_lora/so101_lora_full_v21_10epoch_bs48_20260702_171345/11090`
- 客户端脚本：`easy_use/run_robot_remote_client.sh`
- 上机环境：`lerobot_flex`
- 已验证参数：`action_chunk_steps=30`
- 当前结论：实机效果比此前更好，说明 OpenPI SO101 的 10 epoch LoRA 训练结果已经具备继续做后端工程化改造的价值

因此，这次工作不是从零验证模型是否能跑，而是在“原版远程推理链路已经打通”的前提下，为 `realtimevla_backend` 设计一套独立后端方案，借鉴 `reference_backend/realtime-vla-v2-main` 的服务端/客户端拆分、时序调度与部署组织方式，但不污染现有 OpenPI 原版路径。

本文档目标是为后续在 `my_devs/openpi_train/realtimevla_backend` 内独立开发提供一份技术落地蓝图。

## 2. 当前 OpenPI SO101 状态确认

### 2.1 已完成训练

已经完成并验证过的训练成果为：

- 基座能力：OpenPI SO101
- 微调形式：LoRA
- 训练轮次：10 epoch
- 代表 checkpoint：
  `outputs/checkpoints/pi05_so101_eraser_cup_lora/so101_lora_full_v21_10epoch_bs48_20260702_171345/11090`

这意味着后续做后端替换时，优先复用这份已经可工作的模型资产，而不是再改训练流程。

### 2.2 当前原版 websocket 推理是否成功

根据已知上下文，当前原版远程推理链路已经成功：

- `easy_use/serve_robot_policy_10epoch_bs48.sh` 可以正常加载 10 epoch LoRA checkpoint
- `easy_use/run_robot_remote_client.sh` 在 `lerobot_flex` 环境中已经完成上机验证
- `action_chunk_steps=30` 的实机效果已经被确认比之前更好

因此，对本项目当前阶段的判断是：

**原版 websocket 推理已经成功，不属于“还没跑通”的状态。**

这点非常重要，因为它决定了 `realtimevla_backend` 的目标不是“修通推理”，而是“在推理已成功的基础上，升级后端架构与部署能力”。

## 3. 为什么在推理成功后还要换 RealtimeVLA 风格后端

即便原版 websocket 方案已经可用，仍然有充分理由引入一个独立的 `RealtimeVLA Backend`：

### 3.1 把模型推理服务与机器人控制侧职责拆开

参考实现将系统明确拆成：

- 远端推理服务
- 本地机器人观测/控制客户端
- 中间通过稳定协议通信

这种结构更适合后续演进，因为模型推理机与机器人上机控制机通常有不同的依赖约束、显卡需求和故障模式。

### 3.2 更适合做时序化 action chunk 调度

RealtimeVLA V2 的参考实现并不只是“远程返回动作”，它还显式处理：

- 连续 action chunk 的预填充
- 推理延迟感知
- 本地 heartbeat / control loop
- 本地平滑或 MPC 优化
- 图像采样时间与状态时间对齐

而这些机制对真实机器人时延抖动、动作连续性和控制稳定性都很关键。

### 3.3 便于后续切换不同推理后端

参考代码中，服务端通过 builder 模式把模型后端抽象出来，可切换不同 adapter：

- JAX 版 OpenPI RTC
- Triton 版 OpenPI RTC

迁移到 `realtimevla_backend` 后，也可以沿着这个思路，把当前可工作的 OpenPI SO101 推理链路封装成新的 adapter，从而保留未来继续替换 TensorRT / Triton / Flash 推理后端的空间。

### 3.4 降低对现有原版链路的侵入

因为原版链路已经有效，最稳妥的方式不是直接在原路径上持续堆逻辑，而是：

- 原版链路继续保留，可作为回退方案
- 新后端单独放在 `my_devs/openpi_train/realtimevla_backend`
- 逐步把服务协议、推理调度、客户端执行器迁入新目录

这样能避免开发过程中把现有可用链路弄脏。

## 4. 参考代码来源与借鉴范围

本方案参考的是：

- `my_devs/openpi_train/reference_backend/realtime-vla-v2-main/README.md`
- `my_devs/openpi_train/reference_backend/realtime-vla-v2-main/server/infer_server.py`
- `my_devs/openpi_train/reference_backend/realtime-vla-v2-main/server/builders.py`
- `my_devs/openpi_train/reference_backend/realtime-vla-v2-main/server/model.py`
- `my_devs/openpi_train/reference_backend/realtime-vla-v2-main/server/config.py`
- `my_devs/openpi_train/reference_backend/realtime-vla-v2-main/server/config_chip.yaml`
- `my_devs/openpi_train/reference_backend/realtime-vla-v2-main/server/pi05_infer.py`
- `my_devs/openpi_train/reference_backend/realtime-vla-v2-main/server/pi05rtc_infer.py`
- `my_devs/openpi_train/reference_backend/realtime-vla-v2-main/client/local_client.py`
- `my_devs/openpi_train/reference_backend/realtime-vla-v2-main/client/builders.py`
- `my_devs/openpi_train/reference_backend/realtime-vla-v2-main/client/config.py`
- `my_devs/openpi_train/reference_backend/realtime-vla-v2-main/client/config_chip.yaml`

### 4.1 借鉴到的关键点

1. 服务端采用独立推理进程
2. 客户端与服务端解耦
3. YAML 配置驱动
4. Builder 负责装配模型、执行器、观测器
5. 推理返回的不只是动作，还可以带上 `infer_time`
6. 客户端内部通过多线程拆分状态采集、图像采集、推理请求、控制循环
7. 本地可以做平滑器或 MPC，而不是把所有时序压力都丢给模型

### 4.2 需要特别澄清的一点

参考仓库的推理服务并不是 websocket。

从参考代码可见：

- 服务端使用 `FastAPI + uvicorn`
- 客户端使用 `requests.Session`
- 通信载荷使用 `pickle.dumps / pickle.loads`
- 默认接口为 `POST /infer`

所以本次“换 RealtimeVLA 后端”更准确的含义是：

**从当前原版 websocket 风格链路，迁移到一种更接近 RealtimeVLA V2 的“独立推理服务 + 本地控制客户端 + 时序执行器”架构。**

## 5. 建议的目标架构

建议 `realtimevla_backend` 最终形成下面这套结构：

```text
my_devs/openpi_train/realtimevla_backend/
├── DOC/
│   └── realtimevla_backend_technical_plan.md
├── server/
│   ├── infer_server.py
│   ├── builders.py
│   ├── config.py
│   ├── model.py
│   └── configs/
├── client/
│   ├── local_client.py
│   ├── builders.py
│   ├── config.py
│   ├── executor.py
│   ├── robot_io.py
│   └── configs/
├── scripts/
├── requirements/
└── README.md
```

其中职责建议如下：

- `server/`：只负责模型加载、远程推理服务、可选服务端动作后处理
- `client/`：只负责机器人状态读取、相机读取、网络请求、动作执行和本地时序控制
- `scripts/`：放启动脚本
- `requirements/`：记录后端独立依赖
- `DOC/`：记录开发文档、联调报告、验收报告

## 6. 服务端/客户端架构说明

### 6.1 服务端

服务端建议部署在带 GPU 的模型机上，负责：

- 加载 OpenPI SO101 10 epoch LoRA checkpoint
- 接收来自客户端的观测数据
- 执行一次 action chunk 推理
- 返回动作序列、推理耗时、必要的调试字段

建议服务端请求输入至少包含：

- 当前状态 `state`
- 历史或预填充动作 `actions` / `action_prefill`
- 多路相机图像 `images`
- 时间戳 `timestamp`
- 任务提示 `prompt` 或配置中固化的任务描述

建议服务端响应至少包含：

- `action_list`
- `infer_time`
- 可选 `raw_action_list`
- 可选 `model_time_ms` / `queue_delay_ms`

### 6.2 客户端

客户端继续运行在机器人上机侧，且必须使用 `lerobot_flex` 环境。它的职责是：

- 读取机器人状态
- 读取相机图像
- 维护 action chunk 的本地执行队列
- 周期性向服务端发推理请求
- 处理网络延迟、请求超时与故障回退
- 执行动作平滑、限速或本地 MPC

建议客户端线程结构参考 `reference_backend`：

- 状态采集线程
- 图像采集线程
- 推理请求线程
- heartbeat 线程
- 可选 control 线程

这样做的原因是：

- 采样与控制频率不必完全一致
- 推理耗时波动不会直接阻塞机器人底层控制循环
- 后续更容易做日志对齐和性能分析

## 7. 实现路线

建议分五个阶段推进。

### 阶段 1：建立纯目录内独立骨架

只在 `my_devs/openpi_train/realtimevla_backend` 内创建：

- `server/`
- `client/`
- `scripts/`
- `requirements/`
- `DOC/`

先完成不依赖机器人真实上机的代码骨架和配置骨架。

### 阶段 2：封装 OpenPI SO101 服务端 adapter

目标是把当前已经跑通的 OpenPI SO101 推理能力封装进新的服务端模块，而不是直接复制旧脚本逻辑。

建议做法：

- 在 `server/model.py` 中定义 adapter 抽象
- 先实现一个 `openpi_so101_remote` 或类似命名的 adapter
- adapter 内部调用当前可工作的模型加载与推理逻辑
- 对输入输出做统一格式转换，向客户端暴露稳定协议

此阶段优先保证：

- 单次请求可推理
- 返回 action chunk
- 能记录 infer latency

### 阶段 3：实现最小可用客户端

先不追求 MPC，先做最小闭环：

- 采集状态
- 采集图像
- 发请求
- 收动作
- 执行动作

这一步的目标是完成“新后端路径下的最小远程推理闭环”。

### 阶段 4：引入 RealtimeVLA 风格时序机制

在最小闭环跑通后，再逐步补充：

- action prefill
- pending queue 计数
- heartbeat
- 延迟感知调度
- 动作平滑
- 必要时再引入 on-device MPC

这样推进更稳，因为可以先分离“服务可用性问题”和“控制质量问题”。

### 阶段 5：做真实机器人联调与回退策略

新后端在实机联调时，需要保留两条路径：

- 原版 websocket 方案，作为稳定回退
- 新的 `realtimevla_backend` 方案，作为实验链路

任何时候如果新后端动作不稳定，允许立即切回原版链路。

## 8. 开发边界

这部分是硬边界，开发时必须遵守。

### 8.1 不允许污染其他路径或文件夹

本后端开发只允许在以下路径内创建或修改文件：

- `my_devs/openpi_train/realtimevla_backend`

不允许修改：

- `my_devs/openpi_train/reference_backend`
- `my_devs/openpi_train/openpi_so101`
- `my_devs/openpi_train/easy_use`
- `my_devs/openpi_train/flashrt_backend`
- `my_devs/openpi_train/trt_backend`
- 以及其他 backend 目录

### 8.2 允许该 backend 创建独立 Python 虚拟环境

允许 `realtimevla_backend` 为自己创建独立 Python 虚拟环境或 conda 环境说明文件，目的是：

- 避免服务端推理依赖污染 `lerobot_flex`
- 允许服务端单独管理 `FastAPI`、`uvicorn`、推理框架、图像处理依赖

推荐做法：

- 机器人客户端仍固定在 `lerobot_flex`
- 服务端在 `realtimevla_backend` 内维护独立环境说明

例如可接受的方式包括：

- `environment.yml`
- `requirements/server.txt`
- `requirements/client.txt`
- `venv/` 或文档里记录的专用环境名

### 8.3 服务端模型推理环境与 lerobot_flex 上机客户端环境分离

这是本方案的重要原则：

- 服务端模型推理环境：允许独立，按 GPU 推理需要单独安装依赖
- 客户端上机控制环境：继续使用 `lerobot_flex`

这样可以避免：

- 服务端推理依赖升级影响机器人 SDK
- 机器人上机环境被 FastAPI / Triton / GPU 推理依赖污染
- 联调时难以定位问题归属

## 9. 端口建议

明确要求：**不要使用 8000 端口。**

原因：

- 参考项目默认就是 `8000`
- 容易与其他服务冲突
- 迁移阶段同时保留多后端时更容易混淆

建议优先使用以下端口之一：

- `18080`
- `18081`
- `28080`

其中更推荐：

- 服务端 HTTP 推理端口：`18080`

如果后续还需要额外的健康检查端口、监控端口或调试端口，可继续顺延：

- `18081`
- `18082`

## 10. 通信协议建议

虽然参考代码当前使用 `pickle` 二进制 HTTP 请求，但在本项目里建议分两步走：

### 10.1 第一阶段

先复刻“简单、可跑通”的协议：

- `HTTP POST /infer`
- 请求体为二进制序列化
- 响应体返回动作列表和推理耗时

这样能最快复用参考工程经验。

### 10.2 第二阶段

在链路稳定后，再考虑把协议升级为更可维护的形式，例如：

- `msgpack`
- `numpy + jpeg bytes` 的结构化封装
- 或基于 `pydantic` 的元数据 + 二进制图像混合协议

原因是 `pickle` 在跨版本兼容、安全性和可观测性上都不够理想，更适合内部原型验证，不适合长期作为稳定协议终态。

## 11. 验收标准

由于 Codex 不能代替用户真实上机控制机器人，因此验收标准必须由用户在真实环境中执行并确认。

### 11.1 最小成功标准

用户自行完成以下步骤后，可判定 `realtimevla_backend` 最小成功：

1. 在服务端机器启动新的推理服务
2. 在机器人上机机器使用 `lerobot_flex` 启动新的客户端
3. 客户端能够成功连接到服务端，不报连接错误
4. 服务端能够接收观测并返回 `action_list`
5. 机器人能够持续执行一个完整 action chunk，而不是只动一步就停
6. 多次推理循环过程中没有明显卡死、超时或频繁丢包

### 11.2 质量成功标准

若要认为“迁移到 RealtimeVLA 后端是值得的”，还需要用户观察到以下至少几项：

- 新后端的动作连续性不差于原版 websocket 方案
- 在相同 checkpoint 下，动作抖动更小或节奏更稳定
- 客户端日志中能看见清晰的推理耗时与请求节奏
- 网络偶发波动时，机器人不会立刻进入不可控状态
- 本地队列、平滑器或 MPC 的行为可解释，可调参

### 11.3 明确的用户验收方式

用户的验收动作应当是：

1. 启动服务端
2. 启动客户端
3. 让机器人执行与当前 OpenPI SO101 原版链路相同的一组任务
4. 对比原版链路与 `realtimevla_backend` 链路的以下指标：
   - 是否稳定启动
   - 是否能持续收发动作
   - 是否存在明显空转或断流
   - 动作是否平滑
   - 实机任务成功率是否下降
5. 如果新链路在可重复测试中达到“不低于原版，且日志与时序控制更清晰”的结果，即可判定方案验收成功

换句话说，本项目的“成功”不是 Codex 在终端里打印出一行 `ok`，而是用户在真实机器人上完成闭环验证并认可效果。

## 12. 风险与注意事项

### 12.1 不要把迁移目标混成“重写全部系统”

本次目标是构建独立后端，不是把原版 OpenPI 训练、数据、上机脚本全部推倒重来。

### 12.2 不要一开始就把 MPC、平滑、协议重构全部堆进去

建议先保证：

- 服务通
- 请求通
- 动作能回
- 机器人能执行

然后再逐项加复杂能力。

### 12.3 原版链路必须保留

因为当前原版远程推理已经成功，所以新后端开发期间必须一直保留原版方案作为基线和回退手段。

## 13. 结论

`realtimevla_backend` 的建设前提非常明确：

- OpenPI SO101 10 epoch LoRA 已经完成训练并验证有效
- 原版 websocket 远程推理已经成功
- 因此换后端不是为了“救火”，而是为了获得更清晰的服务端/客户端分层、更好的时序控制能力、更独立的部署环境，以及后续替换多种推理后端的工程空间

后续开发应严格限制在：

- `my_devs/openpi_train/realtimevla_backend`

同时坚持以下原则：

- 不污染其他 backend
- 服务端与客户端环境分离
- 客户端继续使用 `lerobot_flex`
- 默认不要用 `8000`，优先采用 `18080`
- 用户真实上机验证才是最终验收依据

如果后续按本文路线推进，`realtimevla_backend` 可以在不破坏现有可用 OpenPI 链路的前提下，逐步演进成一个更适合真实机器人持续联调的独立后端。
