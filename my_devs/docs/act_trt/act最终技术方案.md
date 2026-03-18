# ACT TensorRT 最终技术方案

> 文档日期：2026-03-06  
> 适用对象：`my_devs/train/act/so101/run_act_infer.py` 当前使用的 ACT 单臂部署链路  
> 本文定位：合并并收敛 `my_devs/docs/act_trt/act加速方案1.md` 与 `my_devs/docs/act_trt/act加速方案2.md`，作为最终执行方案。

---

## 1. 最终结论

最终采用的路线是：

- **保留现有机器人侧控制语义不变**：机器人连接、观测采集、`build_dataset_frame`、`prepare_observation_for_inference`、`policy_preprocessor`、动作队列 / temporal ensemble、`policy_postprocessor`、动作发送全部保留。
- **只替换 ACT 核心网络前向**：将 `policy.model(batch)` 导出为 ONNX，再构建 TensorRT engine，用 TensorRT 输出 `actions_norm`。
- **动作队列与 temporal ensemble 继续留在 Python 侧**：不导出 `ACTPolicy.select_action()`，只在 Python 中复刻其语义。
- **第一版不把 pre/post processor 融入 ONNX / TensorRT**：优先保证数值对齐、部署稳定和可验证性。
- **第一版不做 dynamic batch**：固定静态 `batch=1`，降低构建复杂度与排障成本。
- **第一版 TRT 推理脚本不再加载 Torch policy 权重**：部署时只保留 config、processor stats、engine 与 metadata。

一句话概括：

> **在最大程度复用现有 LeRobot 部署链路的前提下，只把最值得加速的 ACT 核心网络前向替换成 TensorRT，从而以最低风险获得更高的重规划频率。**

---

## 2. 为什么最终选这条路线

### 2.1 当前真正的性能瓶颈，不是整条链路

当前 ACT 配置的关键事实是：

- `chunk_size = 100`
- `n_action_steps = 100`
- `fps = 30`

这意味着：

- 模型并不是每个控制周期都前向。
- 在 `n_action_steps=100` 时，模型大约每 `100 / 30 = 3.33s` 才真正前向一次。
- 大部分控制周期只是从动作队列中取下一步动作。

因此，TensorRT 的价值不在于“让当前配置下的平均 loop 立刻大幅变快”，而在于：

- 让单次 chunk 前向更快；
- 让首个动作响应更快；
- 更重要的是支撑把 `n_action_steps` 从 `100` 逐步下调到 `32 / 16 / 8`，提高重规划频率。

对应重规划周期：

- `n_action_steps=100` -> `3.33s`
- `n_action_steps=32` -> `1.07s`
- `n_action_steps=16` -> `0.53s`
- `n_action_steps=8` -> `0.27s`

### 2.2 ACT 的语义边界决定了导出粒度

`ACTPolicy.select_action()` 的核心不是纯张量计算，而是：

- `deque` 队列管理；
- “队列为空才触发前向”的状态机；
- temporal ensemble 的在线更新逻辑。

这些逻辑：

- 并不适合 ONNX / TensorRT 表达；
- 本身也不是主要耗时点；
- 一旦硬塞进导出图，维护性与验证难度都会明显变差。

所以最终方案只导出：

```text
normalized obs_state + normalized img0 + normalized img1
    -> ACT core model
    -> normalized action chunk
```

这条边界最合理，原因是：

- 它正好对应真正的热点计算；
- 它和当前仓库实现边界完全一致；
- Torch / ONNX / TRT 三方一致性最容易验证；
- 出问题时最容易定位是在导出、engine、runtime 还是 adapter。

### 2.3 推理阶段 `use_vae=true` 不是阻碍

虽然当前 checkpoint 配置中 `use_vae = true`，但推理时并不会走训练态的 VAE 分支损失逻辑；只要保持 eval 推理路径一致，导出 `policy.model` 仍然是合理的。

---

## 3. 当前链路与模型事实

### 3.1 当前推理链路

当前实机推理脚本为：

- `my_devs/train/act/so101/run_act_infer.py`

主链路可概括为：

```text
robot.get_observation()
  -> robot_observation_processor
  -> build_dataset_frame
  -> prepare_observation_for_inference
  -> policy_preprocessor
  -> policy.select_action
  -> policy_postprocessor
  -> make_robot_action / robot.send_action
```

其中真正要替换的热点是：

- `policy.select_action(...)` 内部的 `policy.model(batch)` 前向

### 3.2 当前 checkpoint 的关键事实

以当前默认 ACT checkpoint 为例，关键事实应从 `config.json` 读取，而不能手写猜测：

- policy type: `act`
- `chunk_size = 100`
- `n_action_steps = 100`
- `use_vae = true`
- `temporal_ensemble_coeff = null`
- `observation.state`: shape `[6]`
- `observation.images.top`: shape `[3, 480, 640]`
- `observation.images.wrist`: shape `[3, 480, 640]`
- `action`: shape `[6]`

因此当前模型可视为：

- 状态维度：`6`
- 相机数量：`2`
- 相机顺序：`top -> wrist`
- 动作维度：`6`
- 每次模型前向输出：`(1, 100, 6)`

### 3.3 环境约束

仓库根目录 `AGENTS.md` 已明确要求：

- **开发代码与测试运行必须使用 conda env：`lerobot_flex`**

因此本文所有命令与后续开发默认都应使用：

```bash
conda run -n lerobot_flex ...
```

---

## 4. 最终目标架构

### 4.1 数据流

```text
Robot obs (numpy)
  -> build_dataset_frame
  -> prepare_observation_for_inference (torch, CPU)
  -> policy_preprocessor (torch, CPU)
  -> ActTrtPolicyAdapter._run_engine(...)
       -> TensorRT engine
       -> actions_norm chunk
  -> Python queue / temporal ensemble
  -> policy_postprocessor (torch, CPU)
  -> make_robot_action / send_action
```

### 4.2 这版明确保留在 Python 侧的部分

- 机器人连接与控制循环
- 相机采集与 observation 组织
- `build_dataset_frame`
- `prepare_observation_for_inference`
- `PolicyProcessorPipeline` 的 pre/post processor
- `ACTPolicy.select_action()` 的队列语义
- temporal ensemble 语义
- robot action mapping 与发送

### 4.3 这版明确替换为 ONNX / TensorRT 的部分

- `policy.model(batch)`
- 输入：归一化后的 state / images
- 输出：归一化后的 action chunk，即 `actions_norm`

### 4.4 第一版明确不做的事情

- 不导出 `ACTPolicy.select_action()`
- 不把 pre/post processor 融入 ONNX / TensorRT
- 不做 dynamic batch
- 不做零拷贝指针直喂 TRT
- 不做 INT8 calibration
- 不在 TRT 推理脚本中同时加载 Torch policy 权重

---

## 5. 工程拆分

最终方案拆成 5 个独立层次，逐层完成：

1. **导出层**：Torch `policy.model` -> ONNX
2. **构建层**：ONNX -> TensorRT engine
3. **Runtime 层**：封装 engine 推理
4. **Policy Adapter 层**：复刻 ACT queue / temporal ensemble 语义
5. **接入层**：接回现有 `run_act_infer.py` 主循环

这样拆的好处是：

- 每层都能单独验证；
- 问题定位清晰；
- 允许先拿到“最短可用闭环”，再做性能增强。

---

## 6. 实施方案

### 6.1 步骤 1：导出 ONNX

#### 导出目标

导出对象固定为：

- `policy.model`

导出边界固定为：

- 输入：已预处理、已归一化的 `obs_state_norm`、`img0_norm`、`img1_norm`
- 输出：`actions_norm`

推荐直接复用参考脚本：

- `my_devs/docs/act_trt/reference_docs/onnx_export_reference/export/export_single.py`

推荐产物：

- `outputs/deploy/act_trt/<run_name>/act_single.onnx`
- `outputs/deploy/act_trt/<run_name>/act_single.onnx.data`（如使用 external data）
- `outputs/deploy/act_trt/<run_name>/export_metadata.json`

推荐命令形态：

```bash
CKPT="outputs/train/<run_name>/checkpoints/last/pretrained_model"
OUT="outputs/deploy/act_trt/<run_name>"

conda run -n lerobot_flex python \
  my_devs/docs/act_trt/reference_docs/onnx_export_reference/export/export_single.py \
  --checkpoint "${CKPT}" \
  --output "${OUT}/act_single.onnx" \
  --opset 17 \
  --device cpu \
  --verify \
  --simplify
```

#### 导出要求

- 第一版固定静态 `batch=1`
- 从 checkpoint 读取 camera order，不允许手写猜顺序
- 导出完成后必须有 ORT 验证结果
- `export_metadata.json` 必须记录：
  - 输入输出名
  - shape
  - camera order
  - checkpoint path
  - verification 指标

#### 通过标准

- `act_single.onnx` 可加载
- `export_metadata.json` 完整
- Torch vs ONNX 的 `max_abs_diff <= 1e-4`

### 6.2 步骤 2：构建 TensorRT engine

最终推荐策略是：

1. **先用 `trtexec` 验证 ONNX 能否被 TRT 接受**
2. **再用 Python builder 脚本生成最终可审计产物**

第一版构建原则：

- 静态 shape
- 静态 `batch=1`
- 先 `FP32`，再 `FP16`

原因：

- `FP32` 更适合做数值对齐与排障
- `FP16` 是最终部署候选，但不应跳过 `FP32` 对齐阶段

engine 的固定 IO 应为：

- `obs_state_norm`: `(1, 6)`
- `img0_norm`: `(1, 3, 480, 640)`
- `img1_norm`: `(1, 3, 480, 640)`
- `actions_norm`: `(1, 100, 6)`

最终推荐产物：

- `outputs/deploy/act_trt/<run_name>/act_single_fp32.plan`
- `outputs/deploy/act_trt/<run_name>/act_single_fp16.plan`
- `outputs/deploy/act_trt/<run_name>/build_cache.tcache`
- `outputs/deploy/act_trt/<run_name>/trt_build_summary.json`

Python builder 脚本的职责应包括：

- 读取 ONNX 与 metadata
- 静态 shape 校验
- 选择 FP32 / FP16
- 保存 timing cache
- 输出 build summary

### 6.3 步骤 3：实现 TRT runtime

建议新增：

- `my_devs/train/act/so101/act_trt_runtime.py`

`ActTrtRuntime` 的职责：

- 加载 `.plan` engine
- 建立 execution context
- 根据 tensor name 绑定输入输出
- 做 dtype / shape 自检
- 执行一次推理并返回 `actions_norm`

runtime 设计要求：

- 输入按名字喂，不按位置猜
- 第一版假设静态 shape，但仍建议做 shape 自检
- 输入 dtype 必须与 engine input dtype 匹配
- 输出统一转成 `torch.float32` 或 `numpy.float32` 再交给上层

第一版 runtime 的设备策略：

- **pre/post 与 observation 准备全部留在 CPU**
- TRT runtime 内部负责 H2D / D2H

这样做的原因不是“最快”，而是：

- 路径最稳定；
- 排障最简单；
- 避免早期陷入 Torch CUDA stream 与 TRT stream 同步问题。

### 6.4 步骤 4：实现 `ActTrtPolicyAdapter`

建议新增：

- `my_devs/train/act/so101/act_trt_policy.py`

`ActTrtPolicyAdapter` 的职责不是“重新实现整个 policy”，而是：

- 读取归一化 observation
- 按导出 metadata 中的 camera order 组装 TRT 输入
- 调用 TRT runtime 得到 `actions_norm`
- 在 Python 侧复刻 ACT 的 queue / temporal ensemble 语义
- 对外保持与 `policy.select_action()` 一致的接口语义

关键原则：

- adapter 返回的仍然是 **归一化后的 action**
- action 的反归一化继续由现有 `policy_postprocessor` 完成

队列语义应与现有 ACT 一致：

- `temporal_ensemble_coeff is None`：
  - 队列空 -> 跑一次 engine -> 取前 `n_action_steps` 入队
  - 队列非空 -> 直接弹出下一步 action
- `temporal_ensemble_coeff is not None`：
  - 每步都跑 engine -> 用 Python 侧 temporal ensembler 更新

必须遵守的约束：

- 若启用 temporal ensemble，则 `n_action_steps` 必须为 `1`
- `image_feature_keys` 必须来自 `export_metadata.json`
- 不允许在 adapter 内写死 `top / wrist`

### 6.5 步骤 5：接入推理脚本

最终选择：

- **新增独立脚本** `my_devs/train/act/so101/run_act_trt_infer.py`

而不是直接把 `run_act_infer.py` 改成多 backend 大杂烩。原因：

- 保留现有稳定 Torch 推理脚本作为基线
- TRT 分支能独立迭代
- 更利于问题隔离与线上回退

`run_act_trt_infer.py` 的要求：

- 尽量复用 `run_act_infer.py` 的参数与主循环结构
- 保持 robot / observation / processor 路径一致
- 唯一变化是 `policy` 从 Torch policy 变成 `ActTrtPolicyAdapter`

TRT 版脚本应加载：

- `config.json`
- `policy_preprocessor.json`
- `policy_postprocessor.json`
- 对应 stats safetensors
- `export_metadata.json`
- `.plan` engine

TRT 版脚本不应加载：

- `policy_class.from_pretrained(...)`
- Torch policy 权重

---

## 7. 推荐文件布局

### 7.1 文档

- `my_devs/docs/act_trt/act最终技术方案.md`

### 7.2 代码

- `my_devs/train/act/so101/build_act_trt_engine.py`
- `my_devs/train/act/so101/act_trt_runtime.py`
- `my_devs/train/act/so101/act_trt_policy.py`
- `my_devs/train/act/so101/run_act_trt_infer.py`

### 7.3 部署产物

- `outputs/deploy/act_trt/<run_name>/act_single.onnx`
- `outputs/deploy/act_trt/<run_name>/act_single.onnx.data`（可选）
- `outputs/deploy/act_trt/<run_name>/export_metadata.json`
- `outputs/deploy/act_trt/<run_name>/act_single_fp32.plan`
- `outputs/deploy/act_trt/<run_name>/act_single_fp16.plan`
- `outputs/deploy/act_trt/<run_name>/build_cache.tcache`
- `outputs/deploy/act_trt/<run_name>/trt_build_summary.json`

选择 `outputs/deploy/act_trt/...` 而不是直接把产物写回 checkpoint 目录，原因是：

- 保持训练产物与部署产物职责分离；
- 避免污染 checkpoint 目录；
- 更利于复现、审计与版本管理。

---

## 8. 验证方案

必须严格按层验证，不允许跳层直接上机器人。

### 8.1 第 1 层：Torch vs ONNX

目标：验证导出正确。

输入：同一份 normalized tensor。  
输出：比较 `actions_norm`。

通过标准：

- `max_abs_diff <= 1e-4`

### 8.2 第 2 层：ONNX vs TRT-FP32

目标：验证 TensorRT build / runtime 正确。

通过标准：

- shape 完全一致
- dtype 符合预期
- `max_abs_diff` 在可接受范围内

### 8.3 第 3 层：TRT adapter vs 原 Torch policy

目标：验证 queue / temporal ensemble 语义复刻正确。

对齐方式：

- 用同一帧或同一段 observation 序列
- 一边走原 Torch policy
- 一边走 `ActTrtPolicyAdapter`
- 比较最终输出 action 或中间 `actions_norm`

建议指标：

- `max_abs_diff`
- `mean_abs_diff`
- `cosine_similarity`

### 8.4 第 4 层：机器人在线短时验证

目标：验证真实部署闭环稳定性。

要求：

- 先短时 dry-run / 低风险动作验证
- 先保持 `n_action_steps=100`
- 确认稳定后再逐步下调到 `32 / 16 / 8`

### 8.5 所有阶段都要做的 engine IO 自检

每次接入新的 engine，先确认：

- tensor name 正确
- input shape 正确
- input dtype 与 engine 一致
- output shape 与 metadata 一致
- camera order 与 metadata 一致

---

## 9. 性能评估口径

必须把“模型前向性能”和“整条控制循环性能”分开看。

建议在 `run_act_trt_infer.py` 中记录：

- `t_obs`：采集观测耗时
- `t_prepare`：`prepare_observation_for_inference + preprocessor` 耗时
- `t_model`：TRT forward 耗时
- `t_post`：`postprocessor + action mapping` 耗时
- `t_total`：单循环总耗时

建议统计：

- `p50`
- `p95`

解释口径要统一：

- 在 `n_action_steps=100` 下，`t_model` 提升不一定明显反映到 `t_total`
- 真正体现 TRT 价值的阶段，是把 `n_action_steps` 下调后仍能稳定保持控制频率

---

## 10. 上线顺序

最终推荐按下面顺序推进：

1. **保留 Torch 基线并加 timing 日志**  
   先知道现在慢在哪。
2. **导出 ONNX 并完成 ORT 验证**  
   不通过就不进入 TRT。
3. **用 `trtexec` 验证可编译性**  
   只确认 ONNX 能否被 TRT 接受。
4. **构建 TRT-FP32 engine**  
   用于正确性对齐。
5. **实现 runtime + adapter，并做离线对齐**  
   先不连机器人。
6. **构建 TRT-FP16 engine**  
   在 FP32 对齐完成后再切。
7. **接入 `run_act_trt_infer.py` 做短时在线验证**
8. **逐步下调 `n_action_steps`**  
   推荐顺序：`100 -> 32 -> 16 -> 8`

---

## 11. 风险与规避

### 风险 1：相机顺序错误

现象：动作明显跑偏，但数值上“看起来能跑”。

规避：

- camera order 必须从 `export_metadata.json` 读取
- 不允许手写 `img0=top`, `img1=wrist`

### 风险 2：一开始就上 FP16，排障困难

规避：

- 先完成 Torch -> ONNX -> TRT-FP32 对齐
- 再切 TRT-FP16

### 风险 3：CPU / GPU 来回搬运过多

规避：

- 第一版明确走“CPU pre/post + TRT 内部 H2D/D2H”
- 路线跑稳后再评估零拷贝优化

### 风险 4：外部权重 ONNX 解析问题

规避：

- 若 ONNX 使用 `.onnx.data`，builder 优先使用按文件路径解析的方式
- 不要默认假设单文件 `onnx_bytes` 一定可用

### 风险 5：过早追求全链路 TRT

规避：

- 第一版只替换核心网络前向
- pre/post 融合、零拷贝、INT8 都放到增强阶段

---

## 12. 里程碑定义

### 里程碑 M1：导出闭环成立

产物：

- `act_single.onnx`
- `export_metadata.json`
- Torch vs ONNX 验证通过

### 里程碑 M2：TRT 编译闭环成立

产物：

- `act_single_fp32.plan`
- `trt_build_summary.json`
- runtime 可跑通一次 forward

### 里程碑 M3：语义闭环成立

产物：

- `ActTrtPolicyAdapter`
- 离线对齐报告

### 里程碑 M4：实机闭环成立

产物：

- `run_act_trt_infer.py`
- 短时稳定运行日志
- 性能统计

### 里程碑 M5：重规划频率提升验证通过

产物：

- `n_action_steps` 下调后的实机数据
- 可接受的稳定性与性能结论

---

## 13. 后续增强项

以下项目不属于第一版必需项，但在主路线跑稳后可以评估：

1. **把 pre/post 融入 ONNX / TensorRT**  
   目标：减少 Python / Torch 依赖。
2. **零拷贝输入**  
   目标：减少 CPU/GPU 搬运。
3. **dynamic batch / dynamic resolution**  
   仅在确有需求时评估。
4. **INT8**  
   需要校准，且精度风险更高。

这些增强项的顺序必须排在：

```text
先跑稳核心前向 TRT
  -> 再压缩数据搬运
  -> 最后再做更激进的图融合和量化
```

---

## 14. 最终执行摘要

最终采用的不是“全链路 TRT”方案，而是“**核心前向 TRT + 现有控制语义保留**”方案。

它的技术优点是：

- 边界清晰
- 风险最低
- 可验证性最好
- 与现有 LeRobot 链路最一致
- 最适合作为后续下调 `n_action_steps`、提升重规划频率的基础版本

如果后续要继续深挖性能，正确顺序是：

1. 先把本文方案跑稳
2. 再评估 pre/post 融合与零拷贝
3. 最后才考虑更激进的端到端优化
