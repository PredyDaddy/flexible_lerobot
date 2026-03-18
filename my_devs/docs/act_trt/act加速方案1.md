# ACT 加速方案 1

> 文档日期：2026-03-06  
> 适用对象：`my_devs/train/act/so101/run_act_infer.py` 当前使用的 ACT 单臂部署链路  
> 目标：在尽量不改机器人侧行为语义的前提下，把 ACT 核心网络前向从 PyTorch 切到 ONNX/TensorRT，降低单次前向延迟，并为后续把 `n_action_steps` 从 `100` 下调到 `32/16/8` 提供可落地基础。

---

## 1. 先说结论

这次我建议采用一个**最稳、最容易验证、对现有代码侵入最小**的方案：

- **保留 Python 侧现有链路**：机器人连接、相机采集、`build_dataset_frame`、`prepare_observation_for_inference`、前后处理器、动作队列、机器人动作发送都不动。
- **只替换 ACT 核心网络前向**：把当前 `policy.model(batch)` 这一段导出成 ONNX，再构建 TensorRT engine，用 TensorRT 产生 `actions_norm`。
- **动作队列继续放在 Python 侧**：不要尝试导出 `ACTPolicy.select_action()` 的 deque 逻辑。
- **第一阶段不要把预处理塞进 ONNX/TRT**：先保证语义对齐和部署稳定；后续如果要继续榨性能，再考虑“方案 2”。

这个方案的本质是：

```text
原始观测
  -> build_dataset_frame
  -> prepare_observation_for_inference
  -> policy_preprocessor
  -> TensorRT engine（替代 policy.model）
  -> Python action queue
  -> policy_postprocessor
  -> robot.send_action
```

它的最大优点不是“代码最炫”，而是：

1. **最容易和现有 `run_act_infer.py` 对齐**。
2. **最容易做 Torch / ONNX / TRT 三方一致性验证**。
3. **最适合先上线，再逐步优化**。
4. **一旦验证通过，就能支持把 `n_action_steps` 降低，提升重规划频率**。

---

## 2. 当前链路的真实情况（基于仓库当前代码）

### 2.1 当前推理脚本

当前线上推理脚本是：

- `my_devs/train/act/so101/run_act_infer.py`

脚本主循环本质上是：

1. `robot.get_observation()` 读取机器人状态和两路相机图像。
2. `robot_observation_processor(obs)` 做机器人观测适配。
3. `build_dataset_frame(...)` 生成标准观测字典。
4. `predict_action(...)` 内部继续做：
   - `prepare_observation_for_inference`
   - `policy_preprocessor`
   - `policy.select_action`
   - `policy_postprocessor`
5. `make_robot_action(...)` / `robot_action_processor(...)`
6. `robot.send_action(...)`

所以真正值得替换的热点，不是整条链，而是 `predict_action(...)` 里的这一步：

- `policy.select_action(...)`

对于 ACT，这一步进一步可拆成：

- Python 队列逻辑：`ACTPolicy.select_action()`
- 核心网络前向：`ACTPolicy.predict_action_chunk()` -> `policy.model(batch)`

### 2.2 当前默认 checkpoint 的模型事实

`run_act_infer.py` 当前默认 checkpoint 为：

- `outputs/train/20260305_190147_act_grasp_block_in_bin1_e15/checkpoints/last/pretrained_model`

从该 checkpoint 的 `config.json` 可以确认：

- `type = act`
- `chunk_size = 100`
- `n_action_steps = 100`
- `use_vae = true`
- `temporal_ensemble_coeff = null`
- 输入特征：
  - `observation.state`: shape `[6]`
  - `observation.images.top`: shape `[3, 480, 640]`
  - `observation.images.wrist`: shape `[3, 480, 640]`
- 输出特征：
  - `action`: shape `[6]`

也就是说，你现在这套 ACT 实际是：

- **状态维度**：6
- **相机数量**：2 路
- **相机顺序**：`top -> wrist`
- **动作维度**：6
- **每次前向输出**：`(1, 100, 6)`

### 2.3 一个非常关键的现实：当前平均推理压力其实没你想象中大

因为当前配置是：

- `chunk_size = 100`
- `n_action_steps = 100`
- `fps = 30`

这意味着：

- 模型**每 100 个控制周期**才真正前向一次。
- 按 `30 FPS` 算，相当于**每 3.33 秒前向一次**。

所以如果你保持 `n_action_steps=100` 不变，那么 TRT 加速的收益主要体现在：

- 单次前向更快
- 启动后首次动作延迟更低
- GPU 占用更可控

但**对平均循环周期提升并不会特别夸张**，因为大部分周期都只是“从动作队列里取下一步”。

因此，这个加速方案真正的价值是：

- 在保持 30Hz 控制循环的同时，支持把 `n_action_steps` 从 `100` 降到 `32 / 16 / 8`
- 让机器人更频繁地重规划，而不是每 3.33 秒才重新看一次环境

对应的重规划周期大概是：

- `n_action_steps=100` -> `100 / 30 = 3.33s`
- `n_action_steps=32` -> `1.07s`
- `n_action_steps=16` -> `0.53s`
- `n_action_steps=8` -> `0.27s`

这也是我建议优先做 TRT 的根本原因。

---

## 3. 为什么“方案 1”要只替换核心网络前向

### 3.1 不建议直接导出 `ACTPolicy.select_action()`

`ACTPolicy.select_action()` 包含 Python 侧状态机逻辑：

- `deque`
- `queue empty -> run model`
- `queue non-empty -> pop next action`

这类逻辑不适合直接导出到 ONNX/TensorRT，原因有三点：

1. **动态图状态机不适合 ONNX 表达**。
2. **即使勉强导出，可维护性也很差**。
3. **这部分逻辑本身开销很小，不值得为了它牺牲可读性和可验证性**。

### 3.2 只导出 `policy.model(batch)` 最稳

当前最合理的导出粒度是：

- 输入：**已经做完图像布局变换 + `/255` + mean/std 归一化的张量**
- 输出：**归一化动作块 `actions_norm`**

也就是：

```text
normalized obs_state + normalized img0 + normalized img1
    -> ACT core model
    -> normalized action chunk
```

这样做的优点：

- 前后处理和 checkpoint 统计量完全复用现有实现
- ONNX/TRT 只关注张量计算，不碰业务语义
- 最容易验证 Torch vs ONNX vs TRT 的数学等价性

### 3.3 这个模型虽然 `use_vae=true`，但推理阶段仍然是确定性的

`src/lerobot/policies/act/modeling_act.py` 里可以看到：

- 只有在 `self.config.use_vae and ACTION in batch and self.training` 时，才走 VAE encoder 采样分支
- 推理时模型是 `eval()`
- 推理 batch 不会包含训练目标 `ACTION`

因此部署时会走：

- `latent_sample = zeros([B, latent_dim])`

也就是说，这个导出路径**没有随机采样问题**，更适合稳定部署。

---

## 4. 方案 1 的边界：哪些保留，哪些替换

### 4.1 保留在 Python 侧的部分

这些保持现状，不建议第一阶段改：

- 机器人连接与断开
- 相机采集
- `robot_observation_processor`
- `build_dataset_frame(...)`
- `prepare_observation_for_inference(...)`
- `policy_preprocessor`
- Python 侧 action queue
- `policy_postprocessor`
- `make_robot_action(...)`
- `robot_action_processor(...)`
- `precise_sleep(...)`

### 4.2 替换为 ONNX / TensorRT 的部分

只替换这一段：

```text
policy.select_action(batch)
  └─ queue empty 时，会触发 policy.model(batch)
```

替换后的结构：

```text
ActTrtPolicyAdapter.select_action(batch)
  ├─ queue empty: TensorRT engine infer -> 得到 actions_norm
  ├─ queue fill: 把前 n_action_steps 填进 Python deque
  └─ queue pop: 返回单步 normalized action
```

### 4.3 这一版不建议做的事情

下面这些事情不是不能做，而是**不应该在第一版一起做**：

- 把图像 `/255` + `HWC->CHW` 合进 ONNX
- 把 normalizer / unnormalizer 合进 ONNX
- 把队列逻辑改到 C++/CUDA
- 把机器人 I/O 和引擎执行合并成一个“端到端黑盒”
- 一开始就做 dynamic shape + 多 profile + 多精度同时上线

第一版最重要的是：

- 先导出对
- 先 build 对
- 先跑通一条和当前语义一致的 TRT 推理链

---

## 5. 最推荐的目标架构

```text
robot.get_observation()
    -> robot_observation_processor
    -> build_dataset_frame
    -> prepare_observation_for_inference   # HWC uint8 -> CHW float32/255，保留
    -> policy_preprocessor                 # mean/std 归一化，保留
    -> ActTrtPolicyAdapter.select_action   # 替代原 Torch policy
         -> TensorRT engine infer
         -> Python deque fill/pop
    -> policy_postprocessor                # action 反归一化，保留
    -> make_robot_action
    -> robot_action_processor
    -> robot.send_action
```

这个架构最关键的设计点有两个：

### 5.1 队列仍然在 Python

因为队列逻辑本来就轻，而且和部署策略强相关：

- `n_action_steps=100`、`32`、`16`、`8`
- 未来可能还会接 temporal ensemble

把它放在 Python 更灵活。

### 5.2 前处理建议继续复用 `PolicyProcessorPipeline`

不要在第一版里手写 safetensors 统计量解析。最佳实践是继续复用：

- `policy_preprocessor.json`
- `policy_postprocessor.json`
- 对应 `.safetensors` 状态文件

这样做的好处是：

- 和训练、原始部署链路强一致
- 减少“我以为归一化和原来一样，结果差一个 eps/shape/key”的风险
- 后续换 checkpoint 也更稳

---

## 6. 导出方案：先拿到稳定的 ONNX 中间产物

## 6.1 导出目标

建议导出一个**静态 batch=1** 的 ONNX 模型，输入输出如下：

- 输入：
  - `obs_state_norm`: `(1, 6)`
  - `img0_norm`: `(1, 3, 480, 640)`
  - `img1_norm`: `(1, 3, 480, 640)`
- 输出：
  - `actions_norm`: `(1, 100, 6)`

这里的 `img0_norm` / `img1_norm` 在当前模型中应映射为：

- `img0_norm` -> `observation.images.top`
- `img1_norm` -> `observation.images.wrist`

### 6.2 为什么第一版强烈建议导出静态 batch=1

你的机器人在线推理场景是：

- 单机器人
- 单帧决策
- 固定输入分辨率 `480x640`
- 固定状态维 `6`

因此第一版没有必要引入 dynamic batch。静态导出的好处：

- build 脚本简单很多
- engine 更容易稳定
- 推理脚本更简单
- 排障成本更低

等这条链路完全稳定之后，如果你未来有下面这些需求，再考虑 dynamic shape：

- 批量离线回放
- 不同分辨率相机混跑
- 多 profile benchmark

### 6.3 直接复用现成导出脚本

仓库里已经有很合适的参考导出脚本：

- `my_devs/docs/act_trt/reference_docs/onnx_export_reference/export/export_single.py`

第一阶段建议**直接复用它**，不要另写一份导出脚本。

### 6.4 建议的导出命令

以下命令默认在 `lerobot_flex` 环境里执行；前提是该环境已经安装：

- `torch`
- `onnx`
- `onnxruntime`
- `onnxsim`（可选）

建议命令：

```bash
conda run -n lerobot_flex python \
  my_devs/docs/act_trt/reference_docs/onnx_export_reference/export/export_single.py \
  --checkpoint outputs/train/20260305_190147_act_grasp_block_in_bin1_e15/checkpoints/last/pretrained_model \
  --output outputs/deploy/act_trt/20260305_190147_act_grasp_block_in_bin1_e15/act_single.onnx \
  --device cuda \
  --opset 17 \
  --verify \
  --simplify
```

如果你想极度保守，先走 CPU 导出也可以：

```bash
conda run -n lerobot_flex python \
  my_devs/docs/act_trt/reference_docs/onnx_export_reference/export/export_single.py \
  --checkpoint outputs/train/20260305_190147_act_grasp_block_in_bin1_e15/checkpoints/last/pretrained_model \
  --output outputs/deploy/act_trt/20260305_190147_act_grasp_block_in_bin1_e15/act_single.onnx \
  --device cpu \
  --opset 17 \
  --verify \
  --simplify
```

### 6.5 导出后必须检查的产物

建议导出目录统一放在：

- `outputs/deploy/act_trt/20260305_190147_act_grasp_block_in_bin1_e15/`

预期至少产生：

- `act_single.onnx`
- `export_metadata.json`

其中 `export_metadata.json` 必须重点看这几个字段：

- `camera_order_visual_keys`
- `io_mapping`
- `shapes`
- `verification`
- `simplification`

尤其是当前模型，预期应该是：

```json
{
  "camera_order_visual_keys": [
    "observation.images.top",
    "observation.images.wrist"
  ]
}
```

如果这里顺序错了，后面 TRT 推理就一定错。

### 6.6 导出阶段的通过标准

第一阶段建议用参考脚本的验证门限：

- `Torch vs ONNXRuntime max_abs_diff <= 1e-4`

如果导出脚本已经报告验证通过，说明：

- ONNX 模型和当前 Torch 核心网络基本对齐

这时才值得进入 TensorRT build 阶段。

### 6.7 导出阶段的常见风险

#### 风险 1：相机顺序搞错

务必以 `config.json` / `export_metadata.json` 为准，不要靠主观记忆。

#### 风险 2：导出时触发 torchvision 权重下载

参考导出脚本已经有 `--keep-backbone-pretrained-weights` 开关；默认不保留 pretrained backbone 配置，更适合离线环境。

#### 风险 3：把 `select_action()` 当成导出目标

这是最容易把事情复杂化的错误。导出目标应该始终是：

- `policy.model`

而不是：

- `policy.select_action`

---

## 7. TensorRT 引擎构建方案

### 7.1 第一阶段的 build 目标

建议 build 两个版本，但按顺序来：

1. **先 build FP32 engine**：用于功能对齐
2. **再 build FP16 engine**：用于性能优化

原因很简单：

- FP32 更容易验证数值一致性
- FP16 才是最终部署候选，但不应该跳过 FP32 这个对齐阶段

### 7.2 第一版 engine 建议使用静态 shape

如果导出的是静态 batch=1 ONNX，那么 TensorRT build 脚本也应优先走静态 shape：

- `obs_state_norm = (1, 6)`
- `img0_norm = (1, 3, 480, 640)`
- `img1_norm = (1, 3, 480, 640)`

这种情况下：

- 不需要 optimization profile
- build 脚本更简单
- runtime 也更简单

### 7.3 参考脚本可以借鉴什么，不能直接照搬什么

现有参考 build 脚本：

- `my_devs/docs/act_trt/reference_docs/trt_reference/trt_build.py`

它的优点：

- API 调用顺序正确
- parser error 处理正确
- timing cache 处理成熟
- build 后反序列化验证成熟

但它**不能直接拿来给 ACT 用**，因为它当前更像一个单输入分类模型的 build 参考，尤其这里：

- 动态 profile 只处理了一个 `input_name`
- 示例 shape 是 `(N, 3, 224, 224)`

ACT 版本需要改成：

- 支持 **3 个输入张量**
- 支持 ACT 的实际输入 shape
- 输出是 `actions_norm`，不是分类 logits

### 7.4 推荐新增的 build 脚本职责

建议后续真正落地时新增：

- `my_devs/train/act/so101/build_act_trt_engine.py`

这个脚本应该负责：

1. 读取 `act_single.onnx`
2. 解析 ONNX，失败时打印全部 parser errors
3. 设置 workspace / optimization level / precision
4. 可选加载 timing cache
5. 构建 `.plan`
6. 回读反序列化验证
7. 输出 `build_summary.json`

### 7.5 build 脚本推荐参数

建议 build 脚本具备这些参数：

- `--onnx`
- `--engine`
- `--precision {fp32,fp16}`
- `--workspace-bytes`
- `--opt-level`
- `--timing-cache`
- `--disable-timing-cache`
- `--force-static-shape`
- `--min-shapes` / `--opt-shapes` / `--max-shapes`（仅在未来 dynamic shape 需要时启用）

第一版默认建议：

- `precision = fp32` 或 `fp16`
- `workspace-bytes = 4 GiB` 起步
- `opt-level = 3` 先求稳，稳定后再试更高优化等级

### 7.6 build 命令建议形态

下面是**建议的命令形态**；这里假设未来 build 脚本已经落地到 `my_devs/train/act/so101/build_act_trt_engine.py`：

```bash
conda run -n lerobot_flex python \
  my_devs/train/act/so101/build_act_trt_engine.py \
  --onnx outputs/deploy/act_trt/20260305_190147_act_grasp_block_in_bin1_e15/act_single.onnx \
  --engine outputs/deploy/act_trt/20260305_190147_act_grasp_block_in_bin1_e15/act_single_fp32.plan \
  --precision fp32 \
  --workspace-bytes 4294967296 \
  --opt-level 3
```

然后再 build FP16：

```bash
conda run -n lerobot_flex python \
  my_devs/train/act/so101/build_act_trt_engine.py \
  --onnx outputs/deploy/act_trt/20260305_190147_act_grasp_block_in_bin1_e15/act_single.onnx \
  --engine outputs/deploy/act_trt/20260305_190147_act_grasp_block_in_bin1_e15/act_single_fp16.plan \
  --precision fp16 \
  --workspace-bytes 4294967296 \
  --opt-level 3
```

### 7.7 如果你未来必须支持 dynamic batch，该怎么做

虽然第一版不推荐，但如果后面确实要做动态 batch，那么 optimization profile 必须同时覆盖三个输入：

- `obs_state_norm`
- `img0_norm`
- `img1_norm`

例如：

- `obs_state_norm`: min/opt/max = `(1,6)` / `(1,6)` / `(1,6)`
- `img0_norm`: min/opt/max = `(1,3,480,640)` / `(1,3,480,640)` / `(1,3,480,640)`
- `img1_norm`: min/opt/max = `(1,3,480,640)` / `(1,3,480,640)` / `(1,3,480,640)`

注意：

- 这里即使 batch 维“理论上动态”，如果线上永远只跑 batch=1，也没必要引入真动态 profile
- 对机器人在线部署来说，static shape 的性价比通常最高

### 7.8 build 阶段建议输出的产物

建议 build 脚本最终产出：

- `act_single_fp32.plan`
- `act_single_fp16.plan`
- `act_single_fp16.tcache`（如果启用 timing cache）
- `build_summary_fp32.json`
- `build_summary_fp16.json`

`build_summary_*.json` 建议记录：

- onnx 路径
- engine 路径
- TensorRT 版本
- precision
- workspace bytes
- opt level
- 是否启用 timing cache
- engine 大小
- engine 输入输出名称和 shape

---

## 8. TRT 推理脚本应该怎么设计

### 8.1 不建议一上来就重写整份 `run_act_infer.py`

更稳的做法是：

- 先新增一个 TRT 适配器类
- 再新增一个 TRT 版运行脚本
- 等功能完全稳定后，再考虑把两者合并成一个带 `--policy-backend` 的统一脚本

我建议的落地顺序是：

1. 新增 `ActTrtPolicyAdapter`
2. 新增 `run_act_trt_infer.py`
3. 验证稳定后，再考虑把 `run_act_infer.py` 改成支持 `torch/trt` 双后端

### 8.2 推荐新增的脚本/模块

建议后续新增：

- `my_devs/train/act/so101/act_trt_policy.py`
- `my_devs/train/act/so101/run_act_trt_infer.py`

其中职责建议如下：

#### `act_trt_policy.py`

负责：

- 加载 TensorRT engine
- 管理 CUDA stream / pinned host / device buffer
- 做 3 输入张量的 name-based 推理
- 管理 action queue
- 对外暴露和原 policy 类似的 `reset()` / `select_action()` 接口

#### `run_act_trt_infer.py`

负责：

- 复用当前 `run_act_infer.py` 的机器人配置、相机配置、主循环
- 加载 preprocessor / postprocessor
- 用 `ActTrtPolicyAdapter` 替换原 Torch policy
- 维持原有日志和控制频率

### 8.3 为什么推荐“适配器”而不是重写 `predict_action()`

当前 `predict_action(...)` 已经把主链路封装得很好了：

- 预处理
- preprocessor
- policy.select_action
- postprocessor

如果 `ActTrtPolicyAdapter` 能做到接口兼容，那么大量现有逻辑可以直接复用。

建议适配器最少提供：

- `config`
- `reset()`
- `select_action(batch)`

这样 `predict_action(...)` 可以继续使用，不需要推翻已有调用链。

---

## 9. `ActTrtPolicyAdapter` 的详细设计

### 9.1 适配器的输入输出语义

适配器接收的是：

- **已经过 `prepare_observation_for_inference` 和 `policy_preprocessor` 处理后的 batch**

也就是说，`select_action(batch)` 的输入应已经是：

- `observation.state`: `(1, 6)`，normalized
- `observation.images.top`: `(1, 3, 480, 640)`，normalized
- `observation.images.wrist`: `(1, 3, 480, 640)`，normalized

适配器输出应为：

- 单步 **normalized action**，shape `(1, 6)`

这样 `policy_postprocessor` 就可以像现在一样继续做 action 反归一化。

### 9.2 适配器为什么应该返回 normalized action

因为当前原始链路是：

- `policy.select_action(...)` -> 得到 normalized action
- `policy_postprocessor(...)` -> 反归一化到机器人动作空间

TRT 版如果也返回 normalized action，就可以完全复用：

- `policy_postprocessor`

从工程上看，这是**最小语义改动**。

### 9.3 适配器内部的核心逻辑

内部逻辑应与 `ACTPolicy.select_action()` 保持一致：

1. 如果队列为空：
   - 取当前 normalized observation
   - 按固定 camera order 组装 TRT 输入
   - 执行一次 TensorRT 前向
   - 得到 `actions_norm`，shape `(1, chunk_size, action_dim)`
   - 只取前 `n_action_steps`
   - 以 `transpose(0, 1)` 的方式填进 Python deque
2. 如果队列非空：
   - 直接 `popleft()` 返回下一个动作

### 9.4 建议的伪代码

```python
from collections import deque
import copy
import numpy as np
import torch


class ActTrtPolicyAdapter:
    def __init__(self, policy_cfg, engine, camera_order):
        self.config = copy.deepcopy(policy_cfg)
        self.config.device = "cpu"
        self.config.use_amp = False
        self.engine = engine
        self.camera_order = list(camera_order)
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def reset(self):
        self._action_queue.clear()

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if len(self._action_queue) == 0:
            feed_dict = {
                "obs_state_norm": np.ascontiguousarray(
                    batch["observation.state"].detach().cpu().numpy().astype(np.float32)
                ),
                "img0_norm": np.ascontiguousarray(
                    batch[self.camera_order[0]].detach().cpu().numpy().astype(np.float32)
                ),
                "img1_norm": np.ascontiguousarray(
                    batch[self.camera_order[1]].detach().cpu().numpy().astype(np.float32)
                ),
            }
            actions_norm = self.engine.infer(feed_dict)["actions_norm"]
            actions_norm = actions_norm[:, : self.config.n_action_steps]
            actions_norm = torch.from_numpy(actions_norm.astype(np.float32))
            self._action_queue.extend(actions_norm.transpose(0, 1))

        return self._action_queue.popleft()
```

这个设计的关键点有两个：

- 返回值保持为 `torch.Tensor`
- 返回的是 normalized 单步 action

这样才能和现有 `postprocessor` 无缝衔接。

### 9.5 一个非常重要的设备设计：TRT 路线下，Python 侧建议走 CPU tensor

如果直接复用当前 Torch policy 的配置，`predict_action(...)` 会用：

- `device=get_safe_torch_device(policy.config.device)`

也就是说，如果 `policy.config.device == "cuda"`，那么：

- `prepare_observation_for_inference(...)` 会先把张量放到 GPU
- `policy_preprocessor` 也会在 GPU 上处理
- 而 TensorRT runtime 通常又需要自己做 H2D

这样会产生**没必要的 GPU -> CPU / CPU -> GPU 数据折返**。

因此 TRT 路线我建议：

- `ActTrtPolicyAdapter.config.device = "cpu"`
- `ActTrtPolicyAdapter.config.use_amp = False`

这样 `predict_action(...)` 会让：

- `prepare_observation_for_inference(...)` 产出 CPU tensor
- `policy_preprocessor` 也停留在 CPU
- 然后 TRT runtime 自己完成 H2D

这个设计更合理，也更接近标准 TensorRT runtime 分工。

### 9.6 preprocessor 的设备也建议显式 override 到 CPU

当前 checkpoint 的 `policy_preprocessor.json` 里，`device_processor` 是：

- `device = cuda`

TRT 路线下不建议继续沿用这个值。最佳实践是 load pipeline 时直接 override：

```python
preprocessor = PolicyProcessorPipeline.from_pretrained(
    pretrained_model_name_or_path=str(policy_path),
    config_filename="policy_preprocessor.json",
    overrides={"device_processor": {"device": "cpu"}},
    to_transition=batch_to_transition,
    to_output=transition_to_batch,
)
```

这样你既复用了原始 normalizer，又避免了不必要的 GPU tensor 搬运。

postprocessor 目前本来就是 CPU 路线，可以保持不变。

---

## 10. TensorRT runtime 层应该怎么写

### 10.1 参考脚本里哪些部分可以直接借鉴

现有参考 runtime：

- `my_devs/docs/act_trt/reference_docs/trt_reference/trt_infer.py`

其中这些部分非常值得直接借鉴：

- CUDA error 封装
- pinned host memory
- device memory 管理
- stream 管理
- name-based tensor API
- `context.execute_async_v3(...)`
- output buffer 动态扩容

这些设计都比较成熟。

### 10.2 但 ACT 版 runtime 不能直接照抄分类示例

参考脚本里的 `_infer_once(...)` 当前假设：

- 只有 1 个输入
- 输入是 `[N,3,224,224]`
- 输出类似 logits

ACT runtime 必须改成：

- 支持 3 个输入
- 支持输入名称映射
- 输出 `actions_norm`
- 输出 shape 为 `(1, 100, 6)`

### 10.3 ACT 版 runtime 的推荐职责

建议 TRT runtime 类至少实现：

- `__init__(engine_path)`
- `infer(feed_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]`
- `close()` / `__del__()`

### 10.4 `infer(feed_dict)` 的标准步骤

建议严格按下面顺序执行：

1. 反序列化 engine
2. 创建 execution context
3. 枚举 `engine.num_io_tensors`
4. 按 name-based API 区分 input / output
5. 对每个输入：
   - `context.set_input_shape(name, shape)`
6. `context.infer_shapes()`
7. 为每个输入输出准备 pinned host / device buffer
8. H2D 拷贝
9. `context.set_tensor_address(name, ptr)`
10. `context.execute_async_v3(stream_handle=...)`
11. D2H 拷贝
12. `cudaStreamSynchronize(...)`
13. 把 `actions_norm` 转成 `np.float32`

### 10.5 第一版 TRT runtime 的输入约束建议写死

第一版建议明确约束：

- batch size 只支持 `1`
- 相机只支持 2 路
- 当前 checkpoint 只支持 `state_dim=6`
- 图像固定 `480x640`

原因不是“做不到更通用”，而是：

- 第一版应该优先求稳
- 先把线上这一个模型跑通
- 未来再做通用化重构

### 10.6 输出 dtype 的处理建议

不论 engine 最终输出是：

- FP32
- FP16

在返回 Python 侧之前，都建议统一转成：

- `np.float32`

然后再转成：

- `torch.float32`

原因：

- 后处理和比较逻辑更简单
- 避免 FP16 在 Python 后处理时引入没必要的差异

---

## 11. `run_act_trt_infer.py` 建议长什么样

### 11.1 最小功能目标

这份脚本应该做到：

- 输入参数风格尽量贴近 `run_act_infer.py`
- 机器人和相机配置完全复用
- 主循环结构完全复用
- 唯一区别是 policy 从 Torch 变成 TRT adapter

### 11.2 第一版不要加载 Torch policy 权重

TRT 版推理脚本不应该再做：

- `policy_class.from_pretrained(...)`
- `policy.to(device)`

因为这会：

- 浪费加载时间
- 额外占 GPU 显存
- 混淆真正的部署路径

TRT 版脚本应该只加载：

- `config.json`
- `policy_preprocessor.json`
- `policy_postprocessor.json`
- 对应 safetensors stats
- `.plan` engine

### 11.3 脚本建议新增的参数

建议 TRT 版脚本具备：

- `--policy-path`
- `--trt-engine`
- `--camera-order top,wrist`（通常可选，默认从 metadata 自动读取）
- `--policy-n-action-steps`
- `--policy-temporal-ensemble-coeff`（第一版可以不支持，但参数位可以保留）
- `--fps`
- `--run-time-s`
- `--log-interval`
- `--dry-run`

### 11.4 建议优先从 `export_metadata.json` 自动读取 camera order

不要在 TRT 脚本里手写：

- `img0=top`
- `img1=wrist`

更稳的做法是：

- 默认读取 `export_metadata.json`
- 取 `camera_order_visual_keys`
- 再映射为 TRT 输入顺序

当前这套模型预期应为：

- `img0_norm <- observation.images.top`
- `img1_norm <- observation.images.wrist`

但推荐仍然从 metadata 自动读取，避免后续 checkpoint 变化时踩坑。

### 11.5 主循环中建议增加的 timing 日志

为了后续判断瓶颈，建议 TRT 版脚本按阶段记时：

- `t_obs`: 采集观测耗时
- `t_prepare`: `prepare_observation_for_inference` + preprocessor 耗时
- `t_model`: TRT forward 耗时（只在 queue 为空时触发）
- `t_post`: postprocessor + action mapping 耗时
- `t_total`: 单循环总耗时

如果日志太多，可以只在 `log_interval` 打印一次统计。

---

## 12. 验证方案：一定要分层验证，不要直接上机器人硬跑

### 12.1 验证的总原则

必须分 4 层验证：

1. **Torch vs ONNX**
2. **ONNX vs TRT**
3. **TRT adapter vs 原 Torch policy**
4. **机器人在线短时验证**

不要跳层。

### 12.2 第 1 层：Torch vs ONNX

这个阶段参考导出脚本已经基本覆盖。

通过标准：

- `max_abs_diff <= 1e-4`

### 12.3 第 2 层：ONNX vs TRT

建议单独写一个小测试脚本，输入为**同一份 normalized tensor**，比较：

- ONNXRuntime 输出的 `actions_norm`
- TensorRT 输出的 `actions_norm`

比较维度：

- `max_abs_diff`
- `mean_abs_diff`
- `output_shape`
- `dtype`

建议策略：

1. 先比 FP32 engine
2. 再比 FP16 engine

### 12.4 第 3 层：TRT adapter vs 原 Torch policy（非常关键）

建议做一个“同观测、同队列逻辑”的回放验证：

1. 固定一批离线观测样本（建议先保存成 `.npz`）
2. 用原 Torch policy 跑出一串 step-by-step action
3. 用 TRT adapter 跑同一串观测
4. 比较：
   - normalized action
   - postprocessed action

这一层必须验证的是：

- 不仅单次 chunk 相似
- **队列消耗语义也一致**

### 12.5 第 4 层：机器人在线验证

在线验证建议分两步：

#### 第一步：只看动作，不发机器人

建议加一个 dry-run 模式：

- 正常读观测
- 正常跑 TRT
- 正常打印 action
- 不调用 `robot.send_action(...)`

先看：

- 是否卡顿
- 是否有 shape 问题
- 是否有偶发 engine error

#### 第二步：短时间真实执行

建议先跑：

- `10s ~ 30s`

观察：

- 动作是否平滑
- 是否比 Torch 版本更抖
- 是否有明显相机顺序错误导致的动作偏差

---

## 13. 上线顺序建议

我建议按下面顺序推进，而不是一口气写完所有东西再 debug：

### 第 0 步：保留 Torch 基线，先加时序日志

先在现有 `run_act_infer.py` 或拷贝版脚本里统计：

- 观测耗时
- 预处理耗时
- 模型耗时
- 总循环耗时

目的是拿到一份真实基线。

### 第 1 步：先导出 ONNX

目标：

- 产出 `act_single.onnx`
- 拿到 `export_metadata.json`
- 通过 Torch vs ONNX 验证

### 第 2 步：先 build FP32 TRT engine

目标：

- 产出 `act_single_fp32.plan`
- 单样本推理正确
- ONNX vs TRT-FP32 对齐

### 第 3 步：再 build FP16 TRT engine

目标：

- 产出 `act_single_fp16.plan`
- 精度损失在可接受范围
- 得到更低的单次前向时延

### 第 4 步：实现 `ActTrtPolicyAdapter`

目标：

- 在不改 `predict_action(...)` 调用方式的前提下替换 policy 后端

### 第 5 步：写 `run_act_trt_infer.py`

目标：

- 和 `run_act_infer.py` 保持接近
- 先能稳定跑

### 第 6 步：逐步把 `n_action_steps` 从 100 下调

建议顺序：

1. `100`（仅验证正确性）
2. `32`
3. `16`
4. `8`

每下调一档，都要观察：

- 平均控制周期是否还能稳住 `1 / fps`
- 是否引入抖动
- 是否真正提升任务表现

---

## 14. 这一版方案的关键风险和规避方式

### 风险 1：相机顺序错误

**现象**：动作明显跑偏，但程序不报错。  
**规避**：一律以 `export_metadata.json` 的 `camera_order_visual_keys` 为准。

### 风险 2：前处理设备不合理，造成无谓搬运

**现象**：TRT 明明很快，但总耗时不降反升。  
**规避**：TRT 路线下把 adapter `config.device` 和 preprocessor `device_processor` 都切到 CPU。

### 风险 3：一开始就上 FP16，排障困难

**现象**：有偏差，但不确定是导出问题还是半精度问题。  
**规避**：严格按顺序先对齐 ONNX，再对齐 TRT-FP32，再切 TRT-FP16。

### 风险 4：把 Torch policy 也一起加载，导致显存和路径混乱

**现象**：显存占用高、难判断实际走的是哪条后端。  
**规避**：TRT 推理脚本不加载 `policy_class.from_pretrained(...)`。

### 风险 5：过早追求“全链路 TRT”

**现象**：工程复杂度暴涨，验证难度失控。  
**规避**：第一版只替换核心网络前向。

---

## 15. 最终建议的交付物清单

如果按这个方案推进，建议最终沉淀这些文件：

### 文档

- `my_devs/docs/act_trt/act加速方案1.md`

### 导出与构建产物

- `outputs/deploy/act_trt/20260305_190147_act_grasp_block_in_bin1_e15/act_single.onnx`
- `outputs/deploy/act_trt/20260305_190147_act_grasp_block_in_bin1_e15/export_metadata.json`
- `outputs/deploy/act_trt/20260305_190147_act_grasp_block_in_bin1_e15/act_single_fp32.plan`
- `outputs/deploy/act_trt/20260305_190147_act_grasp_block_in_bin1_e15/act_single_fp16.plan`
- `outputs/deploy/act_trt/20260305_190147_act_grasp_block_in_bin1_e15/*.tcache`
- `outputs/deploy/act_trt/20260305_190147_act_grasp_block_in_bin1_e15/build_summary_*.json`

### 代码（后续建议新增）

- `my_devs/train/act/so101/build_act_trt_engine.py`
- `my_devs/train/act/so101/act_trt_policy.py`
- `my_devs/train/act/so101/run_act_trt_infer.py`

---

## 16. 这份方案最重要的一句话

**方案 1 的核心，不是“把 ACT 全部改成 TRT”，而是“在最大程度复用现有 LeRobot 部署链路的前提下，只把最值得加速的核心网络前向替换成 TensorRT，并保留 Python 侧动作队列与前后处理，从而低风险获得更高的重规划频率”。**

如果后续要继续深挖性能，正确顺序应该是：

1. 先把方案 1 跑稳
2. 再评估是否需要把预处理也并入 ONNX
3. 最后才考虑方案 2 / 方案 3 这种更激进的端到端优化

