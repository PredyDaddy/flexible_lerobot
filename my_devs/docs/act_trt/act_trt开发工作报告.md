# ACT TensorRT 开发工作报告

> 报告日期：2026-03-07  
> 工作范围：`my_devs/act_trt` 路径下的 ACT 导出、TensorRT 引擎构建、Torch / ONNX / TensorRT 三方一致性对齐  
> 关联文档：
> - `my_devs/docs/act_trt/act最终技术方案.md`
> - `my_devs/docs/act_trt/act开发Plan.md`

---

## 1. 本次工作的最终结论

本次工作已经完成以下目标：

1. **完成 ACT 核心模型导出**：已成功将当前目标 checkpoint 的 `policy.model` 导出为 ONNX。
2. **完成 TensorRT 引擎构建**：已成功在 `lerobot_flex` 环境中构建出可用的 `FP32` TensorRT engine。
3. **完成 Torch / ONNX / TensorRT 三方一致性对齐**：最终主路径产物已实现严格对齐，通过验证。
4. **全程未接入机器人**：本次工作严格限定在离线导出、离线构建和离线一致性验证范围内，没有操作机器人。

最终主结论如下：

- `Torch vs ONNX`：通过
- `Torch vs TensorRT`：通过
- `ONNX vs TensorRT`：通过
- 当前 `FP32` 主引擎已达到可接受的一致性精度
- 问题根因已定位：**原始未简化 ONNX 虽然 Torch/ORT 对齐，但 TensorRT 数值漂移明显；对 ONNX 做 simplify 后，TensorRT 与 Torch/ONNX 对齐恢复正常**

这意味着本次工作不仅“把流程打通了”，而且已经把**为什么最开始 TRT 不对齐、后来为什么对齐**这件事完整搞清楚了。

---

## 2. 本次工作的边界

本次工作严格遵守之前技术方案与 Plan 的边界：

### 2.1 做了什么

- 在 `my_devs/act_trt` 下开发了导出、引擎构建、TRT runtime、三方一致性验证脚本；
- 使用当前 ACT checkpoint 完成离线导出；
- 在 `lerobot_flex` 环境中补齐 TensorRT 依赖，使 builder 可以正常工作；
- 对 ONNX / TRT 路线进行了多轮排障与对齐；
- 产出了 ONNX、FP32 engine、timing cache、build summary、consistency report。

### 2.2 没做什么

- 没有操作机器人；
- 没有接入 `run_act_infer.py` 实机闭环；
- 没有实现 ACT TRT policy adapter；
- 没有实现 `select_action` 队列语义复刻；
- 没有做 FP16 engine 对齐；
- 没有做 pre/post processor 融合；
- 没有做零拷贝或动态 shape；
- 没有做 INT8。

这份报告只覆盖：

> **ACT 核心前向导出 + TensorRT engine 构建 + Torch / ONNX / TensorRT 三方一致性对齐**

---

## 3. 目标模型与输入输出事实

本次工作的目标 checkpoint 来自：

- `outputs/train/20260305_190147_act_grasp_block_in_bin1_e15/checkpoints/last/pretrained_model`

在实际读取时，导出元数据中记录到的绝对路径解析为：

- `/data/cqy_workspace/flexible_lerobot/outputs/train/20260305_190147_act_grasp_block_in_bin1_e15/checkpoints/025740/pretrained_model`

这说明 `last` 最终指向了具体训练步目录，本次所有离线工作都围绕这个实际 checkpoint 展开。

### 3.1 模型事实

根据 checkpoint `config.json` 读取到的关键事实如下：

- policy type：`act`
- `chunk_size = 100`
- `n_action_steps = 100`
- state dim：`6`
- camera keys：
  - `observation.images.top`
  - `observation.images.wrist`
- image shape：`[3, 480, 640]`
- action dim：`6`
- 模型输出 shape：`[1, 100, 6]`

### 3.2 导出边界

本次导出的不是 `ACTPolicy.select_action()`，而是更小、更稳定、更适合 TensorRT 的边界：

```text
normalized obs_state + normalized img0 + normalized img1
    -> ACT core model (policy.model)
    -> normalized action chunk
```

也就是说：

- 输入：归一化后的 state / 图像
- 输出：归一化后的 action chunk
- 不包含：queue、temporal ensemble、postprocessor

这条边界与原技术方案保持一致。

---

## 4. 本次新开发的代码

本次所有代码都开发在：

- `my_devs/act_trt`

实际新增文件如下：

- `my_devs/act_trt/__init__.py`
- `my_devs/act_trt/common.py`
- `my_devs/act_trt/export_act_onnx.py`
- `my_devs/act_trt/build_act_trt_engine.py`
- `my_devs/act_trt/trt_runtime.py`
- `my_devs/act_trt/verify_torch_onnx_trt.py`

下面逐个说明每个文件的职责与实现内容。

---

## 5. 各代码文件详细说明

### 5.1 `my_devs/act_trt/common.py`

这个文件承担的是**公共模型信息、checkpoint 解析、测试输入构造、Torch 前向基线**。

主要作用包括：

1. 解析 checkpoint 路径；
2. 解析 deploy 输出目录；
3. 从 `config.json` 中提取模型事实；
4. 固化 camera order；
5. 加载 ACT policy；
6. 提供纯 Torch 的核心前向；
7. 构造一致性验证用的多组输入 case；
8. 提供 JSON 读写工具。

#### 关键设计点

- 定义了 `ActModelSpec`，统一记录：
  - checkpoint
  - visual keys
  - state dim
  - image height / width
  - action dim
  - chunk size
  - n_action_steps
- 所有后续脚本都通过这个 spec 获取输入输出 shape，而不是手写。
- 验证输入 case 不只使用随机输入，还包括：
  - `zeros`
  - `ones`
  - `linspace`
- 这样可以避免“随机输入碰巧没出问题”，提高验证覆盖度。

#### 为什么这个文件重要

它保证了整个开发过程中：

- 输入输出事实不会到处散落；
- 相机顺序不会被手写错；
- Torch 基线始终来自同一个实现；
- 各个脚本共享同一套事实来源。

---

### 5.2 `my_devs/act_trt/export_act_onnx.py`

这个脚本是**项目内的 ONNX 导出入口**。

它本身没有重写一套新的导出逻辑，而是：

- 复用参考导出脚本：
  - `my_devs/docs/act_trt/reference_docs/onnx_export_reference/export/export_single.py`
- 在项目路径下提供统一、稳定、可复用的入口

#### 这个脚本做了什么

1. 解析 checkpoint 与输出目录；
2. 定位参考导出脚本；
3. 透传以下关键参数：
   - `--checkpoint`
   - `--output`
   - `--opset`
   - `--device`
   - `--verify`
   - `--simplify`
   - `--dynamo`
4. 打印实际执行命令，便于排障与复现。

#### 关键开发决策

最重要的一点是：

- **我把 `simplify` 的默认值改成了 `True`**。

原因不是偏好，而是本次实测证明：

- 原始未简化 ONNX：Torch / ORT 对齐，但 TensorRT 数值明显漂移；
- 简化后的 ONNX：TensorRT 与 Torch / ONNX 恢复高精度一致。

也就是说，`simplify=True` 不是 cosmetic，而是本次对齐成功的关键条件之一。

#### 补充说明

这个脚本还加了一个细节：

- 当脚本以 `python my_devs/act_trt/export_act_onnx.py ...` 方式直跑时，会自动补 `sys.path`，确保可以正确 import `my_devs.act_trt.*`。

这是为了让它在命令行直接运行时不依赖额外设置 `PYTHONPATH`。

---

### 5.3 `my_devs/act_trt/build_act_trt_engine.py`

这个脚本负责：

> **把 ONNX 构建成 TensorRT engine，并输出 build summary 与 timing cache**

#### 脚本能力

它支持：

- 输入 ONNX 路径；
- 自动读取 `export_metadata.json`；
- 生成 engine：
  - 默认 `act_single_fp32.plan`
- 生成 timing cache：
  - 默认 `act_single_fp32.tcache`
- 生成 build summary：
  - 默认 `trt_build_summary_fp32.json`

#### 主要实现点

1. 使用 TensorRT Python API 解析 ONNX；
2. 使用 explicit batch 网络定义；
3. 设置 workspace；
4. 设置 builder optimization level；
5. 显式关闭 TF32：
   - 这是为了减少不必要的数值偏差；
6. 支持 timing cache；
7. 构建完成后立刻用 `TensorRTRunner` 反序列化验证 engine 是否可加载；
8. 把 engine tensor 信息写进 build summary。

#### 本次为什么先只做 FP32

这是有意为之。

原因是：

- FP32 更适合做正确性对齐；
- 在没有完成三方一致性之前，不应该上 FP16；
- 按照技术方案，正确顺序必须是：
  - 先 Torch / ONNX / TRT-FP32 对齐
  - 再考虑 TRT-FP16

#### 本次记录的两份 engine

在这次工作中，我实际构建了两版 FP32 engine：

1. `act_single_fp32.plan`
   - 标准 `opt_level=3` 版本
2. `act_single_fp32_opt0.plan`
   - 排障时用的 `opt_level=0` 版本

这个排障分支的意义是：

- 我验证了数值漂移不是单纯由 `opt_level=3` 造成的；
- 即使把优化级别降到 0，未简化 ONNX 下的 TRT 仍然漂移；
- 这帮助我把问题根因进一步收缩到：**ONNX 图本身对 TRT 不够友好，而不是单纯 builder 激进优化**。

---

### 5.4 `my_devs/act_trt/trt_runtime.py`

这个文件实现的是：

> **按名字驱动的 TensorRT runtime 封装**

它不是一个 demo，而是一个可以复用到后续 adapter 阶段的正式 runtime 基础。

#### 主要职责

- 加载 `.plan` engine；
- 创建 execution context；
- 读取 engine 中所有 tensor names；
- 区分 input / output tensors；
- 将 numpy / torch 输入转换为 engine 期望 dtype；
- 将输入移动到 CUDA；
- 绑定 tensor address；
- 调用 `execute_async_v3` 执行推理；
- 返回 output tensors。

#### 本次实现的关键点

##### 1）严格按 tensor name 工作

而不是靠位置或手写固定顺序。

这能避免：

- 输入顺序错位；
- `img0` / `img1` 混淆；
- output name 填错。

##### 2）显式做 dtype 对齐

runtime 会读取 engine 的 tensor dtype，然后把输入转换成对应 torch dtype。

这避免了：

- engine 要 `float32`，结果喂进 `float16`；
- 或者相反。

##### 3）使用独立 CUDA stream

一开始的 runtime 使用默认 stream，TensorRT 会给出警告：

- `Using default stream in enqueueV3() may lead to performance issues ...`

后面我把它改成了：

- `self.stream = torch.cuda.Stream(device=self.device)`
- 推理时在这个专用 stream 上执行

这不是为了这次对齐必须，而是把 runtime 稍微做得更正规一些，也减少默认 stream 下潜在同步副作用。

##### 4）输出保留为 CUDA tensor，再由上层决定是否转 CPU

这让 runtime 语义更清晰：

- runtime 只负责 TRT 执行；
- 上层验证脚本再决定如何 `detach().cpu().numpy()`。

---

### 5.5 `my_devs/act_trt/verify_torch_onnx_trt.py`

这是本次最关键的验证脚本之一。

它负责：

> **对同一组输入，分别跑 Torch / ONNX / TensorRT，然后输出三方差异报告。**

#### 覆盖的对比项

它会计算：

- `torch_vs_onnx`
- `torch_vs_trt`
- `onnx_vs_trt`

每一项都会输出：

- `max_abs_diff`
- `mean_abs_diff`
- `max_rel_diff`
- `cosine_similarity`

#### 验证 case

这次脚本覆盖了以下输入 case：

- `random_seed_0`
- `random_seed_1`
- `random_seed_2`
- `random_seed_3`
- `zeros`
- `ones`
- `linspace`

也就是说，不只是“几组随机数”，而是故意覆盖：

- 随机分布输入；
- 全零输入；
- 全一输入；
- 线性分布输入。

#### 为什么这个脚本重要

因为它让本次“对齐成功”不再是口头结论，而是：

- 有固定脚本；
- 有固定报告；
- 有具体阈值；
- 有结构化 JSON 输出。

这意味着后续如果继续做：

- FP16
- adapter
- 实机接入

都可以沿用同一套验证习惯，而不是临时手工看日志。

---

## 6. 本次实际产出的部署与验证文件

本次主产物落在：

- `outputs/deploy/act_trt/20260305_190147_act_grasp_block_in_bin1_e15`

### 6.1 主产物

- `act_single.onnx`
- `export_metadata.json`
- `act_single_fp32.plan`
- `act_single_fp32.tcache`
- `trt_build_summary_fp32.json`
- `consistency_report_act_single_fp32.json`

### 6.2 排障中间产物

- `act_single_fp32_opt0.plan`
- `act_single_fp32_opt0.tcache`
- `trt_build_summary_fp32_opt0.json`
- `consistency_report_act_single_fp32_opt0.json`

这些中间产物我保留了下来，不是垃圾文件，而是有排障价值的：

- 它们证明我确实做过 builder 优化级别排查；
- 可以帮助后续复盘“为什么某些版本不对齐”。

---

## 7. 环境工作与依赖排障

这部分非常重要，因为本次开发中，环境问题本身就是一个关键障碍。

### 7.1 初始环境状态

在最开始检查 `lerobot_flex` 时，发现：

- `torch`：可用
- `onnx`：可用
- `onnxruntime`：可用
- `onnxsim`：最初脚本检查里显示未安装
- `tensorrt`：不可直接 import
- `cuda-python`：不可用

后续进一步检查发现：

- `lerobot_flex` 里其实存在部分 TensorRT bindings / libs；
- 但包结构不完整，且版本混杂，导致 Python 可以部分 import，却无法正常创建 builder。

### 7.2 遇到的第一个 TensorRT 问题

最初尝试使用较新的 `TensorRT 10.14 / cu13` 路线时，创建 builder 报错：

- `CUDA initialization failure with error: 35`

这说明：

- 当前驱动 / CUDA 运行环境和 `cu13` 版本的 TRT 路线并不匹配；
- 即使 Python 包能 import，也不代表 builder 真能工作。

### 7.3 最终采用的环境修复方案

最终我把 `lerobot_flex` 中的 TRT 依赖切到了：

- `tensorrt 10.13.0.35`
- `tensorrt-cu12 10.13.0.35`

这样做之后：

- `torch.cuda` 仍然正常；
- `tensorrt.Builder` 可以正常创建；
- ONNX parse / build 可以正常进行。

### 7.4 兼容性备注

这个切换过程里，`pip` 给出过一个提示：

- `torch 2.7.1` 要求 `nvidia-cuda-runtime-cu12==12.6.77`
- 安装 TRT 时带来了 `12.9.79`

但本次实际运行结果表明：

- `torch.cuda` 可用；
- TensorRT builder 可用；
- ONNX / TRT 一致性最终可对齐。

也就是说，当前环境在这次任务中是可工作的。

不过从后续维护角度讲，建议未来还是单独做一次环境治理，避免 `torch` 与 TRT 依赖版本在同一环境里长期漂移。

---

## 8. 实际开发过程与关键排障过程

下面按时间顺序总结本次技术推进过程。

### 8.1 第一步：确认 checkpoint 与基础链路事实

我先确认了：

- checkpoint 存在；
- `config.json`、`model.safetensors`、pre/post processor 文件都存在；
- input_features 中视觉顺序确实是：
  - `observation.images.top`
  - `observation.images.wrist`

这一步的意义是：

- 避免后面因为 checkpoint 本身缺文件而浪费时间；
- 提前锁死 camera order，减少后面验证歧义。

### 8.2 第二步：直接使用参考脚本导出 ONNX

我先直接复用参考导出脚本，把核心模型导出成 ONNX。

这一阶段结果很好：

- ONNX 导出成功；
- `Torch vs ONNX` 在参考导出脚本的自带验证里就已经通过；
- `max_abs_diff` 达到 `1e-6` 量级。

这说明：

- 导出边界选得对；
- ONNX 图本身至少在 ORT 下是忠实的。

### 8.3 第三步：让 TensorRT builder 真正跑起来

这一步遇到的主要问题不是模型，而是环境：

- `tensorrt` 初始不可用；
- 后续导入某些版本后 builder 仍然因为 CUDA 初始化失败而起不来；
- 最终通过改用 `TensorRT 10.13 / cu12` 才把 builder 打通。

这是一个很重要的结论：

> **TensorRT 是否能 import，不等于 TensorRT builder 是否可用。**

### 8.4 第四步：第一次 TensorRT 一致性验证失败

在 ONNX 导出成功、FP32 engine 构建成功之后，我第一次跑三方一致性，结果是：

- `Torch vs ONNX`：很好
- `Torch vs TensorRT`：差
- `ONNX vs TensorRT`：差

典型指标大概在：

- `Torch vs TRT max_abs_diff ≈ 0.0659`
- `mean_abs_diff ≈ 0.0173`

这说明：

- 问题不在 Torch / ONNX；
- 问题集中在 TensorRT 这一侧；
- 而且不是小误差，是明显不合格的误差。

### 8.5 第五步：排除 builder opt level 的影响

我尝试把 builder 的 `opt_level` 从 `3` 降到 `0`，重新 build 了一版：

- `act_single_fp32_opt0.plan`

结果显示：

- 偏差几乎没有本质改善；
- 问题不是单纯由激进优化导致。

这一步帮助我排除了一种常见误判：

- “是不是 TRT 优化太激进了，所以数值飘了？”

结论是：

- **不是主要原因。**

### 8.6 第六步：转向 ONNX 图本身的 TRT 友好性

接下来我开始怀疑问题根因在 ONNX 图结构本身。

因为 ACT 在导出时，模型里有一些：

- `list(tensor)`
- `extend(list(...))`
- 这类会导致 trace 图结构比较膨胀、比较不规则的实现

而参考导出时的 warning 里也确实出现了：

- `TracerWarning: Iterating over a tensor might cause the trace to be incorrect`

这并不一定意味着 Torch/ORT 一定错，但它提示：

- 这张图对 TensorRT 未必友好。

### 8.7 第七步：尝试简化 ONNX 图

然后我重新导出 ONNX，并开启：

- `--simplify`

这一步的效果是这次工作最关键的突破。

导出元数据里记录到：

- 原始节点数：`5403`
- 简化后节点数：`1532`
- 减少节点数：`3871`
- 节点减少比例：`71.65%`

这说明 simplify 对这张 ACT 图的结构化简非常明显。

### 8.8 第八步：简化后重新 build 并重测

基于 simplify 后的 ONNX：

- 重新 build FP32 engine；
- 重新跑 Torch / ONNX / TensorRT 三方一致性。

这一次结果完全不一样：

- `Torch vs ONNX`：仍然很好
- `Torch vs TensorRT`：恢复到 `1e-6` 量级
- `ONNX vs TensorRT`：恢复到 `1e-6` 量级
- `passed = true`

到这里，根因已经足够明确：

> **未简化 ONNX 图虽然能被 TRT 编译，但数值实现不稳定；经过 simplify 后，图结构更 TRT-friendly，三方输出恢复高精度一致。**

### 8.9 第九步：把“成功路线”回写到主路径

在 simplify 路线验证成功之后，我没有把成功结果留在临时目录，而是：

- 把主 deploy 路径中的 `act_single.onnx` 更新为 simplify 后的版本；
- 重新在主路径 build `act_single_fp32.plan`；
- 重新在主路径跑三方一致性；
- 确保主路径最终产物本身就是通过验证的产物。

这样做的好处是：

- 你之后直接看主路径就行；
- 主路径不是“第一次失败的版本”，而是“已经验证通过的最终版本”。

---

## 9. 最终导出结果

最终主路径的导出结果如下：

- ONNX：`outputs/deploy/act_trt/20260305_190147_act_grasp_block_in_bin1_e15/act_single.onnx`
- 元数据：`outputs/deploy/act_trt/20260305_190147_act_grasp_block_in_bin1_e15/export_metadata.json`

### 9.1 导出验证结果

导出元数据里记录到：

- `verification.passed = true`
- `max_abs_diff = 8.046627044677734e-07`
- `mean_abs_diff = 1.6782432510353829e-07`
- threshold = `1e-4`

这表示：

- 在 Torch 与 ORT 层面，导出已经非常稳定。

### 9.2 简化结果

导出元数据里还记录到：

- `simplification.success = true`
- `original_nodes = 5403`
- `simplified_nodes = 1532`
- `reduction_percent = 71.65`

这也是本次报告里最值得记住的结论之一：

> **ACT 这张 ONNX 图在本项目里必须保留 simplify 这一步，它不是锦上添花，而是 TensorRT 对齐成功的关键条件。**

---

## 10. 最终 TensorRT 引擎结果

最终主路径的引擎产物如下：

- engine：`outputs/deploy/act_trt/20260305_190147_act_grasp_block_in_bin1_e15/act_single_fp32.plan`
- timing cache：`outputs/deploy/act_trt/20260305_190147_act_grasp_block_in_bin1_e15/act_single_fp32.tcache`
- build summary：`outputs/deploy/act_trt/20260305_190147_act_grasp_block_in_bin1_e15/trt_build_summary_fp32.json`

### 10.1 engine 规格

build summary 记录到的核心信息：

- precision：`fp32`
- workspace：`4 GB`
- opt level：`3`
- `TF32`：显式关闭
- engine IO：
  - `obs_state_norm` -> `FLOAT`
  - `img0_norm` -> `FLOAT`
  - `img1_norm` -> `FLOAT`
  - `actions_norm` -> `FLOAT`

这说明：

- 这是一个纯 `FP32` 输入输出的 engine；
- 本次成功对齐不是靠“降精度换吞吐”，而是基于 `FP32` 的严格对齐。

---

## 11. 最终一致性验证结果

这是本次工作的核心成果。

主报告路径：

- `outputs/deploy/act_trt/20260305_190147_act_grasp_block_in_bin1_e15/consistency_report_act_single_fp32.json`

### 11.1 总体结果

报告中 `passed = true`。

整体 summary 为：

#### Torch vs ONNX

- `max_abs_diff = 3.4086406230926514e-06`
- `mean_abs_diff = 1.3570114560934599e-06`
- `min_cosine_similarity = 0.9999998211860657`

#### Torch vs TensorRT

- `max_abs_diff = 2.5033950805664062e-06`
- `mean_abs_diff = 8.217245408559393e-07`
- `min_cosine_similarity = 0.9999998807907104`

#### ONNX vs TensorRT

- `max_abs_diff = 2.469867467880249e-06`
- `mean_abs_diff = 8.691040420671925e-07`
- `min_cosine_similarity = 0.9999998807907104`

### 11.2 对齐评价

这些结果已经达到一个非常理想的离线对齐水平：

- 绝对误差在 `1e-6` 量级；
- 远小于当前设定阈值 `1e-4`；
- 三方 cosine similarity 基本接近 `1.0`。

从离线数值一致性的角度看，可以认为：

> **当前 simplify 后的 ONNX 与最终 FP32 TensorRT engine，已经和 Torch 核心前向高度一致。**

### 11.3 覆盖 case

一致性验证覆盖了：

- 4 组随机输入；
- `zeros`；
- `ones`；
- `linspace`；

这意味着本次结论不是单一样本下的偶然结果。

---

## 12. 本次最重要的技术发现

本次开发最重要的技术发现，不是“成功导出 ONNX”，而是下面这几条：

### 12.1 未简化 ONNX 可能会“能编译，但不对齐”

这次已经明确证明：

- 原始 ONNX 可以被 TensorRT parse / build；
- 但 build 成功不代表数值一致；
- 未简化图在本项目里会出现明显漂移。

这是一条非常重要的经验。

### 12.2 `simplify` 对 ACT 不是可选增强，而是对齐关键步骤

这次实测中：

- 节点数从 `5403` 降到 `1532`；
- TRT 一致性从失败变为通过。

所以后续这条导出链路里，`simplify` 应该被默认保留。

### 12.3 先做 FP32 对齐是对的

如果这次我一上来就做 FP16：

- 排障会复杂很多；
- 很难知道偏差来自 ONNX、来自 TRT 还是来自半精度。

这次之所以能高效定位问题，核心就在于：

- 先做 Torch / ONNX / TRT-FP32 严格对齐；
- 把问题限定在单一变量范围里。

### 12.4 builder 能创建，比 import tensorrt 更重要

环境排障过程中，真正决定成败的是：

- `trt.Builder(logger)` 能否正常创建

而不是：

- `import tensorrt` 是否成功

这一点后续在任何 TRT 开发里都值得优先检查。

---

## 13. 本次执行过的关键命令类型

下面列的是本次工作中的关键命令类型，方便后续复现。

### 13.1 环境检查

- 检查 `torch / onnx / onnxruntime / tensorrt / cuda-python`
- 检查 GPU 可用性
- 检查 checkpoint 文件完整性

### 13.2 ONNX 导出

项目内入口：

```bash
conda run -n lerobot_flex python my_devs/act_trt/export_act_onnx.py \
  --checkpoint outputs/train/20260305_190147_act_grasp_block_in_bin1_e15/checkpoints/last/pretrained_model \
  --output-dir outputs/deploy/act_trt/20260305_190147_act_grasp_block_in_bin1_e15 \
  --opset 17 \
  --device cpu \
  --verify \
  --simplify
```

### 13.3 TensorRT 引擎构建

```bash
conda run -n lerobot_flex python my_devs/act_trt/build_act_trt_engine.py \
  --onnx outputs/deploy/act_trt/20260305_190147_act_grasp_block_in_bin1_e15/act_single.onnx \
  --precision fp32 \
  --workspace-gb 4 \
  --opt-level 3 \
  --device cuda:0
```

### 13.4 三方一致性验证

```bash
conda run -n lerobot_flex python my_devs/act_trt/verify_torch_onnx_trt.py \
  --checkpoint outputs/train/20260305_190147_act_grasp_block_in_bin1_e15/checkpoints/last/pretrained_model \
  --onnx outputs/deploy/act_trt/20260305_190147_act_grasp_block_in_bin1_e15/act_single.onnx \
  --engine outputs/deploy/act_trt/20260305_190147_act_grasp_block_in_bin1_e15/act_single_fp32.plan \
  --device cuda:0 \
  --random-cases 4 \
  --threshold-max-abs-diff 1e-4
```

---

## 14. 本次没有做但建议后续继续做的内容

虽然这次主目标已经完成，但从完整部署链路看，后续还应继续推进：

### 14.1 FP16 engine 与 FP32/FP16 对齐

当前只完成了 FP32。

后续建议：

- build `act_single_fp16.plan`
- 继续输出：
  - `TRT-FP32 vs TRT-FP16`
  - `Torch vs TRT-FP16`
  - `ONNX vs TRT-FP16`

### 14.2 ACT TRT policy adapter

后续还需要实现：

- `ActTrtPolicyAdapter`
- 在 Python 侧复刻：
  - queue 语义
  - temporal ensemble 语义

### 14.3 `run_act_trt_infer.py`

等 adapter 完成后，再开发：

- 独立 TRT 推理脚本
- 继续保持不改动现有 Torch 基线

### 14.4 实机验证

必须等以下都完成后再碰机器人：

- FP32 / FP16 路线明确；
- adapter 语义对齐；
- offline consistency 稳定；
- 再做短时在线验证。

---

## 15. 本次工作的交付评价

如果只评价这次任务本身，我认为目前交付状态是：

### 15.1 已完成

- [x] `my_devs/act_trt` 路径下基础代码落地
- [x] ONNX 导出
- [x] `export_metadata.json` 产出
- [x] FP32 TensorRT engine 构建
- [x] timing cache 产出
- [x] build summary 产出
- [x] Torch / ONNX / TensorRT 三方一致性通过
- [x] 关键问题根因定位完成
- [x] 全程未操作机器人

### 15.2 未完成但不属于本次目标

- [ ] FP16 engine
- [ ] TRT policy adapter
- [ ] `run_act_trt_infer.py`
- [ ] 实机闭环验证

所以，本次工作可以定义为：

> **ACT TRT 路线的“离线核心前向阶段”已经完成，并且完成得比较扎实。**

---

## 16. 最终一句话总结

本次工作的真正价值，不只是“导出了 ONNX、编出了 engine”，而是：

> **我已经把 ACT 核心前向的 Torch / ONNX / TensorRT 三方链路在 `my_devs/act_trt` 下真正打通，并通过定位“未简化 ONNX 导致 TRT 漂移”这个关键问题，最终把主路径产物稳定对齐到了 `1e-6` 量级，而且全程没有碰机器人。**

这为后续继续做：

- FP16
- policy adapter
- TRT 推理脚本
- 实机验证

提供了可靠的基础版本。
