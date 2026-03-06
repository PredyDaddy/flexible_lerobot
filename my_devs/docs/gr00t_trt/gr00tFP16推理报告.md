# GR00T FP16 推理报告（Torch 导出 ONNX -> TensorRT 引擎构建 -> 引擎推理上机）

最后更新：2026-03-05  
适用仓库：`/data/cqy_workspace/flexible_lerobot`（conda env: `lerobot_flex`）  
方案名称：**gr00tFP16**

---

## 1. 背景与目标

本报告记录并固化一条在当前仓库内可复现的 **GR00T N1.5 (LeRobot Groot policy)** FP16 推理链路：

1. 从 LeRobot 训练产物（`pretrained_model/`）加载模型与处理器配置。
2. 将模型拆分为多个可部署子模块并导出为 **ONNX（FP16）**。
3. 使用 **TensorRT Python API**（非 `trtexec`）从 ONNX 构建 **TensorRT engine（`.engine`）**。
4. 在推理侧使用预构建的 `.engine` 完成推理：
   - 离线一致性验证（Torch vs ONNX / Torch vs TRT）
   - 上机推理（支持 `--mock true` 安全模式，不下发动作）

本报告的目标是：

- 让没有部署经验的同学也能严格按步骤复现。
- 讲清楚每一步用了哪些关键 API、为什么这么设计、以及常见坑怎么排查。

范围说明：

- 本文是 **FP16 baseline**（不涉及量化/FP8/INT8/NVFP4）。量化相关请见同目录 `量化方案.md`。

---

## 2. 本方案的“模块拆分”设计

GR00T N1.5 在 LeRobot 中是一个“Backbone + Action Head”的 policy。为了工程化部署，我们把推理链路拆分为两大块、7 个可部署 engine：

1. Backbone（Eagle2）
2. Action head（diffusion-based action head）

### 2.1 导出与部署的 7 个子模块

ONNX 导出与 TensorRT 引擎构建的目标文件如下（文件名与推理脚本强绑定）：

Backbone（2 个）：

1. `eagle2/vit_fp16.onnx` -> `vit_fp16.engine`
2. `eagle2/llm_fp16.onnx` -> `llm_fp16.engine`

Action head（5 个）：

1. `action_head/vlln_vl_self_attention.onnx` -> `vlln_vl_self_attention.engine`
2. `action_head/state_encoder.onnx` -> `state_encoder.engine`
3. `action_head/action_encoder.onnx` -> `action_encoder.engine`
4. `action_head/DiT_fp16.onnx` -> `DiT_fp16.engine`
5. `action_head/action_decoder.onnx` -> `action_decoder.engine`

### 2.2 为什么不是把整个 policy 一次性导出成一个 engine

主要原因是工程落地的可控性与一致性：

1. **可控性**：单体大图更难定位数值漂移、也更难调 profile。
2. **动态 shape 复杂**：多视角、多序列长度、多步 denoising 的 shape 组合，拆分后 profile 更清晰。
3. **保持与训练侧行为一致**：部分“胶水逻辑（glue）”在 Torch 侧实现更稳定，且能复用 checkpoint 的实现细节。

我们保留在 Torch 侧的 glue（与 export/compare 脚本保持一致）包括：

- ViT 后处理（pixel shuffle + `mlp1`）
- 语言模型 token embedding（`get_input_embeddings()`）与“把 image token 的 embedding 替换成 ViT embedding”的逻辑
- diffusion loop 中的 `future_tokens`、可选 `position_embedding` 与若干拼接/加法

这使得 TRT 负责大头计算（ViT/LLM/DiT 等），Torch 负责可控的 glue。

---

## 3. 产物目录与示例路径（本次实测）

### 3.1 Policy 权重（LeRobot checkpoint）

`--policy-path` 必须指向 LeRobot checkpoint 的 `pretrained_model/` 目录，里面应包含：

- `config.json`
- `model.safetensors`
- `policy_preprocessor.json`
- `policy_postprocessor.json`

本轮实测使用：

```text
/data/cqy_workspace/flexible_lerobot/outputs/train/
  groot_grasp_block_in_bin1_repro_20260302_223413/
    bs32_20260302_223447/checkpoints/last/pretrained_model
```

### 3.2 ONNX / engine 产物目录

本轮产物（ONNX/engine/日志/指标）在：

```text
outputs/trt/consistency_rerun_20260305_102931/
```

其中关键子目录：

- ONNX：`outputs/trt/consistency_rerun_20260305_102931/gr00t_onnx/`
- Engine：`outputs/trt/consistency_rerun_20260305_102931/gr00t_engine_api_trt1013/`

---

## 4. 环境与依赖（必须满足）

### 4.1 Conda 环境

仓库约定所有开发/运行使用 conda env：`lerobot_flex`。

### 4.2 TensorRT 安装方式（推荐）

为了避免把 TensorRT 装到 root 分区导致空间不足，本方案推荐把 TensorRT pip 包安装到 `/data`：

```bash
conda run -n lerobot_flex python -m pip install --target /data/cqy_workspace/third_party/tensorrt_10_13_0_35 \
  tensorrt==10.13.0.35
```

推理或构建引擎时，用下列任一方式让 Python 找到 TensorRT：

1. 在命令里显式传参：`--tensorrt-py-dir /data/cqy_workspace/third_party/tensorrt_10_13_0_35`
2. 或设置环境变量：`TENSORRT_PY_DIR=/data/cqy_workspace/third_party/tensorrt_10_13_0_35`

说明：

- 我们在 `my_devs/groot_trt/trt_utils.py` 里实现了 `import_tensorrt()`，会自动把 `TENSORRT_PY_DIR` 加入 `sys.path`，
  并把 `${TENSORRT_PY_DIR}/tensorrt_libs` 加到 `LD_LIBRARY_PATH`，减少“装了但 import 不到”的问题。

---

## 5. 模型导出（Torch -> ONNX, FP16）

本方案的导出脚本位于：

- `my_devs/groot_trt/export_backbone_onnx.py`
- `my_devs/groot_trt/export_action_head_onnx.py`

### 5.1 导出 backbone（ViT + LLM）

导出命令示例：

```bash
POLICY_PATH=/path/to/pretrained_model
RUN_DIR=outputs/trt/<your_run_id>

conda run -n lerobot_flex python my_devs/groot_trt/export_backbone_onnx.py \
  --policy-path $POLICY_PATH \
  --onnx-out-dir $RUN_DIR/gr00t_onnx \
  --seq-len 296 \
  --video-views 1 \
  --vit-dtype fp16 \
  --llm-dtype fp16 \
  --opset 19 \
  --device cuda
```

输出文件：

- `$RUN_DIR/gr00t_onnx/eagle2/vit_fp16.onnx`
- `$RUN_DIR/gr00t_onnx/eagle2/llm_fp16.onnx`

#### 5.1.1 使用的关键 API（PyTorch / LeRobot）

1. 加载 checkpoint 配置（读取 `config.json`）：
   - `lerobot.configs.policies.PreTrainedConfig.from_pretrained(policy_path)`
2. 根据 `cfg.type` 构造 policy：
   - `lerobot.policies.factory.get_policy_class(cfg.type)`
   - `policy_cls.from_pretrained(policy_path, strict=False)`
3. ONNX 导出：
   - `torch.onnx.export(...)`

#### 5.1.2 导出时的关键设计点

1. **强制注意力实现为 eager**

导出脚本会把 vision/LLM 的 `_attn_implementation` 设为 `eager`，避免 FlashAttention 等实现影响导出稳定性：

- `backbone.eagle_model.vision_model.config._attn_implementation = "eager"`
- `backbone.eagle_model.language_model.config._attn_implementation = "eager"`

2. **ViT wrapper：保留 position_ids 输入**

`VisionModelForOnnx.forward(pixel_values, position_ids)` 中，`position_ids` 不参与 ViT 计算，但会通过一个“0 倍项”注入计算图：

- 目的：确保 ONNX 图里真正存在 `position_ids` 这个输入，便于后续 TensorRT profile 以及与推理侧接口对齐。

3. **LLM wrapper：输入是 inputs_embeds 而不是 input_ids**

我们导出 `LanguageModelForOnnx(inputs_embeds, attention_mask)`，并在 forward 内构造 causal mask，然后调用：

- `language_model.model(inputs_embeds=..., attention_mask=mask_mapping, output_hidden_states=True, use_cache=False, ...)`

这样做的原因：

- 推理侧的“把 image token embedding 替换为 ViT embedding”的逻辑更适合在 Torch 中做（glue），导出为 ONNX 反而复杂。

### 5.2 导出 action head（5 个子模块）

导出命令示例：

```bash
conda run -n lerobot_flex python my_devs/groot_trt/export_action_head_onnx.py \
  --policy-path $POLICY_PATH \
  --onnx-out-dir $RUN_DIR/gr00t_onnx \
  --seq-len 296 \
  --state-horizon 1 \
  --opset 19 \
  --device cuda
```

输出文件：

- `$RUN_DIR/gr00t_onnx/action_head/vlln_vl_self_attention.onnx`
- `$RUN_DIR/gr00t_onnx/action_head/state_encoder.onnx`
- `$RUN_DIR/gr00t_onnx/action_head/action_encoder.onnx`
- `$RUN_DIR/gr00t_onnx/action_head/DiT_fp16.onnx`
- `$RUN_DIR/gr00t_onnx/action_head/action_decoder.onnx`

#### 5.2.1 使用的关键 API

- 同样使用 `PreTrainedConfig.from_pretrained` + `get_policy_class` 加载 policy。
- ONNX 导出核心依旧是 `torch.onnx.export(...)`（opset 19 + dynamic axes）。

#### 5.2.2 一个额外的封装：vlln + vl_self_attention 合并导出

为了减少 engine 数量、简化推理侧调用，我们将两段子网络合并成一个 ONNX：

- `VLLNVLSelfAttention(vlln, vl_self_attention)`，forward 为：
  - `x = vlln(backbone_features)`
  - `x = vl_self_attention(x)`

---

## 6. 构建引擎（ONNX -> TensorRT engine, Python API）

本方案构建引擎的脚本：

- `my_devs/groot_trt/build_engine.py`（核心实现）
- `my_devs/groot_trt/build_engine.sh`（封装 conda/env 与默认 profile）

### 6.1 构建命令（推荐使用 build_engine.sh）

```bash
ONNX_DIR=$RUN_DIR/gr00t_onnx \
ENGINE_DIR=$RUN_DIR/gr00t_engine_api_trt1013 \
VIDEO_VIEWS=2 MAX_BATCH=2 WORKSPACE_GB=8 \
bash my_devs/groot_trt/build_engine.sh
```

构建完成后，`$ENGINE_DIR/` 下应该有 7 个 `.engine`，并生成：

- `build_report.json`（记录 TensorRT 版本、profile、每个 engine 的输入输出信息与构建耗时）

### 6.2 构建实现使用的关键 TensorRT Python API

以下逻辑在 `build_engine.py::_build_engine()` 中：

1. Builder / network / parser
   - `trt.Builder(trt.Logger(...))`
   - `builder.create_network(flags)`
   - `trt.OnnxParser(network, logger)`
   - `parser.parse(onnx_bytes)`
2. BuilderConfig（workspace）
   - `builder.create_builder_config()`
   - `config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)`
3. 动态 shape profile
   - `builder.create_optimization_profile()`
   - `opt_profile.set_shape(input_name, min_shape, opt_shape, max_shape)`
   - `config.add_optimization_profile(opt_profile)`
4. 构建 serialized engine
   - `builder.build_serialized_network(network, config)`
   - `engine_path.write_bytes(serialized)`

### 6.3 精度策略（FP16）

本方案的 FP16 不是依赖 `trt.BuilderFlag.FP16`，而是通过两点实现：

1. 导出 ONNX 时 wrapper/module 被显式 `.to(dtype=torch.float16)`
2. 构建 engine 时启用 `STRONGLY_TYPED`（强类型网络），减少隐式类型转换

结果是构建出来的 engine 以 FP16 为主（并在一致性对比中验证数值表现）。

### 6.4 动态 shape profile（min/opt/max）

构建脚本的默认 profile 由 `build_engine.sh` 根据 `VIDEO_VIEWS` 自动选择：

- 2-view 默认：`MIN_LEN=80 OPT_LEN=568 MAX_LEN=600`
- 1-view 默认：`MIN_LEN=80 OPT_LEN=296 MAX_LEN=300`

你可以通过环境变量覆写：

```bash
export MIN_LEN=80
export OPT_LEN=568
export MAX_LEN=600
```

注意：

- TensorRT engine 只能保证在 profile 的范围内工作；超过 `MAX_LEN` 会直接报错。

---

## 7. 引擎推理（TRT engine -> 推理循环 -> 上机）

推理脚本位于：

- 上机推理（TRT 引擎驱动）：`my_devs/train/groot/so101/run_groot_infer_trt.py`
- 安全 mock 对比（Torch vs TRT，绝不下发动作）：`my_devs/train/groot/so101/run_groot_infer_trt_mock.py`

### 7.1 TRT runtime 封装：TrtSession（`.engine` 加载 + execute_async_v3）

核心实现位于：`my_devs/groot_trt/trt_utils.py`，关键逻辑：

1. 反序列化 engine
   - `trt.Runtime(logger)`
   - `runtime.deserialize_cuda_engine(engine_bytes)`
   - `engine.create_execution_context()`
2. 每次推理时
   - `context.set_input_shape(name, shape)`（动态 shape）
   - 用 `torch.empty(..., device="cuda")` 分配输出 tensor
   - `context.set_tensor_address(name, int(tensor.data_ptr()))` 绑定地址
   - `context.execute_async_v3(stream)` 执行（stream 来自 `torch.cuda.current_stream().cuda_stream`）

这意味着：

- 推理 I/O 全部是 **CUDA Tensor（torch.Tensor）**，零拷贝。
- 动态 shape 支持依赖 engine 内的 profile 范围。

### 7.2 推理侧如何把 7 个 engine 串起来（TrtGrootPolicyAdapter）

上机脚本 `run_groot_infer_trt.py` 内定义了 `TrtGrootPolicyAdapter`，其核心目标是：

- 提供一个“policy-like”的对象，实现 `select_action(batch)`，让 LeRobot 的 `predict_action(...)` 可以直接复用。

推理链路（简化版）：

1. 从预处理器得到 batch（关键键）：
   - `eagle_pixel_values`（多视角图片 tensor）
   - `eagle_input_ids`（文本 + image token 的 token ids）
   - `eagle_attention_mask`
   - `state`、`embodiment_id`
2. Backbone TRT
   - `vit_fp16.engine`：`pixel_values + position_ids -> vit_embeds`
   - Torch glue：pixel shuffle（如开启）+ `mlp1`
   - Torch glue：构造 `inputs_embeds`（token embedding，并把 image token embedding 替换成 vit_embeds）
   - `llm_fp16.engine`：`inputs_embeds + attention_mask -> backbone_features`
3. Action head TRT + Torch glue
   - `vlln_vl_self_attention.engine`：处理 backbone_features 得到 `vl_embs`
   - `state_encoder.engine`：`state -> state_features`
   - diffusion loop（重复 `num_steps` 次）：
     - `action_encoder.engine`：`actions + timesteps -> action_features`
     - Torch glue：加 position embedding（若开启），拼接 `state_features + future_tokens + action_features`
     - `DiT_fp16.engine`：预测中间特征
     - `action_decoder.engine`：得到 velocity，更新 actions

说明：

- `select_action()` 会一次预测一个 action chunk（长度为 action_horizon），并放入队列逐帧弹出，匹配 LeRobot 的控制循环调用方式。

### 7.3 上机脚本的关键参数

`run_groot_infer_trt.py` 常用参数：

- `--policy-path`：checkpoint 的 `pretrained_model/` 目录
- `--engine-dir`：包含 `.engine` 的目录
- `--tensorrt-py-dir`：TensorRT pip target 目录（推荐显式传）
- `--robot-port /dev/ttyACM0`、`--top-cam-index`、`--wrist-cam-index`
- `--task`：语言指令
- `--run-time-s`：运行秒数（>0 到点退出；<=0 持续运行直到 Ctrl+C）
- `--fps`：控制循环频率（同时影响相机采样与 action 下发节奏）
- `--mock true/false`：
  - `true`：**不下发动作**（安全）
  - `false`：下发动作（上机真实控制）

### 7.4 上机推荐流程（先 mock，再真下发）

1. Dry-run（只校验路径与配置，不连接机器人）：

```bash
python my_devs/train/groot/so101/run_groot_infer_trt.py \
  --policy-path /path/to/pretrained_model \
  --engine-dir /path/to/gr00t_engine_api_trt1013 \
  --tensorrt-py-dir /data/cqy_workspace/third_party/tensorrt_10_13_0_35 \
  --dry-run true
```

2. Mock 上机（连接机器人与相机，但不下发动作）：

```bash
python my_devs/train/groot/so101/run_groot_infer_trt.py \
  --policy-path /path/to/pretrained_model \
  --engine-dir /path/to/gr00t_engine_api_trt1013 \
  --tensorrt-py-dir /data/cqy_workspace/third_party/tensorrt_10_13_0_35 \
  --robot-port /dev/ttyACM0 \
  --top-cam-index 4 --wrist-cam-index 6 \
  --task "Put the block in the bin" \
  --run-time-s 10 --fps 5 \
  --mock true
```

3. 真下发动作（建议先小步试探，例如 3 秒、2 FPS）：

```bash
python my_devs/train/groot/so101/run_groot_infer_trt.py \
  --policy-path /path/to/pretrained_model \
  --engine-dir /path/to/gr00t_engine_api_trt1013 \
  --tensorrt-py-dir /data/cqy_workspace/third_party/tensorrt_10_13_0_35 \
  --robot-port /dev/ttyACM0 \
  --top-cam-index 4 --wrist-cam-index 6 \
  --task "Put the block in the bin" \
  --run-time-s 3 --fps 2 \
  --mock false
```

安全提醒（强烈建议遵守）：

1. 第一次真下发动作请确保人手可随时 Ctrl+C 或急停。
2. 先低 FPS、短 run_time_s，确认动作方向/幅度正常，再逐步提高。

---

## 8. 一致性验证（我们做了什么、结果如何）

一致性验证的目标是：在“同一输入”下，验证不同后端输出接近，从而减少上机风险。

本方案提供三种对比：

1. Torch vs ONNX（`compare_torch_onnx.py`）
2. Torch vs TRT（`compare_torch_trt.py`）
3. 上机前安全 mock（Torch vs TRT，`run_groot_infer_trt_mock.py`）

### 8.1 Torch vs ONNX（本轮结果摘录）

本轮 ONNX Runtime 使用 CPU provider（`CPUExecutionProvider`）。端到端 pipeline 指标（action_denoising_pipeline）：

- 1-view：cosine `0.9999999610`（见 `gr00t_onnx/compare_metrics_1view.json`）
- 2-view：cosine `0.9999999590`（见 `gr00t_onnx/compare_metrics_2view.json`）

### 8.2 Torch vs TRT（本轮结果摘录，TensorRT 10.13.0.35）

端到端 pipeline 指标（action_denoising_pipeline）：

- 1-view：cosine `0.9999998024`（见 `gr00t_engine_api_trt1013/compare_metrics_trt_1view.json`）
- 2-view：cosine `0.9999996799`（见 `gr00t_engine_api_trt1013/compare_metrics_trt_2view.json`）

说明：

- 模块级上，ViT/LLM 路径的数值漂移相对更明显（仍然是高 cosine），但端到端 action pipeline 仍然非常接近。

### 8.3 上机 mock 推理（TRT 引擎驱动，实际已跑通）

在上机推理中，我们已完成一次 `--mock true` 的实测运行，表现为：

- TRT engine 可正常加载并执行
- 控制循环可按 `--run-time-s` 正常退出
- 期间出现的 TRT warning（logger/stream）均为性能提示，不影响功能正确性

---

## 9. 常见问题与排查（最容易踩坑的点）

### 9.1 `config.json not found` / `PreTrainedConfig` 解析失败

原因：

- `--policy-path` 传错了（比如传成仓库根目录 `.` 或传成训练 run 的父目录）。

正确做法：

- 传 `.../checkpoints/last/pretrained_model` 这个目录。

脚本侧已做增强：

- `run_groot_infer_trt.py` 会尝试在你给的路径下自动寻找 `pretrained_model/`，并在失败时给出明确报错提示。

### 9.2 `ModuleNotFoundError: No module named 'tensorrt'`

原因：

- TensorRT 没装在当前 conda env site-packages 内，且未设置 `TENSORRT_PY_DIR`。

解决：

1. 推荐安装到 `/data/...`（见 4.2）
2. 运行时传参或设置环境变量：
   - `--tensorrt-py-dir /data/cqy_workspace/third_party/tensorrt_10_13_0_35`
   - 或 `export TENSORRT_PY_DIR=/data/cqy_workspace/third_party/tensorrt_10_13_0_35`

### 9.3 CUDA OOM（显存不足）

GR00T N1.5 + 运行时 glue 模块仍需加载 Torch policy（用于配置与部分 glue），如果 GPU 上同时有其它大进程占用显存，很容易 OOM。

建议：

- `nvidia-smi` 查看显存占用，先清掉其它大进程。
- 初次上机使用 `--mock true` 并缩短 `--run-time-s`。

### 9.4 TRT warning：default stream / logger mismatch

这类 warning 典型含义：

- default stream：性能提示（同步开销），不影响正确性
- logger mismatch：TensorRT 全局 logger 注册行为，通常不影响运行

如果你关心性能再做进一步优化即可，FP16 correctness 验证不依赖消除这些 warning。

---

## 10. 附录：关键脚本清单（你只需要记住这些入口）

导出 ONNX：

- `my_devs/groot_trt/export_backbone_onnx.py`
- `my_devs/groot_trt/export_action_head_onnx.py`

构建 engine（TensorRT Python API）：

- `my_devs/groot_trt/build_engine.py`
- `my_devs/groot_trt/build_engine.sh`

一致性验证：

- `my_devs/groot_trt/compare_torch_onnx.py`
- `my_devs/groot_trt/compare_torch_trt.py`
- `my_devs/train/groot/so101/run_groot_infer_trt_mock.py`（安全 mock 对比，推荐上机前跑）

上机推理（TRT engine 驱动）：

- `my_devs/train/groot/so101/run_groot_infer_trt.py`
