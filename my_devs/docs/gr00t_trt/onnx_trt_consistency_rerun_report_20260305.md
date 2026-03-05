# GROOT ONNX + TensorRT (TRT) 全流程复现与一致性验证工作报告（Rerun: 2026-03-05 10:29:31）

本报告的目标是把 “导出 ONNX -> 校验 Torch vs ONNX -> 用 TensorRT Python API 构建 engine -> 校验 Torch vs TRT”
这一整条链路，**在当前仓库内可复现地跑通**，并把关键结论整理成任何没做过部署的人也能照着做的操作手册。

如果你只想要结论：

- ONNX 导出正确，Torch vs ONNX 一致性非常好（最低 cosine 约 `0.99997`，端到端 `action_denoising_pipeline` 约 `0.99999996`）。
- TensorRT FP16 engine 可成功构建（TensorRT `10.13.0.35`，Python API，Strongly Typed），Torch vs TRT 也整体一致，
  其中 ViT + LLM 路径数值漂移更明显，但端到端 `action_denoising_pipeline` 仍接近 `0.9999997~0.9999998`。

本轮复现的所有产物（ONNX/engine/日志/JSON 指标）都在：

- `outputs/trt/consistency_rerun_20260305_102931/`

---

## 1. 你需要理解的几个概念（给非部署同学的最低门槛解释）

为了看懂后面的步骤，这里用最短解释把关键概念讲清楚：

- **ONNX**：一种“模型计算图”的通用格式。把 PyTorch 模型导出成 ONNX 后，能用 ONNX Runtime 或 TensorRT 等推理框架加载。
- **TensorRT engine（`.engine`）**：TensorRT 把 ONNX（或其它表示）解析后，结合 GPU、精度（FP16/FP8/INT8/…）与
  动态 shape profile 等信息，编译生成的二进制推理计划。engine 绑定了 profile，所以一般不能“一把梭”任意输入 shape。
- **动态 shape profile（min/opt/max）**：告诉 TensorRT 引擎要支持哪些输入形状范围，并指定一个最常用的 `opt` shape
  来让 TensorRT 优化性能。超出 max 会报错。
- **一致性验证（consistency）**：同一输入下，对比两个后端（Torch/ONNX/TRT）的输出差异，常用指标是 cosine 相似度、
  RMSE、绝对误差等。这里我们做两层：
  - 模块级：ViT、LLM、action head 的若干子模块
  - 流程级：最贴近真实推理闭环的 `action_denoising_pipeline`

---

## 2. 复现边界与约束

- **必须使用 conda 环境 `lerobot_flex`**（仓库约定，所有开发/运行都要在这个环境里）。
- 本轮是 **FP16 baseline**（量化/FP8/INT8 不在本报告范围内；见 `my_devs/docs/gr00t_trt/量化方案.md`）。
- 一致性比较使用的是 **合成输入（synthetic inputs）**，用固定随机种子生成，目的是做“可重复的数值一致性”检查，
  不是评估任务成功率。
- 当前环境 `onnxruntime` 没有 CUDA provider，因此 Torch vs ONNX 的 ONNX Runtime 推理是 **CPUExecutionProvider**。

---

## 3. 环境信息（本轮实测）

运行环境（`lerobot_flex`）：

- Python: `3.10.19`
- Torch: `2.7.1+cu126`
- ONNX: `1.17.0`
- ONNX Runtime: `1.20.1`
- ORT providers: `['AzureExecutionProvider', 'CPUExecutionProvider']`（实际使用 CPU）

TensorRT（用于 engine 构建 + TRT 推理）：

- TensorRT: `10.13.0.35`
- 使用方式：**TensorRT Python API**（非 `trtexec`）

说明：由于根目录磁盘空间容易不足，TensorRT 推荐安装到 `/data` 并通过 `TENSORRT_PY_DIR` 指定（下文给命令）。

---

## 4. 本轮使用的模型权重与产物目录

### 4.1 Policy 权重

本轮使用的 checkpoint（同训练产物）：

```text
/data/cqy_workspace/flexible_lerobot/outputs/train/
  groot_grasp_block_in_bin1_repro_20260302_223413/
    bs32_20260302_223447/checkpoints/last/pretrained_model
```

### 4.2 本轮产物目录

本轮 rerun 根目录：

```text
outputs/trt/consistency_rerun_20260305_102931/
```

目录结构（重要）：

```text
outputs/trt/consistency_rerun_20260305_102931/
  gr00t_onnx/
    eagle2/
      vit_fp16.onnx
      llm_fp16.onnx
    action_head/
      vlln_vl_self_attention.onnx
      state_encoder.onnx
      action_encoder.onnx
      DiT_fp16.onnx
      action_decoder.onnx
    compare_metrics_1view.json
    compare_metrics_2view.json
  gr00t_engine_api_trt1013/
    vit_fp16.engine
    llm_fp16.engine
    vlln_vl_self_attention.engine
    state_encoder.engine
    action_encoder.engine
    DiT_fp16.engine
    action_decoder.engine
    build_report.json
    compare_metrics_trt_1view.json
    compare_metrics_trt_2view.json
  logs/
    export_backbone.log
    export_action_head.log
    compare_onnx_1view.log
    compare_onnx_2view.log
    build_trt_engine.log
    compare_trt_1view.log
    compare_trt_2view.log
```

---

## 5. 一键复现的推荐步骤（从零开始）

下面的命令你可以直接复制粘贴执行。

为了避免路径混乱，我用变量把路径写清楚。你需要替换的只有 `POLICY_PATH` 和（可选）`RUN_ID`。

### 5.1 进入仓库根目录

```bash
cd /data/cqy_workspace/flexible_lerobot
```

### 5.2（可选）安装 TensorRT Python 包到 `/data`（推荐做法）

如果你机器的 root 分区空间不够，直接 `pip install tensorrt==...` 到 conda env 往往会失败。
推荐安装到 `/data/...`，然后靠 `TENSORRT_PY_DIR` 指向它。

```bash
conda run -n lerobot_flex python -m pip install --target /data/cqy_workspace/third_party/tensorrt_10_13_0_35 \
  tensorrt==10.13.0.35
```

验证 import（这一步非常重要）：

```bash
TMPDIR=/data/cqy_workspace/tmp \
conda run -n lerobot_flex env \
  PYTHONPATH=/data/cqy_workspace/third_party/tensorrt_10_13_0_35:$PYTHONPATH \
  LD_LIBRARY_PATH=/data/cqy_workspace/third_party/tensorrt_10_13_0_35/tensorrt_libs:$LD_LIBRARY_PATH \
  python -c "import tensorrt as trt; print(trt.__version__)"
```

预期输出：`10.13.0.35`

### 5.3 创建本轮产物目录

```bash
RUN_ID=consistency_rerun_$(date +%Y%m%d_%H%M%S)
RUN_DIR=outputs/trt/$RUN_ID
mkdir -p $RUN_DIR/logs
echo $RUN_DIR
```

### 5.4 导出 ONNX（backbone + action head）

准备权重路径：

```bash
POLICY_PATH=/data/cqy_workspace/flexible_lerobot/outputs/train/groot_grasp_block_in_bin1_repro_20260302_223413/bs32_20260302_223447/checkpoints/last/pretrained_model
```

导出 backbone（ViT + LLM）：

```bash
conda run -n lerobot_flex python my_devs/groot_trt/export_backbone_onnx.py \
  --policy-path $POLICY_PATH \
  --onnx-out-dir $RUN_DIR/gr00t_onnx \
  --seq-len 296 \
  --video-views 1 \
  --vit-dtype fp16 \
  --llm-dtype fp16 \
  --device cuda \
  > $RUN_DIR/logs/export_backbone.log 2>&1
```

导出 action head（5 个子模块）：

```bash
conda run -n lerobot_flex python my_devs/groot_trt/export_action_head_onnx.py \
  --policy-path $POLICY_PATH \
  --onnx-out-dir $RUN_DIR/gr00t_onnx \
  --seq-len 296 \
  --device cuda \
  > $RUN_DIR/logs/export_action_head.log 2>&1
```

你可以用这条命令确认导出了 7 个 ONNX：

```bash
find $RUN_DIR/gr00t_onnx -type f -name '*.onnx' | wc -l
```

预期输出：`7`

### 5.5 Torch vs ONNX 一致性验证（1-view / 2-view）

1-view（`seq_len=296, video_views=1`）：

```bash
conda run -n lerobot_flex python my_devs/groot_trt/compare_torch_onnx.py \
  --policy-path $POLICY_PATH \
  --onnx-dir $RUN_DIR/gr00t_onnx \
  --seq-len 296 \
  --video-views 1 \
  --vit-dtype fp16 --llm-dtype fp16 --dit-dtype fp16 \
  --device cuda \
  --json-out $RUN_DIR/gr00t_onnx/compare_metrics_1view.json \
  > $RUN_DIR/logs/compare_onnx_1view.log 2>&1
```

2-view（`seq_len=568, video_views=2`）：

```bash
conda run -n lerobot_flex python my_devs/groot_trt/compare_torch_onnx.py \
  --policy-path $POLICY_PATH \
  --onnx-dir $RUN_DIR/gr00t_onnx \
  --seq-len 568 \
  --video-views 2 \
  --vit-dtype fp16 --llm-dtype fp16 --dit-dtype fp16 \
  --device cuda \
  --json-out $RUN_DIR/gr00t_onnx/compare_metrics_2view.json \
  > $RUN_DIR/logs/compare_onnx_2view.log 2>&1
```

### 5.6 用 TensorRT Python API 构建 engine

本仓库脚本位置（你要求的路径）：

- `my_devs/groot_trt/build_engine.py`：TensorRT Python API 真正 build engine 的实现
- `my_devs/groot_trt/build_engine.sh`：包一层 conda/env 变量，方便一条命令跑通

构建 engine（profile 默认按 2-view 配置，覆盖 1-view/2-view 两种输入）：

```bash
ONNX_DIR=$RUN_DIR/gr00t_onnx \
ENGINE_DIR=$RUN_DIR/gr00t_engine_api_trt1013 \
VIDEO_VIEWS=2 MAX_BATCH=2 \
WORKSPACE_GB=8 \
bash my_devs/groot_trt/build_engine.sh \
  > $RUN_DIR/logs/build_trt_engine.log 2>&1
```

你可以用这条命令确认导出了 7 个 engine：

```bash
find $RUN_DIR/gr00t_engine_api_trt1013 -maxdepth 1 -type f -name '*.engine' | wc -l
```

预期输出：`7`

### 5.7 Torch vs TRT 一致性验证（1-view / 2-view）

说明：TRT 对比脚本会自动通过 `TENSORRT_PY_DIR` 寻找 TensorRT Python 包（默认也会尝试本机固定路径）。

1-view：

```bash
TMPDIR=/data/cqy_workspace/tmp \
conda run -n lerobot_flex env TENSORRT_PY_DIR=/data/cqy_workspace/third_party/tensorrt_10_13_0_35 \
  python my_devs/groot_trt/compare_torch_trt.py \
    --policy-path $POLICY_PATH \
    --engine-dir $RUN_DIR/gr00t_engine_api_trt1013 \
    --seq-len 296 \
    --video-views 1 \
    --device cuda \
    --json-out $RUN_DIR/gr00t_engine_api_trt1013/compare_metrics_trt_1view.json \
    > $RUN_DIR/logs/compare_trt_1view.log 2>&1
```

2-view：

```bash
TMPDIR=/data/cqy_workspace/tmp \
conda run -n lerobot_flex env TENSORRT_PY_DIR=/data/cqy_workspace/third_party/tensorrt_10_13_0_35 \
  python my_devs/groot_trt/compare_torch_trt.py \
    --policy-path $POLICY_PATH \
    --engine-dir $RUN_DIR/gr00t_engine_api_trt1013 \
    --seq-len 568 \
    --video-views 2 \
    --device cuda \
    --json-out $RUN_DIR/gr00t_engine_api_trt1013/compare_metrics_trt_2view.json \
    > $RUN_DIR/logs/compare_trt_2view.log 2>&1
```

---

## 6. 本轮结果汇总（关键指标）

本轮 run id：`consistency_rerun_20260305_102931`

### 6.1 Torch vs ONNX（CPU ORT）结果

1-view（seq=296）：

- 最低 cosine：`llm_from_vit_pipeline = 0.9999742818`
- 端到端：`action_denoising_pipeline = 0.9999999610`

2-view（seq=568）：

- 最低 cosine：`llm_from_vit_pipeline = 0.9999869419`
- 端到端：`action_denoising_pipeline = 0.9999999590`

解释：

- 这说明 ONNX 导出与 PyTorch 参考实现 **高度一致**，可以作为 TRT 构建前的基线。
- 误差最低点出现在 `vit -> llm` 组合路径很正常：该路径包含更复杂的 attention/MLP 计算，且 tensor 规模更大。

### 6.2 Torch vs TRT（TensorRT 10.13.0.35, FP16 engine）结果

1-view（seq=296）：

- 最低 cosine：`llm_from_vit_pipeline = 0.9992854137`
- 端到端：`action_denoising_pipeline = 0.9999998024`

2-view（seq=568）：

- 最低 cosine：`llm_from_vit_pipeline = 0.9995578666`
- 端到端：`action_denoising_pipeline = 0.9999996799`

解释（很重要）：

- TRT 相比 ONNX 会引入更明显的数值漂移，这是预期现象（tactic 选择、融合、FP16 算子实现差异等）。
- 但端到端 denoising pipeline 仍然非常接近 1.0，说明 “工程加速后整体推理闭环” 的数值行为仍然稳定。
- 如果未来要进一步降低漂移，可以从：
  - profile 更贴近真实输入分布（opt shape）
  - 减少某些层的融合或强制精度（更强约束）
  - 或进入量化/FP8/INT8 的系统化评估
  这几个方向继续做工程迭代。

---

## 7. 本轮 engine 构建耗时（来自 build_report.json）

engine 构建耗时（仅供参考，不同机器会不同）：

- `vit_fp16.engine`：约 `7.32s`
- `llm_fp16.engine`：约 `6.92s`
- `vlln_vl_self_attention.engine`：约 `5.22s`
- `state_encoder.engine`：约 `4.91s`
- `action_encoder.engine`：约 `3.73s`
- `DiT_fp16.engine`：约 `7.04s`
- `action_decoder.engine`：约 `4.72s`

---

## 8. 常见问题与排障

### 8.1 `import tensorrt` 失败

优先检查你是否设置了：

- `TENSORRT_PY_DIR=/data/cqy_workspace/third_party/tensorrt_10_13_0_35`

并确保该目录下存在：

- `tensorrt/`
- `tensorrt_libs/`（里面有 `libnvinfer.so.10` 等）

### 8.2 构建 engine 时报内存不够或很慢

建议逐步调小 profile：

- `MAX_BATCH` 从 `2` 调小到 `1`
- `MAX_LEN` 从 `600` 调小到更贴近真实推理的长度
- `WORKSPACE_GB` 适当调整（过小可能 build 失败，过大可能触发系统压力）

### 8.3 “为什么 ONNX 是 CPU，但 TRT 是 GPU？”

这和当前 `onnxruntime` 的安装有关：它只有 `CPUExecutionProvider`。
一致性验证的逻辑是：

- Torch reference：GPU（cuda）
- ONNX Runtime：CPU（所以它验证的是“导出正确性”，不是 GPU 性能）
- TensorRT：GPU（验证的是“部署后数值漂移”）

如果你希望 ONNX Runtime 也跑 GPU，需要额外安装/配置 `onnxruntime-gpu` 并确保 CUDA provider 可用。

---

## 9. 相关脚本索引（仓库内位置）

全部脚本都在你指定的路径：`my_devs/groot_trt/`

- `export_backbone_onnx.py`：导出 backbone（ViT + LLM）ONNX
- `export_action_head_onnx.py`：导出 action head 5 子模块 ONNX
- `compare_torch_onnx.py`：Torch vs ONNX（模块级 + 流程级）一致性验证
- `build_engine.py`：TensorRT Python API 构建 engine（Strongly Typed, FP16 baseline）
- `build_engine.sh`：build_engine.py 的 wrapper（负责 conda/env/TMPDIR）
- `compare_torch_trt.py`：Torch vs TensorRT engine（模块级 + 流程级）一致性验证

