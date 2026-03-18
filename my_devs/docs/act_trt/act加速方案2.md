# ACT 加速方案 2：PyTorch ACTPolicy → ONNX → TensorRT Engine → 接入 `run_act_infer.py`

> 适用对象：`my_devs/train/act/so101/run_act_infer.py` 的 ACT 实机推理（SO100/SO101 follower）  
> 目标：在 **不改变机器人控制语义（action queue / temporal ensemble）** 的前提下，把 **ACT 核心网络前向**从 PyTorch 替换为 TensorRT 推理。
>
> 本方案是“方案 2”：**TensorRT engine 加速**。  
> （对比：你们已有的参考资料里已经沉淀了 ONNX/ORT 路线；TRT 路线需要额外补齐 engine 构建与 runtime 接入。）

---

## 0. 你需要先知道的 3 个事实（避免走弯路）

### 0.1 `run_act_infer.py` 当前推理链路长什么样

脚本：`my_devs/train/act/so101/run_act_infer.py`

核心链路（简化版）：

1. `robot.get_observation()` 拿到原始观测（numpy）
2. `robot_observation_processor(obs)` 对齐 observation features（numpy）
3. `build_dataset_frame(...)` 拼出和训练一致的 `observation_frame`（numpy）
4. `predict_action(...)`：
   - `prepare_observation_for_inference`: numpy → torch，图像 `/255`，`HWC→CHW`，加 batch
   - `policy_preprocessor`：用 checkpoint 里保存的 stats 做归一化 / 对齐
   - `policy.select_action`：ACT 的 chunk 推理 + 队列语义（**性能瓶颈**）
   - `policy_postprocessor`：反归一化
5. `make_robot_action` / `robot_action_processor` / `robot.send_action`

我们要替换的只有第 4 步中的 **核心网络前向**（即 `policy.model(batch)` 那一段），其它保持一致。

---

### 0.2 ACT 的“动作队列语义”决定了你应该导出什么

ACT 的 `select_action()` 并不是每一步都完整前向：

- 当 `temporal_ensemble_coeff is None` 时：
  - 队列为空：跑一次 `predict_action_chunk()` 得到 `(B, chunk_size, action_dim)`，只取前 `n_action_steps` 入队
  - 队列非空：只 `popleft()`，不再前向
- 当 `temporal_ensemble_coeff is not None` 时：
  - 每步都需要一个 action chunk（然后做 temporal ensemble）

因此：

- **不建议导出 `ACTPolicy.select_action`**（里面有 Python deque/状态机，ONNX/TRT 不适合表达）
- **建议导出 `policy.model`（ACT 核心网络）**：
  - 输入：已预处理 + 已归一化的 state / images
  - 输出：归一化的 action chunk
  - 队列 / temporal ensemble 留在 Python 侧复刻

这与 `my_devs/docs/act_trt/reference_docs/onnx_export_reference/infer_docs/模型导出方案.md` 的结论一致。

---

### 0.3 你不能“猜 IO”，必须从 checkpoint 里读

以当前 SO101 示例 checkpoint 为例（`run_act_infer.py` 的默认 checkpoint 同系列）：

- checkpoint：`outputs/train/20260305_190147_act_grasp_block_in_bin1_e15/checkpoints/last/pretrained_model`
- `config.json` 里写死了：
  - `observation.state`: shape `[6]`
  - `observation.images.top`: shape `[3, 480, 640]`
  - `observation.images.wrist`: shape `[3, 480, 640]`
  - `action`: shape `[6]`
  - `chunk_size=100`
  - `n_action_steps=100`

任何导出/engine/profile 必须跟这些事实一致。

快速检查命令（务必每个新 checkpoint 都跑一遍）：

```bash
conda run -n lerobot_flex python - <<'PY'
import json
from pathlib import Path

ckpt = Path("outputs/train/20260305_190147_act_grasp_block_in_bin1_e15/checkpoints/last/pretrained_model")
cfg = json.loads((ckpt / "config.json").read_text())

visual = [k for k, v in cfg.get("input_features", {}).items() if v.get("type") == "VISUAL"]
print("VISUAL keys (order matters):", visual)
print("state shape:", cfg["input_features"]["observation.state"]["shape"])
print("action shape:", cfg["output_features"]["action"]["shape"])
print("chunk_size:", cfg.get("chunk_size"))
print("n_action_steps:", cfg.get("n_action_steps"))
PY
```

---

## 1. 方案 2 总览（你最终会得到什么）

### 1.1 最终产物（建议放在 checkpoint 目录内，方便“一键部署”）

对每个 `<ckpt_dir>=.../pretrained_model`，建议产物如下：

- `<ckpt_dir>/act_single.onnx`  
  由 ONNX 导出脚本生成（核心网络，normalized in/out）
- `<ckpt_dir>/act_single.onnx.data`（可选）  
  若 ONNX 使用 external data 保存权重，会有这个文件
- `<ckpt_dir>/export_metadata.json`  
  记录相机顺序、输入输出名、shape、导出参数、验证指标
- `<ckpt_dir>/act_single_fp16.plan`  
  TensorRT engine（FP16 推荐）
- `<ckpt_dir>/act_single_fp16.tcache`（可选）  
  timing cache（强烈建议，二次构建会快很多）
- `<ckpt_dir>/trt_build_summary.json`（建议新增）  
  记录 TRT 构建参数（workspace/opt_level/profile/precision 等），用于审计与复现

> 命名可以按你们习惯调整，但建议固定规则，避免现场拿错 engine。

---

### 1.2 数据流（部署时的真实闭环）

```
Robot obs (np)                       Checkpoint artifacts
┌────────────────────┐              ┌──────────────────────────────┐
│ top/wrist images    │              │ policy_preprocessor.json     │
│ state               │              │ policy_postprocessor.json    │
└─────────┬──────────┘              │ (un)normalizer *.safetensors │
          │                          └───────────────┬──────────────┘
          ▼                                          │
  prepare_observation_for_inference (torch)          │
          ▼                                          │
  policy_preprocessor (torch)  ──────────────────────┘
          ▼
  ┌───────────────────────────────────────────────────────────┐
  │                 ActTrtPolicy.select_action                 │
  │  - gather inputs by camera order                            │
  │  - TensorRT engine forward: actions_norm chunk              │
  │  - queue / temporal ensemble (Python)                       │
  └───────────────────────────────────────────────────────────┘
          ▼
  policy_postprocessor (torch)
          ▼
  make_robot_action / send_action
```

**关键点：** pre/post processor 仍然来自 checkpoint，本方案只替换“核心网络前向”。

---

## 2. 环境准备（版本要锁死，别靠“应该能跑”）

### 2.1 仓库规范要求

仓库根目录 `AGENTS.md` 明确要求：

- **开发代码与测试运行必须使用 conda env：`lerobot_flex`**

因此建议路线：

1. 在 `lerobot_flex` 内补齐 TRT 相关依赖（推荐）
2. 如果 `lerobot_flex` 不方便安装 TRT（常见原因：驱动/权限/离线），可以单独建一个 *部署 env*，但：
   - 代码修改、静态检查、pytest 仍然在 `lerobot_flex`
   - 部署 env 需要 `pip install -e .` 或 `PYTHONPATH` 指向本仓库

---

### 2.2 依赖清单（最小可跑）

**导出 ONNX**（CPU 即可）：

- `torch`（已有）
- `onnx`
- `onnxruntime`（用于验证导出，强烈建议）
- `onnxsim`（可选，用于简化）

**构建 + 推理 TensorRT**（需要 NVIDIA GPU 环境）：

- `tensorrt`（Python bindings）
- CUDA runtime（driver/Toolkit）
- `cuda-python`（或等价方案）用于 `cudaMalloc/cudaMemcpy/stream`（参考 `trt_reference/trt_infer.py`）
- `numpy`

在当前环境下，我这边检查 `lerobot_flex` 里**默认没有** `tensorrt`：

```bash
conda run -n lerobot_flex python -c "import tensorrt"
# -> ModuleNotFoundError: No module named 'tensorrt'
```

所以你需要自己按机器条件安装（不同机器/不同 CUDA/不同 TRT 版本安装方式差异很大）。

> 本文不强行指定“唯一安装命令”，但会把“怎么验证安装正确”写清楚。

---

### 2.3 版本验证（必须能打印版本号）

建议你在目标机器上确认以下命令都能跑通：

```bash
conda run -n lerobot_flex python - <<'PY'
import sys
print("python:", sys.version.split()[0])
import torch
print("torch:", torch.__version__)

try:
    import onnx, onnxruntime as ort
    print("onnx:", onnx.__version__)
    print("onnxruntime:", ort.__version__)
except Exception as e:
    print("[WARN] onnx/onnxruntime not ready:", type(e).__name__, e)

try:
    import tensorrt as trt
    print("tensorrt:", trt.__version__)
except Exception as e:
    print("[WARN] tensorrt not ready:", type(e).__name__, e)

try:
    from cuda import cudart
    err, count = cudart.cudaGetDeviceCount()
    print("cudaGetDeviceCount:", err, count)
except Exception as e:
    print("[WARN] cuda-python not ready:", type(e).__name__, e)
PY
```

你应该看到：

- `tensorrt` 能 import 并打印版本
- `cudaGetDeviceCount` 返回 `count>0`

否则不要急着写代码，先把环境搞定（TRT 环境不对，后面全是无效劳动）。

---

## 3. 步骤 1：导出 ONNX（ACT 核心网络，normalized in/out）

### 3.1 直接复用参考脚本（推荐起步方式）

参考脚本在你给的资料里已经实现好了：

- ONNX 导出脚本：`my_devs/docs/act_trt/reference_docs/onnx_export_reference/export/export_single.py`
- 导出设计解释：`my_devs/docs/act_trt/reference_docs/onnx_export_reference/infer_docs/模型导出方案.md`

这个脚本的关键特性：

- 导出的是 `policy.model`（不是 `ACTPolicy.select_action`）
- 输入输出都是 **归一化后的 float32**
- 自动从 `<ckpt>/config.json` 推导：
  - `state_dim`
  - 图像 H/W
  - camera 顺序（VISUAL keys 的插入顺序）
  - `chunk_size`、`action_dim`
- 支持：
  - `--verify` 用 onnxruntime 对比 PyTorch 输出（默认开）
  - `--simplify` 用 onnxsim 简化（默认开）

---

### 3.2 导出命令（以 SO101 checkpoint 为例）

```bash
CKPT="outputs/train/20260305_190147_act_grasp_block_in_bin1_e15/checkpoints/last/pretrained_model"

conda run -n lerobot_flex python my_devs/docs/act_trt/reference_docs/onnx_export_reference/export/export_single.py \
  --checkpoint "${CKPT}" \
  --output "${CKPT}/act_single.onnx" \
  --opset 17 \
  --device cpu \
  --verify \
  --simplify
```

说明：

- `--device cpu`：导出时 forward 在 CPU 做即可（更稳定，不依赖 GPU 环境）
- `--opset 17`：通常足够；若后续 TRT parse 报不支持，可以再尝试 14/16/18

---

### 3.3 导出后验收（必须满足）

导出完成后，至少检查这些点：

1. ONNX 文件存在：
   - `${CKPT}/act_single.onnx`
   - 若有 external weights：`${CKPT}/act_single.onnx.data`
2. 元数据存在：`${CKPT}/export_metadata.json`
3. `export_metadata.json` 里：
   - `camera_order_visual_keys` 与你期望一致（例如 top/wrist）
   - `verification.passed == true`（或至少 `max_abs_diff <= 1e-4`）

快速查看：

```bash
conda run -n lerobot_flex python - <<'PY'
import json
from pathlib import Path
ckpt = Path("outputs/train/20260305_190147_act_grasp_block_in_bin1_e15/checkpoints/last/pretrained_model")
meta = json.loads((ckpt / "export_metadata.json").read_text())
print("onnx_path:", meta["onnx_path"])
print("camera order:", meta["camera_order_visual_keys"])
print("shapes:", meta["shapes"])
print("verify:", meta.get("verification", {}))
PY
```

---

### 3.4 导出时的 TensorRT 友好设置（经验项，遇到 TRT parse 问题再用）

导出脚本 `export_single.py` 已经足够覆盖“能导出/能验证/能跑 ORT”。但 TensorRT 对 ONNX 的支持不是 100% 全覆盖，遇到 parse 失败时，通常按下面顺序尝试，性价比最高：

1. **先保持导出边界不变：只导出核心网络**
   - 也就是仍然导出 `policy.model`，不要把队列、pre/post、task 文本等塞进图里
2. **切换导出器：`--dynamo`**
   - `export_single.py` 支持 `--dynamo`（Torch ONNX dynamo exporter）
   - 很多模型在 dynamo 导出下会得到更“规整”的图，TRT 更容易 parse
3. **调整 opset**
   - 常用尝试：`--opset 14`、`--opset 16`、`--opset 18`
   - 一般来说：opset 越高算子越新，但 TRT 未必支持得更好；需要实际试
4. **关闭/调整 external data（如果你后续写 Python builder 时只会 `parser.parse(bytes)`）**
   - 当前参考脚本 export 时固定 `external_data=True`
   - 如果你确定模型权重远小于 2GB，可以考虑在你自己的导出脚本里把 `external_data=False`，让所有权重内嵌到一个 `.onnx`，这样：
     - TensorRT 可以直接 `parser.parse(onnx_bytes)`
     - 不需要担心 `.onnx.data` 的路径解析
   - 如果你不想改导出脚本：那 Python builder 必须使用 `parse_from_file`（见第 4.4 节）
5. **ONNX 图过大/parse 超慢**
   - 参考资料里已经提示：ACT 的视觉 token 构造存在 `list(tensor)` 展开，可能导致 ONNX 图膨胀
   - 这种情况本质是“模型实现方式不利于导出”，需要对 `ACT.forward` 做导出友好的改造（属于增强项，不是方案 2 最小闭环必需）

---

## 4. 步骤 2：构建 TensorRT Engine（ONNX → `.plan`）

这一段是方案 2 的核心：把 ONNX 编译成 TensorRT engine。

你给的参考资料里已经沉淀了 TRT 的 build/infer 规范：

- 设计与流程手册：`my_devs/docs/act_trt/reference_docs/trt_reference/TENSORRT_ENGINE_BUILD_INFER_PLAYBOOK.md`
- Python build 样例：`my_devs/docs/act_trt/reference_docs/trt_reference/trt_build.py`
- Python infer 样例：`my_devs/docs/act_trt/reference_docs/trt_reference/trt_infer.py`

但是：`trt_build.py` / `trt_infer.py` 当前是以 **ResNet 单输入**为例写的，接入 ACT 时需要改造成 **三输入**。

本节给出“ACT 版本”的改造方案（按可落地程度从易到难排序）。

---

### 4.1 Engine 构建方式选择（建议先 A 后 B）

#### 方式 A：先用 `trtexec` 跑通（最快验证 ONNX 是否可被 TRT 接受）

优点：

- 命令行简单
- 最快知道“这份 ONNX 能不能被 TRT parse”

缺点：

- 不好做审计/复现（参数多、输出杂）
- 不利于后续直接接入 `run_act_infer.py`

#### 方式 B：用 Python builder 脚本固化 build 参数（推荐最终形态）

优点：

- 参数统一、可审计（输出 JSON summary）
- timing cache 复用更方便
- 后续同一套脚手架可以迁移其它模型

缺点：

- 需要你们维护一个 ACT 专用的 build 脚本（但这是值得的）

下面会把两种方式都写清楚。

---

### 4.2 输入输出名与 shape（构建 profile 的唯一依据）

导出脚本已经固定了 ONNX 的 input/output names：

- inputs：
  - `obs_state_norm`: `[1, state_dim]`
  - `img0_norm`: `[1, 3, H, W]`
  - `img1_norm`: `[1, 3, H, W]`
- outputs：
  - `actions_norm`: `[1, chunk_size, action_dim]`

其中：

- `state_dim / H / W / chunk_size / action_dim` 以 `<ckpt>/export_metadata.json` 为准
- `img0_norm/img1_norm` 的相机顺序由 `camera_order_visual_keys` 决定

在示例 checkpoint（top/wrist, 480x640）里，shape 就是：

- `obs_state_norm`: `[1, 6]`
- `img0_norm`: `[1, 3, 480, 640]`（top）
- `img1_norm`: `[1, 3, 480, 640]`（wrist）
- `actions_norm`: `[1, 100, 6]`

---

### 4.3 方式 A：`trtexec` 快速构建（仅用于验证）

> 注意：不同 TensorRT 版本 `trtexec` 参数略有差异；以下是常见写法，你需要按实际版本调整。

```bash
CKPT="outputs/train/20260305_190147_act_grasp_block_in_bin1_e15/checkpoints/last/pretrained_model"
ONNX="${CKPT}/act_single.onnx"
ENGINE="${CKPT}/act_single_fp16.plan"

# 这是一种典型写法：静态 shape（batch=1）
trtexec \
  --onnx="${ONNX}" \
  --saveEngine="${ENGINE}" \
  --fp16 \
  --workspace=4096 \
  --verbose \
  --dumpProfile \
  --separateProfileRun \
  --buildOnly
```

如果你的 ONNX 是 dynamic batch（导出时开了 `--dynamic-batch`），则需要给每个输入指定 profile：

```bash
trtexec \
  --onnx="${ONNX}" \
  --saveEngine="${ENGINE}" \
  --fp16 \
  --minShapes=obs_state_norm:1x6,img0_norm:1x3x480x640,img1_norm:1x3x480x640 \
  --optShapes=obs_state_norm:1x6,img0_norm:1x3x480x640,img1_norm:1x3x480x640 \
  --maxShapes=obs_state_norm:1x6,img0_norm:1x3x480x640,img1_norm:1x3x480x640 \
  --workspace=4096 \
  --buildOnly
```

你要的结果是：

- 能成功生成 `${ENGINE}`
- 且 `trtexec` 不报 `unsupported node/op` / `parse error`

如果 `trtexec` 都失败，先不要写 Python builder，先解决 ONNX 本身（见第 8 节排障）。

---

### 4.4 方式 B：Python Builder 脚本（推荐最终落地）

#### 4.4.1 你应该写一个 ACT 专用 build 脚本（建议文件名）

建议新增一个脚本（最终位置你们可调整，我这里建议放在和 ACT 部署脚本同目录，方便管理）：

- `my_devs/train/act/so101/act_trt_build_engine.py`（建议新增）

脚本职责：

1. 读取 ONNX（最好 **parse from file**，以兼容 `.onnx.data` external weights）
2. 设置 builder config（workspace/opt_level/fp16）
3. （可选）设置动态 batch profile：对 3 个输入都 `profile.set_shape(...)`
4. 构建 `.plan` 并写入
5. 写一份 `trt_build_summary.json`（参数审计）
6. 反序列化回读验证 engine 可用

#### 4.4.2 必须遵守的 builder API 顺序

按 `TENSORRT_ENGINE_BUILD_INFER_PLAYBOOK.md` 的规范（强烈建议照抄顺序）：

1. `logger = trt.Logger(trt.Logger.WARNING)`
2. `trt.init_libnvinfer_plugins(logger, "")`
3. `builder = trt.Builder(logger)`
4. `network = builder.create_network(flags)`
5. `parser = trt.OnnxParser(network, logger)`
6. `parser.parse_from_file(str(onnx_path))`（优先）或 `parser.parse(onnx_bytes)`
7. `config = builder.create_builder_config()`
8. `config.set_memory_pool_limit(...)`
9. `config.builder_optimization_level = opt_level`
10. precision（FP16/FP32）
11. profile（若动态 batch）
12. timing cache（可选但推荐）
13. `plan = builder.build_serialized_network(network, config)`
14. 写 `.plan`
15. 回读反序列化验证

#### 4.4.3 动态 batch profile（ACT 多输入写法要点）

如果你想支持 batch>1（不一定需要，机器人上通常 batch=1 就够），profile 需要对 **每个输入**设置：

```python
profile = builder.create_optimization_profile()
profile.set_shape("obs_state_norm", (1, state_dim), (opt, state_dim), (max, state_dim))
profile.set_shape("img0_norm", (1, 3, H, W), (opt, 3, H, W), (max, 3, H, W))
profile.set_shape("img1_norm", (1, 3, H, W), (opt, 3, H, W), (max, 3, H, W))
config.add_optimization_profile(profile)
```

注意：

- ACT 推理部署很少需要 batch>1；**建议先做静态 batch=1 引擎**，跑稳后再加动态 batch

---

#### 4.4.4 Builder 脚本骨架（建议你直接复制成 `act_trt_build_engine.py`）

下面是一份“接近可直接运行”的脚本骨架（写法严格对齐 `trt_reference/trt_build.py` + playbook），重点改动点：

- 支持 3 个输入的 profile（obs_state_norm/img0_norm/img1_norm）
- 默认从 `export_metadata.json` 读取 shape（避免手写错）
- 优先使用 `parser.parse_from_file(...)`（兼容 `.onnx.data` external weights）

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import tensorrt as trt


def _check_cuda_available() -> None:
    from cuda import cudart  # type: ignore

    err, count = cudart.cudaGetDeviceCount()
    if err != cudart.cudaError_t.cudaSuccess or count is None or int(count) <= 0:
        raise RuntimeError(f"No CUDA device available (cudaGetDeviceCount -> {err}, {count}).")


def _parse_major_minor(version: str) -> tuple[int, int]:
    parts = version.split(".")
    if len(parts) < 2:
        return (0, 0)
    return int(parts[0]), int(parts[1])


def _is_trt_before_10() -> bool:
    major, _minor = _parse_major_minor(trt.__version__)
    return major < 10


def _to_plan_bytes(serialized_plan) -> bytes:
    try:
        return bytes(memoryview(serialized_plan))
    except TypeError:
        with serialized_plan as buffer:
            return bytes(buffer)


def _save_timing_cache(config: trt.IBuilderConfig, timing_cache_path: Path) -> None:
    timing_cache = config.get_timing_cache()
    serialized = timing_cache.serialize()
    try:
        data = memoryview(serialized)
        timing_cache_path.write_bytes(data.tobytes())
    except TypeError:
        with serialized as buffer:
            timing_cache_path.write_bytes(bytes(buffer))


def _load_export_metadata(export_metadata: Path) -> dict[str, Any]:
    with export_metadata.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build ACT TensorRT engine from ONNX.")
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--export-metadata", type=Path, default=None)
    p.add_argument("--engine", type=Path, default=None)
    p.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    p.add_argument("--workspace-bytes", type=int, default=(4 << 30))
    p.add_argument("--opt-level", type=int, default=3)
    p.add_argument("--timing-cache", type=Path, default=None)
    p.add_argument("--disable-timing-cache", action="store_true")
    p.add_argument("--dynamic-batch", action="store_true")
    p.add_argument("--min-batch", type=int, default=1)
    p.add_argument("--opt-batch", type=int, default=1)
    p.add_argument("--max-batch", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.onnx.is_file():
        raise FileNotFoundError(args.onnx)

    _check_cuda_available()

    export_metadata = args.export_metadata
    if export_metadata is None:
        export_metadata = args.onnx.parent / "export_metadata.json"
    if not export_metadata.is_file():
        raise FileNotFoundError(export_metadata)
    meta = _load_export_metadata(export_metadata)
    shapes = meta["shapes"]
    state_dim = int(shapes["obs_state_norm"][1])
    h = int(shapes["img0_norm"][2])
    w = int(shapes["img0_norm"][3])

    engine_path = args.engine or (args.onnx.parent / f"{args.onnx.stem}_{args.precision}.plan")
    timing_cache_path = args.timing_cache or (args.onnx.parent / f"{args.onnx.stem}_{args.precision}.tcache")

    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")

    builder = trt.Builder(logger)
    network_flags = 0
    if _is_trt_before_10():
        network_flags |= 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)

    parser = trt.OnnxParser(network, logger)
    parsed_ok = False
    if hasattr(parser, "parse_from_file"):
        parsed_ok = bool(parser.parse_from_file(str(args.onnx)))
    if not parsed_ok:
        # fallback: parse from bytes (may not work with external_data onnx)
        parsed_ok = bool(parser.parse(args.onnx.read_bytes()))

    if not parsed_ok:
        for i in range(parser.num_errors):
            print(str(parser.get_error(i)))
        raise RuntimeError(f"ONNX parse failed: {args.onnx}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(args.workspace_bytes))
    config.builder_optimization_level = int(args.opt_level)

    if args.precision == "fp16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            print("[WARN] platform_has_fast_fp16 is False; build as FP32.")

    if args.dynamic_batch:
        profile = builder.create_optimization_profile()
        min_b, opt_b, max_b = int(args.min_batch), int(args.opt_batch), int(args.max_batch)
        profile.set_shape("obs_state_norm", (min_b, state_dim), (opt_b, state_dim), (max_b, state_dim))
        profile.set_shape("img0_norm", (min_b, 3, h, w), (opt_b, 3, h, w), (max_b, 3, h, w))
        profile.set_shape("img1_norm", (min_b, 3, h, w), (opt_b, 3, h, w), (max_b, 3, h, w))
        config.add_optimization_profile(profile)

    if not args.disable_timing_cache:
        cache_bytes = timing_cache_path.read_bytes() if timing_cache_path.exists() else b""
        timing_cache = config.create_timing_cache(cache_bytes)
        config.set_timing_cache(timing_cache, ignore_mismatch=True)

    serialized_plan = builder.build_serialized_network(network, config)
    if serialized_plan is None:
        raise RuntimeError("build_serialized_network returned None")
    plan_bytes = _to_plan_bytes(serialized_plan)

    engine_path.write_bytes(plan_bytes)
    if not args.disable_timing_cache:
        _save_timing_cache(config, timing_cache_path)

    # Validate deserialization.
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan_bytes)
    if engine is None:
        raise RuntimeError("Engine deserialization failed after build")

    summary = {
        "onnx": str(args.onnx),
        "export_metadata": str(export_metadata),
        "engine": str(engine_path),
        "precision": args.precision,
        "workspace_bytes": int(args.workspace_bytes),
        "opt_level": int(args.opt_level),
        "dynamic_batch": bool(args.dynamic_batch),
        "profile": None
        if not args.dynamic_batch
        else {
            "obs_state_norm": {
                "min": [int(args.min_batch), state_dim],
                "opt": [int(args.opt_batch), state_dim],
                "max": [int(args.max_batch), state_dim],
            },
            "img0_norm": {
                "min": [int(args.min_batch), 3, h, w],
                "opt": [int(args.opt_batch), 3, h, w],
                "max": [int(args.max_batch), 3, h, w],
            },
            "img1_norm": {
                "min": [int(args.min_batch), 3, h, w],
                "opt": [int(args.opt_batch), 3, h, w],
                "max": [int(args.max_batch), 3, h, w],
            },
        },
        "timing_cache": None if args.disable_timing_cache else str(timing_cache_path),
        "engine_size_bytes": int(len(plan_bytes)),
        "versions": {"tensorrt": trt.__version__, "python": platform.python_version()},
    }
    (engine_path.parent / "trt_build_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

你可以先在目标机上把这个脚本保存成 `my_devs/train/act/so101/act_trt_build_engine.py`，然后这样调用：

```bash
CKPT="outputs/train/20260305_190147_act_grasp_block_in_bin1_e15/checkpoints/last/pretrained_model"

conda run -n lerobot_flex python my_devs/train/act/so101/act_trt_build_engine.py \
  --onnx "${CKPT}/act_single.onnx" \
  --export-metadata "${CKPT}/export_metadata.json" \
  --engine "${CKPT}/act_single_fp16.plan" \
  --precision fp16 \
  --workspace-bytes $((4<<30)) \
  --opt-level 3
```

---

## 5. 步骤 3：TensorRT 推理 runtime（Engine → 输出 actions_norm）

这一步的目标是把 `.plan` 跑起来，并且能够输出 `actions_norm`，然后我们才能接入 ACT 的队列语义。

你们参考资料里已经给出了一套“标准 TRT runtime 骨架”：

- `my_devs/docs/act_trt/reference_docs/trt_reference/trt_infer.py`

它的优点是：

- 全程使用 TRT 10 推荐的 **name-based API**（不依赖 binding index）
- 采用 pinned host + device + stream 的标准写法
- 支持动态 shape：每次 `set_input_shape` 后 `infer_shapes()`，按真实 shape 扩容 buffer

接入 ACT 时你需要做的核心改造：

1. 输入不再是 1 个，而是 3 个（state + 2 images）
2. 不再需要 ResNet 的 preprocess/postprocess（这部分由 LeRobot pre/post processor 承担）
3. 输出是 `actions_norm`，shape `[1, chunk_size, action_dim]`

建议你抽象一个通用 runner：

```python
class TrtRunner:
    def __init__(self, engine_path: Path, profile_index: int = 0): ...
    def run(self, feed_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]: ...
```

并确保：

- `feed_dict` 的 key 必须精确匹配 engine input tensor name
- 输入必须是 contiguous 的 numpy 数组
- dtype 必须匹配 engine input dtype（一般是 FP16/FP32）

### 5.1 TRT Runner 参考实现骨架（多输入 / name-based API）

下面这段代码“写法风格”与 `my_devs/docs/act_trt/reference_docs/trt_reference/trt_infer.py` 一致，你可以把它作为 ACT 版本 runtime 的核心骨架（省略了大量错误处理/扩容细节，完整细节请直接参考 `trt_infer.py` 的实现并迁移）。

**关键差异：**这里的 `run(feed_dict)` 支持多输入，且每轮会对所有输入先 `set_input_shape` 再 `infer_shapes()`。

```python
from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorrt as trt
from cuda import cudart  # type: ignore


def _check_cuda_available() -> None:
    err, count = cudart.cudaGetDeviceCount()
    if err != cudart.cudaError_t.cudaSuccess or count is None or int(count) <= 0:
        raise RuntimeError(f"No CUDA device available (cudaGetDeviceCount -> {err}, {count}).")


def cuda_call(call):
    err = call[0]
    rest = call[1:]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {err}")
    return rest[0] if len(rest) == 1 else rest


def trt_dtype_to_numpy_dtype(trt_dtype: trt.DataType) -> np.dtype:
    # tensorrt.nptype covers common dtypes; if遇到异常，再参考 trt_infer.py 的 byte-fallback
    return np.dtype(trt.nptype(trt_dtype))


@dataclass
class HostDeviceMem:
    nbytes: int
    host: np.ndarray
    device: int

    @classmethod
    def allocate(cls, nbytes: int) -> "HostDeviceMem":
        host_ptr = cuda_call(cudart.cudaMallocHost(int(nbytes)))
        host = np.ctypeslib.as_array(ctypes.cast(host_ptr, ctypes.POINTER(ctypes.c_uint8)), (int(nbytes),))
        device = int(cuda_call(cudart.cudaMalloc(int(nbytes))))
        return cls(nbytes=int(nbytes), host=host, device=device)

    def free(self) -> None:
        if self.device:
            cuda_call(cudart.cudaFree(self.device))
            self.device = 0
        if self.host is not None:
            cuda_call(cudart.cudaFreeHost(int(self.host.ctypes.data)))
            self.host = None  # type: ignore[assignment]


class TrtRunner:
    def __init__(self, engine_path: Path, profile_index: int = 0) -> None:
        _check_cuda_available()

        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, "")
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {engine_path}")
        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError("Failed to create execution context")

        stream = int(cuda_call(cudart.cudaStreamCreate()))

        self.engine = engine
        self.context = context
        self.stream = stream

        self.names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        self.input_names = [n for n in self.names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.output_names = [n for n in self.names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]

        if engine.num_optimization_profiles > 1 or profile_index != 0:
            ok = context.set_optimization_profile_async(int(profile_index), stream)
            if not ok:
                raise RuntimeError(f"Failed to set profile {profile_index}")
            cuda_call(cudart.cudaStreamSynchronize(stream))

        # 简化：先按 engine 静态 shape 分配 buffer；动态 shape 时需要按每次 infer_shapes() 结果扩容
        self.buffers: dict[str, HostDeviceMem] = {}
        for name in self.names:
            shape = tuple(engine.get_tensor_shape(name))
            # 如果有 -1，建议按 profile max shape 分配；详见 trt_infer.py:_shape_for_allocation
            numel = int(trt.volume(shape if all(d > 0 for d in shape) else (1,)))
            dtype = trt_dtype_to_numpy_dtype(engine.get_tensor_dtype(name))
            nbytes = int(numel * dtype.itemsize)
            self.buffers[name] = HostDeviceMem.allocate(max(nbytes, 1))

    def run(self, feed_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        # 1) set shapes for each input
        for name in self.input_names:
            arr = feed_dict[name]
            if not self.context.set_input_shape(name, tuple(arr.shape)):
                raise RuntimeError(f"set_input_shape failed: {name} {arr.shape}")

        unresolved = self.context.infer_shapes()
        if unresolved:
            raise RuntimeError(f"infer_shapes unresolved: {[str(x) for x in unresolved]}")

        # 2) copy inputs to pinned host, then H2D
        for name in self.input_names:
            dtype = trt_dtype_to_numpy_dtype(self.engine.get_tensor_dtype(name))
            arr = np.ascontiguousarray(feed_dict[name], dtype=dtype)
            mem = self.buffers[name]
            if arr.nbytes > mem.nbytes:
                raise RuntimeError(f"Buffer too small for {name}: need {arr.nbytes}, have {mem.nbytes}")
            mem.host[: arr.nbytes] = np.frombuffer(arr.tobytes(), dtype=np.uint8)
            cuda_call(
                cudart.cudaMemcpyAsync(
                    mem.device,
                    mem.host,
                    int(arr.nbytes),
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    self.stream,
                )
            )

        # 3) set tensor addresses
        for name in self.names:
            mem = self.buffers[name]
            if not self.context.set_tensor_address(name, int(mem.device)):
                raise RuntimeError(f"set_tensor_address failed: {name}")

        # 4) execute
        if not self.context.execute_async_v3(stream_handle=self.stream):
            raise RuntimeError("execute_async_v3 failed")

        # 5) D2H outputs (这里简化为同步复制；更完整写法见 trt_infer.py)
        outputs: dict[str, np.ndarray] = {}
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = trt_dtype_to_numpy_dtype(self.engine.get_tensor_dtype(name))
            nbytes = int(trt.volume(shape) * dtype.itemsize)
            mem = self.buffers[name]
            if nbytes > mem.nbytes:
                raise RuntimeError(f"Output buffer too small for {name}")
            cuda_call(
                cudart.cudaMemcpyAsync(
                    mem.host,
                    mem.device,
                    nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    self.stream,
                )
            )
            cuda_call(cudart.cudaStreamSynchronize(self.stream))
            data = np.frombuffer(mem.host[:nbytes].tobytes(), dtype=dtype).reshape(shape)
            outputs[name] = np.asarray(data, dtype=np.float32)  # 统一 float32 交给后处理
        return outputs

    def close(self) -> None:
        for mem in self.buffers.values():
            mem.free()
        if self.stream:
            cuda_call(cudart.cudaStreamDestroy(self.stream))
            self.stream = 0
```

这份骨架的定位是“帮助你把 ACT 跑通”，不是“终极性能版本”。如果你追求性能与稳定性，请直接把 `trt_reference/trt_infer.py` 的：

- buffer 扩容策略（按真实 output shape）
- dtype/byte-fallback
- stream 同步位置
- engine tensor location（HOST/DEVICE）判断

整套搬过来，然后把 ResNet preprocess/postprocess 换成 ACT 的 feed_dict 和输出张量名即可。

> 这部分如果你希望我后续直接落地成脚本/模块，也可以在你确认方案后再开一个任务，我再按仓库风格把代码写进去。

---

## 6. 步骤 4：复刻 ACT 语义（把 TRT 输出接回 `select_action`）

### 6.1 你需要实现的“最小 ACT TRT Policy”

建议实现一个轻量 wrapper（不用继承太深，目标是能接 `predict_action`）：

```python
class ActTrtPolicy:
    def __init__(self, *, engine_path, image_feature_keys, chunk_size, n_action_steps, temporal_ensemble_coeff):
        ...

    def reset(self): ...

    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        # 1) gather normalized inputs from batch
        # 2) if queue empty (or temporal-ensemble): run TRT forward -> actions_norm chunk
        # 3) queue / temporal ensemble -> next action_norm (single step)
        # 4) return torch.Tensor with shape [1, action_dim] (normalized)
```

关键点：

- 输入 `batch` 是 `policy_preprocessor` 之后的结果，已经是 **归一化** 的
- `batch` 里图像仍然是按 key 存的（例如 `observation.images.top`），所以你必须按 `image_feature_keys` 顺序 gather
- 输出必须是 torch.Tensor，因为后面还要走 `policy_postprocessor`

### 6.2 相机顺序（必须用 export_metadata 固化）

不要在代码里写死 top/wrist，也不要按字母排序。

正确做法：

1. ONNX 导出时记录了 `export_metadata.json`：
   - `camera_order_visual_keys = ["observation.images.top", "observation.images.wrist"]`
2. TRT Policy 初始化时读取并使用这个顺序

这样才能保证：

- **导出/engine/推理三者一致**
- 换 checkpoint 时不会 silently 行为跑偏

### 6.3 queue 与 temporal ensemble 的兼容

建议照抄 `src/lerobot/policies/act/modeling_act.py` 的行为：

- `temporal_ensemble_coeff is None`：走 deque queue
- `temporal_ensemble_coeff is not None`：走 `ACTTemporalEnsembler`（可以直接 import 复用）

### 6.4 `ActTrtPolicy.select_action()` 参考实现（接近可用代码）

下面的实现目标是：

- 输入：`policy_preprocessor` 输出的 batch（torch.Tensor，已经 normalized）
- 输出：**normalized 的单步 action**（torch.Tensor），交给 `policy_postprocessor` 做反归一化
- 复刻 ACT 语义：
  - queue 模式：仅在队列为空时跑一次 TRT 前向
  - temporal ensemble 模式：每步都跑 TRT 前向（然后 ensemble）

你可以把它放在一个文件里，例如：

- `my_devs/train/act/so101/act_trt_policy.py`

```python
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
from lerobot.utils.constants import OBS_STATE


@dataclass
class ActTrtPolicyConfig:
    # 只保留 run_act_infer.py / predict_action 需要的字段
    device: str = "cpu"  # 建议先固定 cpu：pre/post 在 CPU，TRT 自己用 GPU
    use_amp: bool = False

    chunk_size: int = 100
    n_action_steps: int = 100
    temporal_ensemble_coeff: float | None = None


class ActTrtPolicy:
    def __init__(
        self,
        *,
        engine_path: Path,
        image_feature_keys: list[str],
        cfg: ActTrtPolicyConfig,
        profile_index: int = 0,
    ) -> None:
        # TrtRunner 建议用第 5 节的 runner（完整版本可直接迁移 trt_reference/trt_infer.py）
        from .act_trt_runner import TrtRunner  # 你需要自己提供（或直接内联）

        self.config = cfg
        self.chunk_size = int(cfg.chunk_size)
        self.n_action_steps = int(cfg.n_action_steps)
        self.temporal_ensemble_coeff = cfg.temporal_ensemble_coeff

        self.image_feature_keys = list(image_feature_keys)
        if len(self.image_feature_keys) != 2:
            raise ValueError(f"Expected 2 camera keys, got {self.image_feature_keys}")

        self.runner = TrtRunner(engine_path, profile_index=profile_index)

        if self.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(
                temporal_ensemble_coeff=float(self.temporal_ensemble_coeff),
                chunk_size=int(self.chunk_size),
            )
        else:
            self._action_queue: deque[torch.Tensor] = deque([], maxlen=self.n_action_steps)

    def reset(self) -> None:
        if self.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue.clear()

    def _run_engine(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        # batch 已经 normalized
        obs_state = batch[OBS_STATE]  # shape: [B, state_dim]
        img0 = batch[self.image_feature_keys[0]]  # shape: [B,3,H,W]
        img1 = batch[self.image_feature_keys[1]]

        # 方案 2 最小闭环：先全部放 CPU numpy，然后喂给 TRT（TRT runtime 内部做 H2D）
        obs_state_np = np.ascontiguousarray(obs_state.detach().to("cpu").numpy())
        img0_np = np.ascontiguousarray(img0.detach().to("cpu").numpy())
        img1_np = np.ascontiguousarray(img1.detach().to("cpu").numpy())

        outputs = self.runner.run(
            {
                "obs_state_norm": obs_state_np,
                "img0_norm": img0_np,
                "img1_norm": img1_np,
            }
        )
        actions_norm = outputs["actions_norm"]  # shape: [B, chunk_size, action_dim], float32
        return torch.from_numpy(actions_norm)

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        # temporal ensemble：每步跑一次 chunk，然后 ensembler pop 出一个 action
        if self.temporal_ensemble_coeff is not None:
            actions = self._run_engine(batch)  # [B, chunk, action_dim]
            action = self.temporal_ensembler.update(actions)  # [B, action_dim]
            return action

        # queue：只有队列空时才跑一次 engine
        if len(self._action_queue) == 0:
            actions = self._run_engine(batch)  # [B, chunk, action_dim]
            actions = actions[:, : self.n_action_steps]  # [B, n_action_steps, action_dim]
            self._action_queue.extend(actions.transpose(0, 1))  # each: [B, action_dim]
        return self._action_queue.popleft()
```

实现建议（非常重要）：

- 第一次接入建议把 `cfg.device="cpu"` 固定：
  - 这样 `prepare_observation_for_inference` + `policy_preprocessor` + `policy_postprocessor` 都在 CPU
  - TRT runner 自己用 GPU（通过 cuda-python）
  - 避免 torch GPU tensor 与 TRT stream 同步的复杂度
- 如果你未来要追求极限性能，再做“高级版”：
  - preprocessor 在 GPU
  - runner 直接绑定 torch CUDA tensor 的 device ptr（零拷贝）

---

### 6.5 “normalized in/out” 与 LeRobot pre/post processor 的对接方式

导出的 ONNX/TRT engine 输入叫 `*_norm`，意味着它期望的数据已经过“checkpoint 同款归一化”。

在 `run_act_infer.py` 里，你已经在用 checkpoint 的 processor：

- `load_pre_post_processors(policy_path)`：
  - `policy_preprocessor.json`（里面引用 `policy_preprocessor_step_3_normalizer_processor.safetensors`）
  - `policy_postprocessor.json`（里面引用 `policy_postprocessor_step_0_unnormalizer_processor.safetensors`）

这两条 pipeline 正好把：

- 原始 observation → normalized batch（输入 engine）
- normalized action → unnormalized action（发给机器人）

因此 **最稳** 的集成方式是：

1. `predict_action(...)` 仍然保留
2. `preprocessor/postprocessor` 仍然从 checkpoint 加载
3. 只把 `policy` 从 PyTorch `ACTPolicy` 换成上面的 `ActTrtPolicy`

这能最大程度保证“行为对齐”，避免你手抄 mean/std 细节导致漂移。

---

## 7. 步骤 5：接入 `run_act_infer.py` 的两种落地方式

### 7.1 方式 1（更稳）：新建一个 `run_act_infer_trt.py`

优点：

- 不破坏现有稳定脚本
- TRT 分支可以独立演进

建议做法：

1. 复制 `my_devs/train/act/so101/run_act_infer.py` → `run_act_infer_trt.py`
2. 新增 CLI 参数：
   - `--backend {torch,trt}`（默认 torch）
   - `--trt-engine-path`（默认 `<policy-path>/act_single_fp16.plan`）
   - `--export-metadata`（默认 `<policy-path>/export_metadata.json`）
   - `--trt-profile-index`（默认 0）
3. 当 `backend=trt`：
   - 不再 `policy_class.from_pretrained(...)`
   - 读取 `PreTrainedConfig` 仅用于拿到 `chunk_size/n_action_steps/temporal_ensemble_coeff/image_features`
   - 初始化 `ActTrtPolicy`
4. 其余逻辑（robot/pre/post/processors）保持不变

### 7.2 方式 2（更集成）：在原 `run_act_infer.py` 里加 backend 分支

优点：

- 一个脚本就能跑 torch 与 trt

缺点：

- 脚本复杂度会上升

如果你们更偏向“一份脚本”，可以做这种方式；建议仍然保持默认 torch，TRT 仅在显式参数下启用。

### 7.3 `run_act_infer_trt.py` 关键代码片段（建议你照着拼）

下面给出“最小改动”的初始化逻辑片段，帮助你把 TRT policy 换进去。

核心思路：

- 仍然用 `PreTrainedConfig.from_pretrained(...)` 读取 checkpoint config
- 仍然复用 `apply_act_runtime_overrides(...)`（你在 `run_act_infer.py` 已经写好了）
- 但 **不再加载 PyTorch 权重**，改为：
  - 读 `export_metadata.json` 拿相机顺序
  - 加载 `.plan` engine
  - 初始化 `ActTrtPolicy`
- 并且建议把 `policy.device` 固定到 CPU（让 pre/post 在 CPU，先跑稳）

```python
import json
from pathlib import Path

from lerobot.configs.policies import PreTrainedConfig

from my_devs.train.act.so101.act_trt_policy import ActTrtPolicy, ActTrtPolicyConfig


policy_path = Path(args.policy_path).expanduser()
policy_cfg = PreTrainedConfig.from_pretrained(str(policy_path))
policy_cfg.pretrained_path = policy_path

# 复用你现有的 runtime override（n_action_steps / temporal_ensemble_coeff）
apply_act_runtime_overrides(
    policy_cfg=policy_cfg,
    policy_n_action_steps=args.policy_n_action_steps,
    policy_temporal_ensemble_coeff=args.policy_temporal_ensemble_coeff,
)

# 推荐先固定 CPU：避免 torch<->trt GPU stream 复杂度
policy_cfg.device = "cpu"

export_meta_path = Path(args.export_metadata or (policy_path / "export_metadata.json"))
engine_path = Path(args.trt_engine_path or (policy_path / "act_single_fp16.plan"))

meta = json.loads(export_meta_path.read_text())
image_feature_keys = list(meta["camera_order_visual_keys"])

trt_cfg = ActTrtPolicyConfig(
    device="cpu",
    use_amp=False,
    chunk_size=int(policy_cfg.chunk_size),
    n_action_steps=int(policy_cfg.n_action_steps),
    temporal_ensemble_coeff=policy_cfg.temporal_ensemble_coeff,
)

policy = ActTrtPolicy(
    engine_path=engine_path,
    image_feature_keys=image_feature_keys,
    cfg=trt_cfg,
    profile_index=int(args.trt_profile_index),
)
```

跑实机时的命令示例（假设你已经导出 ONNX 并 build engine）：

```bash
CKPT="outputs/train/20260305_190147_act_grasp_block_in_bin1_e15/checkpoints/last/pretrained_model"

conda run -n lerobot_flex python my_devs/train/act/so101/run_act_infer_trt.py \
  --policy-path "${CKPT}" \
  --trt-engine-path "${CKPT}/act_single_fp16.plan" \
  --export-metadata "${CKPT}/export_metadata.json" \
  --policy-n-action-steps 16 \
  --robot-port /dev/ttyACM0 \
  --top-cam-index 4 \
  --wrist-cam-index 6 \
  --fps 30 \
  --task "Put the block in the bin" \
  --run-time-s 120
```

说明：

- `--policy-n-action-steps 16` 是为了让你能真实观察到“前向频率变高后的加速收益”
  - 如果你仍然用 `n_action_steps=100`，ACT 本来就 100 步才跑一次前向，TRT 的收益会不明显（甚至被其它开销淹没）

---

## 8. 验收与排障（不做这些，你不知道自己是否真的“加速且正确”）

### 8.1 正确性验收（强制项）

建议你做一个离线对齐脚本（不连机械臂）：

1. 从一帧观测（state + top/wrist image）构造 `observation_frame`
2. 用原 PyTorch policy 走一遍 `predict_action_chunk` 或 `select_action`
3. 用 TRT policy 走一遍
4. 对比：
   - `max_abs_diff`
   - `mean_abs_diff`
   - `cosine_similarity`
5. 在允许误差范围内（FP16 通常会比 FP32 误差大一些）

如果对齐不过：

- 先查相机顺序
- 再查归一化 stats 是否使用一致
- 再查 TRT 是否 FP16 造成误差（可切 FP32 engine 验证）

#### 8.1.1 先做一次“engine IO 自检”（避免低级错误）

很多“跑不起来/输出不对”的问题，本质是：

- engine 的 tensor name 你填错了（例如你用的是 `actions` 但实际叫 `actions_norm`）
- engine dtype 是 FP16，但你喂了 FP32（有的 runtime 会报错，有的会隐式 cast，性能也会变）
- 你以为 shape 是固定的，但 engine 其实是 dynamic（需要 `set_input_shape` + `infer_shapes`）

建议在构建完 `.plan` 后，第一时间跑这个 IO dump（注意：必须在装了 TRT 的环境里跑）：

```bash
ENGINE="outputs/train/20260305_190147_act_grasp_block_in_bin1_e15/checkpoints/last/pretrained_model/act_single_fp16.plan"

conda run -n lerobot_flex python - <<'PY'
from pathlib import Path
import tensorrt as trt

engine_path = Path("outputs/train/20260305_190147_act_grasp_block_in_bin1_e15/checkpoints/last/pretrained_model/act_single_fp16.plan")

logger = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(logger, "")
runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
assert engine is not None, "deserialize engine failed"

print("num_io_tensors:", engine.num_io_tensors)
print("num_optimization_profiles:", engine.num_optimization_profiles)
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    mode = engine.get_tensor_mode(name)
    dtype = engine.get_tensor_dtype(name)
    shape = engine.get_tensor_shape(name)
    print(f"[{i}] {mode} {name} dtype={dtype} shape={shape}")
PY
```

你应该看到类似：

- INPUT：`obs_state_norm/img0_norm/img1_norm`
- OUTPUT：`actions_norm`

#### 8.1.2 离线对齐脚本骨架（强烈建议你做成可重复的回归）

建议你写一个独立脚本（例如 `my_devs/train/act/so101/compare_torch_vs_trt.py`），只做一件事：

> 同一帧 observation，torch policy 与 trt policy 输出 action 是否对齐？

骨架如下（你需要自己提供一帧可复现的 observation，例如从 dataset 中取一帧、或把实机观测 dump 成 `.npz`）：

```python
import json
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class
from lerobot.utils.control_utils import prepare_observation_for_inference

from my_devs.train.act.so101.act_trt_policy import ActTrtPolicy, ActTrtPolicyConfig
from my_devs.train.act.so101.run_act_infer import load_pre_post_processors, apply_act_runtime_overrides


def load_observation_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {k: data[k] for k in data.files}


def main() -> None:
    ckpt = Path("outputs/train/20260305_190147_act_grasp_block_in_bin1_e15/checkpoints/last/pretrained_model")
    obs_npz = Path("your_single_observation_dump.npz")

    # 1) load preprocessors
    preprocessor, postprocessor = load_pre_post_processors(ckpt)

    # 2) prepare a single observation (raw numpy, HWC uint8)
    observation_np = load_observation_npz(obs_npz)

    # 3) run torch policy once
    policy_cfg = PreTrainedConfig.from_pretrained(str(ckpt))
    apply_act_runtime_overrides(policy_cfg, policy_n_action_steps=1, policy_temporal_ensemble_coeff=None)
    policy_cfg.device = "cpu"

    policy_class = get_policy_class(policy_cfg.type)
    torch_policy = policy_class.from_pretrained(str(ckpt), config=policy_cfg, strict=False)
    torch_policy.reset()

    obs_torch = prepare_observation_for_inference(observation_np, device=torch.device("cpu"))
    obs_torch = preprocessor(obs_torch)
    with torch.inference_mode():
        torch_actions = torch_policy.predict_action_chunk(obs_torch)  # [1, chunk, action_dim]

    # 4) run trt policy once
    meta = json.loads((ckpt / "export_metadata.json").read_text())
    trt_policy = ActTrtPolicy(
        engine_path=(ckpt / "act_single_fp16.plan"),
        image_feature_keys=list(meta["camera_order_visual_keys"]),
        cfg=ActTrtPolicyConfig(
            device="cpu",
            use_amp=False,
            chunk_size=int(policy_cfg.chunk_size),
            n_action_steps=int(policy_cfg.n_action_steps),
            temporal_ensemble_coeff=policy_cfg.temporal_ensemble_coeff,
        ),
        profile_index=0,
    )
    trt_policy.reset()
    with torch.inference_mode():
        trt_actions = trt_policy._run_engine(obs_torch)  # [1, chunk, action_dim]

    # 5) compare
    diff = (torch_actions - trt_actions).abs()
    print("max_abs_diff:", float(diff.max()))
    print("mean_abs_diff:", float(diff.mean()))


if __name__ == "__main__":
    main()
```

说明：

- 这里对齐的是 **normalized action chunk**（也就是 engine 输出），最干净、最容易定位问题
- 如果你想对齐“最终发给机器人的 action”，就再加一层 `postprocessor` 对比（但那样误差来源更多）

### 8.2 性能验收（建议项）

你需要区分两类 latency：

1. **模型前向 latency**（TRT 的收益主要在这里）
2. **整条控制循环 latency**（还包含相机采集、processor、串口通信、sleep）

建议在 `run_act_infer(_trt).py` 里增加两段统计：

- `t_model_ms`：只包住 policy 前向部分（含 pre/post 或不含都要说清楚）
- `t_loop_ms`：整次循环

并打印 p50/p95。

补充建议（让数据更可信）：

- TRT forward benchmark 时尽量 **只测 engine**：
  - `preprocessor/postprocessor/robot IO` 这些在不同机器差异很大，会淹没 TRT 的收益
- 需要 warmup：
  - 建议先 warmup 10~50 次（让 CUDA kernel、TRT tactic 进入稳定态）
- 对 ACT 来说要注意“前向频率”：
  - `n_action_steps=100`：100 步才 forward 一次，TRT 的收益对 end-to-end loop 影响可能很小
  - `n_action_steps=1`：每步 forward，一旦 TRT 做快了，你会明显看到 loop latency 降低

### 8.3 常见问题速查

#### 问题 A：TensorRT parse 失败（Unsupported op / parse error）

优先处理顺序：

1. 用 `--simplify` 简化 ONNX（导出脚本默认开）
2. 换 opset（14/16/17/18）
3. 尝试 torch 的 dynamo exporter（导出脚本支持 `--dynamo`）
4. 如果 ONNX 图巨大（ACT list 展开导致），考虑改 ACT 模型导出路径（需要修改模型实现，属于“方案 2.1”增强点）

#### 问题 A.1：ONNX 使用了 `.onnx.data`，但你的 builder 用的是 `parser.parse(bytes)`

现象：

- parse 报错（找不到外部权重）
- 或者 parse 成功但输出异常（权重没加载对）

处理：

- builder 优先使用 `parser.parse_from_file(str(onnx_path))`
- 或者把导出改成 `external_data=False`（前提是权重文件 <2GB）

#### 问题 B：engine 构建很慢

- 开 timing cache（`.tcache`），并在后续构建复用
- 固定输入 shape（静态 batch=1），别一上来就动态 profile

#### 问题 B.1：engine build 成功，但 runtime 推理时报 shape unresolved

常见原因：

- 你只对一个输入 `set_input_shape`，其它输入没设
- 你忘了在每轮推理前调用 `context.infer_shapes()`

处理：

- 对所有 INPUT tensor 都调用 `context.set_input_shape(name, arr.shape)`
- 然后 `unresolved = context.infer_shapes()`，不为空就直接 fail 并打印名字

#### 问题 C：动作明显跑偏

99% 是三件事之一：

- 相机顺序不一致（img0/img1 反了）
- 归一化/反归一化 stats 不一致（eps、mean/std 文件路径错）
- 图像预处理不一致（是否 /255、是否 CHW、是否 contiguous）

#### 问题 D：跑得起来，但比 PyTorch 还慢

常见原因：

- 你把 `n_action_steps` 设得很大（forward 本来就很少跑，差异不明显）
- 你每步都在做大量 CPU↔GPU 来回拷贝（例如先把 tensor 放 GPU 又拷回 CPU 再喂 TRT）
- engine 实际没用到 FP16（builder 没开 FP16 flag 或平台不支持）

处理建议：

- 先把对比口径固定为“只测 engine forward”（第 8.2 节）
- dump engine IO dtype（第 8.1.1 节），确认输入输出是不是 half
- 先走最小闭环：pre/post 全 CPU，TRT 只负责 forward（第 6.4 节）

---

## 9. 建议的落地里程碑（按“最短闭环”排序）

1. **导出 ONNX + verify 通过**  
   产物：`act_single.onnx` + `export_metadata.json`
2. **用 `trtexec` build 成功**（只验证可编译）  
   产物：`act_single_fp16.plan`
3. **Python TRT runtime 能跑通一次 forward**（随机输入即可）  
   产物：能输出 `actions_norm`，shape 对
4. **ActTrtPolicy 复刻 queue 语义**（不连机器人，离线对齐 torch 输出）  
   产物：对齐报告（diff 指标）
5. **接入 `run_act_infer.py` 实机验证**  
   产物：稳定运行日志 + 性能数据

---

## 10. 后续可选增强（不影响“方案 2 最小可用”）

1. **把 pre/post 融合进 ONNX/TRT**（减少 Python/torch 依赖）  
   代价：要精确复刻 processor pipeline，且一旦换 checkpoint 就可能不兼容
2. **零拷贝输入**（torch GPU tensor 指针直接喂 TRT）  
   收益：减少 H2D；代价：实现复杂，需要严格的 stream 同步与生命周期管理
3. **INT8**（需要 calibration）  
   收益：更低延迟；代价：校准与精度风险更高

---

## 附录 A：参考资料清单（对应你给的目录）

- ONNX 导出参考：
  - `my_devs/docs/act_trt/reference_docs/onnx_export_reference/infer_docs/模型导出方案.md`
  - `my_devs/docs/act_trt/reference_docs/onnx_export_reference/export/export_single.py`
  - `my_devs/docs/act_trt/reference_docs/onnx_export_reference/infer_docs/导出推理报告.md`
  - `my_devs/docs/act_trt/reference_docs/onnx_export_reference/infer_docs/onnxruntime推理计划.md`
- TRT build/infer 参考：
  - `my_devs/docs/act_trt/reference_docs/trt_reference/TENSORRT_ENGINE_BUILD_INFER_PLAYBOOK.md`
  - `my_devs/docs/act_trt/reference_docs/trt_reference/trt_build.py`
  - `my_devs/docs/act_trt/reference_docs/trt_reference/trt_infer.py`
- ACT 推理脚本参考：
  - `my_devs/train/act/so101/run_act_infer.py`
  - `my_devs/train/act/so101/ACT推理验证脚本_审核分析.md`
