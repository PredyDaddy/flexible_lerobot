# VLA 模型 TensorRT 部署 Playbook（基于本仓库脚本，先 FP16 再量化）

本文是一份“从零到可上线”的部署手册，目标读者是**没有模型部署经验**的人：你照着做就能把一个 VLA（Vision-Language-Action）模型跑成可服务化的推理系统，并且能把 PyTorch 推理替换为 TensorRT engine 加速。

本文以本仓库 `deployment_scripts/` 与推理服务脚本为唯一参考，覆盖：

- 如何先把 **FP16** 全链路跑通（最稳妥）
- 如何导出 ONNX、构建 TensorRT engine、做一致性验证
- 如何把 engine 组织成“可部署产物”（目录结构与命名）
- 如何启用 ZMQ 推理服务（不改机器人端协议）
- 如何做量化（ViT/DiT FP8，LLM NVFP4/FP8），以及需要的依赖
- 如何写/调用“API”（ZMQ/HTTP），包含可复制的最小客户端代码

> 强约束：开发/部署调试统一使用 `gr00t` conda 环境（仓库规范）。Python 推理、ONNX 导出、服务启动都用 `conda run -n gr00t ...`。

---

## 0. 你将得到什么（产物定义）

一次“可上线”的部署，最终你应该得到一个**产物目录**（建议固定命名），例如：

```
outputs/trt/<ARTIFACT_NAME>/
  gr00t_engine/               # 推理服务运行时真正需要的东西（.engine）
    vit_fp16.engine
    llm_fp16.engine
    DiT_fp16.engine
    vlln_vl_self_attention.engine
    state_encoder.engine
    action_encoder.engine
    action_decoder.engine
  gr00t_onnx/                  # 审计/复现需要（运行时可不带）
    eagle2/...
    action_head/...
  export_onnx.log
  build_engine.log
  compare_pytorch_vs_trt.log
  server.log
  healthcheck.log
  run_meta.txt                 # 关键版本与参数留痕（强烈建议）
```

并且你能做到：

1. `deployment_scripts/gr00t_inference.py --inference-mode compare` 通过（PyTorch vs TRT）
2. `scripts/inference_service.py --server --use-tensorrt ...` 启动成功
3. `docs/infer_service/server_healthcheck.sh ...` 返回 `status=ok` 且 `get_action` 成功

---

## 1. 概念速成（给完全新手）

### 1.1 ONNX 是什么？

ONNX 是一个“跨框架的计算图格式”。你把 PyTorch 模型导出成 ONNX，相当于把模型结构固化成一个标准图，方便后续给 TensorRT 这类推理引擎做编译优化。

在本仓库里，导出由：

- `deployment_scripts/export_onnx.py`

负责。

### 1.2 TensorRT engine 是什么？

TensorRT engine（本仓库后缀是 `.engine`）是一个**已编译好的推理计划（plan）**，通常与以下强绑定：

- GPU 架构（例如 A100/H100/4090 等）
- TensorRT 版本（例如 10.13）
- CUDA/driver 组合
- 你导出 ONNX 的 dtype、输入 shape profile 等

所以：**engine 必须在最终要跑推理服务的那台机器上构建**，不要在别的机器 build 了拷过去（除非硬件与软件栈完全一致）。

在本仓库里，engine 构建由：

- `deployment_scripts/build_engine.sh`（内部调用 `trtexec`）

负责。

### 1.3 “先 FP16 再量化”为什么是正确路径？

部署最常见的坑是“栈不一致”或“profile 不匹配”。FP16 是最稳的基线：

- 不依赖量化校准（`modelopt` 等）
- 数值最接近 PyTorch
- 只要 TRT/驱动版本对齐，最容易跑通

跑通 FP16 后，你再逐步尝试：

- `vit=fp8`
- `dit=fp8`
- `llm=nvfp4`（或 `fp8`）

每前进一步都用 compare + healthcheck 验证，风险可控。

---

## 2. 环境与版本：先把地基打牢（最重要）

### 2.1 必须满足的三件事

1. 有 GPU，且驱动可用（能 `nvidia-smi`）
2. 系统有 `trtexec`，并且路径满足本仓库脚本要求：
   - `deployment_scripts/build_engine.sh` 会检查 `/usr/src/tensorrt/bin/trtexec`
3. Python 能 `import tensorrt`（推理服务加载 `.engine` 时需要）

### 2.2 强制要求：构建侧与加载侧 TensorRT 版本一致

你需要同时对齐：

- **构建侧**：`/usr/src/tensorrt/bin/trtexec --version`
- **加载侧**：`conda run -n gr00t python -c "import tensorrt as trt; print(trt.__version__)"`

如果它们不一致，常见后果是：engine 反序列化失败（例如 “Version tag does not match”）。

### 2.3 本仓库推荐的 Python TRT 版本（deploy 口径）

`pyproject.toml` 提供了：

- `pip install -e ".[deploy]"`
- 其中固定 `tensorrt-cu12==10.13.0.35`

建议你把系统 `trtexec` 也固定到同一主版本（10.13）。

### 2.4 最小自检命令（照抄即可）

在 repo 根目录执行：

```bash
cd /app/my_gr00tN1.5

# 1) Python TRT 版本
conda run -n gr00t python -c "import tensorrt as trt; print('tensorrt', trt.__version__)"

# 2) trtexec 版本（构建侧）
/usr/src/tensorrt/bin/trtexec --version | head -n 3

# 3) GPU 是否可用（可选，但强烈建议）
conda run -n gr00t python -c "import torch; print('cuda', torch.cuda.is_available()); print('torch', torch.__version__, 'cuda', torch.version.cuda)"
```

验收标准：

- `tensorrt` 与 `trtexec` 均为 10.13（例如 `10.13.0.35` / `v101300`）
- `torch.cuda.is_available()` 为 `True`（在 GPU 机器/容器内）

### 2.5 从零安装（新机器/新容器常用）

如果你是在一台全新的机器或新容器里从头开始，建议按下面顺序做（尽量避免“系统 python / base 环境”污染）：

1. 创建 `gr00t` conda 环境（Python >= 3.10）
2. 在 `gr00t` 环境里安装仓库依赖
3. 安装 Python TensorRT（运行时）
4. 另外确保系统 `trtexec` 可用（构建时）

最小命令（示例）：

```bash
cd /app/my_gr00tN1.5

# 1) 创建环境
conda create -y -n gr00t python=3.10

# 2) 安装 repo 依赖（包含 torch/torchvision/torchcodec 等）
conda run --no-capture-output -n gr00t pip install -e ".[base]"

# 3) 安装 Python TRT（运行时加载 engine 需要）
conda run --no-capture-output -n gr00t pip install -e ".[deploy]"

# 4) 自检
conda run -n gr00t python -c "import tensorrt as trt; print(trt.__version__)"
```

说明：

- `pip install -e ".[deploy]"` 只解决 “Python 能 import tensorrt”；不保证系统有 `trtexec`
- `deployment_scripts/build_engine.sh` 构建 engine 依赖 `/usr/src/tensorrt/bin/trtexec`
- 最可控的方式是使用“固定版本的镜像/容器”来避免系统 TRT 漂移

---

## 3. 选定“部署对象”（checkpoint + data config + 参数）

一次部署必须“钉死”的参数（否则不可审计、不可复现）：

1. `CKPT_PATH`：你要上线的 checkpoint 目录（建议固定到 `checkpoint-XXXXX`）
2. `DATASET_PATH`：用于导出/校准/验证的数据集（LeRobot 格式目录）
3. `DATA_CONFIG`：决定输入输出 modality keys 与 transforms（例如 `so100_dualcam`）
4. `EMBODIMENT_TAG`：例如 `new_embodiment`
5. `DENOISING_STEPS`：例如 `4`
6. `VIDEO_BACKEND`：`decord` 或 `torchcodec`

推荐你用环境变量固定它们，后续所有命令都复用，避免输错：

```bash
export REPO_ROOT="$(pwd)"

export CKPT_PATH="outputs/gr00t_agilex_right_box_h264_train_4gpu_7654_scratch_decord_w1/checkpoint-40000"
export DATASET_PATH="my_data/agilex_right_box_h264"

export DATA_CONFIG="so100_dualcam"
export EMBODIMENT_TAG="new_embodiment"
export DENOISING_STEPS="4"
export VIDEO_BACKEND="torchcodec"
```

---

## 4. FP16 全链路（推荐你第一次照这个做）

### 4.1 规划产物目录（不要覆盖）

建议每次部署创建唯一目录（包含时间戳/版本信息），例如：

```bash
export TRT_VERSION_TAG="trt1013"
export PRECISION_TAG="fp16"
export CKPT_TAG="ckpt40000"
export TS="$(date +%Y%m%d_%H%M%S)"

export TRT_ROOT="outputs/trt/agilex_right_box_${CKPT_TAG}_${PRECISION_TAG}_${TRT_VERSION_TAG}_${TS}"
mkdir -p "${TRT_ROOT}"
```

强烈建议你立刻写一份 `run_meta.txt`（用于复现/审计/排查）：

```bash
cat > "${TRT_ROOT}/run_meta.txt" <<EOF
date: ${TS}
repo_root: ${REPO_ROOT}
ckpt_path: ${CKPT_PATH}
dataset_path: ${DATASET_PATH}
data_config: ${DATA_CONFIG}
embodiment_tag: ${EMBODIMENT_TAG}
denoising_steps: ${DENOISING_STEPS}
video_backend: ${VIDEO_BACKEND}
vit_dtype: fp16
llm_dtype: fp16
dit_dtype: fp16
python_tensorrt: $(conda run -n gr00t python -c "import tensorrt as trt; print(trt.__version__)")
trtexec: $(/usr/src/tensorrt/bin/trtexec --version 2>/dev/null | head -n 1 || true)
torch: $(conda run -n gr00t python -c "import torch; print(torch.__version__)" || true)
torch_cuda: $(conda run -n gr00t python -c "import torch; print(torch.version.cuda)" || true)
EOF
```

### 4.2 Step 1：导出 ONNX（FP16）

本仓库脚本入口：

- `deployment_scripts/export_onnx.py`

它会输出（FP16 情况）：

- `gr00t_onnx/eagle2/vit_fp16.onnx`
- `gr00t_onnx/eagle2/llm_fp16.onnx`
- `gr00t_onnx/action_head/DiT_fp16.onnx`
- `gr00t_onnx/action_head/vlln_vl_self_attention.onnx`
- `gr00t_onnx/action_head/state_encoder.onnx`
- `gr00t_onnx/action_head/action_encoder.onnx`
- `gr00t_onnx/action_head/action_decoder.onnx`

执行命令（建议把日志落盘）：

```bash
set -o pipefail

conda run --no-capture-output -n gr00t python "${REPO_ROOT}/deployment_scripts/export_onnx.py" \
  --dataset-path "${DATASET_PATH}" \
  --model-path "${CKPT_PATH}" \
  --onnx-model-path "${TRT_ROOT}/gr00t_onnx" \
  --data-config "${DATA_CONFIG}" \
  --embodiment-tag "${EMBODIMENT_TAG}" \
  --denoising-steps "${DENOISING_STEPS}" \
  --vit-dtype fp16 \
  --llm-dtype fp16 \
  --dit-dtype fp16 \
  --calib-size 10 \
  --video-backend "${VIDEO_BACKEND}" \
  2>&1 | tee "${TRT_ROOT}/export_onnx.log"
```

导出后自检：

```bash
find "${TRT_ROOT}/gr00t_onnx" -type f -name "*.onnx" | sort
```

如果你看到的 `.onnx` 数量不对，先不要进入 build engine。

### 4.3 Step 2：构建 TensorRT engine（FP16）

本仓库构建脚本：

- `deployment_scripts/build_engine.sh`

它要求你在 **产物目录**里执行（因为它写死相对路径 `gr00t_onnx/...`，并输出到 `gr00t_engine/`）。

关键参数：

- `VIDEO_VIEWS`
  - 单路视频：`VIDEO_VIEWS=1`
  - 双路视频（例如 `so100_dualcam`）：**必须** `VIDEO_VIEWS=2`
- `MAX_BATCH`
  - 对 `so100_dualcam`，推荐 `MAX_BATCH=2`（匹配 2 views 的 batch 维）
- dtype 必须与 ONNX 文件名一致：`VIT_DTYPE/LLM_DTYPE/DIT_DTYPE`

执行命令：

```bash
set -o pipefail

cd "${TRT_ROOT}"

VIDEO_VIEWS=2 \
MAX_BATCH=2 \
VIT_DTYPE=fp16 \
LLM_DTYPE=fp16 \
DIT_DTYPE=fp16 \
bash "${REPO_ROOT}/deployment_scripts/build_engine.sh" \
  2>&1 | tee "${TRT_ROOT}/build_engine.log"
```

构建后自检（必须看到 `.engine`）：

```bash
ls -lh "${TRT_ROOT}/gr00t_engine"
```

你至少应看到：

- `vit_fp16.engine`
- `llm_fp16.engine`
- `DiT_fp16.engine`
- `vlln_vl_self_attention.engine`
- `state_encoder.engine`
- `action_encoder.engine`
- `action_decoder.engine`

> 注意：`deployment_scripts/build_engine.sh` 脚本本身没有 `set -e`，即使 `trtexec` 失败也可能打印 “Build Complete”。所以你必须靠 “是否生成 .engine + log 是否报错” 来判定成功。

### 4.4 Step 3：离线一致性验证（PyTorch vs TRT）

本仓库验证脚本：

- `deployment_scripts/gr00t_inference.py`

推荐用 `--inference-mode compare`，它会确保 PyTorch 与 TRT 在 action head 的初始噪声一致（减少随机性影响），并打印 cosine similarity 与 L1 距离。

执行命令：

```bash
set -o pipefail

conda run --no-capture-output -n gr00t python "${REPO_ROOT}/deployment_scripts/gr00t_inference.py" \
  --model-path "${CKPT_PATH}" \
  --dataset-path "${DATASET_PATH}" \
  --data-config "${DATA_CONFIG}" \
  --embodiment-tag "${EMBODIMENT_TAG}" \
  --denoising-steps "${DENOISING_STEPS}" \
  --video-backend "${VIDEO_BACKEND}" \
  --inference-mode compare \
  --trt-engine-path "${TRT_ROOT}/gr00t_engine" \
  --vit-dtype fp16 \
  --llm-dtype fp16 \
  --dit-dtype fp16 \
  2>&1 | tee "${TRT_ROOT}/compare_pytorch_vs_trt.log"
```

验收建议（经验值，不是硬标准）：

- cosine similarity 趋近 1（例如 `> 0.999`）
- L1 mean/max 足够小（FP16 通常非常接近）

如果 compare 失败，优先排查：

1. engine 与 runtime 版本是否一致（10.13 vs 10.13）
2. dtype 参数是否一致（fp16 engine 却用默认 fp8/nvfp4 去加载）
3. `VIDEO_VIEWS` 是否匹配（dualcam 没用 2）

---

## 5. 启动推理服务（ZMQ/HTTP）并做健康检查

### 5.1 ZMQ 服务（推荐，机器人端通常用这个）

服务入口：

- `scripts/inference_service.py`
- ZMQ server 实现：`gr00t/eval/service.py` + `gr00t/eval/robot.py`

ZMQ 端点（endpoint）定义：

- `ping`（无输入）
- `kill`（无输入）
- `get_modality_config`（无输入）
- `get_action`（需要 `observation` dict）

启动命令（FP16 TRT）：

```bash
set -o pipefail

conda run --no-capture-output -n gr00t python "${REPO_ROOT}/scripts/inference_service.py" \
  --server \
  --port 5555 \
  --model-path "${CKPT_PATH}" \
  --data-config "${DATA_CONFIG}" \
  --embodiment-tag "${EMBODIMENT_TAG}" \
  --denoising-steps "${DENOISING_STEPS}" \
  --api-token "REPLACE_ME" \
  --use-tensorrt \
  --trt-engine-path "${TRT_ROOT}/gr00t_engine" \
  --vit-dtype fp16 \
  --llm-dtype fp16 \
  --dit-dtype fp16 \
  2>&1 | tee "${TRT_ROOT}/server.log"
```

看到类似日志代表 engine 加载成功：

- `Setting up TensorRT engines from: ...`
- `TensorRT engines loaded successfully!`
- `Server is ready and listening on tcp://0.0.0.0:5555`

### 5.2 健康检查（推荐用 dataset replay）

本仓库脚本：

- `docs/infer_service/server_healthcheck.sh`

它会复用 `scripts/agilex_groot_client_mvp.py`，并支持从 LeRobot 数据集构造真实 observation（更接近线上）。

执行命令：

```bash
set -o pipefail

bash "${REPO_ROOT}/docs/infer_service/server_healthcheck.sh" \
  --host 127.0.0.1 \
  --port 5555 \
  --api-token "REPLACE_ME" \
  --dataset-path "${DATASET_PATH}" \
  --episode-index 7 \
  --frame-index 0 \
  --video-backend "${VIDEO_BACKEND}" \
  2>&1 | tee "${TRT_ROOT}/healthcheck.log"
```

验收标准：

- `ping` 返回 `status: ok`
- `get_modality_config` 成功
- `get_action` 返回 action（例如 `action.single_arm: (16, 6)` 等）

### 5.3 HTTP 服务（可选，适合 Web 集成/调试）

HTTP server 类在：

- `gr00t/eval/http_server.py`

启动方式（例）：

```bash
conda run --no-capture-output -n gr00t python "${REPO_ROOT}/scripts/inference_service.py" \
  --server \
  --http-server \
  --host 0.0.0.0 \
  --port 8000 \
  --model-path "${CKPT_PATH}" \
  --data-config "${DATA_CONFIG}" \
  --embodiment-tag "${EMBODIMENT_TAG}" \
  --denoising-steps "${DENOISING_STEPS}" \
  --use-tensorrt \
  --trt-engine-path "${TRT_ROOT}/gr00t_engine" \
  --vit-dtype fp16 \
  --llm-dtype fp16 \
  --dit-dtype fp16
```

注意：HTTP 里 numpy 的 JSON 编码依赖 `json-numpy`，更适合 Python 客户端（见附录）。

### 5.4 如何停止服务（避免手动 kill -9）

如果你只是想优雅停止服务，有两种方式：

1. 在 server 前台运行时直接 `Ctrl+C`
2. 调用 ZMQ 的 `kill` endpoint（协议里内置了）

Python 示例（复用仓库客户端）：

```python
from gr00t.eval.robot import RobotInferenceClient

client = RobotInferenceClient(host="127.0.0.1", port=5555, api_token="REPLACE_ME")
client.kill_server()
```

如果你用的是 `scripts/agilex_groot_client_mvp.py` 的轻量 client，可以按 `gr00t/eval/service.py` 协议发送 `{"endpoint": "kill"}`。

---

## 6. “API 怎么写”：客户端最小实现（可直接复制）

### 6.1 ZMQ 协议（强烈推荐用仓库实现）

你可以直接复用：

- `scripts/agilex_groot_client_mvp.py`

它实现了 msgpack + numpy 序列化协议，和服务端 `gr00t/eval/service.py` 完全一致。

#### 6.1.1 直接用脚本调用（最省事）

```bash
conda run --no-capture-output -n gr00t python scripts/agilex_groot_client_mvp.py \
  --host 127.0.0.1 --port 5555 --api-token "REPLACE_ME" \
  --dataset-path "${DATASET_PATH}" --episode-index 7 --frame-index 0 \
  --video-backend "${VIDEO_BACKEND}"
```

#### 6.1.2 自己写 Python 客户端（最小代码）

ZMQ 请求格式（概念）：

- request：`{"endpoint": "...", "data": {...}, "api_token": "..."}`（有些 endpoint 不需要 data）
- response：`{"status": "...", ...}` 或 action dict

最小客户端（建议直接复制 `scripts/agilex_groot_client_mvp.py` 里的 `ZmqInferenceClient`）：

```python
from scripts.agilex_groot_client_mvp import ZmqInferenceClient, build_random_obs

client = ZmqInferenceClient(host="127.0.0.1", port=5555, api_token="REPLACE_ME", timeout_ms=5000)

print(client.ping())
print(client.get_modality_config())

obs = build_random_obs()
action = client.get_action(obs)
print({k: v.shape for k, v in action.items()})
```

> observation 的 key/shape 必须匹配 data_config（见附录 “Observation schema”）。

### 6.2 HTTP API（Python requests 方式）

HTTP `/act` 的 payload 结构：

- `{"observation": <dict>}`

建议 Python 调用（依赖 `json_numpy`）：

```python
import json_numpy
import numpy as np
import requests

json_numpy.patch()

obs = {
    "video.front": np.random.randint(0, 256, (1, 480, 640, 3), dtype=np.uint8),
    "video.wrist": np.random.randint(0, 256, (1, 480, 640, 3), dtype=np.uint8),
    "state.single_arm": (np.random.rand(1, 6).astype(np.float32) - 0.5) * 2.0,
    "state.gripper": np.random.rand(1, 1).astype(np.float32) * 0.07,
    "annotation.human.task_description": ["do something"],
}

resp = requests.post("http://127.0.0.1:8000/act", json={"observation": obs}, timeout=30)
resp.raise_for_status()
action = resp.json()
print(action.keys())
```

---

## 7. 量化（进阶）：FP8 / NVFP4 的正确打开方式

本仓库的量化发生在 **ONNX 导出前**，由 `deployment_scripts/export_onnx.py` 内部调用 `modelopt` 完成校准与量化（没有 `modelopt` 时会报错）。

### 7.1 你需要知道的事实

1. FP16 导出不需要 `modelopt`
2. 量化（`vit=fp8` / `dit=fp8` / `llm=nvfp4|fp8`）需要 `modelopt`
3. 量化校准依赖真实数据分布，`--calib-size` 建议从 `100` 起步（甚至更大）
4. NVFP4（W4A4）通常要求更严格的引擎构建 shape（`build_engine.sh` 会把 nvfp4 的 LLM batch 固定为 1）

在开始量化前，先确认 `modelopt` 是否已安装（不安装一定会失败）：

```bash
conda run -n gr00t python -c "import modelopt.torch.quantization as mtq; print('modelopt OK')"
```

如果这里 import 失败：先不要继续 fp8/nvfp4，回到 FP16 路线把全链路跑通；量化依赖的安装请按 NVIDIA ModelOpt 的官方安装说明做（不同机器/镜像差异很大，不建议靠猜版本）。

### 7.2 量化路线建议（按风险从低到高）

1. 只量化 ViT：`vit=fp8, llm=fp16, dit=fp16`
2. 再量化 DiT：`vit=fp8, llm=fp16, dit=fp8`
3. 最后量化 LLM：`llm=nvfp4`（或 `fp8`）

每一步都重复“导出 ONNX -> build engine -> compare -> server -> healthcheck”。

### 7.3 量化导出命令示例

#### 7.3.1 ViT FP8（其他保持 FP16）

```bash
conda run --no-capture-output -n gr00t python "${REPO_ROOT}/deployment_scripts/export_onnx.py" \
  --dataset-path "${DATASET_PATH}" \
  --model-path "${CKPT_PATH}" \
  --onnx-model-path "${TRT_ROOT}/gr00t_onnx" \
  --data-config "${DATA_CONFIG}" \
  --embodiment-tag "${EMBODIMENT_TAG}" \
  --denoising-steps "${DENOISING_STEPS}" \
  --vit-dtype fp8 \
  --llm-dtype fp16 \
  --dit-dtype fp16 \
  --calib-size 200 \
  --video-backend "${VIDEO_BACKEND}"
```

然后 build engine：

```bash
cd "${TRT_ROOT}"
VIDEO_VIEWS=2 MAX_BATCH=2 VIT_DTYPE=fp8 LLM_DTYPE=fp16 DIT_DTYPE=fp16 \
  bash "${REPO_ROOT}/deployment_scripts/build_engine.sh"
```

启动 server 时 dtype 必须一致：

```bash
... --vit-dtype fp8 --llm-dtype fp16 --dit-dtype fp16
```

#### 7.3.2 LLM NVFP4（注意 full-layer-quant 与 onnx 2dq）

`export_onnx.py` 支持：

- `--llm-dtype nvfp4`
- `--full-layer-quant`（是否做“全层量化”；默认是 selective，禁用部分层以减少精度损失）

示例：

```bash
conda run --no-capture-output -n gr00t python "${REPO_ROOT}/deployment_scripts/export_onnx.py" \
  --dataset-path "${DATASET_PATH}" \
  --model-path "${CKPT_PATH}" \
  --onnx-model-path "${TRT_ROOT}/gr00t_onnx" \
  --data-config "${DATA_CONFIG}" \
  --embodiment-tag "${EMBODIMENT_TAG}" \
  --denoising-steps "${DENOISING_STEPS}" \
  --vit-dtype fp16 \
  --llm-dtype nvfp4 \
  --dit-dtype fp16 \
  --calib-size 200 \
  --video-backend "${VIDEO_BACKEND}"
```

脚本内部会在 nvfp4 情况下把 ONNX 做额外转换（fp4 qdq -> 2dq），并可能产生外部数据文件（`*.onnx_data`）。因此你在打包产物时要确保 `gr00t_onnx/eagle2/` 里的外部数据一并带上。

`build_engine.sh` 中对 nvfp4 的特殊点：

- 当 `LLM_DTYPE` 是 `nvfp4`/`nvfp4_full` 时，会把 `LLM_MAX_BATCH` 固定为 1（Myelin 限制）

因此你启动 server 时，如果要 nvfp4，请确保你的服务不会对 LLM 做大 batch 合并（本仓库当前是单请求推理，通常 OK）。

---

## 8. “把 engine 转成合适格式”：本仓库的约定与打包方式

### 8.1 运行时真正需要的 engine 文件名

`deployment_scripts/trt_model_forward.py` 里加载 engine 的规则是：

- `vit_{vit_dtype}.engine`
- `llm_{llm_dtype}.engine`
- `DiT_{dit_dtype}.engine`
- `vlln_vl_self_attention.engine`
- `state_encoder.engine`
- `action_encoder.engine`
- `action_decoder.engine`

所以你的“合适格式”其实就是：

1. 所有 engine 放在同一目录（例如 `${TRT_ROOT}/gr00t_engine`）
2. 文件名严格按上面规则
3. server 启动时传 `--trt-engine-path <这个目录>`

### 8.2 建议的“上线打包”最小集合

线上推理通常只需要：

- `${TRT_ROOT}/gr00t_engine/`（必选）

建议保留（用于审计/复现/回滚）：

- `${TRT_ROOT}/gr00t_onnx/`
- `${TRT_ROOT}/*.log`
- `${TRT_ROOT}/run_meta.txt`

打包示例（只打 engine）：

```bash
tar -C "${TRT_ROOT}" -czf "${TRT_ROOT}.engines.tgz" gr00t_engine
```

打包示例（engine + onnx + logs）：

```bash
tar -C "${TRT_ROOT}" -czf "${TRT_ROOT}.full.tgz" gr00t_engine gr00t_onnx *.log run_meta.txt || true
```

> ONNX 目录可能很大（外部数据文件也可能很大）。你可以按公司流程放对象存储/制品库，不一定要随 engine 一起分发到线上机器。

---

## 9. 针对“部署其他 VLA 模型”的迁移清单（通用方法）

当你要部署“不是 GR00T 的 VLA 模型”，你仍然可以复用本手册的思想，但需要你明确三件事：

1. 你的模型有哪些“可拆分模块”（vision / language / action head 等）
2. 每个模块的输入输出张量 shape/dtype 是什么（包含动态维度范围）
3. 你的线上服务协议是什么（输入 observation 的 schema，输出 action 的 schema）

建议你用下面流程迁移：

1. 先用 PyTorch 做一个稳定的 server（哪怕慢，但功能正确）
2. 找出耗时最大的子模块（通常是 vision backbone 和 LLM）
3. 为每个子模块写 ONNX 导出（输入输出清晰、可单测）
4. 用 `trtexec` 为每个子模块 build engine（把动态 shape profile 想清楚）
5. 用 “PyTorch vs TRT” 在同一批输入上做一致性比较（先 FP16）
6. 接入 server：只替换 forward，不改协议（先小流量验证）
7. 最后再逐步量化（每次只动一个模块）

如果你希望保持和本仓库一样的 ZMQ 协议，可以直接复用：

- `gr00t/eval/service.py` 的 msgpack+npy 协议

只需要你在 server 端把 “get_action(observation) -> action” 实现替换成你的模型即可。

---

## 10. 常见故障排查（按出现频率排序）

1. engine 反序列化失败（Version tag 不匹配）
   - 检查 `trtexec --version` 与 `import tensorrt` 是否同一主版本
2. engine 文件不存在
   - 99% 是 dtype 参数没传，server 默认会找 `vit_fp8.engine / llm_nvfp4.engine / DiT_fp8.engine`
3. dualcam shape/profile 不匹配
   - build engine 时必须 `VIDEO_VIEWS=2`
4. prompt/序列长度超出 profile
   - `build_engine.sh` 默认 `MAX_LEN` 单路 300、双路 600；如果你任务文本更长，需要扩大 profile 并重建 engine
5. `export_onnx.py` 做 fp8/nvfp4 报 `ImportError: modelopt is required`
   - 说明你没装 `modelopt`，先回到 FP16 跑通，或按 NVIDIA 文档安装 modelopt

---

## 附录 A：Observation schema（so100_dualcam 示例）

`scripts/agilex_groot_client_mvp.py` 里构造的 observation（dataset replay）是：

- `video.front`: `np.uint8`，形状 `(1, H, W, 3)`
- `video.wrist`: `np.uint8`，形状 `(1, H, W, 3)`
- `state.single_arm`: `np.float32`，形状 `(1, 6)`
- `state.gripper`: `np.float32`，形状 `(1, 1)`
- `annotation.human.task_description`: `list[str]`，长度 1

你的 observation 必须与 `--data-config so100_dualcam` 定义一致，否则模型 transform 会报错或输出异常。

---

## 附录 B：自定义 data_config（给部署其他 VLA/其他输入 schema 用）

`gr00t/experiment/data_config.py` 支持通过字符串 `module:ClassName` 加载外部 data config。

最小步骤：

1. 在 repo 根目录新建文件 `my_configs.py`
2. 写一个继承 `BaseDataConfig` 的类，实现 `transform()` 与 keys/indices
3. 运行时传：`--data-config my_configs:MyConfig`

示例骨架（仅演示结构，具体 keys 以你的 schema 为准）：

```python
# my_configs.py
from gr00t.experiment.data_config import BaseDataConfig
from gr00t.data.transform.base import ComposedModalityTransform, ModalityTransform
from gr00t.data.transform.video import VideoToTensor, VideoResize, VideoToNumpy
from gr00t.data.transform.state_action import StateActionToTensor, StateActionTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.model.transforms import GR00TTransform


class MyConfig(BaseDataConfig):
    video_keys = ["video.front"]
    state_keys = ["state.single_arm", "state.gripper"]
    action_keys = ["action.single_arm", "action.gripper"]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def transform(self) -> ModalityTransform:
        transforms = [
            VideoToTensor(apply_to=self.video_keys),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoToNumpy(apply_to=self.video_keys),
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(apply_to=self.state_keys, normalization_modes={k: "min_max" for k in self.state_keys}),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(apply_to=self.action_keys, normalization_modes={k: "min_max" for k in self.action_keys}),
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(state_horizon=len(self.observation_indices), action_horizon=len(self.action_indices), max_state_dim=64, max_action_dim=32),
        ]
        return ComposedModalityTransform(transforms=transforms)
```

然后你就可以用：

```bash
--data-config my_configs:MyConfig
```

来替换默认配置。

---

## 附录 C：ZMQ 线协议（想自己实现客户端/跨语言对接时用）

服务端协议实现位于：

- `gr00t/eval/service.py`

客户端参考实现位于：

- `scripts/agilex_groot_client_mvp.py`

协议要点：

1. 底层传输：ZeroMQ（server 用 `REP`，client 用 `REQ`）
2. 消息序列化：`msgpack`
3. `numpy.ndarray` 的编码方式：
   - 用 `np.save(BytesIO, array, allow_pickle=False)` 把数组写成 `.npy` 字节流
   - 用一个 marker dict 包起来：`{"__ndarray_class__": True, "as_npy": <bytes>}`
4. request 基本结构：
   - 有输入的 endpoint（例如 `get_action`）：
     - `{"endpoint": "get_action", "data": <observation_dict>, "api_token": "..."}`
   - 无输入的 endpoint（例如 `ping/get_modality_config/kill`）：
     - `{"endpoint": "ping", "api_token": "..."}`
5. response 基本结构：
   - 成功：业务 dict（action dict / modality config / ping status）
   - 失败：`{"error": "<message>"}`

> 注意：这是一个“内网开发协议”，token 只是明文字符串校验，没有加密与权限系统。上生产请至少放在受控网络/加防火墙。

---

## 附录 D：build_engine.sh 的 profile 参数怎么理解和调整

`deployment_scripts/build_engine.sh` 使用 `trtexec` 为每个子模块构建 engine，并指定 `--minShapes/--optShapes/--maxShapes`。

你需要理解的关键点：

1. profile 决定 “运行时允许的输入 shape 范围”
2. 超出 `maxShapes` 会在运行时报错（shape/profile 不匹配），不是自动回退
3. profile 越大，engine 越大、build 越慢、可能更耗显存

本仓库里最容易踩的 profile 是 LLM/DiT 的序列长度：

- 单路视频（`VIDEO_VIEWS=1`）：
  - `MIN_LEN/OPT_LEN/MAX_LEN = 80/296/300`
- 双路视频（`VIDEO_VIEWS=2`）：
  - `MIN_LEN/OPT_LEN/MAX_LEN = 80/568/600`

这些长度本质上是“transform 后的 token 序列长度”，通常与：

- 图像 token 数量（多视角会变长）
- 文本 prompt 长度

强相关。

当你遇到 “prompt 变长导致运行时报 shape/profile 不匹配”，处理方式是：

1. 估计你线上可能出现的最大 prompt/token 长度（保守一点）
2. 在 `deployment_scripts/build_engine.sh` 里把 `MAX_LEN` 调大（同时可调 `OPT_LEN`）
3. 重新 build engine（必须 rebuild，engine 不是运行时可改的）

如果你不确定当前真实的序列长度，你可以在 PyTorch 侧打印一次 transform 产物的 `eagle_attention_mask` 长度（示意）：

```bash
conda run -n gr00t python - <<'PY'
import os

from gr00t.experiment.data_config import load_data_config
from gr00t.model.policy import Gr00tPolicy, unsqueeze_dict_values
from gr00t.data.dataset import LeRobotSingleDataset

ckpt = os.environ["CKPT_PATH"]
dataset_path = os.environ["DATASET_PATH"]
data_config = os.environ["DATA_CONFIG"]
embodiment = os.environ["EMBODIMENT_TAG"]
denoising_steps = int(os.environ["DENOISING_STEPS"])
video_backend = os.environ.get("VIDEO_BACKEND", "decord")

cfg = load_data_config(data_config)
policy = Gr00tPolicy(
    model_path=ckpt,
    embodiment_tag=embodiment,
    modality_config=cfg.modality_config(),
    modality_transform=cfg.transform(),
    denoising_steps=denoising_steps,
    device="cuda",
)

ds = LeRobotSingleDataset(
    dataset_path=dataset_path,
    modality_configs=policy.modality_config,
    embodiment_tag=embodiment,
    video_backend=video_backend,
)
step = ds[0]
if not policy._check_state_is_batched(step):
    step = unsqueeze_dict_values(step)
norm = policy.apply_transforms(step)
mask = norm["eagle_attention_mask"]
print("eagle_attention_mask shape:", getattr(mask, "shape", None))
PY
```

---

## 附录 E：脚本参数速查（照着填，不用翻代码）

下面是本 playbook 里会用到的脚本与关键参数。你开会/对齐时可以直接用这张表检查“有没有漏传、有没有用错默认值”。

### E.1 `deployment_scripts/export_onnx.py`（导出 ONNX，可选量化）

你最常用的一组参数（FP16/量化都适用）：

- `--dataset-path`：LeRobot 数据集目录（用于校准/示例输入）
- `--model-path`：checkpoint 或 HF 模型目录
- `--onnx-model-path`：输出 ONNX 的根目录（脚本会创建 `gr00t_onnx/...` 结构）
- `--data-config`：例如 `so100_dualcam`
- `--embodiment-tag`：例如 `new_embodiment`
- `--denoising-steps`：例如 `4`
- `--video-backend`：`decord` 或 `torchcodec`
- `--calib-size`：校准样本数（FP16 不敏感；量化建议 100+）
- `--vit-dtype`：`fp16` 或 `fp8`
- `--llm-dtype`：`fp16` / `fp8` / `nvfp4`
- `--dit-dtype`：`fp16` 或 `fp8`
- `--full-layer-quant`：仅对 `llm=nvfp4` 有意义（是否全层量化；默认 selective）

### E.2 `deployment_scripts/build_engine.sh`（ONNX -> engine）

它没有命令行参数，靠环境变量控制（必须在 `${TRT_ROOT}` 目录执行）：

- `VIDEO_VIEWS`：`1` 或 `2`
- `MAX_BATCH`：engine 支持的最大 batch（会影响 engine 体积与兼容性）
- `VIT_DTYPE`：`fp16` 或 `fp8`（对应 `vit_<dtype>.onnx/engine`）
- `LLM_DTYPE`：`fp16` / `fp8` / `nvfp4` / `nvfp4_full`
- `DIT_DTYPE`：`fp16` 或 `fp8`

### E.3 `deployment_scripts/gr00t_inference.py`（离线推理/对比）

- `--inference-mode`：`pytorch` / `tensorrt` / `compare`（推荐 compare）
- `--trt-engine-path`：engine 目录（通常是 `${TRT_ROOT}/gr00t_engine`）
- `--vit-dtype --llm-dtype --dit-dtype`：必须与 engine 文件名一致
- `--video-backend`：建议与导出/线上保持一致
- 其他与模型绑定参数：`--model-path --dataset-path --data-config --embodiment-tag --denoising-steps`

### E.4 `scripts/inference_service.py`（起服务：ZMQ/HTTP）

ZMQ 模式关键参数：

- `--server`：启动 server
- `--port`：ZMQ 端口（server bind `tcp://*:<port>`）
- `--api-token`：可选，开启后 client 必须带同样 token
- `--model-path --data-config --embodiment-tag --denoising-steps`：必须与训练/导出一致

启用 TensorRT 的关键参数（强烈建议每次都显式传）：

- `--use-tensorrt`
- `--trt-engine-path`
- `--vit-dtype --llm-dtype --dit-dtype`

HTTP 模式额外参数：

- `--http-server`
- `--host`（HTTP 才会用到；ZMQ 的 host 固定 `*`）

### E.5 `docs/infer_service/server_healthcheck.sh`（一键健康检查）

常用参数：

- `--host --port --api-token`
- `--timeout-ms`：默认 5000
- dataset replay：
  - `--dataset-path --episode-index --frame-index --video-backend`
- `--task`：覆盖任务文本（可选）
