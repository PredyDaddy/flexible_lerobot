<!--
NOTE:
- This document is written for /data/cqy_workspace/flexible_lerobot.
- It is intentionally practical (commands + artifacts + integration points).
- It summarizes upstream Isaac-GR00T-n1.5-release deployment scripts and maps them to this repo's LeRobot GROOT implementation.
-->

# LeRobot GROOT TensorRT 集成方案（完整落地版）

目标：在 **不改变现有数据处理逻辑**（preprocess / postprocess）的前提下，把 **GROOT（GR00T N1.5）** 的推理链路加速到 TensorRT，并把导出、构建、部署流程“集成进当前仓库”，日常开发落在 `my_devs/` 下。

本方案基于两套代码：

1. 上游参考实现（只作“参考与对齐”）：`my_devs/docs/gr00t_trt/Isaac-GR00T-n1.5-release/`
2. 本仓库真实推理实现（我们已经在用）：`src/lerobot/policies/groot/*` + 你的推理脚本 `my_devs/train/groot/so101/run_groot_infer.py`

你已经写的 playbook：`my_devs/docs/gr00t_trt/playbook_vla_trt_deployment.md` 是非常好的“运行手册草稿”。本方案会：

- 保留你 playbook 的组织方式（先 FP16 跑通，再 FP8/NVFP4）
- 修正与上游脚本不一致/容易踩坑的点
- 给出“如何在本仓库内做集成”的目录结构与接口设计

---

## 1. 结论先行（你应该怎么做）

我建议分 3 个阶段推进，每个阶段都有明确验收标准。

### 本仓库当前已落地的内容（2026-03-03）

- TRT runtime（可选依赖、lazy import tensorrt）：
  - `src/lerobot/policies/groot/trt_runtime/engine.py`
  - `src/lerobot/policies/groot/trt_runtime/patch.py`
- 机器人推理脚本支持一行参数切换 torch/TRT：
  - `my_devs/train/groot/so101/run_groot_infer.py --backend {pytorch|tensorrt}`
  - 支持 `--trt-engine-path`、`--vit-dtype/--llm-dtype/--dit-dtype`、`--trt-action-head-only`
- 导出/构建脚手架（先从 action head FP16 开始）：
  - `my_devs/groot_trt/export_action_head_onnx.py`
  - `my_devs/groot_trt/build_engine.sh`

### 阶段 A：快速跑通（不改 `src/`，只在 `my_devs/` 开发）

目的：最短路径验证“TRT engine 构建 + 加速可行”，先把工程跑通。

- 在 `my_devs/groot_trt/` 放置：
  - `export_action_head_onnx.py`（已提供：先导出 action head ONNX，最容易先跑通）
  - `build_engine.sh`（已提供：从 ONNX 构建 engines，支持可配置 `ONNX_DIR/ENGINE_DIR/TRTEXEC`）
  - `README.md`（说明如何从 action-head-only 起步）
- 验收：`my_devs/train/groot/so101/run_groot_infer.py --backend=tensorrt --trt-action-head-only=true` 能出 action，
  且循环频率提升明显（至少 action head 侧 latency 下降）。

### 阶段 B：集成到本仓库推理（推荐，开始动 `src/`）

目的：让你现在已经能跑的推理脚本（比如 `my_devs/train/groot/so101/run_groot_infer.py`）**一行参数切换** torch/TRT。

- 在 `src/lerobot/policies/groot/` 加一个 **可选依赖** 的 TRT runtime（lazy import tensorrt），核心是：
  - Engine wrapper（等价于上游 `deployment_scripts/trt_torch.py`）
  - `setup_tensorrt_engines(...)`（等价于上游 `deployment_scripts/trt_model_forward.py` 的 monkey patch 逻辑）
- 可选：在 `GrootConfig` 增加 TRT 相关字段（backend、engine 路径、dtype 等），把 TRT 开关从“脚本参数”
  下沉成“配置驱动”（方便 `lerobot-record` / 服务化复用）。
- 验收：你的 `run_groot_infer.py --backend=tensorrt` 能运行，且动作与 torch 模式一致性满足阈值。

### 阶段 C：工程化部署（Orin/Thor/Server 化）

目的：把“导出 + 构建 engine + 服务化启动 + 健康检查”变成稳定 SOP。

- 输出 `outputs/trt/<artifact>/` 作为可部署产物
- 支持 Jetson（Orin/Thor）与 x86（4090/A100 等）两条路径
- 验收：一次完整部署从零机器到上线，手动步骤最少且可复现

---

## 2. 上游 TRT 方案拆解（你需要理解的“原理与产物”）

上游提供的 **核心思路** 是“模块化导出 ONNX + trtexec 编译 engine + Python 侧组装推理”。

### 2.1 ONNX 导出（上游 `deployment_scripts/export_onnx.py`）

它不会导出一个端到端整图，而是拆成 7 个 ONNX：

- `gr00t_onnx/eagle2/vit_{vit_dtype}.onnx`
  - inputs: `pixel_values`, `position_ids`
  - output: `vit_embeds`
- `gr00t_onnx/eagle2/llm_{llm_dtype}.onnx`
  - inputs: `inputs_embeds`, `attention_mask`
  - output: `embeddings`
  - `nvfp4` 会额外生成外置权重文件 `llm_nvfp4.onnx_data`
- `gr00t_onnx/action_head/vlln_vl_self_attention.onnx`
  - input: `backbone_features`
  - output: `output`
- `gr00t_onnx/action_head/state_encoder.onnx`
  - inputs: `state`, `embodiment_id`
- `gr00t_onnx/action_head/action_encoder.onnx`
  - inputs: `actions`, `timesteps_tensor`, `embodiment_id`
- `gr00t_onnx/action_head/DiT_{dit_dtype}.onnx`
  - inputs: `sa_embs`, `vl_embs`, `timesteps_tensor`
- `gr00t_onnx/action_head/action_decoder.onnx`
  - inputs: `model_output`, `embodiment_id`

这一点非常重要：**TRT 加速不是“把整个 policy 编成一个 engine”，而是把关键子模块编 engine。**

### 2.2 Engine 构建（上游 `deployment_scripts/build_engine.sh`）

engine 输出目录固定为：

```
gr00t_engine/
  vit_{VIT_DTYPE}.engine
  llm_{LLM_DTYPE}.engine
  vlln_vl_self_attention.engine
  state_encoder.engine
  action_encoder.engine
  DiT_{DIT_DTYPE}.engine
  action_decoder.engine
  *.log
```

它用 `trtexec` 构建，并且强依赖：

- `/usr/src/tensorrt/bin/trtexec` 存在
- profile（MIN/OPT/MAX）匹配你的输入长度
- 典型 profile（单 view）是 `MIN_LEN/OPT_LEN/MAX_LEN = 80/296/300`
- 双 view 则是 `80/568/600`

### 2.3 推理时如何切 TRT（上游 `deployment_scripts/trt_model_forward.py`）

上游的“集成方式”非常工程化，建议直接照搬：

1. 反序列化 engine（`trt_torch.Engine`）
2. 删除大模块节省显存（vision_model、language_model、action_head 的各子模块）
3. monkey patch：
   - `policy.model.backbone.forward = eagle_tensorrt_forward`
   - `policy.model.action_head.get_action = action_head_tensorrt_forward`

并且，**扩散（denoising）循环仍在 Python**，每个 step 里跑 TRT engine：

```
action_encoder -> DiT -> action_decoder
actions = actions + dt * pred_velocity
```

所以你不用担心“TRT 要支持复杂控制流”。

### 2.4 一个关键差异：上游数据集格式 != 本仓库 LeRobot 数据集格式

上游 `Isaac-GR00T-n1.5-release` 的 `gr00t/data/dataset.py::LeRobotSingleDataset` 依赖 `meta/modality.json`、
并且常见 key 是 `video.front` / `state.single_arm` 这套“GR00T 训练数据 schema”。

而本仓库 `lerobot-record` 产生的数据、以及我们当前推理链路使用的 frame key，是：

- 图像：`observation.images.<camera_key>`
- 状态：`observation.state`（names 里是 `shoulder_pan.pos` 等）
- 动作：`action`（names 同上）

因此：

- **不要指望** 直接用上游 `export_onnx.py` 读取你本仓库的数据集就能跑通
- 我们集成版的“导出/校准/统计 profile”应该以 **本仓库的 preprocess 输出（`eagle_*`）** 为准

这也是我建议把“导出/构建脚本”放在 `my_devs/groot_trt/`，并直接复用本仓库的 processor 的原因。

---

## 3. 本仓库 GROOT 推理链路（TRT 应该插哪里）

本仓库的 GROOT 推理非常清晰地被拆成：

1. preprocess（`src/lerobot/policies/groot/processor_groot.py`）
2. model forward（`src/lerobot/policies/groot/modeling_groot.py` -> `src/lerobot/policies/groot/groot_n1.py`）
3. postprocess（同一个 processor pipeline 的 postprocessor）

关键点：**我们不应该改 preprocess/postprocess**，否则会引入训练/推理不一致问题。

### 3.1 preprocess / postprocess 保持不动

`processor_groot.py` 做了：

- 从 `observation.images.*` 聚合 video（多相机视角）
- 语言 instruction（`task` 字段）打包
- state/action padding 到 `max_state_dim/max_action_dim`
- Eagle processor 生成 `eagle_*` 张量（input_ids / attention_mask / pixel_values 等）
- DeviceProcessor 把 tensor 放到 `cfg.policy.device`
- postprocess 里做 action slice + 反归一化

TRT 插入点应该在：**模型 forward**（`_groot_model.get_action(...)`）内部。

### 3.2 最稳的插入点（建议从这里开始）

你可以按风险从低到高分 3 层：

1. **只加速 DiT（最低风险）**：只把 `FlowmatchingActionHead.model` 替换成 TRT，保留其余 torch
2. 加速整个 action head（中等风险）：`vlln + state_encoder + action_encoder + DiT + action_decoder` 都走 TRT
3. 加速 backbone（最高收益也最高复杂度）：ViT + LLM engine

这与上游一致，只是你可以“分步落地”。

---

## 4. “集成到当前仓库”的目录结构建议

你希望在 `my_devs/` 下开发导出/部署脚本，同时最终能在 `src/` 的 policy 里启用 TRT。

我建议把工作分成两层：

### 4.1 `my_devs/`：导出与构建工具链（重依赖、偏运维）

建议新增目录（你可以改名）：

```
my_devs/groot_trt/
  README.md
  export_onnx_lerobot_groot.py
  build_engine.sh
  compare_torch_trt.py
  run_export_fp16.sh
  run_build_fp16.sh
  run_compare_fp16.sh
  artifacts.md
```

这个目录的职责是：

- 从 **LeRobot checkpoint**（`pretrained_model/`）导出 ONNX
- 调 `trtexec` 构建 engine
- 生成“可部署产物目录”（`outputs/trt/<artifact>/...`）

### 4.2 `src/`：runtime 支持（轻依赖、面向推理）

建议新增（或同名模块）：

```
src/lerobot/policies/groot/trt_runtime/
  engine.py          # Engine wrapper (lazy import tensorrt)
  patch.py           # setup_tensorrt_engines(...) / enable_trt(...)
  __init__.py
```

以及改动：

- `src/lerobot/policies/groot/configuration_groot.py`
  - 添加 TRT 配置字段（backend、engine_path、dtype、video_views、profile 等）
- `src/lerobot/policies/groot/modeling_groot.py`
  - 根据 config 启用 TRT（只在推理时启用，训练默认 torch）

这样做的好处：

- 你可以在 `my_devs/` 快速迭代导出/构建，不污染核心包
- runtime 只需要 `tensorrt`（可选）+ torch，不依赖 modelopt
- 启用 TRT 对现有脚本透明：`run_groot_infer.py`、`lerobot-record`、`async_inference` 都能复用

---

## 5. 产物定义（强烈建议标准化）

建议把一次 TRT 部署产物固定成下面结构（与你 playbook 的思路一致）：

```
outputs/trt/<ARTIFACT_NAME>/
  gr00t_engine/
    *.engine
    *.log
  gr00t_onnx/                  # 可选（调试/审计用）
    eagle2/...
    action_head/...
  logs/
    export_onnx.log
    build_engine.log
    compare.log
  run_meta.txt                 # 强烈建议写：CUDA/TRT/driver/git sha/命令行参数
```

**上线最小必需**：`gr00t_engine/`（以及你推理服务/脚本需要的配置文件）。

---

## 6. Runbook：端到端 FP16 基线（强烈建议先跑通）

这一节给你一套“可执行的落地流程”。其中导出脚本名字是建议的；你也可以先复用上游脚本做 PoC。

### 6.0 先避一个坑：checkpoint 加载 `strict=False`

如果你加载 LeRobot 的 GROOT checkpoint 时看到类似错误：

```
Missing key(s) in state_dict: ... embed_tokens.weight
```

这通常不是“权重坏了”，而是因为 **LLM 的词嵌入与 lm_head 权重是 tied weight**，
保存 safetensors 时只存了一份（例如只存 `lm_head.weight`）。因此加载时应该用 `strict=False`。

本仓库已将 `GrootPolicy.from_pretrained(..., strict=...)` 的默认值改成 `strict=False`；
如果你在旧脚本/旧环境里仍遇到报错，显式加上 `strict=False` 即可。

### 6.1 环境约束（必须对齐）

你 playbook 里强调的点完全正确：

- `trtexec` 与 Python `import tensorrt` 的 **主版本必须一致**
- engine 与硬件强绑定：建议在最终部署机器上构建

建议自检命令（在最终部署环境跑）：

```bash
nvidia-smi
/usr/src/tensorrt/bin/trtexec --version | head -n 3
python -c "import tensorrt as trt; print('tensorrt', trt.__version__)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

### 6.2 确定“配置与 shape”（避免 profile 不匹配）

你需要钉死：

- 相机数量（`VIDEO_VIEWS=1/2`）
- `max_state_dim` / `max_action_dim`（通常 64/32）
- `action_horizon`（通常 16）
- LLM 输出 sequence length profile（`MIN_LEN/OPT_LEN/MAX_LEN`）

建议做法：从“真实 preprocess 输出”统计 `eagle_attention_mask.shape[1]`，用它做 OPT_LEN，并留冗余做 MAX_LEN。

下面给一个**最小脚本**（直接跑在本仓库、直接读取你 checkpoint 的 preprocessor），用来打印：

- `VIDEO_VIEWS`（你当前有几个相机）
- `SEQ_LEN`（`eagle_attention_mask` 的长度）

你需要提供一条可用的 observation（有两种方式）：

1. 直接连机器人抓一帧（最真实）
2. 从某个本仓库 LeRobotDataset 里取一帧（离线）

为了避免在文档里塞太多业务代码，这里给你一个“连机器人抓一帧”的模板（离线版本后续可以补）。

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot_flex

POLICY_PATH=/path/to/pretrained_model \
TASK="Put the block in the bin" \
ROBOT_PORT=/dev/ttyACM0 TOP_CAM_INDEX=4 WRIST_CAM_INDEX=6 \
python - <<'PY'
import os
from pathlib import Path

import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline, make_default_processors
from lerobot.processor.converters import batch_to_transition, transition_to_batch
from lerobot.robots import make_robot_from_config
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.utils.constants import OBS_STR

policy_path = Path(os.environ["POLICY_PATH"]).expanduser()
task = os.environ.get("TASK", "Perform the task.")

pre = PolicyProcessorPipeline.from_pretrained(
    pretrained_model_name_or_path=str(policy_path),
    config_filename="policy_preprocessor.json",
    overrides={"device_processor": {"device": "cpu"}},
    to_transition=batch_to_transition,
    to_output=transition_to_batch,
)

robot_cfg = SOFollowerRobotConfig(
    id="tmp",
    calibration_dir=Path("/tmp"),
    port=os.environ.get("ROBOT_PORT", "/dev/ttyACM0"),
    cameras={
        "top": OpenCVCameraConfig(
            index_or_path=int(os.environ.get("TOP_CAM_INDEX", "4")),
            width=640,
            height=480,
            fps=30,
        ),
        "wrist": OpenCVCameraConfig(
            index_or_path=int(os.environ.get("WRIST_CAM_INDEX", "6")),
            width=640,
            height=480,
            fps=30,
        ),
    },
)
robot = make_robot_from_config(robot_cfg)
robot.connect()
obs = robot.get_observation()
robot.disconnect()

_, robot_action_processor, robot_observation_processor = make_default_processors()
dataset_features = combine_feature_dicts(
    aggregate_pipeline_dataset_features(
        pipeline=robot_action_processor,
        initial_features=create_initial_features(action=robot.action_features),
        use_videos=True,
    ),
    aggregate_pipeline_dataset_features(
        pipeline=robot_observation_processor,
        initial_features=create_initial_features(observation=robot.observation_features),
        use_videos=True,
    ),
)

obs_processed = robot_observation_processor(obs)
observation_frame = build_dataset_frame(dataset_features, obs_processed, prefix=OBS_STR)

# Convert to the exact tensor format expected by the policy processors.
batch = prepare_observation_for_inference(
    observation_frame,
    torch.device("cpu"),
    task=task,
    robot_type=robot.robot_type,
)

out = pre(batch)
mask = out.get("eagle_attention_mask")
pixel_values = out.get("eagle_pixel_values")

print("eagle_attention_mask:", None if mask is None else tuple(mask.shape))
print("eagle_pixel_values:", None if pixel_values is None else tuple(pixel_values.shape))
if mask is not None:
    print("SEQ_LEN =", int(mask.shape[1]))
if pixel_values is not None:
    print("VIDEO_VIEWS =", int(pixel_values.shape[0]))
PY
```

说明：

- 这个脚本已经完整复用了本仓库的“构造 observation frame -> preprocessor”的流程，所以打印出来的 `SEQ_LEN/VIDEO_VIEWS`
  可以直接用于 `build_engine.sh` 的 profile 选取。
- 如果你要离线统计（不连机器人），把 `obs = robot.get_observation()` 换成 “从 `LeRobotDataset` 取一帧” 即可。

### 6.3 导出 ONNX（FP16）

FP16 baseline 你只需要导出 fp16 版本：

- `vit_fp16.onnx`
- `llm_fp16.onnx`
- `DiT_fp16.onnx`

action head 其他 onnx 固定。

**推荐起步（action head only）**：先只导出 action head（含 `vlln/state/action_encoder/DiT/action_decoder`），
不导出 ViT/LLM，先验证“扩散循环 TRT 加速”是否成立：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot_flex

REPO_ROOT="$(pwd)"  # repo root
POLICY_PATH=/path/to/pretrained_model
ARTIFACT=outputs/trt/my_artifact_$(date +%Y%m%d_%H%M%S)

python my_devs/groot_trt/export_action_head_onnx.py \
  --policy-path "${POLICY_PATH}" \
  --onnx-out-dir "${ARTIFACT}/gr00t_onnx" \
  --seq-len 568 \
  --device cuda
```

说明：

- `--seq-len` 建议先用双相机默认 `568`（单相机用 `296`），后续再按你的真实 `eagle_attention_mask.shape[1]`
  统计值调整到更贴近线上分布的 `OPT_LEN`。
- 这一步只生成 `${ARTIFACT}/gr00t_onnx/action_head/*`，`build_engine.sh` 会自动跳过缺失的 ViT/LLM ONNX。

**完整链路（ViT+LLM+ActionHead）**：需要额外处理 Eagle/Qwen3 的 ONNX 导出与量化（fp8/nvfp4），建议先复用上游
`Isaac-GR00T` 的 `deployment_scripts/export_onnx.py` 作为参考实现，再逐步迁移到本仓库（见第 8.2）。

### 6.4 构建 engine（FP16）

构建命令（按上游 build_engine.sh 的 env 变量设计）：

```bash
export VIDEO_VIEWS=2          # 双相机
export VIT_DTYPE=fp16
export LLM_DTYPE=fp16
export DIT_DTYPE=fp16
export MAX_BATCH=8            # 先用上游默认

# 这三个长度建议从数据统计来，而不是硬写
export MIN_LEN=80
export OPT_LEN=568
export MAX_LEN=600

cd "${ARTIFACT}"
# 如果你的 trtexec 不在 /usr/src/tensorrt/bin/trtexec，请显式设置 TRTEXEC=/path/to/trtexec
bash "${REPO_ROOT}/my_devs/groot_trt/build_engine.sh"
```

### 6.5 离线一致性验证（Torch vs TRT）

必须做 compare，否则后面全是“盲飞”：

- 固定随机初始噪声（扩散模型必须固定 init_actions）
- 比较 action_pred 的 cosine similarity / L1

验收建议（经验阈值，可调整）：

- cosine similarity > 0.99
- mean L1 < 1e-2（不同模块/量化方案会不同；FP16 应该很接近）

### 6.6 接入你现有推理脚本

当阶段 B 完成后（TRT runtime 集成进 `src/`），你应该能这样切换：

```bash
python my_devs/train/groot/so101/run_groot_infer.py \
  --policy-path /path/to/pretrained_model \
  --task "Put the block in the bin" \
  --backend tensorrt \
  --trt-engine-path outputs/trt/<ARTIFACT_NAME>/gr00t_engine \
  --trt-action-head-only true \
  --vit-dtype fp16 --llm-dtype fp16 --dit-dtype fp16
```

---

## 7. 你 playbook 里需要修正/补强的关键点

这些点我建议你在后续把 playbook 做一次“可直接执行的修订版”。（我也可以帮你直接改那份文档。）

1. `deployment_scripts/README.md` 里命令写成了 `--inference_mode`，但脚本实际参数是 `--inference-mode`
2. 上游 `export_onnx.py` 顶部无条件 `import modelopt...`，即使 FP16 也会 import 失败
   - 建议在我们集成版本里把 modelopt 变成“只有量化才需要的可选依赖”
3. `nvfp4_full` 在 build/export 中存在，但 `gr00t_inference.py` / `scripts/inference_service.py` 的 CLI choices 没包含它
   - 要么加到 choices，要么明确不支持
4. HTTP server 的 `api_token` 目前不校验（ZMQ 才校验），需要在文档里写清楚
5. `MAX_BATCH` 和 `VIDEO_VIEWS` 语义需要更清晰
   - `VIDEO_VIEWS` 影响 OPT/MAX_LEN
   - `MAX_BATCH` 是 engine profile 的 batch 上限，不等于“相机数”

---

## 8. 集成实现清单（你可以按这个开工）

下面是“把 TRT 集成进本仓库”的最小实现清单（建议顺序）。

### 8.1 最小可用（只加速 DiT）

**当前仓库已实现（action head only）**：

- `src/lerobot/policies/groot/trt_runtime/engine.py`：TRT `.engine` 加载与执行（lazy import）
- `src/lerobot/policies/groot/trt_runtime/patch.py`：
  - `setup_tensorrt_engines(model, trt_engine_path, ..., action_head_only=True)`
  - monkey patch `action_head.get_action` 走 TRT engines（同时可选 patch backbone）
- `my_devs/groot_trt/export_action_head_onnx.py`：导出 action head ONNX（FP16）
- `my_devs/groot_trt/build_engine.sh`：用 `trtexec` 构建 engines（缺失 ViT/LLM ONNX 时会自动跳过）
- `my_devs/train/groot/so101/run_groot_infer.py`：`--backend=tensorrt` 一行切换

验收：

- `run_groot_infer.py --backend=tensorrt --trt-action-head-only=true` 能跑起来并输出动作
- 记录 loop 中 action head 的耗时明显下降（建议你在脚本里加简单的 timing 日志）

### 8.2 上游同等能力（全链路 TRT：ViT+LLM+ActionHead）

全链路已经具备 runtime 接口（同一个 `setup_tensorrt_engines(..., action_head_only=False)`），还差的是：

- 导出 backbone ONNX（ViT + LLM）到 `${ONNX_DIR}/eagle2/`：
  - 这一步建议先“参考上游 export_onnx.py”，再按本仓库 `EagleBackbone` 实现迁移
- 构建 `vit_*.engine` / `llm_*.engine` 并在推理时指定 `--vit-dtype/--llm-dtype`

验收：

- `run_groot_infer.py --backend=tensorrt --trt-action-head-only=false` 能跑通
- 端到端 latency 明显下降（backbone + action head 同时加速）

### 8.3 Async / Server 化（可选）

- 扩展 `src/lerobot/async_inference/helpers.py` 的 `RemotePolicyConfig`：增加 backend/engine path
- server 在加载 policy 后启用 TRT
- 验收：robot client 不改协议即可获得加速

---

## 9. 建议的下一步（我可以直接帮你做）

如果你希望我下一步直接开始落地代码集成，我建议先做这两件事（收益最大、风险最小）：

1. 把 backbone（ViT + LLM）的 ONNX 导出脚本迁移到 `my_devs/groot_trt/`（FP16 先跑通）
2. 加一个离线 `compare_torch_trt.py`（固定 init_actions，对齐验证作为门禁）

这样我们能把加速从 action-head-only 平滑扩展到全链路，并用 compare 作为门禁避免“盲飞”。

---

## 附录 A：现有推理/录制脚本怎么用（以及“运行时间”在哪设置）

### A.1 纯推理（不录数据）：`my_devs/train/groot/so101/run_groot_infer.py`

- 运行时长（你问的“运行时间”）：`--run-time-s` 或环境变量 `RUN_TIME_S`
  - `--run-time-s <= 0` 表示一直跑到 `Ctrl+C`
- 控制循环频率：`--fps`（默认 30）
- 切 TRT：
  - `--backend=tensorrt`
  - `--trt-engine-path=/path/to/gr00t_engine`
  - 起步推荐：`--trt-action-head-only=true`

示例：

```bash
conda run -n lerobot_flex python my_devs/train/groot/so101/run_groot_infer.py \
  --robot-port /dev/ttyACM0 \
  --top-cam-index 4 --wrist-cam-index 6 \
  --policy-path /path/to/pretrained_model \
  --task "Put the block in the bin" \
  --run-time-s 120
```

### A.2 录制评测数据（policy 驱动）：`my_devs/train/groot/so101/run_groot_eval_record.py`

- 每个 episode 录制时长：`--episode-time-s`（或 `EPISODE_TIME_S`）
- episode 数量：`--num-episodes`（或 `NUM_EPISODES`）
- reset 等待：`--reset-time-s`（或 `RESET_TIME_S`）

示例：

```bash
conda run -n lerobot_flex python my_devs/train/groot/so101/run_groot_eval_record.py \
  --policy-path /path/to/pretrained_model \
  --dataset-repo-id admin123/eval_run_03 \
  --dataset-task "Put the block in the bin" \
  --num-episodes 5 \
  --episode-time-s 40 \
  --reset-time-s 10
```

### A.3 Bash 版本录制：`my_devs/train/groot/so101/run_groot_eval_record.sh`

这份脚本最终会执行 `lerobot-record`。运行时间相关参数是：

- `NUM_EPISODES`
- `EPISODE_TIME_S`
- `RESET_TIME_S`
