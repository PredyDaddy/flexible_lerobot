# LeRobot PI0.5 TensorRT 后端开发方案

## 目标

把当前 SO101 PI0.5 实时推理链路里的模型后端从 PyTorch 换成 TensorRT，同时严格保持开发产物只落在：

```text
my_devs/openpi_trt/
```

现有实时推理入口是只读参考：

```text
my_devs/train/pi/so101/run_pi05_infer.py
```

这个脚本目前已经能完成：

1. 连接 SO101 follower 机器人和 top/wrist 摄像头。
2. 读取 checkpoint 下保存的 `policy_preprocessor.json` 和 `policy_postprocessor.json`。
3. 用 LeRobot `predict_action(...)` 执行 preprocessor -> policy -> postprocessor。
4. 把 policy 输出动作转换成机器人 action 并发送。

本方案不建议一开始改动这条实时控制链路，而是先在 `my_devs/openpi_trt` 下做一个可验证的 TensorRT 后端，再接入。

## 核心判断

这件事不应该按 OpenPI 原生 JAX 后端的思路做。

原因：

- 当前 checkpoint 是 LeRobot `PI05Policy` / `PI05Pytorch` 路径，不是 OpenPI 原生 JAX policy。
- 当前 `model.safetensors` 已经能通过 LeRobot 的 `PI05Policy.from_pretrained(...)` 恢复成 PyTorch 模型。
- 当前推理链路的关键边界是：

```text
predict_action(...)
  -> preprocessor(observation)
  -> policy.select_action(batch)
  -> PI05Policy.predict_action_chunk(batch)
  -> PI05Pytorch.sample_actions(images, img_masks, tokens, masks)
  -> postprocessor(action)
```

所以我们要做的是：从 LeRobot 版 PyTorch 模型导出 TensorRT 可执行图，然后让 `PI05Policy.predict_action_chunk(...)` 或它内部的 `model.sample_actions(...)` 使用 TensorRT。

OpenPI on Thor 教程里的代码仍然有参考价值，但它主要参考这些部分：

- ONNX wrapper 的设计。
- 固定输入名和输出名。
- `trtexec` 动态 shape profile。
- TensorRT runtime wrapper。
- 用固定 noise 做 Torch / TRT 数值对齐。

不应直接照搬的部分：

- OpenPI 原生 `policy_config.create_trained_policy(...)`。
- OpenPI/JAX 数据 transform。
- OpenPI 自己的 checkpoint/config 结构。
- 教程中假设的 `openpi.models_pytorch.pi0_pytorch` 包结构。

## 当前 checkpoint 的真实形状

从默认 checkpoint：

```text
outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model/config.json
```

读取到的关键配置：

```text
type: pi05
device: cuda
dtype: bfloat16
paligemma_variant: gemma_2b
action_expert_variant: gemma_300m
input_features:
  observation.state: [6]
  observation.images.top: [3, 480, 640]
  observation.images.wrist: [3, 480, 640]
output_features:
  action: [6]
chunk_size: 50
n_action_steps: 50
max_state_dim: 32
max_action_dim: 32
num_inference_steps: 10
image_resolution: [224, 224]
tokenizer_max_length: 200
normalization_mapping:
  ACTION: MEAN_STD
  STATE: MEAN_STD
  VISUAL: IDENTITY
```

注意：LeRobot PI0.5 的 policy 内部没有把 state 单独作为 TensorRT 输入传给模型。state 已经在 preprocessor 的 `Pi05PrepareStateTokenizerProcessorStep` 中离散化并拼进语言 prompt，然后经过 tokenizer 变成 `observation.language.tokens` 和 `observation.language.attention_mask`。

因此 TensorRT 后端最直接的输入不是原始 observation，而是 policy 内部预处理后的张量：

```text
images: list[Tensor], 每个 Tensor 形状 [B, 3, 224, 224]
img_masks: list[Tensor], 每个 Tensor 形状 [B]
tokens: Tensor, [B, 200]
masks: Tensor, [B, 200]
noise: Tensor, [B, 50, 32]
```

输出：

```text
actions: Tensor, [B, 50, 32]
```

之后 `PI05Policy.predict_action_chunk(...)` 会截断成真实动作维度：

```text
actions[:, :, :6]
```

再交给 postprocessor 反归一化。

## 推荐的导出边界

### 第一阶段：单体 `sample_actions` 图

先导出完整推理图：

```text
images + img_masks + tokens + masks + noise -> actions
```

其中 `noise` 必须显式作为输入，而不是在 TensorRT 图里随机生成。

原因：

- Torch / ONNX / TRT 对齐必须使用同一份 noise。
- TensorRT 图里放随机数会导致不可复现实验。
- 当前 `PI05Pytorch.sample_actions(...)` 本身支持传入 `noise`。

建议把 list 输入展平成稳定的张量输入名，例如当前 SO101 只有两路相机：

```text
image_0: [B, 3, 224, 224]  # top
image_1: [B, 3, 224, 224]  # wrist
img_mask_0: [B]
img_mask_1: [B]
tokens: [B, 200]
masks: [B, 200]
noise: [B, 50, 32]
```

输出：

```text
actions: [B, 50, 32]
```

导出 wrapper 内部再还原成 LeRobot 模型需要的：

```python
images = [image_0, image_1]
img_masks = [img_mask_0, img_mask_1]
actions = model.sample_actions(images, img_masks, tokens, masks, noise=noise)
```

### 第二阶段：拆分 prefix cache / denoise step

如果单体图导出困难或 engine 太大，再拆成两段：

```text
prefix graph:
  images + img_masks + tokens + masks -> prefix_pad_masks + past_key_values

denoise graph:
  prefix_pad_masks + past_key_values + x_t + timestep -> v_t
```

Python runtime 保留 10 步 denoise loop：

```text
x_t = noise
for step in range(num_inference_steps):
    v_t = denoise_engine(prefix_cache, x_t, timestep)
    x_t = x_t + dt * v_t
actions = x_t
```

这个路线更贴近模型结构，也更适合优化 latency，但开发复杂度明显更高，因为 `past_key_values` 的 TensorRT I/O 数量、shape、dtype 都需要精确管理。

建议优先做第一阶段单体图，除非 ONNX 导出或 TensorRT build 明确卡死。

## 代码组织建议

全部开发都放在：

```text
my_devs/openpi_trt/
```

建议目录：

```text
my_devs/openpi_trt/
  docs/
    lerobot_pi05_tensorrt方案.md
    lerobot_pi05_tensorrt工作报告.md
  reference/
    openpi_on_thor/                 # 可选：后续可把当前上游参考代码移动/复制到 reference
  scripts/
    export_lerobot_pi05_onnx.py
    build_lerobot_pi05_engine.sh
    verify_lerobot_pi05_torch_onnx.py
    verify_lerobot_pi05_torch_trt.py
    run_pi05_trt_infer_so101.py
  runtime/
    trt_engine.py
    trt_pi05_policy.py
    input_capture.py
  artifacts/
    .gitkeep
```

其中 `runtime/trt_pi05_policy.py` 提供一个包装器，不改 `src/lerobot`：

```text
LeRobotPI05TensorRTPolicyWrapper
```

它包装原始 `PI05Policy`，复用：

- 原始 config。
- 原始 `_preprocess_images(...)`。
- 原始 action queue 逻辑。
- 原始 postprocessor。

只替换：

```text
PI05Pytorch.sample_actions(...)
```

或在 wrapper 的 `predict_action_chunk(...)` 中直接调用 TensorRT engine。

## 开发步骤

### Step 1：做输入捕获与 Torch baseline

先写一个离线脚本，从 checkpoint 加载 policy 和 pre/post processors，不连接机器人也能构造一批输入。

输入来源两种：

- 合成输入：随机图像、固定 task、零 state。
- 从真实运行中保存一帧 preprocessor 之后的 batch。

建议保存成：

```text
my_devs/openpi_trt/artifacts/pi05_so101_sample_batch.pt
```

内容包含：

```text
batch["observation.images.top"]
batch["observation.images.wrist"]
batch["observation.language.tokens"]
batch["observation.language.attention_mask"]
golden_noise
torch_actions
```

这一步的目标不是 TensorRT，而是确认我们能稳定复现：

```text
policy.predict_action_chunk(batch)
```

并拿到固定 noise 下的 baseline。

### Step 2：写 ONNX export wrapper

新增：

```text
my_devs/openpi_trt/scripts/export_lerobot_pi05_onnx.py
```

核心逻辑：

1. `PreTrainedConfig.from_pretrained(policy_path)`。
2. `get_policy_class(config.type).from_pretrained(policy_path, strict=False)`。
3. `policy.eval().to("cuda")`。
4. 构造 wrapper：

```python
class PI05SampleActionsONNXWrapper(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, image_0, image_1, img_mask_0, img_mask_1, tokens, masks, noise):
        images = [image_0, image_1]
        img_masks = [img_mask_0, img_mask_1]
        return self.policy.model.sample_actions(images, img_masks, tokens, masks, noise=noise)
```

5. 用 `torch.onnx.export(...)` 导出：

```text
input_names:
  image_0
  image_1
  img_mask_0
  img_mask_1
  tokens
  masks
  noise
output_names:
  actions
```

建议先固定 batch=1，避免一开始引入复杂 dynamic axes。SO101 实时推理本来也是 batch=1。

首版 ONNX 文件：

```text
my_devs/openpi_trt/artifacts/pi05_so101_sample_actions_b1.onnx
```

### Step 3：Torch vs ONNX 数值验证

新增：

```text
my_devs/openpi_trt/scripts/verify_lerobot_pi05_torch_onnx.py
```

固定输入和固定 noise，比较：

```text
torch_actions = policy.model.sample_actions(...)
onnx_actions = onnxruntime.InferenceSession(...).run(...)
```

输出指标：

```text
shape
mean_abs_diff
max_abs_diff
cosine_similarity
前几个 action 值
```

通过标准建议：

- FP32/BF16 导出时，先追求 `max_abs_diff` 在可解释范围内。
- 如果 ONNXRuntime 不支持某些 BF16 op，可以先强制导出 wrapper 使用 FP32/FP16。
- 不要在 Torch vs ONNX 还没过的时候进入 TensorRT。

### Step 4：构建 TensorRT engine

新增：

```text
my_devs/openpi_trt/scripts/build_lerobot_pi05_engine.sh
```

第一版固定 batch=1：

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx="${ONNX_PATH}" \
  --saveEngine="${ENGINE_PATH}" \
  --fp16 \
  --stronglyTyped \
  --verbose \
  --profilingVerbosity=detailed \
  --dumpProfile \
  --dumpLayerInfo
```

如果需要 dynamic shape，再加 profile：

```text
image_0: 1x3x224x224
image_1: 1x3x224x224
img_mask_0: 1
img_mask_1: 1
tokens: 1x200
masks: 1x200
noise: 1x50x32
```

注意：教程里的 `build_engine.sh` 假设三路图像和 `ACTION_HORIZON=15`，不适合直接用于当前 SO101 checkpoint。当前应该使用两路图像和 `chunk_size=50`。

### Step 5：Torch vs TensorRT 验证

新增：

```text
my_devs/openpi_trt/scripts/verify_lerobot_pi05_torch_trt.py
```

复用 `openpi_on_thor/trt_torch.py` 的 runtime 思路，但最好在本目录重写一个更贴合 LeRobot PI0.5 的轻量版：

```text
my_devs/openpi_trt/runtime/trt_engine.py
```

验证逻辑：

1. 加载 policy。
2. 加载 sample batch。
3. 加载 engine。
4. 设置 runtime tensor shape。
5. 调用 engine。
6. 比较 Torch / TRT 输出。

输出同样保存报告：

```text
my_devs/openpi_trt/artifacts/verify_torch_trt_YYYYmmdd_HHMMSS.json
my_devs/openpi_trt/docs/lerobot_pi05_tensorrt工作报告.md
```

### Step 6：接入实时推理

在不改 `my_devs/train/pi/so101/run_pi05_infer.py` 的前提下，建议新增一个镜像入口：

```text
my_devs/openpi_trt/scripts/run_pi05_trt_infer_so101.py
```

它可以复制原脚本的控制流程，但改为加载 TensorRT wrapper：

```text
policy = PI05Policy.from_pretrained(...)
policy = LeRobotPI05TensorRTPolicyWrapper(policy, engine_path)
```

然后继续调用原 LeRobot `predict_action(...)`。

这样做的好处：

- 原 PyTorch 实时脚本保持不动，随时可回退。
- TensorRT 入口完全在 `my_devs/openpi_trt` 下。
- preprocessor/postprocessor/机器人控制流程保持一致，方便做 A/B 测试。

## 运行命令建议

所有命令都用 `lerobot_flex` 环境：

```bash
conda run -n lerobot_flex python my_devs/openpi_trt/scripts/export_lerobot_pi05_onnx.py \
  --policy-path outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
  --output my_devs/openpi_trt/artifacts/pi05_so101_sample_actions_b1.onnx
```

```bash
conda run -n lerobot_flex python my_devs/openpi_trt/scripts/verify_lerobot_pi05_torch_onnx.py \
  --policy-path outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
  --onnx-path my_devs/openpi_trt/artifacts/pi05_so101_sample_actions_b1.onnx
```

TensorRT build 通常必须在目标 TensorRT 环境里跑：

```bash
conda run -n lerobot_flex bash my_devs/openpi_trt/scripts/build_lerobot_pi05_engine.sh \
  my_devs/openpi_trt/artifacts/pi05_so101_sample_actions_b1.onnx \
  my_devs/openpi_trt/artifacts/pi05_so101_sample_actions_b1.engine
```

如果目标设备的 `trtexec` 不在 conda 环境里，脚本内部需要检测并报清楚：

```text
/usr/src/tensorrt/bin/trtexec
trtexec
```

## 主要风险

### 风险 1：ONNX 不支持完整 transformer/cache 图

`PI05Pytorch.sample_actions(...)` 内部包含：

- SigLIP image embedding。
- Gemma language embedding。
- PaliGemma + expert Gemma forward。
- `past_key_values`。
- Python denoise loop。

完整单体图可能遇到 ONNX 导出失败或 graph 过大。

应对：

- 先固定 batch=1、固定 token length=200、固定 chunk_size=50。
- `torch.onnx.export(..., dynamo=False)` 和 `dynamo=True` 都准备尝试。
- 如果单体导出不可行，切换到 prefix graph + denoise graph 两段式。

### 风险 2：BF16 / FP16 精度差异

当前 checkpoint 配置是 `dtype=bfloat16`，TensorRT 路径大概率会用 FP16 或 FP8。

应对：

- 第一个可用版本先不追求 FP8/NVFP4，先做 FP16 engine。
- 固定 noise，先看 action chunk 的整体趋势和 cosine similarity。
- 真机前必须做限幅和短时 dry-run。

### 风险 3：preprocessor 与导出输入不一致

当前 state 被 preprocessor 编入 prompt/token，而不是模型单独输入。如果导出脚本绕过 preprocessor 直接喂 state，结果会错。

应对：

- exporter 和 verifier 都使用 checkpoint 自带的 `policy_preprocessor.json`。
- TensorRT 接入点放在 policy 内部，接收 tokenized 后的 `tokens/masks`。

### 风险 4：动作维度截断/反归一化错位

模型内部输出 `[B, 50, 32]`，真实 SO101 动作是 `[B, 50, 6]`。

应对：

- TensorRT runtime 输出仍保持 `[B, 50, 32]`。
- wrapper 中复用 `PI05Policy.predict_action_chunk(...)` 的截断规则，或明确执行：

```python
actions = actions[:, :, : policy.config.output_features["action"].shape[0]]
```

- postprocessor 继续用 checkpoint 自带的 unnormalizer。

## 验收标准

离线验收：

- 能加载默认 checkpoint。
- 能生成或读取 sample batch。
- Torch baseline 能稳定输出 `[1, 50, 32]`。
- ONNX 导出成功。
- Torch vs ONNX 输出 shape 一致。
- TensorRT engine build 成功。
- Torch vs TRT 输出 shape 一致，并输出误差报告。

真机验收：

- 新增 TRT 实时入口可以加载 policy、processors、engine。
- `--check-policy-load` 类似能力可在不连接机器人时验证加载。
- dry-run 不触碰硬件。
- 真机短时运行可以打印 step latency。
- PyTorch 入口保留，随时回退。

## 推荐优先级

第一版只做这些：

1. `export_lerobot_pi05_onnx.py`
2. `verify_lerobot_pi05_torch_onnx.py`
3. `build_lerobot_pi05_engine.sh`
4. `verify_lerobot_pi05_torch_trt.py`

确认离线链路跑通以后，再做：

5. `trt_pi05_policy.py`
6. `run_pi05_trt_infer_so101.py`

不要一开始直接上真机。PI0.5 输出是连续动作，TensorRT 后端一旦有 token、noise、shape、归一化任意一处错位，真机表现会很难定位。先把固定输入的 Torch / ONNX / TRT 对齐打牢，再接实时控制。

