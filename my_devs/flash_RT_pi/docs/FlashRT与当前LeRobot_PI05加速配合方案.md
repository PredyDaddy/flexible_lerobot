# FlashRT 与当前 LeRobot PI0.5 加速配合方案

日期：2026-06-22  
仓库：`/data/cqy_workspace/flexible_lerobot`  
参考源码：`my_devs/flash_RT_pi/reference_source_code/FlashRT-main`  
目标：在不走 TensorRT/T32D 的前提下，研究如何用 FlashRT 的 VLA 小 batch 实时推理能力，加速当前仓库里的 LeRobot PI0.5 机器人推理链路，并与现有 Laravel/服务化链路配合。

## 0. 前置处理状态

本次先完成两个仓库卫生动作：

1. `FlashRT-main` 解压目录当前没有 `.git/` 目录，因此它已经不是一个会独立连接 GitHub 远端的子仓库。
2. 当前 `.gitignore` 已经忽略：

```text
*.zip
my_devs/flash_RT_pi/reference_source_code/FlashRT-main/
```

也就是说：

```text
my_devs/flash_RT_pi/reference_source_code/FlashRT-main.zip
my_devs/flash_RT_pi/reference_source_code/FlashRT-main/
```

都不会作为当前仓库的提交内容进入 Git。

## 1. 结论先行

当前最合理的 FlashRT 接入方式不是把 `src/lerobot/policies/pi05/modeling_pi05.py` 直接改成 FlashRT，也不是继续沿用 `my_devs/openpi_trt` 里的 split TensorRT patch 方式，而是新增一个隔离的 FlashRT PI0.5 推理后端：

```text
Laravel / Web UI / 调度层
  -> 当前已有 VLA 服务化 API
  -> flashrt_pi05 backend
  -> FlashRT VLAModel.predict(...)
  -> action_chunk
  -> 现有异步 action chunk / robot client / safety / robot processor
```

推荐路线：

```text
Phase 0: 保持 FlashRT 源码只读参考，先不改主包
Phase 1: 在 my_devs/flash_RT_pi/ 做最小 smoke：load_model + predict
Phase 2: 在 vlash_iner server 增加 backend=flashrt_pi05
Phase 3: Laravel 只作为任务、相机帧、状态、动作结果的编排入口
Phase 4: 如需深度融合，再做 LeRobot policy adapter，让上层看起来仍是 PI05Policy
```

核心原则：

- 训练仍然走当前 LeRobot PI0.5。
- 当前已经能跑的 PyTorch 实机命令是黄金基线，后续所有 FlashRT Phase 都必须围绕同一份 `policy-path`、同一组相机、同一个 task 去替换推理后端，而不是换模型、换 processor 或换 robot runtime。
- checkpoint、processor、normalization、robot schema 不在第一阶段迁移。
- FlashRT 只接管推理热点路径，不接管机器人 I/O、安全限幅和服务编排。
- FlashRT 不产生 TensorRT `.engine`，不要把它纳入 `ONNX -> engine` 那套流程。
- 先走服务化旁路，验证延迟、一致性和上机稳定性后，再考虑主包级 adapter。

## 2. 当前 PyTorch 实机基线

你当前已经在用 PyTorch 后端直接跑训练出来的 PI0.5 / SO101 权重，命令是：

```bash
python my_devs/train/pi/so101/run_pi05_infer.py \
      --policy-path outputs/pi05_lora_eraser_cup_multi_task_runs/20260604_212835_full/checkpoints/last/pretrained_model \
      --robot-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00 \
      --top-cam /dev/video4 \
      --wrist-cam /dev/video6 \
      --task "Put the eraser into the small box" \
      --run-time-s 120
```

这条命令在本方案里不是普通示例，而是后续 FlashRT 接入的对照基线。它定义了我们必须保持不变的上层事实：

```text
policy_path:
  outputs/pi05_lora_eraser_cup_multi_task_runs/20260604_212835_full/checkpoints/last/pretrained_model

robot:
  SO101/SO follower
  port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00

cameras:
  top=/dev/video4
  wrist=/dev/video6
  shape in config: observation.images.top=[3,480,640], observation.images.wrist=[3,480,640]

state/action:
  observation.state=[6]
  action=[6]

task:
  Put the eraser into the small box

runtime:
  120 秒真实机器人推理运行
```

`run_pi05_infer.py` 当前做的事情是：

```text
1. 解析 policy_path、robot port、top/wrist camera、task、run_time_s。
2. 校验 checkpoint artifact：
   - config.json
   - adapter_model.safetensors 或 model.safetensors
   - policy_preprocessor.json
   - policy_preprocessor_step_2_normalizer_processor.safetensors
   - policy_postprocessor.json
   - policy_postprocessor_step_0_unnormalizer_processor.safetensors
   - train_config.json
3. 校验 SO101 feature shape：
   - observation.state = [6]
   - observation.images.top = [3,480,640]
   - observation.images.wrist = [3,480,640]
   - action = [6]
4. 如果是 LoRA checkpoint，通过 adapter_config.json 找 base_model_name_or_path：
   - 先加载 base PI0.5 policy
   - 再用 PeftModel.from_pretrained(...) 加载 LoRA adapter
5. 加载 checkpoint 自带的 preprocessor/postprocessor。
6. 连接 SO101 follower 和两路 OpenCV 相机。
7. 在控制循环里执行：
   robot.get_observation()
     -> robot_observation_processor
     -> build_dataset_frame(...)
     -> predict_action(...)
     -> policy preprocessor
     -> PI05Policy.select_action(...)
     -> PI05Policy.predict_action_chunk(...)
     -> PI05Pytorch.sample_actions(...)  # 当前是 PyTorch 后端
     -> policy postprocessor
     -> make_robot_action(...)
     -> robot_action_processor
     -> robot.send_action(...)
8. 循环 120 秒或直到 Ctrl+C。
```

这里最关键的是：当前实机基线已经证明这份 LoRA 权重、checkpoint processor、normalizer/unnormalizer、SO101 robot schema、top/wrist 相机、task 文本和 PyTorch `sample_actions(...)` 是能组成一条完整执行链路的。

FlashRT 后续的目标不是重新训练一份模型，也不是绕过这些 processor，而是把上面链路里的这一段：

```text
PI05Pytorch.sample_actions(...)  # PyTorch
```

逐步替换成：

```text
FlashRT PI0.5 backend
  -> load 同一份 policy_path 对应的权重或转换后的等价权重
  -> 使用同一份 normalizer / prompt / camera / action schema
  -> 输出可被当前 postprocessor 或 robot client 正确消费的 action_chunk
```

因此，后续每个 Phase 的判断标准都应该回到这个问题：

```text
在不改变 policy_path、task、top/wrist camera、SO101 action schema 的前提下，
能不能把当前 PyTorch 推理后端替换成 FlashRT，并保持动作语义、安全检查和实机稳定性？
```

## 3. 当前仓库 PI0.5 推理边界

当前 LeRobot PI0.5 的上层推理链路是：

```text
robot observation
  -> policy preprocessor
  -> PI05Policy.select_action(...)
  -> PI05Policy.predict_action_chunk(...)
  -> PI05Pytorch.sample_actions(...)
  -> policy postprocessor
  -> robot action
```

关键代码位置：

```text
src/lerobot/policies/pi05/modeling_pi05.py
  PI05Policy.select_action(...)
  PI05Policy.predict_action_chunk(...)
  PI05Pytorch.sample_actions(...)

src/lerobot/policies/pi05/processor_pi05.py
  make_pi05_pre_post_processors(...)
  Pi05PrepareStateTokenizerProcessorStep(...)
```

其中 `PI05Pytorch.sample_actions(...)` 是最重的部分。已有 `my_devs/openpi_trt` 的 TensorRT 路线就是围绕它做后端替换：

```text
policy.model.sample_actions = trt_sample_actions
```

但 FlashRT 的工作方式不同：

- 它不是导出 ONNX。
- 它不是编译 TensorRT engine。
- 它直接加载 checkpoint，然后用手写 CUDA kernel、FP8/NVFP4、CUDA Graph capture 执行固定形状推理。
- 它已经有 PI0.5 VLA frontend，包括 RTX、Thor、Orin 相关路径。

因此 FlashRT 最自然的边界不是 patch `sample_actions(...)`，而是直接在 action chunk 服务边界提供一个新的后端：

```text
observation dict + prompt
  -> FlashRT model.predict(images, prompt, state)
  -> np.ndarray[action_horizon, action_dim]
```

这更接近 `my_devs/vla_engineering/vlash_iner` 当前服务化后端的设计。

## 4. FlashRT 能提供什么

根据参考源码的 `README.md`、`USAGE.md`、`flash_rt/api.py`、`docs/architecture.md`、`docs/rtc_lite_design.md`，FlashRT 对本项目最有价值的能力是：

1. **PI0.5 已有前端**

   参考源码中存在：

   ```text
   flash_rt/frontends/torch/pi05_rtx.py
   flash_rt/frontends/torch/pi05_thor.py
   flash_rt/frontends/torch/pi05_thor_fp4.py
   ```

   这说明我们不是从零实现 PI0.5 kernel frontend，而是优先研究如何让当前 checkpoint 和当前 observation schema 适配 FlashRT 已有 PI0.5 API。

2. **小 batch 实时推理定位**

   FlashRT 目标是 small-batch、latency-sensitive workload。PI0.5 机器人控制正好是 batch=1、小 action chunk、低延迟需求。

3. **不走 TensorRT engine**

   FlashRT 的主线是：

   ```text
   safetensors / Orbax checkpoint
     -> FlashRT weight loader
     -> FP8/NVFP4 calibration
     -> CUDA Graph capture
     -> graph replay
   ```

   这和 `my_devs/openpi_trt` 的：

   ```text
   model.safetensors -> PyTorch wrapper -> ONNX -> TensorRT engine
   ```

   是两条不同路线。

4. **稳定公开 API**

   FlashRT 文档里的稳定调用形式是：

   ```python
   import flash_rt

   model = flash_rt.load_model(
       checkpoint="/path/to/pi05_checkpoint",
       config="pi05",
       framework="torch",
       num_views=2,
       autotune=3,
   )

   actions = model.predict(
       images=[base_img, wrist_img],
       prompt="pick up the red block",
       state=state,
   )
   ```

   对 PI0.5，`predict()` 负责把 **normalized state** 编成 OpenPI 风格 prompt token，并返回 action chunk。这里的 state 不能直接使用机器人原始关节值，必须先和当前 checkpoint 的 LeRobot preprocessor / normalizer stats 对齐。

5. **可选 RTC-lite**

   FlashRT 自带 `ActionChunkAdapter`、`CallablePolicyAdapter`、`AsyncChunkRunner` 思路，但它不管理机器人 I/O、不管理相机、不管理安全检查。这个边界和我们当前 `vlash_iner` 的机器人客户端职责是一致的。

## 5. 与 Laravel 的配合方式

这里的 Laravel 不应该直接 import Python 模型，也不应该直接调用 CUDA。Laravel 更适合作为上层业务和控制台：

```text
Laravel
  - 任务创建
  - 机器人/相机配置
  - prompt 和 task_id 管理
  - 实验记录、指标展示
  - 启停推理 session
  - 查看 action chunk、latency、deadline miss、安全停机原因

Python VLA Service
  - 加载 LeRobot/FlashRT 后端
  - 接收 observation
  - 推理 action chunk
  - 输出动作和诊断指标

Robot Client / Executor
  - 采集相机和关节状态
  - 调 Python service
  - 执行动作 chunk
  - 做 safety check 和 robot processor
```

推荐接口分层：

```text
Laravel Web / API
  -> HTTP/WebSocket 控制 Python service session
  -> 不传大模型权重，不跑 CUDA

Python inference service
  -> FastAPI 或现有 vlash_iner server
  -> backend: torch | torch_compile | tensorrt_split | flashrt_pi05

Robot runtime
  -> 仍复用 my_devs/vla_engineering/vlash_iner 或 my_devs/pi05_engineering
```

长期建议新增的服务 API：

```text
POST /sessions
  backend: "flashrt_pi05"
  checkpoint_path: "..."
  robot_type: "so101_follower"
  camera_names: ["top", "wrist"]
  task: "..."
  num_views: 2
  hardware: "auto"
  use_fp8: true
  use_fp4: false

POST /sessions/{id}/infer
  images:
    top: uint8 image
    wrist: uint8 image
  state: float array
  prompt: optional string

返回:
  actions: float array [T, A]
  backend: "flashrt_pi05"
  latency_ms:
    preprocess
    flashrt_predict
    postprocess
    total
  diagnostics:
    first_call
    calibration_cache_hit
    action_shape
    safety_status
```

这属于 Laravel 控制面成熟后的目标接口。第一阶段更贴近当前 `vlash_iner` 代码的做法是：保持一个启动时固定后端的 Python service 进程，扩展现有 `--backend` 分支，继续复用当前 `/health`、`/reset`、`/infer` 协议；等 `flashrt_pi05` 后端通过 smoke/离线/低速上机后，再抽象 session manager。

Laravel 侧只保存 session 配置、调用记录和指标，不保存 GPU 进程内部状态。GPU 进程重启后，由 Python service 重新 `load_model()`、calibrate/capture。

### 为什么不让 Laravel 直接接 FlashRT

不建议 Laravel 直接接 FlashRT，原因：

- FlashRT 是 Python/CUDA 运行时，生命周期、显存、CUDA Graph capture 都应该留在 Python worker。
- Laravel/PHP 进程模型不适合持有 GPU graph 和模型权重。
- 大图像帧和 action chunk 传输需要明确序列化协议，放在 Python service 更容易复用现有 `vlash_iner` 客户端。
- 机器人急停、安全阈值、deadline miss 不应该依赖 Web 请求线程。

Laravel 的正确位置是“控制面”和“记录面”，Python service/robot client 是“数据面”和“执行面”。

## 6. FlashRT 加速当前仓库代码的三种方案

### 方案 A：服务化旁路后端，推荐第一阶段

新增：

```text
my_devs/flash_RT_pi/
  flashrt_pi05_backend.py
  run_flashrt_pi05_smoke.py
  README.md

my_devs/vla_engineering/vlash_iner/server/
  增加 backend=flashrt_pi05
```

后端职责：

```python
class FlashRTPi05Backend:
    def __init__(self, checkpoint_path, num_views=2, hardware="auto", use_fp8=True, use_fp4=False):
        ...

    def predict_action_chunk(self, observation: dict, prompt: str) -> np.ndarray:
        ...
```

输入适配：

```text
当前 observation:
  observation.images.top
  observation.images.wrist
  observation.state              # 机器人原始状态或 LeRobot observation 状态

FlashRT predict:
  images=[top_img_224, wrist_img_224]
  prompt=task
  state=normalized_state         # 必须是和 PI0.5 normalizer 等价的 [-1, 1] 空间
```

这个方案的优点：

- 不改 `src/lerobot`。
- 不影响训练。
- 可以和当前 `vlash_iner` 的 torch、torch_compile、tensorrt_split 后端并列。
- Laravel 只需要选择 `backend=flashrt_pi05`。
- 失败时可以回退到 torch 或 TensorRT split。

需要重点确认：

- 当前 checkpoint 是否能被 FlashRT 直接加载。FlashRT 文档说 torch path 支持 safetensors，但还需要验证 key 命名、state/action 维度、num_views、normalization 是否一致。
- 当前 LeRobot preprocessor 是否输出的图像/state 与 FlashRT `predict()` 期待一致。
- FlashRT `predict()` 返回的 action 是否已经是物理动作空间，还是仍处于 normalized action 空间。必须通过离线对比确认，不能凭 API 名称假设。
- FlashRT 后端必须加载当前 checkpoint 的 normalizer / unnormalizer stats，先把 `observation.state` 转成和 `Pi05PrepareStateTokenizerProcessorStep` 一致的 normalized state，再传给 FlashRT。禁止把 SO101 原始关节值直接传给 `model.predict(..., state=...)`。

### 方案 B：LeRobot policy adapter，中期可选

如果方案 A 验证通过，可以新增一个 LeRobot 风格 wrapper：

```text
my_devs/flash_RT_pi/policy_adapter.py
```

目标是让上层仍然调用：

```text
policy.predict_action_chunk(...)
```

但内部改用 FlashRT：

```text
batch
  -> 提取 images/state/tokens 或 prompt
  -> FlashRT predict(...)
  -> torch.Tensor action chunk
```

注意这个 adapter 不建议直接修改 `PI05Policy`，而是做独立类或 monkey patch 到服务化后端内部。原因是 FlashRT 的 `predict()` 直接吃 image/prompt/state，而当前 `PI05Policy.predict_action_chunk(...)` 已经吃过 LeRobot processor 后的 token/mask/image tensor。两者边界并不完全相同。

### 方案 C：深度移植 FlashRT frontend，不建议第一阶段做

深度移植意味着把 FlashRT 的 PI0.5 frontend、weight spec、kernel 调用和 CUDA Graph capture 融入当前仓库。这会带来大量维护成本：

- CMake/CUDA build 接入。
- pybind `.so` 产物管理。
- FP8/NVFP4 calibration 缓存管理。
- 不同 GPU arch dispatch。
- checkpoint key 映射维护。
- 与当前 LeRobot processor 的职责重叠。

除非后续确认 FlashRT 作为正式主推后端长期维护，否则第一阶段不建议把这些内容搬进 `src/lerobot`。

## 7. 与现有 TensorRT/VLASH 工程的关系

当前已有两条相关工程线：

```text
my_devs/openpi_trt
  -> split TensorRT sample_actions 后端
  -> prefix_cache engine + denoise_step engine

my_devs/vla_engineering/vlash_iner
  -> 当前仓库 PI0.5 的同步/异步/服务化/验收工作区
  -> 已支持 torch、torch_compile、tensorrt_split 后端
```

FlashRT 应该复用 `vlash_iner` 的服务化和上机链路，而不是复制一套机器人 runtime。

推荐后端矩阵：

```text
backend=torch
  作用：正确性基线

backend=torch_compile
  作用：PyTorch compile 基线

backend=tensorrt_split
  作用：已有 TensorRT split 加速基线

backend=flashrt_pi05
  作用：不用 ONNX/TensorRT engine 的 FlashRT 小 batch 实时后端
```

这样 Laravel 或命令行只需要切后端参数，就能做 A/B：

```text
同一个 checkpoint
同一个 prompt
同一组 replay observation
同一个 action safety
对比 torch / tensorrt_split / flashrt_pi05
```

## 8. 推荐实施步骤

所有 Phase 都围绕同一条主线展开：

```text
当前 PyTorch 实机基线：
  run_pi05_infer.py
  --policy-path outputs/pi05_lora_eraser_cup_multi_task_runs/20260604_212835_full/checkpoints/last/pretrained_model
  --top-cam /dev/video4
  --wrist-cam /dev/video6
  --task "Put the eraser into the small box"

最终目标：
  同一份训练权重、同一组相机、同一个 task、同一个 SO101 action schema
  只把 PyTorch sample_actions 推理热点换成 FlashRT 后端
```

### Phase 0：固定 PyTorch 黄金基线

目标：

- 先把当前 PyTorch 实机链路作为可回退、可对比、可验收的黄金基线。
- 后续 FlashRT 的所有结果都和这条命令对齐，而不是换 checkpoint 或换 task。

基线命令：

```bash
python my_devs/train/pi/so101/run_pi05_infer.py \
      --policy-path outputs/pi05_lora_eraser_cup_multi_task_runs/20260604_212835_full/checkpoints/last/pretrained_model \
      --robot-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00 \
      --top-cam /dev/video4 \
      --wrist-cam /dev/video6 \
      --task "Put the eraser into the small box" \
      --run-time-s 120
```

建议在正式替换后端前，先至少保存一次 PyTorch baseline 记录：

```text
policy_path
task
robot port
camera device
run_time_s
平均控制循环耗时
policy 推理耗时，如果脚本后续加计时
是否完成 120 秒
是否触发安全限制或人工中断
机器人任务表现：是否能把 eraser 放进 small box
```

这个 Phase 的产物不是新模型，而是一份对照证据。FlashRT 后续哪怕更快，也必须先证明它在同样输入和同样动作语义下工作。

### Phase 1：环境和源码边界确认

目标：

- FlashRT 参考源码保持只读。
- 不把 FlashRT 解压源码提交到当前仓库。
- 不修改 `src/lerobot`。

检查项：

```bash
conda run -n lerobot_flex git check-ignore -v \
  my_devs/flash_RT_pi/reference_source_code/FlashRT-main/README.md
```

预期：

```text
.gitignore:... my_devs/flash_RT_pi/reference_source_code/FlashRT-main/
```

### Phase 2：最小 FlashRT PI0.5 smoke

新增脚本：

```text
my_devs/flash_RT_pi/run_flashrt_pi05_smoke.py
```

脚本只做：

```text
1. import flash_rt
2. 使用和 PyTorch baseline 相同的 policy_path：
   outputs/pi05_lora_eraser_cup_multi_task_runs/20260604_212835_full/checkpoints/last/pretrained_model
3. flash_rt.load_model(checkpoint, config="pi05", framework="torch", num_views=2)
4. 检查本地 PaliGemma tokenizer 和 checkpoint normalizer / unnormalizer stats
5. 构造或读取一帧 top/wrist/raw_state/prompt
6. raw_state -> 使用当前 checkpoint stats 做 LeRobot 等价归一化 -> normalized_state
7. model.predict(images=[top, wrist], prompt=task, state=normalized_state)
8. 打印 action shape、dtype、horizon、dim、数值范围、首帧 latency、warm latency
```

验收门槛：

```text
actions.ndim == 2
actions.shape[0] > 1
actions.shape[0] 记录为 flashrt_horizon，后续 runtime 参数不得超过它
actions.shape[1] 必须与 checkpoint action feature 对齐，或存在显式验证过的维度映射
无 NaN / Inf
连续 20 次 warm predict 不崩溃
同一 raw_state 经 LeRobot preprocessor 和 FlashRT 后端归一化后的 normalized_state 逐项可解释
```

特别注意：FlashRT PI0.5 参考路径常见返回是 `T=10` 的 action chunk，而当前 LeRobot/openpi_trt 链路常见内部 chunk 是 50 步。第一阶段不能沿用原来 50 步 chunk 的运行参数，必须以实际 `flashrt_horizon` 重算 `n_action_steps`、`actions_per_chunk`、`overlap_steps` 和 `queue_low_watermark`。

这一阶段只回答一个问题：

```text
FlashRT 能不能加载并推理这份当前已经通过 PyTorch 实机验证的 LoRA PI0.5 权重？
```

如果 FlashRT 不能直接加载 LoRA adapter checkpoint，就不能直接跳到机器人。此时要先研究 FlashRT 是否需要：

```text
1. 合并 LoRA 到 base model 后再加载；
2. 适配 adapter_model.safetensors 的 key；
3. 增加只放在 my_devs/flash_RT_pi 下的 checkpoint conversion/adapter；
4. 或改为 FlashRT 支持的等价 full checkpoint。
```

### Phase 3：离线一致性对比

新增对比脚本：

```text
my_devs/flash_RT_pi/compare_lerobot_torch_vs_flashrt.py
```

对同一组 replay observation，对比：

```text
LeRobot torch PI05Policy.predict_action_chunk(...)
FlashRT VLAModel.predict(...)
```

其中 LeRobot torch 侧必须复用 PyTorch baseline 的加载方式：

```text
load_policy(policy_path, policy_cfg)
load_pre_post_processors(policy_path)
task = "Put the eraser into the small box"
top/wrist/state schema = run_pi05_infer.py 当前 schema
```

需要记录：

```text
action_shape
action_horizon
source_action_dim
target_action_dim
action_dim_mapping
mean_abs_diff
max_abs_diff
cosine_similarity
torch_latency_ms
flashrt_first_latency_ms
flashrt_warm_latency_ms
state_normalization_diff
state_prompt_mode
warmed_prompt_lengths
```

注意：如果 FlashRT 使用 FP8/NVFP4，不能要求逐元素完全一致。更合理的是建立任务可接受阈值，并结合短时上机 smoke 判断动作是否平滑、是否触发 safety。

动作维度必须作为 hard gate：不允许因为当前 SO101 action dim 是 6，就把 FlashRT 返回的 7 维或其他维度 action 盲目截断为前 6 维。只有在明确 source/target action 含义、joint 顺序、归一化/反归一化空间，并通过离线 replay、只读、低速真实动作 gate 后，才允许加显式维度映射。

### Phase 4：接入 vlash_iner 服务后端

新增后端：

```text
my_devs/vla_engineering/vlash_iner/server/backends/flashrt_pi05.py
```

或者如果当前 server 还没有 backends 子目录，则先在现有 `run_pi05_async_server.py` 的后端分支里加最小实现，稳定后再抽目录。

接口对齐现有服务：

```text
request:
  observation images/state
  task/prompt

response:
  action_chunk
  latency metrics
  backend diagnostics
```

不要让 FlashRT 后端直接发机器人动作。它只返回 action chunk。

第一阶段实现时，建议贴合当前 `vlash_iner.server.run_pi05_async_server` 的形态：

```text
--backend flashrt_pi05
--policy-path outputs/pi05_lora_eraser_cup_multi_task_runs/20260604_212835_full/checkpoints/last/pretrained_model
--flashrt-checkpoint-path ...
--flashrt-hardware auto
--flashrt-state-prompt-mode fixed|exact
```

服务端仍然是单 active backend、单 `/infer` endpoint。`/sessions` 是后续 Laravel 控制面成熟后的抽象，不作为第一版必须项。

这里的目标仍然不是换 robot client，而是让服务端具备这样的能力：

```text
同一份 policy_path:
  backend=torch          -> 当前 PyTorch 语义基线
  backend=flashrt_pi05   -> FlashRT 后端

同一份 /infer request:
  observation + task

同一种 response:
  action_chunk + diagnostics
```

只有服务化后端对齐后，才考虑把当前单机 `run_pi05_infer.py` 也改造成可选后端版本。例如长期可以演进成：

```bash
python my_devs/train/pi/so101/run_pi05_infer.py \
      --backend flashrt_pi05 \
      --policy-path outputs/pi05_lora_eraser_cup_multi_task_runs/20260604_212835_full/checkpoints/last/pretrained_model \
      --robot-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00 \
      --top-cam /dev/video4 \
      --wrist-cam /dev/video6 \
      --task "Put the eraser into the small box" \
      --run-time-s 120
```

注意：这只是目标形态，当前 `run_pi05_infer.py` 还没有 `--backend` 参数。第一阶段更稳的是先在服务化链路里加 `flashrt_pi05`，等通过 L1/L2/L3 后，再决定是否回填到单机脚本。

### Phase 5：Laravel 控制面接入

Laravel 新增或复用已有配置项：

```text
backend = flashrt_pi05
checkpoint_path = outputs/pi05_lora_eraser_cup_multi_task_runs/20260604_212835_full/checkpoints/last/pretrained_model
robot_id
camera_profile
task
fps
actions_per_chunk
queue_low_watermark
safety thresholds
```

Laravel 发起 session：

```text
POST Python service /sessions
```

Laravel 展示指标：

```text
first_call_latency_ms
warm_predict_latency_ms
action_chunk_horizon
deadline_miss_count
held_action_count
safety_reject_count
backend_error
```

Laravel 不应该：

- 直接加载模型。
- 直接接触 CUDA。
- 直接写机器人串口。
- 在 Web 请求线程里等待长时间 GPU 初始化。

### Phase 6：真实机器人保守验收

沿用当前 `vlash_iner` 和 `pi05_engineering` 已经沉淀的保守上机原则：

```text
1. check-policy-load / check-backend-load
2. camera + state 只读
3. 离线 replay observation
4. dry-run action chunk
5. 短时低 fps 上机
6. 30 秒
7. 120 秒
```

不建议一上来就用高 fps 和低 buffer。参数必须先根据 FlashRT 实际输出 horizon 重算。例如如果 `flashrt_horizon=10`，则 `actions_per_chunk` 不能再写成 12 或 50，`queue_low_watermark` 也必须小于可执行 chunk 长度。先用保守参数区间：

```text
fps=10 或 15
actions_per_chunk<=flashrt_horizon
queue_low_watermark < actions_per_chunk
max_action_delta 开启
```

等确认 no starvation、no safety reject、no deadline miss 后再提升频率。

FlashRT 实机验收的最终对照命令仍然是 PyTorch baseline：

```bash
python my_devs/train/pi/so101/run_pi05_infer.py \
      --policy-path outputs/pi05_lora_eraser_cup_multi_task_runs/20260604_212835_full/checkpoints/last/pretrained_model \
      --robot-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00 \
      --top-cam /dev/video4 \
      --wrist-cam /dev/video6 \
      --task "Put the eraser into the small box" \
      --run-time-s 120
```

FlashRT 后端必须证明：

```text
1. 加载的是同一训练权重或由同一 LoRA checkpoint 合并/转换得到的等价权重。
2. 使用同一 task、同一 top/wrist camera、同一 SO101 state/action schema。
3. 不绕过 checkpoint 的 normalizer/unnormalizer 语义。
4. action chunk 可以被现有 safety 和 robot processor 正确消费。
5. 真实机器人低速验收不比 PyTorch baseline 更危险。
6. 在通过安全验收后，warm latency 或 deadline miss 明显优于 PyTorch baseline。
```

## 9. 关键风险和需要验证的问题

### 9.1 checkpoint 兼容风险

FlashRT 支持 PI0.5 safetensors，但当前仓库的 LeRobot checkpoint 可能存在：

- key remap 差异。
- `time_mlp_*` / `action_time_mlp_*` 命名差异。
- state/action dim 差异。
- processor json 与 FlashRT 内置 prompt 构造差异。
- top/wrist camera 名称和顺序差异。

必须用 Phase 1/2 先验证，不能直接假设可加载。

### 9.2 action 空间风险

当前 LeRobot PI0.5 上层有 preprocessor/postprocessor 和 unnormalizer。FlashRT `predict()` 返回值是否已经 unnormalize，需要实测确认。

判断方式：

```text
同一 observation:
  LeRobot postprocessor 后 action
  FlashRT action
```

对比数值范围和动作维度。如果 FlashRT 返回 normalized action，需要在服务后端补当前 checkpoint 的 postprocessor 或 stats unnormalize。

动作维度也必须严格验证。FlashRT 参考 PI0.5 路线可能默认面向 LIBERO 风格动作维度；当前 SO101 checkpoint 的 action feature 是另一套维度和关节顺序。任何 `7 -> 6`、`32 -> 6`、`10步 -> N步` 的转换都必须有显式映射和验收记录，不能用“取前 N 维”作为默认策略。

### 9.3 horizon 兼容风险

当前 LeRobot PI0.5 / openpi_trt 链路常见内部 action chunk 是 `[B, 50, 32]`，随后上层截到真实 action dim。FlashRT PI0.5 公开示例常见返回是 `[10, 7]`。这会直接影响异步 chunk 调度：

```text
actions_per_chunk
n_action_steps
overlap_steps
queue_low_watermark
miss_policy / hold-last 触发概率
```

因此接入 `vlash_iner` 前必须先记录 FlashRT 实际返回 `T`，并用这个 `T` 重算调度参数。若 `T=10`，第一阶段要围绕 10 步 chunk 做低速保守验收，不能沿用 50 步链路的直觉。

### 9.4 图像预处理风险

当前 LeRobot processor 会处理 image tensor、resize、normalization、device。FlashRT `predict()` 文档示例期待 `(224, 224, 3)` uint8 numpy images。

需要明确：

- top/wrist 原始相机是 `[3,480,640]` 还是 HWC。
- resize 是否由 FlashRT 内部负责。
- 是否需要我们在后端先做 `resize_with_pad` 到 224。
- camera 顺序必须固定为 `top -> wrist`，不能和训练 schema 反。

### 9.5 首次调用和 state prompt graph 风险

FlashRT 首次调用包含 calibration + CUDA Graph capture。README 中描述 first call 约数秒级，warm call 才是实时路径。

服务化时必须设计：

```text
session create
  -> load_model
  -> warmup/calibrate/capture
  -> session ready
```

不要在机器人已经开始执行动作后才触发第一次 `predict()`。

另外 PI0.5 的 state 会进入 prompt。默认 `state_prompt_mode="exact"` 时，不同 state 数值可能带来不同 token length，从而在真实运行中触发新的 graph capture。第一阶段必须二选一：

```text
1. 使用 state_prompt_mode="fixed"，设置能覆盖真实状态 token 长度的 state_prompt_fixed_max_len；
2. 或在启动时用 replay/真实状态样本调用 warm_state_prompt_buckets(...)，预热代表性的 prompt length。
```

服务 diagnostics 里建议暴露：

```text
state_prompt_mode
warmed_prompt_lengths
first_call_latency_ms
new_length_capture_latency_ms
warm_predict_latency_ms
```

### 9.6 多进程和显存风险

FlashRT graph 和权重在 Python 进程内持有。Laravel 多用户或多 session 不能无限创建 GPU worker。

建议第一阶段：

```text
单 GPU
单 active FlashRT session
一个 Python service 进程
显式 unload/restart 策略
```

后续再做 session pool。

## 10. 建议新增文件清单

第一阶段建议新增：

```text
my_devs/flash_RT_pi/README.md
my_devs/flash_RT_pi/run_flashrt_pi05_smoke.py
my_devs/flash_RT_pi/flashrt_pi05_backend.py
my_devs/flash_RT_pi/compare_lerobot_torch_vs_flashrt.py
my_devs/docs/flash_RT_pi/FlashRT与当前LeRobot_PI05加速配合方案.md
```

第二阶段再接服务：

```text
my_devs/vla_engineering/vlash_iner/server/backends/flashrt_pi05.py
my_devs/vla_engineering/vlash_iner/server/run_pi05_backend_acceptance.py
```

如果 Laravel 项目在本仓库内，建议只增加 Laravel 的 session/backend 配置和页面展示，不把 FlashRT Python 代码放进 Laravel 目录。

## 11. 最小代码草图

下面是后续实现时可以参考的最小后端形态。它不是最终代码，只定义边界：

```python
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np


@dataclass
class FlashRTPi05BackendConfig:
    checkpoint_path: str
    num_views: int = 2
    hardware: str = "auto"
    framework: str = "torch"
    autotune: int = 3
    use_fp8: bool = True
    use_fp4: bool = False


class FlashRTPi05Backend:
    def __init__(self, config: FlashRTPi05BackendConfig):
        import flash_rt

        self.config = config
        self.model = flash_rt.load_model(
            checkpoint=config.checkpoint_path,
            config="pi05",
            framework=config.framework,
            num_views=config.num_views,
            hardware=config.hardware,
            autotune=config.autotune,
            use_fp8=config.use_fp8,
            use_fp4=config.use_fp4,
        )

    def predict_action_chunk(self, observation: dict[str, Any], prompt: str) -> tuple[np.ndarray, dict[str, Any]]:
        images = self._extract_images(observation)
        state = self._extract_state(observation)

        start = perf_counter()
        actions = self.model.predict(images=images, prompt=prompt, state=state)
        latency_ms = (perf_counter() - start) * 1000.0

        actions = np.asarray(actions, dtype=np.float32)
        self._validate_actions(actions)
        return actions, {"flashrt_predict_ms": latency_ms, "action_shape": list(actions.shape)}

    def _extract_images(self, observation: dict[str, Any]) -> list[np.ndarray]:
        top = observation["observation.images.top"]
        wrist = observation["observation.images.wrist"]
        return [self._to_hwc_uint8_224(top), self._to_hwc_uint8_224(wrist)]

    def _extract_state(self, observation: dict[str, Any]) -> np.ndarray:
        raw_state = np.asarray(observation["observation.state"], dtype=np.float32)
        # 实现时必须使用当前 checkpoint 的 LeRobot normalizer stats，
        # 得到和 Pi05PrepareStateTokenizerProcessorStep 输入一致的 normalized state。
        normalized_state = self._normalize_state_with_checkpoint_stats(raw_state)
        return normalized_state

    def _normalize_state_with_checkpoint_stats(self, raw_state: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Load checkpoint normalizer stats and match LeRobot PI0.5 preprocessing")

    def _to_hwc_uint8_224(self, image: Any) -> np.ndarray:
        # 实现时必须和当前 LeRobot/FlashRT 图像约定对齐。
        arr = np.asarray(image)
        return arr

    def _validate_actions(self, actions: np.ndarray) -> None:
        if actions.ndim != 2:
            raise ValueError(f"Expected [T, A] actions, got {actions.shape}")
        if not np.isfinite(actions).all():
            raise ValueError("FlashRT returned NaN or Inf actions")
```

## 12. 当前建议

我建议下一步不要直接改主框架，而是先完成：

```text
1. 安装/构建 FlashRT 到 lerobot_flex 环境
2. 写 run_flashrt_pi05_smoke.py
3. 用当前 PI0.5 checkpoint 跑 load_model + predict
4. 做 torch vs flashrt 离线 action chunk 对比
5. 通过后再把 flashrt_pi05 接入 vlash_iner server
6. Laravel 只加 backend/session 控制项
```

如果 Phase 1 发现当前 checkpoint 不能直接被 FlashRT 加载，再进入第二条路线：研究 FlashRT 的 `WEIGHT_SPEC` 和当前 LeRobot `model.safetensors` key 差异，做一个只放在 `my_devs/flash_RT_pi` 下的 checkpoint adapter，而不是修改 `src/lerobot`。
