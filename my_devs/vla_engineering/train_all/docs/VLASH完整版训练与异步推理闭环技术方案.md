# VLASH 完整版训练与异步推理闭环技术方案

## 1. 背景与目标

当前 `my_devs/vla_engineering` 已经完成了两类重要探索：

1. `my_devs/vla_engineering/train`
   - 用 `vlash-main` 的训练体系完成了 PI0.5 + SO101 数据集上的 VLASH LoRA 微调。
   - 产出了 `pi05_so101_vlash_lora_10epochs_r192_bs1_accum8` 等 checkpoint。
   - 提供了 `serve_vlash_policy.py` / `run_vlash_remote_client.py` 这类服务端和客户端脚本，用于验证 VLASH 训练后的 checkpoint 是否能在真实机器人链路中产生 action chunk。

2. `my_devs/vla_engineering/vlash_iner`
   - 面向当前 LeRobot PI0.5 checkpoint，做了隔离的本地同步、本地异步、服务化异步推理工程。
   - 引入了 `AsyncChunkManager`、`background_inference`、`inference_overlap_steps`、`future_state_aware`、`chunk_blend_steps` 等运行时机制。
   - 增加了真实机器人安全保护、只读验收、运行报告、服务端健康检查，以及 torch / torch_compile / TensorRT split / TensorRT pure 等后端方向。

这两条工作都很有价值，但它们目前没有组成一条真正完整的 VLASH 闭环。也就是说，现阶段我们还没有做到：

```text
VLASH delay-aware 训练
  -> VLASH checkpoint 标准产物
  -> 原生 async runtime 加载
  -> future-state-aware / overlap / latency-aware 执行
  -> compile / fusion / TensorRT 加速
  -> SO101 上机验收
  -> 指标化调参和报告
```

因此，`my_devs/vla_engineering/train_all` 的目标是：在一个新的隔离目录下，重新整理并承载完整的 VLASH 训练、异步推理、加速优化和真实机器人验收闭环。

本目录必须避免继续污染或耦合已有历史模块：

```text
my_devs/vla_engineering/train_all/
```

后续所有新开发、文档、脚本、运行报告和验收记录，优先放在这个目录下。已有 `train`、`vlash_iner`、`vlash-main` 只作为参考来源，不再作为新主线继续堆叠逻辑。

## 2. 已有训练模块回顾

### 2.1 已完成内容

已有训练模块主要位于：

```text
my_devs/vla_engineering/train/
```

核心文件包括：

```text
train_vlash.sh
run_vlash_pi05_lora_smoke.sh
pi05_so101_vlash_lora_10epochs_r192_bs1_accum8.yaml
pi05_so101_vlash_lora_smoke.yaml
smoke_vlash_pi05_checkpoint.py
serve_vlash_policy.py
serve_vlash_policy.sh
run_vlash_remote_client.py
run_vlash_remote_client.sh
```

训练侧已经做到：

- 使用 `vlash-main` 的训练入口完成 PI0.5 LoRA 微调。
- 使用 SO101 任务数据集 `desk_cleanup_v1/eraser_cup_multi_task`。
- 支持 LoRA adapter、额外可训练模块、gradient accumulation、checkpoint 保存。
- 引入了 VLASH 的 temporal delay augmentation 配置，例如：

```yaml
max_delay_steps: 1
shared_observation: false
```

这说明训练阶段已经开始尝试让模型适应异步推理中的 stale observation / delayed action chunk 问题。

训练产物示例：

```text
my_devs/vla_engineering/train/outputs/
  pi05_so101_vlash_lora_10epochs_r192_bs1_accum8/
    checkpoints/033275/pretrained_model/
      config.json
      model.safetensors
      train_config.json
      lora_adapters/
        adapter_config.json
        adapter_model.safetensors
```

### 2.2 训练模块的当前缺陷

训练模块最大的问题不是“不能训练”，而是训练产物没有和完整版运行时形成稳定契约。

当前训练产物缺少 `vlash_iner` 异步服务端所期待的 LeRobot policy bundle 文件：

```text
policy_preprocessor.json
policy_preprocessor_step_2_normalizer_processor.safetensors
policy_postprocessor.json
policy_postprocessor_step_0_unnormalizer_processor.safetensors
```

这导致一个直接后果：

```text
VLASH LoRA checkpoint 可以被 train/serve_vlash_policy.py 加载，
但不能直接被 vlash_iner.server.run_pi05_async_server 加载。
```

也就是说，训练模块产出的 checkpoint 和异步运行时模块之间没有真正打通。

此外，当前训练配置中的 `max_delay_steps=1` 比较保守。它可以作为 smoke 或小幅延迟增强，但无法充分覆盖真实异步部署中常见的 overlap 设置，例如：

```text
n_action_steps = 50
inference_overlap_steps = 13~15
control_fps = 30~45
```

如果训练只让模型见过 1 步 delay，而推理时让模型在 10 多步 overlap 场景下接续新 chunk，那么模型未必真正学会了这种异步分布。这样运行时只能靠 future-state approximation 和 chunk blending 去补救，无法从模型本身解决 chunk 边界不连续。

训练模块当前还没有形成系统化的 delay 实验矩阵，例如：

```text
max_delay_steps = 0
max_delay_steps = 4
max_delay_steps = 8
max_delay_steps = 12
max_delay_steps = 15
```

也没有把 `shared_observation=true` 的高效 delay 训练路线纳入主线验收。

## 3. 已有推理模块回顾

### 3.1 `train` 下的远程推理验证

`train` 目录下已经有一组远程推理脚本：

```text
serve_vlash_policy.py
serve_vlash_policy.sh
run_vlash_remote_client.py
run_vlash_remote_client.sh
```

这组脚本的核心行为是：

```text
客户端采集 observation
  -> POST /infer 给服务端
  -> 服务端调用 policy.predict_action_chunk(...)
  -> 返回完整 action chunk
  -> 客户端顺序执行前 action_chunk_steps 步
  -> 再请求下一段 chunk
```

这条链路的价值是验证 VLASH 训练后的 checkpoint 是否能在真实 SO101 机器人任务中产生有效动作。它对于回答“VLASH 训练是否有帮助”是有意义的。

但它不是完整版 VLASH async runtime。当前客户端是在执行完整 chunk 后才请求下一段，整体行为更接近：

```text
blocking remote chunk inference
```

而不是：

```text
execute current chunk while prefetching next chunk
```

因此，这条链路无法完整验证 VLASH 异步推理优化的价值。

### 3.2 `vlash_iner` 下的异步推理工程

`vlash_iner` 已经实现了一套更接近工程化部署的异步运行时。

核心能力包括：

- `AsyncChunkManager`
- `background_inference`
- `inference_overlap_steps`
- `future_state_aware`
- `chunk_blend_steps`
- `reuse_observation_within_chunk`
- 服务端 `/health`、`/reset`、`/infer`
- 只读和真机客户端
- 运行报告 JSON
- wait_count、request_latency、server_infer、switch_delta 等指标
- torch / torch_compile / TensorRT 后端探索

这些能力对于“丝滑”上机非常重要。它们解决的是：

```text
推理耗时阻塞控制循环
chunk 之间等待
chunk 切换突然跳变
机器人动作不稳定
```

已有调参记录也说明，较好的配置通常类似：

```text
n_action_steps = 50
inference_overlap_steps = 15
background_inference = true
future_state_aware = true
chunk_blend_steps = 4
reuse_observation_within_chunk = true
```

这说明异步运行时方向是正确的。

### 3.3 推理模块的当前缺陷

`vlash_iner` 的最大缺陷是：它不是为当前 VLASH LoRA checkpoint 原生设计的运行时闭环。

它最初定位是：

```text
面向当前 LeRobot PI0.5 checkpoint 的隔离推理工程
```

因此，它期待 checkpoint 目录中存在 LeRobot policy preprocessor/postprocessor artifact。当前 `train` 目录下的 VLASH LoRA checkpoint 不满足这个契约。

这造成了两条路线分裂：

```text
VLASH checkpoint
  -> 可以走 train/serve_vlash_policy.py
  -> 但这条是同步 remote chunk 验证链路

LeRobot PI0.5 bundle
  -> 可以走 vlash_iner async runtime
  -> 但这条不能直接验证 VLASH delay-aware 训练产物
```

所以当前推理模块无法真正验证：

```text
VLASH 训练 + VLASH async runtime 是否共同提升丝滑程度
```

另外，`vlash_iner` 虽然提供 `future_state_aware`，但它目前只是把当前 chunk 的末尾 action 近似当成下一次推理的 state。这个近似在 SO101 上能跑，是因为 state/action 都是 6 维，但它并不等价于完整的 delay-aware 或 RTC 机制。

当前还缺少：

- 显式的 `inference_delay` 估计。
- `prev_chunk_left_over` / leftover alignment。
- 基于真实机器人 state 和执行误差的 future state 估计。
- 真正的 RTC 或类 RTC chunk continuity correction。
- action_quant_ratio 的完整执行语义。
- VLASH checkpoint 原生加载和 processor/statistics 桥接。

因此，`vlash_iner` 当前更像一个优秀的异步推理工程骨架，而不是完整版 VLASH 部署栈。

## 4. 为什么需要 `train_all`

新目录 `train_all` 应该解决一个核心问题：

```text
不要再让训练、checkpoint、服务端、异步客户端、加速后端分裂在不同历史目录里。
```

它应该成为完整 VLASH 闭环的新主线：

```text
my_devs/vla_engineering/train_all
  -> 训练配置
  -> delay-aware 数据策略
  -> checkpoint 产物契约
  -> 原生 policy loader
  -> async server
  -> async robot client
  -> 加速后端
  -> 只读验收
  -> 真机验收
  -> 调参报告
```

`train_all` 的原则是：

1. 新开发只放在 `my_devs/vla_engineering/train_all` 下。
2. 不修改 `my_devs/vla_engineering/train`。
3. 不修改 `my_devs/vla_engineering/vlash_iner`。
4. 不直接改 `my_devs/vla_engineering/vlash-main`，除非后续明确需要同步参考实现。
5. 可以读取和参考已有模块，但新主线代码和文档必须自洽。

### 4.1 数据、环境与任务基线

`train_all` 第一阶段不是重新选择数据、任务和环境，而是必须和已有 `my_devs/vla_engineering/train` 的训练基线保持一致。这样才能保证后续对比的是“完整 VLASH 闭环是否更好”，而不是因为换了数据、换了环境或换了依赖导致结果不可比。

第一阶段固定使用当前已经验证过的 SO101 桌面清理多任务数据集：

```text
dataset.repo_id: desk_cleanup_v1/eraser_cup_multi_task
dataset.root: /data/cqy_workspace/flexible_lerobot/datasets/desk_cleanup_v1/eraser_cup_multi_task
dataset.video_backend: pyav
dataset.use_imagenet_stats: false
```

这里继续使用 `pyav`，原因与旧 `train` 保持一致：当前历史训练链路中已经验证 `pyav` 可用，而 `lerobot_flex` 里的 `torchcodec` 曾存在 FFmpeg 动态库问题。`train_all` 不应在第一阶段更换视频后端，否则训练速度、解码行为和错误来源都会变成新的变量。

第一阶段固定使用当前本地 PI0.5 基座权重：

```text
policy.type: pi05
policy.pretrained_path: /data/cqy_workspace/flexible_lerobot/assets/modelscope/lerobot/pi05_base
policy.dtype: bfloat16
policy.device: cuda
policy.state_cond: true
policy.gradient_checkpointing: true
policy.normalization_mapping:
  VISUAL: IDENTITY
  STATE: MEAN_STD
  ACTION: MEAN_STD
```

输入输出 schema 与已有训练产物保持一致：

```text
observation.images.top:   [3, 480, 640]
observation.images.wrist: [3, 480, 640]
observation.state:        [6]
action:                   [6]
```

真实机器人默认硬件配置也保持一致：

```text
robot.type: so101_follower
robot.id: hfy_follower
robot.port: /dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00
calibration_dir: /home/cqy/.cache/huggingface/lerobot/calibration/robots/so_follower
top camera: /dev/video4, 640x480, 30fps, YUYV
wrist camera: /dev/video6, 640x480, 30fps, MJPG
```

第一阶段固定任务文本至少包括当前已在历史脚本中使用过的三类任务：

```text
eraser_to_box:
  Put the eraser into the small box

cup_to_upper_right:
  Move the cup back to the upper-right corner

eraser_then_cup:
  First put the eraser into the small box, then move the cup back to the upper-right corner
```

其中当前上机和单任务验证的主任务是：

```text
Put the eraser into the small box
```

### 4.2 训练环境基线

`train_all` 第一阶段的模型训练环境应与已有 `my_devs/vla_engineering/train` 保持一致：

```text
conda env: vlash_train
python: 3.10
```

旧训练环境由 `my_devs/vla_engineering/train/env_setup.sh` 创建，核心安装逻辑是：

```bash
conda create -y -n vlash_train python=3.10
conda install -y -c conda-forge ffmpeg=7.1.1

python -m pip install -e "/data/cqy_workspace/flexible_lerobot[feetech,smolvla]"
python -m pip install -e "/data/cqy_workspace/flexible_lerobot/my_devs/vla_engineering/vlash-main" --no-deps

python -m pip install \
  "accelerate>=1.12.0" \
  "transformers==4.53.3" \
  "peft==0.18.1" \
  "bitsandbytes==0.48.2" \
  "termcolor" \
  "torchcodec"
```

训练和 VLASH policy 服务端默认使用 `vlash_train`，机器人客户端仍然使用 `lerobot_flex`。这与旧 `train` 的真实机器人推理约定一致：

```text
VLASH 模型训练:      vlash_train
VLASH 模型服务端:    vlash_train
SO101 机器人客户端:  lerobot_flex
```

第一阶段不建议把所有东西强行合并到一个环境里。当前目标是先保证与历史训练链路可比，并把完整版闭环补齐。后续如果要把训练、服务端和机器人客户端统一迁移到 `lerobot_flex`，应该作为单独的环境迁移任务，并单独做依赖、性能和上机回归验证。

训练和服务端运行时继续使用离线模式变量：

```bash
export PYTHONPATH=/data/cqy_workspace/flexible_lerobot/my_devs/vla_engineering/vlash-main:/data/cqy_workspace/flexible_lerobot/src:${PYTHONPATH}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false
```

### 4.3 训练超参基线

`train_all` 第一阶段应该复刻已有主训练配置，再在 delay 维度上做受控实验。当前历史主配置是：

```text
config: my_devs/vla_engineering/train/pi05_so101_vlash_lora_10epochs_r192_bs1_accum8.yaml
output_dir: my_devs/vla_engineering/train/outputs/pi05_so101_vlash_lora_10epochs_r192_bs1_accum8
job_name: pi05_so101_vlash_lora_10epochs_r192_bs1_accum8
batch_size: 1
grad_accum_steps: 8
steps: 66550
num_workers: 2
seed: 1000
save_freq: 6655
log_freq: 50
wandb.enable: false
```

优化器和 scheduler 基线：

```text
optimizer: adamw
lr: 5.0e-05
betas: [0.9, 0.95]
eps: 1.0e-08
weight_decay: 1.0e-10
grad_clip_norm: 1.0

scheduler: cosine_decay_with_warmup
num_warmup_steps: 1000
peak_lr: 5.0e-05
decay_lr: 2.5e-06
num_decay_steps: 66550
```

LoRA 基线：

```text
lora.enable: true
lora.backend: peft
lora.r: 192
lora.alpha: 192
lora.dropout: 0.05
```

额外可训练模块继续沿用：

```text
action_in_proj
action_out_proj
time_mlp_in
time_mlp_out
state_proj
state_mlp_in
state_mlp_out
input_layernorm
post_attention_layernorm
```

LoRA target modules 继续沿用：

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
out_proj
fc1
fc2
```

历史训练中的 delay baseline 是：

```text
max_delay_steps: 1
shared_observation: false
```

`train_all` 后续做 delay 矩阵时，应以这个配置作为可复现 baseline，然后只改变 delay 相关变量，例如：

```text
delay0:  max_delay_steps = 0,  shared_observation = false
delay1:  max_delay_steps = 1,  shared_observation = false
delay4:  max_delay_steps = 4,  shared_observation = false
delay8:  max_delay_steps = 8,  shared_observation = false
delay12: max_delay_steps = 12, shared_observation = false
delay15: max_delay_steps = 15, shared_observation = false
```

在这些基线跑通后，再单独评估：

```text
shared_observation = true
```

这样可以保证实验结论只对应 delay-aware 训练和 async runtime 本身，而不是被数据、环境、batch、LoRA rank 或 optimizer 改动污染。

## 5. 完整版闭环设计

### 5.1 总体链路

目标链路应该是：

```text
delay-aware VLASH training
  -> runtime-ready checkpoint
  -> checkpoint audit
  -> async inference server
  -> async robot client
  -> latency and continuity metrics
  -> acceleration backend
  -> real robot acceptance
```

更具体地说：

```text
configs/train/*.yaml
  -> scripts/train_delay_lora.sh
  -> outputs/*/checkpoints/*/pretrained_model
  -> scripts/audit_checkpoint.py
  -> server/serve_delay_policy.py
  -> client/run_async_robot_client.py
  -> reports/*.json
  -> docs/调参记录.md
```

### 5.2 建议目录结构

建议后续将 `train_all` 组织为：

```text
my_devs/vla_engineering/train_all/
  README.md

  configs/
    train/
      pi05_so101_delay_lora_smoke.yaml
      pi05_so101_delay_lora_r192.yaml
    runtime/
      pi05_so101_async_torch.yaml
      pi05_so101_async_compile.yaml
      pi05_so101_async_trt.yaml

  scripts/
    train_delay_lora.sh
    smoke_checkpoint.py
    audit_checkpoint.py
    export_runtime_bundle.py
    benchmark_latency.py

  runtime/
    async_manager.py
    policy_loader.py
    observation_builder.py
    action_safety.py
    metrics.py

  server/
    serve_delay_policy.py

  client/
    run_async_robot_client.py
    run_readonly_client.py
    run_mock_client.py

  docs/
    VLASH完整版训练与异步推理闭环技术方案.md
    运行命令.md
    验收标准.md
    调参记录.md
    工作报告.md

  reports/
    smoke/
    readonly/
    robot/
    latency/

  outputs/
```

### 5.3 Checkpoint 产物契约

`train_all` 必须先定义 runtime-ready checkpoint contract。

一个合格的产物目录至少应该能回答：

```text
policy type 是什么？
chunk_size 是多少？
n_action_steps 是多少？
state/action/image feature schema 是什么？
normalization stats 在哪里？
LoRA adapter 是否存在？
adapter 是否已经 merge？
运行时是否需要 policy_preprocessor/postprocessor？
能否直接 mock observation -> action_chunk？
```

推荐在 `scripts/audit_checkpoint.py` 中做强检查。

目标不是简单检查文件存在，而是确保：

```text
训练产物可以被 train_all/server/serve_delay_policy.py 原生加载。
```

这样后续不再出现“训练能产出，但异步 server 加载不了”的问题。

### 5.4 训练策略

训练侧需要从单点配置升级为 delay 实验矩阵。

建议至少保留以下实验：

```text
delay0:
  max_delay_steps = 0
  用作普通 PI0.5/VLASH LoRA baseline

delay4:
  max_delay_steps = 4
  小延迟增强

delay8:
  max_delay_steps = 8
  中等延迟增强

delay12:
  max_delay_steps = 12
  接近真实 async overlap 区间

delay15:
  max_delay_steps = 15
  对齐 overlap15 的激进配置
```

训练时不要假设 delay 越大越好。delay 太大可能让模型过度预测未来，导致对当前观测响应变慢。

建议第一阶段：

```text
max_delay_steps = 4 或 8
shared_observation = false
```

跑通以后再评估：

```text
max_delay_steps = 12 或 15
shared_observation = true
```

如果要真正利用 VLASH 的高效训练思想，后续需要把 `shared_observation` 纳入主线实验，而不是只保留在参考文档里。

### 5.5 异步运行时策略

`train_all` 的 async runtime 需要以 VLASH 思想为基础，但要比当前零散实现更明确。

核心参数：

```text
n_action_steps
inference_overlap_steps
background_inference
future_state_aware
chunk_blend_steps
reuse_observation_within_chunk
control_fps
late_chunk_policy
```

其中 `late_chunk_policy` 建议显式定义，用来处理下一段 chunk 没有按时返回的情况：

```text
wait:
  等下一段 chunk，最简单，但可能造成停顿。

hold_last:
  继续保持上一帧 action，动作可能更连续，但需要安全限制。

slowdown:
  降低动作发送频率，为推理争取时间。

safe_stop:
  超过阈值后进入安全停止。
```

当前 `vlash_iner` 主要依赖 `wait_count` 观察是否等待，但没有把 late chunk 策略上升为一等配置。`train_all` 中应该明确这件事。

### 5.6 Future State 与 Delay 对齐

当前 future-state-aware 的简化做法是：

```text
future_state = current_chunk[-1]
```

即把当前 chunk 最后一个 action 当成下一次推理的 state。

在 `train_all` 中，应该逐步比较不同 future state 模式：

```text
none:
  不使用 future state，只用当前观测。

chunk_final_action:
  使用当前 chunk 的最后一个 action。

chunk_overlap_action:
  使用当前 chunk 在预计服务端返回时刻附近的 action。

measured_state_plus_delta:
  使用真实机器人当前 state 加预测 delta。
```

最终目标不是只靠 `chunk_final_action` 这个近似，而是让训练 delay 分布、运行时 overlap、future state 估计三者一致。

### 5.7 RTC / Leftover Alignment

完整版不能只停留在 overlap + blend。

后续需要评估是否接入：

```text
prev_chunk_left_over
inference_delay
execution_horizon
leftover alignment
RTC / Real-Time Chunking
```

这类机制的目标是让新 chunk 在生成阶段就知道旧 chunk 已经执行了多少、还剩多少，而不是在生成后用 blend 去糊边界。

当前 `chunk_blend_steps` 仍然有价值，但它应该是安全补偿，不应该成为唯一的连续性手段。

### 5.8 加速优化策略

加速优化应分阶段接入。

第一阶段优先：

```text
bfloat16
num_inference_steps 可控
torch.compile
compile warmup
fuse_qkv
fuse_gate_up
server latency profiling
```

第二阶段再接：

```text
TensorRT split
prefix cache
CUDA graph
pure TensorRT
```

原因是当前首要问题不是极限速度，而是训练产物、运行时语义、chunk 连续性没有闭环。只有 torch backend 的主线闭环稳定以后，TensorRT 才应该作为替换后端进入。

## 6. 验收标准

### 6.1 L0: Checkpoint 验收

目标：证明训练产物可以被 `train_all` 原生加载。

验收项：

```text
能加载 config.json
能加载 model.safetensors
能识别 LoRA adapter
能解析 feature schema
能解析 normalization stats
能 mock observation -> action_chunk
action_chunk shape 正确
无 NaN / Inf
```

### 6.2 L1: 服务端推理验收

目标：证明 async server 可稳定服务。

验收项：

```text
/health 正常
/reset 正常
/infer 正常
连续 100 次 mock infer 无异常
server_infer_s p50 / p95 / max 有报告
输出 action chunk shape = [n_action_steps, action_dim]
```

### 6.3 L2: 只读机器人验收

目标：证明真实 observation 到 action chunk 的链路稳定，但不发动作。

验收项：

```text
连接 SO101 和相机
读取真实 observation
发送到 server
返回 action chunk
运行 60s
request_latency_s 有报告
server_infer_s 有报告
wait_count = 0 或在可接受范围
observed_hz 接近目标 control_fps
```

### 6.4 L3: 真机执行验收

目标：证明 async runtime 可以真实执行任务。

验收项：

```text
运行 60s 或 120s
没有明显卡顿
没有明显 chunk 边界回退
wait_count = 0
switch_delta 可控
direction_flip_count 有记录
任务是否成功有人工标注
保留 output_json 和视频/观察记录
```

### 6.5 L4: 对比验收

目标：回答每个技术点是否真的有收益。

对比组：

```text
普通 PI0.5 / 非 delay checkpoint
VLASH LoRA sync remote chunk
VLASH delay async torch
VLASH delay async torch_compile
VLASH delay async fusion
VLASH delay async TensorRT
```

最终需要分别回答：

```text
delay training 是否提升？
async runtime 是否提升？
future-state-aware 是否提升？
chunk blend 是否只是掩盖问题？
compile/fusion 是否降低延迟且不伤效果？
TensorRT 是否值得进入主线？
```

## 7. 推荐实施阶段

### 阶段 0: 文档与边界

当前阶段只做文档和边界定义。

目标：

```text
明确 train_all 是新主线
明确旧 train / vlash_iner 的问题
明确完整闭环的模块划分
明确验收标准
```

### 阶段 1: Checkpoint Loader

目标：

```text
让 train_all 原生加载当前 VLASH LoRA checkpoint
支持 mock observation -> action_chunk
解决 processor/statistics/artifact contract
```

### 阶段 2: 同步服务端基线

目标：

```text
在 train_all 中复刻 train/serve_vlash_policy.py 的能力
形成自己的同步 server/client baseline
确保不会依赖旧 train 目录
```

### 阶段 3: Async Runtime

目标：

```text
接入 AsyncChunkManager
支持 background_inference
支持 overlap
支持 future_state_aware
支持 chunk_blend_steps
支持 readonly 和 real robot 两种客户端
```

### 阶段 4: Delay 训练矩阵

目标：

```text
训练 delay0 / delay4 / delay8 / delay12 / delay15
对比 sync 和 async 效果
找到最适合 SO101 当前任务的 delay 分布
```

### 阶段 5: 加速优化

目标：

```text
接入 torch.compile + warmup
接入 fuse_qkv / fuse_gate_up
记录服务端延迟和真机效果
确认加速不会破坏动作质量
```

### 阶段 6: 连续性增强

目标：

```text
显式测量 inference_delay
引入 leftover alignment
评估 RTC 或类 RTC chunk continuity correction
减少对 chunk_blend_steps 的依赖
```

### 阶段 7: TensorRT 后端

目标：

```text
在 torch 主线稳定后接 TensorRT split / pure
保持同一套 server/client/runtime API
用报告证明 TensorRT 的收益
```

## 8. 结论

当前 `train` 和 `vlash_iner` 已经分别证明了两个方向：

```text
train:
  VLASH LoRA 训练可以产出可推理 checkpoint。

vlash_iner:
  异步运行时可以改善等待和执行节奏。
```

但这两个方向现在还没有形成真正的完整版：

```text
VLASH 训练产物不能直接进入 VLASH-style async runtime；
async runtime 也没有完整利用 VLASH delay-aware 训练的产物。
```

`train_all` 的价值就是把它们重新合成一条主线：

```text
训练时就考虑 delay；
产物格式直接服务 runtime；
runtime 原生异步、可观测、可验收；
加速后端作为同一 API 下的替换实现；
最终用真实机器人指标判断是否真的更丝滑。
```

后续所有新增开发都应该围绕这条闭环展开，避免继续在历史目录中追加分散脚本。
