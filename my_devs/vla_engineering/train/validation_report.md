# VLASH PI0.5/PI0 LoRA smoke 验证报告

## 目标

验证当前仓库中 `my_devs/vla_engineering/vlash-main` 是否能在隔离目录 `my_devs/vla_engineering/train/` 内完成：

- 基于本地 SO101 LeRobot 数据集的数据读取与 temporal delay augmentation。
- 基于本地 `pi05_base` 的 PI0.5 LoRA fine-tuning smoke。
- 基于本地 `pi05_base` 与 `pi0_base` 的 12GiB 训练计算显存探针。
- 训练后 checkpoint 的最小推理加载与 action chunk 生成。
- 真实机器人 async inference 配置脚本准备。

## 输入资产

- 数据集：`datasets/desk_cleanup_v1/eraser_cup_multi_task`
- PI0.5 基座权重：`assets/modelscope/lerobot/pi05_base`
- PI0 基座权重：`assets/modelscope/lerobot/pi0_base`
- tokenizer：`assets/modelscope/google/paligemma-3b-pt-224`，通过 `google/paligemma-3b-pt-224` symlink 离线加载。

## 结论

当前修补后的 `vlash-main` 可以在本地 SO101 数据集上跑通 PI0.5 LoRA smoke fine-tuning，并能保存 inference-ready checkpoint 后重新加载生成动作块。

对 README 中的“LoRA fine-tuning for π0.5/π0 under 12G GPU memory”结论：在本机 `vlash_train` 环境、`batch_size=1`、`r=4`、`bf16`、`gradient_checkpointing=true`、`fuse_qkv=false`、`fuse_gate_up=false` 条件下，PI0.5 与 PI0 的单 batch forward/backward 训练计算峰值均低于 12GiB：

- PI0.5 LoRA：`max_allocated=10.84GiB`，`max_reserved=11.09GiB`。
- PI0 LoRA：`max_allocated=9.91GiB`，`max_reserved=10.21GiB`。

完整端到端 smoke 脚本在 checkpoint 保存阶段会执行 LoRA merge、写出约 7.0G 的 `model.safetensors`，之后还会重新加载 checkpoint 做推理探针；这些阶段的整卡 `nvidia-smi` 采样不等价于训练 step 的 12GiB 显存口径。

## 本次新增/修改

- `my_devs/vla_engineering/vlash-main/vlash/configs/run_config.py`
  - 兼容当前仓库的机器人模块路径：`so_follower`、`bi_so_follower`。
- `my_devs/vla_engineering/vlash-main/vlash/policies/factory.py`
  - 让预训练 PI0.5/PI0 配置按当前数据集覆盖输入/输出 features，匹配 SO101 的 `top`/`wrist` 双相机。
- `my_devs/vla_engineering/vlash-main/pyproject.toml`
  - 改为 setuptools 自动发现 `vlash*` 子包，避免独立环境安装后丢失 `vlash.policies`、`vlash.lora`、`vlash.layers`。
- `my_devs/vla_engineering/vlash-main/vlash/policies/gemma_compat.py`
  - 新增 Gemma RMSNorm 兼容层，支持当前 Transformers 中缺失 `cond_dim`/adaRMS 变体的问题。
- `my_devs/vla_engineering/vlash-main/vlash/policies/pi05/modeling_pi05.py`
  - 兼容当前 Transformers 缺失 `_gated_residual` 的情况。
  - 为 PI0.5 action expert 接入 adaRMS RMSNorm 兼容层。
  - 修复 bf16 视觉塔、LoRA 投影输入、MLP/attention 投影的 dtype 对齐。
- `my_devs/vla_engineering/vlash-main/vlash/policies/pi0/modeling_pi0.py`
  - 同步 `_gated_residual` fallback、RMSNorm 兼容层、bf16/dtype 对齐修复。
- `my_devs/vla_engineering/vlash-main/vlash/layers/attention.py`
  - 手写 attention 使用 fp32 计算 score/softmax，并在 weighted-sum 前转回 value dtype，避免 bf16 LoRA 训练中 `attn_weights` 与 `v` dtype 不一致。
- `my_devs/vla_engineering/train/`
  - 新增环境、离线资产、训练配置、训练脚本、数据探针、训练显存探针、checkpoint 推理探针、真实机器人 async inference YAML。

## 复现命令

```bash
cd /data/cqy_workspace/flexible_lerobot
bash my_devs/vla_engineering/train/env_setup.sh
bash my_devs/vla_engineering/train/run_vlash_pi05_lora_smoke.sh
```

12GiB 显存探针：

```bash
conda run -n vlash_train python my_devs/vla_engineering/train/probe_vlash_train_memory.py \
  --config-path my_devs/vla_engineering/train/pi05_so101_vlash_lora_smoke.yaml

conda run -n vlash_train python my_devs/vla_engineering/train/probe_vlash_train_memory.py \
  --config-path my_devs/vla_engineering/train/pi0_so101_vlash_lora_memory_probe.yaml
```

## 验证记录

### 环境

- conda env：`vlash_train`
- Python package versions：
  - `torch 2.7.1+cu126`
  - `transformers 4.53.3`
  - `peft 0.18.1`
  - `bitsandbytes 0.48.2`
  - `lerobot 0.4.3`
  - `vlash /data/cqy_workspace/flexible_lerobot/my_devs/vla_engineering/vlash-main/vlash/__init__.py`

### 数据探针

命令：

```bash
conda run -n vlash_train python my_devs/vla_engineering/train/probe_vlash_dataset.py
```

结果：

- `num_frames=53235`
- `num_episodes=157`
- 相机 keys：`observation.images.top`、`observation.images.wrist`
- sample action shape：`[50, 6]`
- sample state shape：`[6]`
- sample task：`Put the eraser into the small box`

### PI0.5 LoRA smoke 训练

命令：

```bash
bash my_devs/vla_engineering/train/run_vlash_pi05_lora_smoke.sh
```

关键日志：

- 日志目录：`my_devs/vla_engineering/train/logs/20260701_220046_pi05_lora_smoke/`
- LoRA 注入：`Cast 414 LoRA modules`
- 参数量：`total_params=3745429204`，`trainable_params=126538656`
- step 1：`loss=0.439`，`grad_norm=0.717`
- step 2：`loss=0.153`，`grad_norm=0.634`
- checkpoint：`Policy checkpointed at step 2`
- 训练结束：`End of training`

checkpoint 输出：

- `my_devs/vla_engineering/train/outputs/pi05_so101_vlash_lora_smoke/checkpoints/last/pretrained_model/config.json`
- `my_devs/vla_engineering/train/outputs/pi05_so101_vlash_lora_smoke/checkpoints/last/pretrained_model/train_config.json`
- `my_devs/vla_engineering/train/outputs/pi05_so101_vlash_lora_smoke/checkpoints/last/pretrained_model/model.safetensors`，约 `7.0G`
- `my_devs/vla_engineering/train/outputs/pi05_so101_vlash_lora_smoke/checkpoints/last/pretrained_model/lora_adapters/`

### checkpoint 推理

`run_vlash_pi05_lora_smoke.sh` 训练后自动执行：

```bash
python my_devs/vla_engineering/train/smoke_vlash_pi05_checkpoint.py \
  --policy-path my_devs/vla_engineering/train/outputs/pi05_so101_vlash_lora_smoke/checkpoints/last/pretrained_model \
  --device cuda \
  --num-inference-steps 2
```

结果：

- `policy_type=pi05`
- `action_shape=[1, 50, 6]`
- `action_dtype=torch.float32`
- `finite=true`

### async inference 配置解析

未连接真实机器人，只做 `RunConfig` 解析检查：

```bash
conda run -n vlash_train python -c "import draccus; from vlash.configs.run_config import RunConfig; cfg=draccus.parse(config_class=RunConfig, config_path='my_devs/vla_engineering/train/pi05_so101_vlash_async_infer.yaml', args=[]); print(type(cfg.robot).__name__, cfg.policy.type, cfg.policy.compile_model, cfg.inference_overlap_steps)"
```

结果：

- robot config：`SOFollowerRobotConfig`
- policy type：`pi05`
- policy path：`my_devs/vla_engineering/train/outputs/pi05_so101_vlash_lora_smoke/checkpoints/last/pretrained_model`
- `compile_model=True`
- `inference_overlap_steps=4`
- `action_quant_ratio=1`

### 12GiB 显存探针

PI0.5：

```json
{
  "batch_size": 1,
  "grad_accum_steps": 1,
  "lora_enabled": true,
  "qlora_enabled": false,
  "after_policy": {
    "allocated_gib": 7.117728233337402,
    "reserved_gib": 7.1796875
  },
  "after_lora": {
    "allocated_gib": 7.572732448577881,
    "reserved_gib": 7.634765625
  },
  "after_backward_step": {
    "allocated_gib": 8.505343914031982,
    "reserved_gib": 11.091796875,
    "max_allocated_gib": 10.841865062713623,
    "max_reserved_gib": 11.091796875
  }
}
```

PI0：

```json
{
  "batch_size": 1,
  "grad_accum_steps": 1,
  "lora_enabled": true,
  "qlora_enabled": false,
  "after_policy": {
    "allocated_gib": 6.679938316345215,
    "reserved_gib": 6.78125
  },
  "after_lora": {
    "allocated_gib": 6.708878993988037,
    "reserved_gib": 6.814453125
  },
  "after_backward_step": {
    "allocated_gib": 6.788737773895264,
    "reserved_gib": 10.208984375,
    "max_allocated_gib": 9.909279346466064,
    "max_reserved_gib": 10.208984375
  }
}
```

端到端脚本的 `nvidia-smi` 采样在 checkpoint merge/save 阶段观测到更高整卡 used 值；该阶段包含额外的 checkpoint 合并和重新加载，不用于判断训练计算是否 under 12GiB。

## 当前限制

- smoke 配置只跑 2 个 optimizer step，用于证明训练链路可执行，不代表任务质量。
- PI0 目前只做了同配置 1-batch memory probe，尚未单独保存 PI0 checkpoint 或跑 PI0 checkpoint 推理。
- 12GiB 结论对应 `batch_size=1` 和当前 LoRA 配方；更大 batch、更多 trainable modules、关闭 gradient checkpointing、开启 fused qkv/gate-up 后需要重新测。
- `pi05_so101_vlash_async_infer.yaml` 是真实机器人运行模板，实际运行前需要确认串口和摄像头路径。
