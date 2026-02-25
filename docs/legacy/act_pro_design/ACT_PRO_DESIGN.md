# ACT Pro 优化设计方案

> **状态说明 (lerobot 0.4.3 baseline)**  
> 当前仓库 0.4.3 baseline 并不包含 `act_pro`/`act_dinov2` 对应实现代码；本文为设计备忘录，不包含可直接运行的步骤。  

> ACT (Action Chunking Transformer) 全面优化方案，针对精细操作场景

## 1. 需求分析

### 1.1 优化目标

| 维度 | 目标 |
|------|------|
| 模型性能 | 提升任务成功率、泛化能力、动作精度 |
| 训练效率 | 加快收敛速度、减少数据需求、降低显存占用 |
| 推理速度 | 降低延迟、提高实时性、优化部署 |
| 架构升级 | 更换骨干网络、修复已知bug、增加新功能 |

### 1.2 当前问题

- 精度不足：任务成功率不够高，泛化能力差
- 收敛慢：训练需要大量数据和时间
- 延迟高：推理延迟无法满足实时控制
- 显存不足：无法使用大batch训练

### 1.3 硬件环境

- 训练：单卡高端GPU (RTX 3090/4090, 24GB显存)
- 部署：同上或边缘设备

### 1.4 应用场景

- 精细操作：桌面级抓取、装配等需要高精度的任务

---

## 2. Phase 1: 架构升级

### 2.1 视觉骨干升级: DINOv2-Large

**当前实现：** ResNet-18 (11M参数)

**升级目标：** DINOv2-Large (300M参数)

| 属性 | ResNet-18 | DINOv2-Large |
|------|-----------|--------------|
| 参数量 | 11M | 300M |
| 特征维度 | 512 | 1024 |
| 预训练方式 | ImageNet监督 | 自监督 |
| 空间理解 | 一般 | 强 |

**DINOv2配置：**

```python
dinov2_model_name: str = "facebook/dinov2-large"
dinov2_image_size: int = 224  # 必须是14的倍数
dinov2_output_mode: str = "grid"  # 保留空间信息
```

**分阶段微调策略：**

```
阶段1 (前50%步数): 完全冻结DINOv2
├─ 目的: 稳定收敛，避免破坏预训练特征
└─ 学习率: backbone=0, transformer=1e-4

阶段2 (后50%步数): 解冻最后6层
├─ 目的: 微调高层特征，适应具体任务
└─ 学习率: backbone=1e-6, transformer=1e-4
```

### 2.2 修复 Decoder 层数 Bug

**问题：** 原始ACT实现只使用1层解码器（bug）

**修复：** 增加到4层，与编码器一致

```python
n_decoder_layers: int = 4  # 原来是1
```

**预期效果：** 更强的动作序列建模能力

### 2.3 Pre-Norm + RoPE 位置编码

**Pre-Norm：**
- 当前：Post-Norm (norm在残差之后)
- 升级：Pre-Norm (norm在残差之前)
- 优势：训练更稳定，支持更深网络

**RoPE位置编码：**
- 当前：固定正弦位置编码
- 升级：Rotary Position Embedding
- 优势：更好的相对位置建模

```python
pre_norm: bool = True
use_rope: bool = True
```

### 2.4 多步观察支持

**当前限制：** 只支持单帧观察 (n_obs_steps=1)

**升级：** 支持2帧观察 + 时序注意力融合

```python
n_obs_steps: int = 2
use_temporal_attention: bool = True
```

**时序注意力融合：**
- 对多帧特征添加时序位置编码
- 通过自注意力学习时序关系
- 输出融合后的特征序列

---

## 3. Phase 2: 训练优化

### 3.1 混合精度训练 (AMP)

```python
use_amp: bool = True
amp_dtype: str = "float16"
```

**预期效果：**
- 显存减少 40-50%
- 训练速度提升 30-50%

### 3.2 梯度检查点

```python
use_gradient_checkpointing: bool = True
checkpoint_layers: list = ["backbone", "encoder"]
```

**预期效果：**
- 额外减少 20-30% 显存
- 训练速度略降 10-15%

### 3.3 学习率策略

```python
optimizer_lr: float = 1e-4
optimizer_lr_backbone: float = 1e-6
lr_scheduler: str = "cosine"
warmup_steps: int = 1000
min_lr_ratio: float = 0.01
```

**分层学习率：**

| 组件 | 学习率 |
|------|--------|
| DINOv2 (冻结阶段) | 0 |
| DINOv2 (解冻阶段) | 1e-6 |
| Transformer | 1e-4 |

### 3.4 数据增强

**颜色增强：**
```python
color_jitter:
  brightness: 0.2
  contrast: 0.2
  saturation: 0.2
  hue: 0.1
```

**空间增强：**
```python
random_crop:
  scale: [0.9, 1.0]
  ratio: [0.95, 1.05]
```

**动作噪声：**
```python
action_noise:
  std: 0.01
  apply_prob: 0.5
```

---

## 4. Phase 3: 推理优化

### 4.1 Flash Attention

```python
use_flash_attention: bool = True
```

**预期效果：**
- 推理速度提升 20-40%
- 显存减少 30%

### 4.2 FP16 推理

```python
inference_dtype: str = "float16"
compile_model: bool = True  # torch.compile加速
```

### 4.3 推理延迟估算

| 配置 | 单帧延迟 |
|------|----------|
| FP32 (当前) | ~50ms |
| FP16 + Flash Attention | ~20ms |
| TensorRT FP16 (可选) | ~10ms |

---

## 5. 实现计划

### 5.1 文件结构

```
src/lerobot/policies/act_pro/
├── __init__.py
├── configuration_act_pro.py    # 配置类
├── modeling_act_pro.py         # 模型实现
├── backbone_dinov2.py          # DINOv2骨干
├── temporal_fusion.py          # 时序融合模块
└── processor_act_pro.py        # 数据处理
```

### 5.2 实现步骤

1. 创建配置类 `ACTProConfig`
2. 实现 DINOv2 骨干网络
3. 实现时序注意力融合模块
4. 实现 RoPE 位置编码
5. 整合为 `ACTProPolicy`
6. 注册到 factory.py
7. 编写单元测试
8. 编写训练/推理脚本

### 5.3 配置总览

```python
@dataclass
class ACTProConfig:
    # Phase 1: 架构升级
    use_dinov2: bool = True
    dinov2_model_name: str = "facebook/dinov2-large"
    dinov2_image_size: int = 224
    freeze_backbone: bool = True
    unfreeze_at_step: int = 50000
    n_encoder_layers: int = 4
    n_decoder_layers: int = 4
    pre_norm: bool = True
    use_rope: bool = True
    n_obs_steps: int = 2
    use_temporal_attention: bool = True

    # Phase 2: 训练优化
    use_amp: bool = True
    use_gradient_checkpointing: bool = True
    optimizer_lr: float = 1e-4
    optimizer_lr_backbone: float = 1e-6
    lr_scheduler: str = "cosine"
    warmup_steps: int = 1000

    # Phase 3: 推理优化
    use_flash_attention: bool = True
    inference_dtype: str = "float16"
    compile_model: bool = True
```

---

## 6. 资源估算

### 6.1 参数量

| 组件 | 参数量 |
|------|--------|
| DINOv2-Large | 300M |
| Transformer | ~20M |
| 其他 | ~5M |
| **总计** | **~325M** |

### 6.2 显存估算

| 配置 | 显存占用 |
|------|----------|
| batch=8, FP32 | ~18GB |
| batch=8, AMP | ~10GB |
| batch=16, AMP | ~16GB |
| batch=16, AMP+Checkpoint | ~12GB |
