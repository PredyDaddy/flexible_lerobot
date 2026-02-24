# ACT Pro 实现计划

> 基于设计文档的详细实现计划

## 概述

**目标：** 实现 ACT Pro 策略，全面优化 ACT 算法

**设计文档：** `docs/act_pro_design/ACT_PRO_DESIGN.md`

---

## 实现任务清单

### Phase 1: 基础架构 (必须先完成)

- [ ] 1.1 创建 `act_pro` 目录结构
- [ ] 1.2 实现 `ACTProConfig` 配置类
- [ ] 1.3 实现 DINOv2 骨干网络
- [ ] 1.4 实现 RoPE 位置编码
- [ ] 1.5 实现时序注意力融合模块

### Phase 2: 核心模型

- [ ] 2.1 实现 Pre-Norm Transformer 层
- [ ] 2.2 实现 `ACTPro` 神经网络
- [ ] 2.3 实现 `ACTProPolicy` 策略类
- [ ] 2.4 注册到 factory.py

### Phase 3: 训练优化

- [ ] 3.1 实现混合精度训练支持
- [ ] 3.2 实现梯度检查点
- [ ] 3.3 实现分阶段微调逻辑
- [ ] 3.4 实现数据增强

### Phase 4: 推理优化

- [ ] 4.1 实现 Flash Attention
- [ ] 4.2 实现 FP16 推理
- [ ] 4.3 实现 torch.compile 支持

### Phase 5: 测试与脚本

- [ ] 5.1 编写单元测试
- [ ] 5.2 编写训练脚本
- [ ] 5.3 编写推理脚本
- [ ] 5.4 Smoke 测试验证

---

## 详细实现说明

### 1.1 创建目录结构

```bash
mkdir -p src/lerobot/policies/act_pro
touch src/lerobot/policies/act_pro/__init__.py
touch src/lerobot/policies/act_pro/configuration_act_pro.py
touch src/lerobot/policies/act_pro/modeling_act_pro.py
touch src/lerobot/policies/act_pro/backbone_dinov2.py
touch src/lerobot/policies/act_pro/temporal_fusion.py
```

### 1.2 ACTProConfig 配置类

**文件：** `configuration_act_pro.py`

**关键字段：**

```python
@dataclass
class ACTProConfig(ACTConfig):
    # DINOv2 配置
    use_dinov2: bool = True
    dinov2_model_name: str = "facebook/dinov2-large"
    dinov2_image_size: int = 224
    freeze_backbone: bool = True
    unfreeze_at_step: int = 50000

    # Transformer 配置
    n_decoder_layers: int = 4
    pre_norm: bool = True
    use_rope: bool = True

    # 多步观察
    n_obs_steps: int = 2
    use_temporal_attention: bool = True

    # 训练优化
    use_amp: bool = True
    use_gradient_checkpointing: bool = True

    # 推理优化
    use_flash_attention: bool = True
    inference_dtype: str = "float16"
```

### 1.3 DINOv2 骨干网络

**文件：** `backbone_dinov2.py`

**核心实现：**

```python
class DINOv2Backbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dinov2 = Dinov2Model.from_pretrained(config.dinov2_model_name)
        self.image_size = config.dinov2_image_size
        self.proj = nn.Conv2d(1024, config.dim_model, kernel_size=1)

    def forward(self, images):
        # Resize to DINOv2 input size
        images = F.interpolate(images, size=(self.image_size, self.image_size))

        # DINOv2 forward
        outputs = self.dinov2(images)
        patch_tokens = outputs.last_hidden_state[:, 1:]  # 去掉CLS

        # Reshape to 2D feature map
        h = w = self.image_size // 14
        features = patch_tokens.transpose(1, 2).reshape(-1, 1024, h, w)

        return {"feature_map": self.proj(features)}
```

### 1.4 RoPE 位置编码

**核心实现：**

```python
class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()
```

### 1.5 时序注意力融合模块

**文件：** `temporal_fusion.py`

```python
class TemporalAttentionFusion(nn.Module):
    def __init__(self, dim, n_obs_steps=2, n_heads=4):
        super().__init__()
        self.temporal_attn = nn.MultiheadAttention(dim, n_heads)
        self.temporal_pos_embed = nn.Embedding(n_obs_steps, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, features_list):
        # 添加时序位置编码
        for i, feat in enumerate(features_list):
            feat = feat + self.temporal_pos_embed.weight[i]

        # 拼接并融合
        x = torch.cat(features_list, dim=1)
        x = x.transpose(0, 1)
        x = self.temporal_attn(x, x, x)[0]
        return self.norm(x.transpose(0, 1))
```

---

### 5.2 训练脚本

**文件：** `agilex_scripts/train_act_pro.sh`

```bash
#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
export HF_HUB_OFFLINE=1

lerobot-train \
  --dataset.repo_id=agilex_dataset1 \
  --policy.type=act_pro \
  --policy.device=cuda \
  --policy.dinov2_model_name=/path/to/dinov2-large \
  --batch_size=8 \
  --steps=100000 \
  --output_dir=outputs/act_pro
```

### 5.4 Smoke 测试

**CPU Smoke (验证基本功能)：**

```bash
lerobot-train \
  --policy.type=act_pro \
  --policy.device=cpu \
  --steps=1 --batch_size=1 \
  --output_dir=outputs/_smoke_act_pro
```

**GPU Smoke (验证显存)：**

```bash
CUDA_VISIBLE_DEVICES=7 lerobot-train \
  --policy.type=act_pro \
  --policy.device=cuda \
  --steps=10 --batch_size=2 \
  --output_dir=outputs/_smoke_act_pro_gpu
```

---

## 依赖关系

```
1.1 目录结构
    ↓
1.2 配置类 ← 1.3 DINOv2 ← 1.4 RoPE ← 1.5 时序融合
    ↓
2.1 Pre-Norm层 → 2.2 ACTPro网络 → 2.3 ACTProPolicy
    ↓
2.4 注册factory → 5.1 单元测试 → 5.4 Smoke测试
```
