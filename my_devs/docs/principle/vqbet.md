# VQ-BeT

## 策略背景

VQ-BeT（Vector-Quantized Behavior Transformer）出自 *Behavior Generation with Latent Actions*。它针对模仿学习中的动作多峰问题：连续回归容易平均化，diffusion 表达力强但部署更重。核心思路是先把短时动作片段离散化，再让 Transformer 预测潜在动作。

## 核心原理

论文原理可概括为三步：先用 Residual VQ-VAE 将 action chunk 映射到多层离散 code，得到粗粒度潜在动作；再用 Transformer 根据观测历史预测各层 code；最后预测连续 offset，对码本解码结果做精修。最终动作可理解为“离散中心 + 连续偏移”。训练通常先学离散化，再冻结码本训练行为模型；推理时一次前向即可得到动作块。

## 在本仓库中的实现解读

以下是本仓库实现，不等同于论文唯一做法。`configuration_vqbet.py` 默认使用 `n_obs_steps=5`、`n_action_pred_token=3`、`action_chunk_size=5`，视觉主干为 `resnet18`，状态和动作做 `min-max` 归一化，视觉保持 `identity`。注释提到多相机，但 `validate_features()` 实际强制只能输入一张图像。

`modeling_vqbet.py` 中，`VQBeTRgbEncoder` 用 ResNet + `SpatialSoftmax` 编码图像，状态经 MLP 投影；每个时刻组成“图像 token + 状态 token + action query token”，再追加未来查询 token 后送入 GPT。`VQBeTHead` 同时预测 code logits 和 offset；RVQ 层数被固定为 2，默认同时采样两层 code，也支持顺序采样。

`processor_vqbet.py` 实现工程侧流水线：观测重命名、加 batch 维、搬到设备、按数据集统计量归一化；输出端再反归一化并搬回 CPU。这部分属于仓库接入逻辑，不属于论文算法本体。

## 训练与推理机制

训练分两阶段。第一阶段在 `VQBeTPolicy.forward()` 中优先训练 `VqVae`：对动作序列做滑窗切块，优化重建误差和量化损失，并统计码本利用率；达到 `n_vqvae_training_steps` 后冻结码本。第二阶段再训练 GPT 与 action head，损失由 code 分类的 focal loss 和动作块的 L1 loss 组成。

推理时，`select_action()` 维护观测队列和动作队列。只有动作队列为空时才基于最近 `n_obs_steps` 步观测预测一个 action chunk，随后逐步弹出执行。虽然模型会学习“当前 + 未来”多个动作 token，但 rollout 实际只取当前对齐的那个 chunk，更像分块重规划。

## 关键超参数及影响

- `n_obs_steps`：历史更长，时序信息更充分，但计算更重。
- `action_chunk_size`、`n_action_pred_token`：决定一次预测多远、多长；越大越平滑，也更易开环漂移。
- `n_vqvae_training_steps`、`vqvae_n_embed`、`vqvae_embedding_dim`：决定离散动作空间容量。
- `offset_loss_weight`、`primary_code_loss_weight`、`secondary_code_loss_weight`：平衡“选对桶”和“精修动作”；默认更偏重拟合精度。
- `bet_softmax_temperature`、`sequentially_select`：影响采样保守度与层间决策方式。

## 适用场景、优势与局限

VQ-BeT 适合图像加状态输入、动作明显多模态、且可按短时片段建模的机器人模仿学习任务，如抓取、推拉、接触操作。它相比简单回归更能表达“一观测多动作”，也更利于低延迟部署。

局限也很明显：它依赖先学好的动作码本，训练比单阶段策略更复杂；若码本覆盖不足，后续 Transformer 很难补救。本仓库实现还固定了 2 层 RVQ、单图像输入和 chunk 式执行，灵活性有限；对强闭环、高频修正任务，过大的 action chunk 也会放大误差累积。

## 参考资料

1. 论文：<https://arxiv.org/abs/2403.03181>
2. 项目页：<https://sjlee.cc/vq-bet/>
3. 官方实现：<https://github.com/jayLEE0301/vq_bet_official>
4. BeT 前作论文：<https://arxiv.org/abs/2206.11251>
