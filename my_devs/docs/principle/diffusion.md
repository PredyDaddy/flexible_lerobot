# Diffusion Policy

## 策略背景

论文通用原理：Diffusion Policy 不直接回归单步动作，而是在观测条件下生成一段未来动作，因此更适合多峰动作分布、接触操作和时间平滑控制。  
本仓库具体实现：`src/lerobot/policies/diffusion/` 采用 `1D Conditional U-Net + DDPM/DDIM scheduler` 的工程版本，并非论文中的完整 transformer 变体。

## 核心原理

论文通用原理：对真实动作轨迹 $x_0$ 加噪，

$$
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,\quad \epsilon\sim\mathcal N(0,I)
$$

训练条件网络预测噪声并做 MSE；推理时从噪声反向去噪，得到长度为 `horizon` 的动作序列，再按 receding horizon 只执行一段。  
本仓库具体实现：`compute_loss()` 中 `prediction_type="epsilon"` 时拟合噪声，`"sample"` 时拟合干净动作；条件 `global_cond` 由机器人状态、图像特征和可选环境状态拼接得到。

## 在本仓库中的实现解读（模块与数据流）

`configuration_diffusion.py`：定义 `DiffusionConfig`，默认 `n_obs_steps=2`、`horizon=16`、`n_action_steps=8`，并约束图像 shape 与 `horizon % 2^len(down_dims) == 0`。  
`processor_diffusion.py`：前处理是“补 batch -> 搬 device -> 归一化”，后处理是“动作反归一化 -> CPU”；训练目标动作也会归一化。  
`modeling_diffusion.py`：`DiffusionPolicy` 负责观测/动作队列，`DiffusionModel` 负责扩散生成；图像编码器是“裁剪 + ResNet18 + SpatialSoftmax”，动作网络是带 FiLM 条件的 `DiffusionConditionalUnet1d`。

数据流：观测前处理后进入 `select_action()`，更新观测队列；若动作队列为空，则编码观测得到 `global_cond`，从高斯噪声迭代去噪，生成完整动作轨迹，再截取当前应执行的 `n_action_steps` 放入动作队列。

## 训练与推理机制

训练：`compute_loss()` 对 `batch["action"]` 加噪，再用 U-Net 预测噪声或干净动作，损失是 MSE；若开启 `do_mask_loss_for_padding`，则用 `action_is_pad` 屏蔽 padding。训练脚本还会用 `drop_n_last_frames=7` 减少尾部 padding。  
推理：`reset()` 清空队列；首次观测会复制以填满窗口；之后模型每 `n_action_steps` 步重规划一次。

## 关键超参数及影响

- `n_obs_steps`：观测历史长度。  
- `horizon` / `n_action_steps`：生成长度与重规划频率。  
- `down_dims`：U-Net 容量。  
- `num_inference_steps`：越小越快，但质量可能下降。

## 适用场景、优势与局限

适用于视觉条件模仿学习和需要平滑 chunked action 的机械臂任务。优势是比单步回归更能表示多峰动作；局限是推理较慢。

## 参考资料

1. Diffusion Policy 项目主页：https://diffusion-policy.cs.columbia.edu/  
2. Diffusion Policy 论文：https://arxiv.org/abs/2303.04137  
3. 官方代码仓库：https://github.com/real-stanford/diffusion_policy
