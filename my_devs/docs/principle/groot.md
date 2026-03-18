# Groot 策略原理与仓库实现解读

## 策略背景

GR00T 是 NVIDIA 的机器人基础模型家族。N1 论文强调“双系统”结构：System 2 做视觉语言理解与语义推理，System 1 把条件快速变成连续动作；N1.5 继续沿这一方向做后训练与数据扩展，并引入 FLARE。  

这些属于论文/官方资料中的通用原理；本仓库的 `groot` 则是把 GR00T N1.5 接到 LeRobot 中，默认底座为 `nvidia/GR00T-N1.5-3B`。

## 核心原理

论文通用原理可概括为“视觉语言条件 + 动作生成头”：先用 VLM 把多视角图像和文本编码成条件 token，再结合状态与 embodiment 条件生成动作块。  

本仓库动作头采用 flow matching。训练时，对真实动作块 `a` 与高斯噪声 `eps` 线性插值，得到

`x_t = (1 - t) * eps + t * a`

并以速度场

`v* = a - eps`

作为监督，最小化掩码 MSE：

`L = ||v_theta(x_t, t, c) - v*||^2`

其中 `c` 表示视觉语言、状态和 embodiment 条件。推理时从高斯噪声初始化动作，再用 Euler 积分更新

`x <- x + dt * v_theta(x, t, c)`

得到整段动作。  

N1.5 的 FLARE 在这里更偏“论文背景”；本仓库当前前向里明确写了 `not using flare now`，因此不能视为已启用训练项。

## 在本仓库中的实现解读

`configuration_groot.py` 是包装层配置：要求至少一路视觉输入，默认 `max_state_dim=64`、`max_action_dim=32`，并把基础模型设为 N1.5。虽然配置里 `chunk_size=50`，但 `processor_groot.py` 实际把 `action_horizon` 限制为 `min(chunk_size, 16)`，所以当前接入结构本质上按最多 16 步动作块运行。  

`processor_groot.py` 的主链路是：重命名键 -> 加 batch -> 打包图像/状态/动作/语言/embodiment -> Eagle 编码 -> 整理成 `eagle_*` 张量 -> 移到设备。它会把多相机图像整理成 `video`，语言默认读 `task`，对 `state/action` 做 min-max 归一化、补零并生成 mask，再把 `embodiment_tag` 映射成离散 ID。这里应以 processor 的真实行为为准，因为配置里写的是 `MEAN_STD`，但预处理实际执行的是 min-max。  

`modeling_groot.py` 中的 `GrootPolicy` 很薄，只负责调用 `GR00TN15.from_pretrained(...)`、筛选需要的输入键并转发到底层模型。更底层的 `groot_n1.py` 将模型拆成 `EagleBackbone + FlowmatchingActionHead`：前者提取视觉语言 token，后者把状态、future tokens、噪声动作与视觉语言条件拼接后送入 DiT，输出动作速度。

## 训练与推理机制

训练阶段，Eagle 先产出视觉语言 token，动作头对加噪动作块预测速度场，最终返回单个 `loss`；实现中支持 bf16 autocast。推理阶段，动作头从随机噪声出发做多步积分得到动作块，`predict_action_chunk()` 输出整段动作，`select_action()` 用队列逐步消费；若后处理拿到三维动作张量，则会取最后时刻、裁剪到环境动作维并做反归一化。

## 关键超参数及影响

- `base_model_path`：决定预训练底座。  
- `tune_llm`、`tune_visual`、`tune_projector`、`tune_diffusion_model`：决定微调范围。  
- `max_state_dim`、`max_action_dim`：决定截断/补零边界。  
- `chunk_size` 与真实 `action_horizon`：当前最关键的约束是 16 步上限。  
- `embodiment_tag`、`use_bf16`、`optimizer_lr`：分别影响条件化是否正确、显存/吞吐与收敛速度。

## 适用场景、优势与局限

适用场景是“多相机观测 + 语言指令 + 低维机器人状态”的模仿学习或后训练微调。优势是能直接复用 GR00T 预训练知识、支持多 embodiment 条件化、并以 chunk 而非单步回归生成动作。局限是对外部 GR00T/Eagle 资产依赖较强，动作时域当前受 16 步上限约束，且 FLARE 论文能力未在当前前向中真正启用。

## 参考资料

1. GR00T N1 论文页  
   https://research.nvidia.com/publication/2025-03_nvidia-isaac-gr00t-n1-open-foundation-model-humanoid-robots
2. GR00T N1 arXiv  
   https://arxiv.org/abs/2503.14734
3. GR00T N1.5 官方页  
   https://research.nvidia.com/labs/gear/gr00t-n1_5/
4. FLARE arXiv  
   https://arxiv.org/abs/2505.15659
