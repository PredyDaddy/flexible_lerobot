# XVLA 策略原理与仓库实现解读

## 策略背景

XVLA 属于 VLA 路线，重点是缓解不同机械臂、相机布局、动作空间和数据域造成的分布偏移。公开资料目前主要是 arXiv、OpenReview 与项目页；我没有查到更完整的正式期刊版，因此下文以这些资料和本仓库代码为边界。

## 核心原理

核心是“软提示 + 连续动作 chunk 生成”。`domain_id` 会索引 soft prompts，并进入 domain-aware 线性层，让同一主干区分不同 embodiment 和数据域；视觉骨干用 Florence-2，第一视角先与文本融合，其余视角作辅助视觉 token。训练时按时间 `t` 将真实动作与高斯噪声线性插值，再回归干净动作；推理时从噪声出发，多步细化整段动作。

## 在本仓库中的实现解读

`configuration_xvla.py` 定义输入输出契约和训练默认值：至少一个视觉输入，`use_proprio=True` 时要求 `observation.state`，并可自动补 `num_image_views`、`empty_cameras`。它还给出 Florence-2 配置、soft prompt 长度、动作模式和冻结开关。需注意：注释写“默认冻结 VLM”，但默认值其实是不冻结。

`modeling_xvla.py` 中，`XVLAModel` 先按 `action_mode` 构造动作空间，再实例化 Florence-2 并删除 decoder/lm head。`forward_vlm()` 负责多视角编码与“首视角+文本”融合；`SoftPromptedTransformer` 再把带噪动作、状态、时间、视觉语言 token 和 soft prompts 拼成序列，最后只解码动作段。`XVLAPolicy` 负责 padding、`domain_id` 提取和 action chunk 队列化执行。

`processor_xvla.py` 定义前后处理：默认会做 tokenizer、图像 `[0,255] -> [0,1]`、ImageNet 标准化、补 `domain_id` 和状态/动作归一化。LIBERO 专用流水线会把嵌套 `robot_state` 展平成 20 维状态，并把 6D 旋转动作转成 axis-angle。

## 训练与推理机制

训练时，processor 先完成 tokenization 与图像标准化，`forward()` 再做状态/动作 pad、`domain_id` 组装和噪声插值。损失由 `action_hub.py` 的动作空间决定，如 `ee6d` 对位置/旋转用加权 MSE、对夹爪用 BCE；`XVLAAdamW` 默认让 VLM 只用 1/10 学习率。推理时，`generate_actions()` 从高斯噪声初始化 `chunk_size` 动作并按 `num_denoising_steps` 细化，`select_action()` 再逐步弹出。

## 关键超参数及影响

`chunk_size`、`n_action_steps` 决定预测时域与控制粒度；`num_denoising_steps` 越大通常越稳但更慢。`len_soft_prompts`、`num_domains`、`domain_feature_key` 影响跨 embodiment 适配；`hidden_size`、`depth`、`num_heads`、`max_len_seq` 决定容量与显存；`action_mode`、`max_action_dim`、`max_state_dim` 决定接口兼容性。

## 适用场景、优势与局限

XVLA 适合“语言条件 + 多相机 + 连续控制 + 跨机器人迁移”的操作任务，优势是跨 embodiment 设计明确、动作空间可注册、soft prompt 适配成本较低。局限是实现高度依赖 Florence-2，首视角在融合中具有特权，`domain_id` 错配会伤害泛化，多步迭代推理通常慢于一步式 BC；另外部分训练 recipe 仍主要依赖代码和官方文档推断。

## 参考资料

1. X-VLA OpenReview: https://openreview.net/forum?id=kt51kZH4aG
2. X-VLA arXiv: https://arxiv.org/abs/2510.10274
3. X-VLA 项目页: https://thu-air-dream.github.io/X-VLA/
