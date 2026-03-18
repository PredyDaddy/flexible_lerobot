# SmolVLA 策略原理与仓库实现解读

## 策略背景

Vision-Language-Action（VLA）模型把图像、语言和机器人状态统一到一个策略里。其演进大致是：2023 年 RT-2 验证视觉语言模型可迁移到机器人控制；2024 年 OpenVLA 把路线做成开源通用模型；随后 TinyVLA、SmolVLA 开始强调“小型化”，追求更低训练成本、更轻部署和更低时延。SmolVLA 就属于这一趋势。

## 核心原理

VLA 的共同点是把多模态观测编码成上下文，再预测未来动作；区别主要在动作头。RT-2、OpenVLA 偏向离散动作 token；本仓库的 SmolVLA 则是连续动作 chunk 的 flow matching。训练时采样噪声 `epsilon` 和时间 `t`，构造 `x_t = t * epsilon + (1 - t) * action`，目标为 `u_t = epsilon - action`，模型学习预测速度场 `v_theta(x_t, t)` 并用 MSE 拟合；推理时从噪声出发，多步迭代还原动作 chunk。

## 在本仓库中的实现解读

`src/lerobot/policies/smolvla/configuration_smolvla.py` 定义输入输出契约和训练默认值。默认 `chunk_size=50`、`n_action_steps=50`，状态与动作补到 `32` 维，图像默认 resize+pad 到 `512x512`，语言最大长度 `48`，推理迭代 `num_steps=10`。小型化相关旋钮包括 `num_vlm_layers=16`、`expert_width_multiplier=0.75`、`self_attn_every_n_layers=2`；默认骨干是 `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`。

`src/lerobot/policies/smolvla/processor_smolvla.py` 定义前后处理流水线：补 batch 维、给 `task` 末尾补换行、调用骨干 tokenizer、移动到设备、按数据集统计量归一化；输出端再把动作反归一化并转回 CPU。补换行是为了和预训练 tokenizer 的 prompt 习惯保持一致。

`src/lerobot/policies/smolvla/modeling_smolvla.py` 是主体。`prepare_images()` 保持宽高比缩放，并把像素从 `[0,1]` 映射到 `[-1,1]`；缺失相机可用全 `-1` 图像和 mask 补位。`prepare_state()`、`prepare_action()` 负责维度补零。`VLAFlowMatching` 把图像、语言、状态组成 prefix，把“带噪动作 + 时间嵌入”组成 suffix，再交给 `SmolVLMWithExpertModel`；推理时 prefix 会缓存 `past_key_values`，`SmolVLAPolicy` 则用内部队列把 chunk 动作拆成单步控制。

## 训练与推理机制

训练时，processor 先完成 tokenization 与归一化，`forward()` 再做图像预处理、状态/动作 padding、时间 `t` 的 Beta 采样、噪声采样和 flow matching 损失计算；若 batch 含 `actions_id_pad`，越过 episode 边界的损失会被屏蔽。推理时，`predict_action_chunk()` 直接返回整段动作，`select_action()` 则把 chunk 放进队列后逐步消费；启用 `rtc_config` 时，还可借助 RTCProcessor 把推理与执行解耦。

## 关键超参数及影响

`chunk_size/n_action_steps` 决定预测时域与控制粒度；`num_steps` 决定去噪质量与时延；`max_state_dim/max_action_dim` 决定和机器人接口是否匹配；`freeze_vision_encoder`、`train_expert_only`、`load_vlm_weights` 决定迁移学习范围；`num_vlm_layers`、`num_expert_layers`、`expert_width_multiplier` 影响容量、显存和吞吐；`use_cache` 与 `rtc_config` 主要影响在线时延。

## 适用场景、优势与局限

SmolVLA 适合“多模态输入 + 自然语言任务 + 中短时域动作 chunk”的低成本操作任务，尤其适合单卡或消费级硬件部署。它的优势是复用 VLM 先验、连续动作生成更贴近机械臂控制，并支持 KV cache 与异步 RTC。局限在于它仍然依赖预训练 VLM、tokenizer 和数据集统计量，对 prompt/维度配置较敏感；同时迭代去噪的单次时延通常高于一步式 BC。

## 参考资料

1. RT-2: https://arxiv.org/abs/2307.15818
2. OpenVLA: https://openvla.github.io/
3. TinyVLA: https://arxiv.org/abs/2409.12514
4. SmolVLA 论文: https://arxiv.org/abs/2506.01844
5. Hugging Face 博客: https://huggingface.co/blog/smolvla
