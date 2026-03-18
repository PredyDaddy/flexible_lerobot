# ACT（Action Chunking Transformer）

## 策略背景

[论文通用原理] ACT 出自《Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware》，面向机器人模仿学习里的细粒度、长时程操作。它针对两个常见问题：单步行为克隆容易累积误差；同一观测下常有多种合理动作。ACT 因此不再只预测下一步，而是直接预测一段未来动作。

## 核心原理

[论文通用原理] 在时刻 $t$，策略根据观测 $o_t$ 预测长度为 $k$ 的动作块 $A_t=(a_t,\dots,a_{t+k-1})$。训练时使用条件 VAE：$q_\phi(z|s_t,A_t)=\mathcal N(\mu,\sigma^2I)$，再由 $\hat A_t=\pi_\theta(o_t,z)$ 重建动作块，目标可写为
$\mathcal L=\|A_t-\hat A_t\|_1+\beta D_{KL}(q_\phi\|\mathcal N(0,I))$。  
动作块提升了有效时域；若相邻 chunk 对同一时刻给出多个候选动作，还可做时间集成：$\bar a_t=\sum_i w_i\hat a_t^{(i)}/\sum_i w_i,\; w_i=e^{-mi}$。

## 在本仓库中的实现解读（模块与数据流）

[本仓库实现]

- `configuration_act.py` 定义 `ACTConfig`。关键约束是：`n_obs_steps=1`、`n_action_steps <= chunk_size`，且启用 `temporal_ensemble_coeff` 时 `n_action_steps` 必须为 1。默认 `n_decoder_layers=1`，是为了对齐原始 ACT 代码的实际行为。
- `modeling_act.py` 中 `ACTPolicy` 负责训练/推理封装，`ACT` 负责网络本体。主数据流为：图像 -> ResNet `layer4` 特征 -> `1x1` 卷积映射到 `dim_model` -> 2D 位置编码；状态、环境状态、latent 走线性投影；所有 token 进入 Transformer encoder，decoder 用长度为 `chunk_size` 的查询向量回归整段动作。
- 训练时额外启用 VAE encoder，输入为 `[CLS, robot_state, action_sequence]`，输出 `mu` 与 `log_sigma^2`，经重参数化采样 latent。损失是带 `action_is_pad` 掩码的 L1，加上可选 KL 项。
- 推理时不走 VAE encoder，而是把 latent 置零。若未启用时间集成，就预测一个 chunk 并按队列依次执行前 `n_action_steps` 个动作；若启用时间集成，则每步都预测 chunk，并用 `ACTTemporalEnsembler` 对重叠动作做指数加权。
- `processor_act.py` 负责前后处理：加 batch 维、迁移到设备、按数据集统计量归一化；输出动作再反归一化并移回 CPU。训练目标动作也会被归一化，因此损失是在归一化动作空间中计算的。

## 训练与推理机制

[论文通用原理] 训练目标是“根据当前观测复原未来动作块”，并用 KL 正则约束 latent 接近标准高斯。  
[本仓库实现] VAE 分支只在训练且 batch 含 `ACTION` 时启用；推理时直接使用零 latent。虽然配置注释说 `observation.state` 可选，但当前 `modeling_act.py` 多处直接访问 `batch[OBS_STATE]` 取 device，因此实际使用时更稳妥的做法是始终提供机器人状态。

## 关键超参数及影响

- `chunk_size`：一次预测多长未来，直接决定 chunking 强度。
- `n_action_steps`：每次前向后实际执行几步；小于 `chunk_size` 时尾部动作会被丢弃。
- `temporal_ensemble_coeff`：控制是否启用时间集成及其平滑强度。
- `use_vae`、`latent_dim`、`kl_weight`：决定多模态建模能力与正则强度。
- `vision_backbone`、`dim_model`、层数/头数：影响表达能力、显存和速度。

## 适用场景、优势与局限

[论文通用原理] ACT 适合多相机、强时序相关、细粒度操作任务，如双臂装配、抓取转移等；优势是能同时建模长时程动作与示教多模态性。局限是依赖高质量示教数据，并对 chunk 长度、控制频率较敏感。  
[本仓库实现] 当前实现仅支持 `n_obs_steps=1`，并且实践上更依赖 `observation.state`；如果只给环境状态或希望引入更长观测窗口，需要额外改代码。

## 参考资料

1. ACT 原论文: https://arxiv.org/abs/2304.13705
2. ALOHA / ACT 项目页: https://tonyzhaozh.github.io/aloha/
3. ACT 原始代码: https://github.com/tonyzhaozh/act
4. LeRobot 官方 ACT 文档: https://huggingface.co/docs/lerobot/act
5. VAE 基础论文: https://arxiv.org/abs/1312.6114
