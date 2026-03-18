# TD-MPC（Temporal Difference Learning for Model Predictive Control）

## 策略背景

[论文通用原理] TD-MPC 由 Nicklas Hansen 等人在 2022 年提出，目标是把 MPC 的短视域规划，与 TD 价值学习结合起来：短期回报由模型滚动预测，长期收益由价值函数补齐。这样既比纯 model-free 更省样本，也比长视域显式规划更便宜。  
[本仓库实现] 本仓库的 `tdmpc` 不是原始论文的逐行复刻，而是混合了 FOWM（Finetuning Offline World Models in the Real World）的若干训练设计。

## 核心原理

[论文通用原理] TD-MPC 先把观测编码为潜变量 `z_t=h(o_t)`，再在潜空间学习动力学、奖励、策略和价值。推理时不在原始观测空间搜索，而是在潜空间对长度 `H` 的动作序列做采样优化，目标近似为  
`sum gamma^t r_t + gamma^H V(z_H)`。  
原始方法通常用 MPPI/CEM 做滚动规划，并让策略网络提供候选动作，以减少纯随机采样的搜索难度。

## 在本仓库中的实现解读

[本仓库实现] `configuration_tdmpc.py` 默认 `n_obs_steps=1`、`horizon=5`、`use_mpc=True`、`n_action_repeats=2`。动作必须做 `MIN_MAX` 归一化，因为规划阶段会把动作裁到 `[-1,1]`；当前只支持单图像，且要求方形输入。  

`modeling_tdmpc.py` 中的 `TDMPCPolicy` 维护在线模型和 EMA target 模型；其核心 `TDMPCTOLD` 包含观测编码器、latent dynamics、reward head、policy head、Q ensemble 和 V head。图像经 4 层卷积，状态/环境状态经 MLP，最后都投到 `latent_dim` 并做均值融合。  

与论文最不同的是训练目标：代码同时计算 latent consistency、reward regression、Q 的 TD loss、V 的 expectile loss，以及 advantage-weighted policy regression。这说明仓库实现更接近“TD-MPC 推理框架 + FOWM 训练配方”。  

`processor_tdmpc.py` 负责前后处理，实际顺序是：重命名、加 batch、迁移设备、归一化；输出端再反归一化并回 CPU。动作也参与归一化，因此训练和规划都在统一的归一化动作空间中进行。

## 训练与推理机制

[论文通用原理] 训练时用真实轨迹监督潜空间滚动预测，并用 TD 目标学习价值；推理时采用 receding horizon control，每步只执行规划序列前一段，再重新规划。  

[本仓库实现] `forward()` 会先转成时间优先张量，对图像做 random shift，再用真实动作 rollout 预测未来 latent 和 reward。训练目标中，`q_targets = reward + discount * V(next_z)`，`v_targets` 来自 target Q 的保守最小值。推理时若 `use_mpc=True`，`plan()` 会把策略采样轨迹与高斯采样轨迹一起送入 MPPI/CEM；每轮按 `estimate_value()` 打分、选 elite、更新均值方差，并复用上一步均值做 warm start。这里还显式加入了 `lambda * std(Q)` 形式的不确定性惩罚；若关闭 MPC，则直接执行 `pi(z)`。

## 关键超参数及影响

- `horizon`：规划能看多远；越大越慢，也更依赖模型准确性。  
- `n_gaussian_samples`、`n_pi_samples`、`cem_iterations`、`n_elites`：决定 CEM 搜索强度与算力开销。  
- `min_std`、`max_std`、`gaussian_mean_momentum`：影响采样分布的探索范围和稳定性。  
- `uncertainty_regularizer_coeff`：越大越保守，更抑制分布外动作。  
- `latent_dim`、`mlp_dim`、`q_ensemble_size`：影响表达能力、稳定性与显存。  
- `consistency_coeff`、`reward_coeff`、`value_coeff`、`pi_coeff`、`temporal_decay_coeff`：决定模型学习与策略拟合的权衡。  
- `expectile_weight`、`advantage_scaling`：控制 V 的乐观程度和策略对高优势样本的偏置。

## 适用场景、优势与局限

[论文通用原理] TD-MPC 适合连续控制、动作维度中等、需要在线再规划的任务，优势是样本效率和规划能力兼顾。  
[本仓库实现] 当前版本更适合“单步观测 + 单图像/状态 + 连续动作”的机器人设定。优势是已把 FOWM 的实用训练技巧并入 LeRobot 接口；局限是多帧、多相机尚未支持，图像必须方形，且代码也明确提示其训练结果未系统验证完全复现实验室原版结果。

## 参考资料

1. TD-MPC 原论文（arXiv）：https://arxiv.org/abs/2203.04955
2. TD-MPC 项目页：https://td-mpc.github.io/
3. TD-MPC 官方代码：https://github.com/nicklashansen/tdmpc
4. FOWM 论文（Finetuning Offline World Models in the Real World）：https://arxiv.org/abs/2310.16029
5. TD-MPC2 项目页（了解后续演化）：https://www.tdmpc2.com/
