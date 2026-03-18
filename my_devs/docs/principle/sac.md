# SAC（Soft Actor-Critic）

## 策略背景

[论文通用原理] SAC 是 2018 年提出的最大熵离策略 actor-critic。它在累计回报之外显式鼓励高熵策略，因此通常比 DDPG 一类确定性方法更稳、更会探索、也更能复用历史数据。  
[本仓库实现] `src/lerobot/policies/sac/` 将其落成机器人控制版本，支持图像/状态混合观测、在线与离线回放，以及连续动作上附加离散夹爪分支。

## 核心原理

[论文通用原理] SAC 的目标可写为
$$
J(\pi)=\sum_t \mathbb E_{(s_t,a_t)\sim \pi}\left[r(s_t,a_t)+\alpha \mathcal H(\pi(\cdot|s_t))\right].
$$
其中 `alpha` 平衡回报与探索。现代 SAC 用双 Q 和软 Bellman 目标
$$
y=r+\gamma\left(\min_i Q_i(s',a')-\alpha \log \pi(a'|s')\right),\ a'\sim \pi(\cdot|s').
$$
actor 最小化 `alpha * log_pi - min(Q1,Q2)`；策略常用 squashed Gaussian，即高斯采样后再经 `tanh` 压到动作范围内。若启用自动温度调节，则继续优化 `alpha` 使熵逼近目标值。

## 在本仓库中的实现解读

[本仓库实现] `configuration_sac.py` 的 `SACConfig` 管理 `discount`、`temperature_init`、`num_critics`、`utd_ratio`、视觉编码器、共享编码器和离散动作分支，并要求输入至少含状态或图像、输出必须含 `action`。  
`modeling_sac.py` 中，`SACObservationEncoder` 编码图像/状态，critic 用多个 `CriticHead` 组成 Q 集成并维护 target 网络，actor 的 `Policy` 输出均值和方差，再经 `TanhMultivariateNormalDiag` 采样连续动作。若设置 `num_discrete_actions`，还会增加 `DiscreteCritic`，在推理时把离散 `argmax` 拼到连续动作后。  
`processor_sac.py` 只负责前后处理：补 batch、搬 device、归一化输入与动作目标，输出端再反归一化。  
`reward_model/*` 的定位是独立奖励分类器，而不是 SAC 子网络。`configuration_classifier.py` 定义配置，`modeling_classifier.py` 用冻结视觉编码器做成功/失败分类，`processor_classifier.py` 处理归一化。系统运行时，环境 processor 调用它的 `predict_reward()` 改写 `reward/done`，SAC 只是消费这个奖励去学习。

## 训练与推理机制

[本仓库实现] learner 从 online replay buffer 采样，也可拼接 offline 数据；critic 可按 `utd_ratio` 连续更新，actor 与温度参数按 `policy_update_freq` 更新，target critic 用软更新同步。若视觉编码器冻结，则会缓存当前帧与下一帧图像特征。  
推理时 `select_action()` 只输出单步动作，不支持 chunk；若有离散分支，再拼接离散 critic 的最优动作。这里默认仍从随机策略采样，而不是直接取均值动作。

## 关键超参数及影响

- `discount`：决定长期回报权重，过大更易累积 bootstrap 误差。
- `temperature_init`、`temperature_lr`、`target_entropy`：控制探索强度。
- `num_critics`、`num_subsample_critics`：影响 Q 高估抑制和高 UTD 稳定性。
- `critic_target_update_weight`、`utd_ratio`、`policy_update_freq`：决定学习节奏。
- `shared_encoder`、`freeze_vision_encoder`、`vision_encoder_name`：影响视觉质量和算力。
- `num_discrete_actions`、`use_backup_entropy`：决定混合动作分支与熵备份。

## 适用场景、优势与局限

[论文通用原理] SAC 适合连续控制和需要稳定探索的机器人任务。  
[本仓库实现] 它尤其适合机械臂位姿控制这类“连续平移 + 离散夹爪”场景；若环境原生奖励弱，还可接 `reward_model` 把视觉成功检测转成奖励。优势是离策略、可复用数据、容易接视觉编码器；局限是仍依赖奖励设计与温度调参，而 `reward_model` 的分类误差也会直接影响 RL 目标。

## 参考资料

1. SAC 原论文（PMLR 2018）：https://proceedings.mlr.press/v80/haarnoja18b.html
2. SAC 原论文（arXiv 版本）：https://arxiv.org/abs/1801.01290
3. Soft Actor-Critic: Algorithms and Applications：https://arxiv.org/abs/1812.05905
4. OpenAI Spinning Up SAC 文档：https://spinningup.openai.com/en/latest/algorithms/sac.html
