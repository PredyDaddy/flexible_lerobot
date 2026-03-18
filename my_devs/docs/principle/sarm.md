# SARM 策略原理与仓库实现解读

## 策略背景

SARM（Stage-Aware Reward Modeling）来自论文《SARM: Stage-Aware Reward Modeling for Long Horizon Robot Manipulation》，目标是给长时程机器人操作学习一个稳定、可解释的任务进度 reward。奖励建模的出发点很直接：手工奖励难写，纯行为克隆又会把低质量演示一并学进去。偏好式奖励学习和视觉语言奖励都在尝试“从数据里学 reward”；SARM 的区别是显式建模“阶段”和“阶段内进度”，更适合衣物折叠、装配这类可分阶段但速度差异很大的任务。LeRobot 里的部分工程细节在公开材料里描述较少，因此下文部分内容以仓库实现为主。

## 核心原理

SARM 把每一帧的监督写成 `y=k+τ`：`k` 是当前阶段，`τ∈[0,1]` 是该阶段内部进度。标签由子任务名及其起止帧生成，而不是直接按帧序号回归，所以不同演示即使快慢不同，只要语义状态相近，进度也更容易对齐。推理时再结合阶段时间比例先验把 `k+τ` 映射到 `[0,1]`，得到更接近“真实完成度”的 reward。相比只给整段视频打一个分数的 reward model，SARM 更强调分段结构与时序一致性。

## 在本仓库中的实现解读

`configuration_sarm.py` 定义了三种模式：`single_stage` 把整段任务视为一个阶段，`dense_only` 用细粒度标注并自动补一个稀疏 `"task"` 头，`dual` 同时训练 sparse/dense 两套头。默认序列是 9 帧双向观测窗加最多 4 帧 rewind，视觉和文本都走 512 维 CLIP 特征，状态补齐到 `max_state_dim=32`。  

`processor_sarm.py` 负责把原始图像、状态、任务文本转成 `video_features`、`state_features`、`text_features`，并生成 `sparse_targets` / `dense_targets`。训练时它会做 rewind augmentation 和 language perturbation，分别增强时间鲁棒性与语言鲁棒性。  

`modeling_sarm.py` 把模型拆成 `StageTransformer` 和 `SubtaskTransformer`：前者判阶段，后者在阶段先验条件下回归 `τ`。最终 reward 先算 `stage_idx + tau_pred`，再通过 `normalize_stage_tau()` 按 `temporal_proportions` 归一化到 `[0,1]`。当前实现 `num_cameras=1` 是写死的，因此以单相机路径为主。

## 训练与推理机制

训练时，sparse 头总会参与，dense 头按模式可选。每个头都是“先分类阶段，再回归阶段内进度”，并用 75%/25% 的 teacher forcing 混合 ground-truth stage 与预测 stage，减轻训练推理不一致。损失函数是阶段交叉熵加 `τ` 的 MSE；双头模式下两套损失直接相加。需要注意的是，配置里虽然有 `stage_loss_weight`，但当前 `forward()` 并没有真正使用它。  

推理时会关闭增强，`calculate_rewards()` 可返回 reward、阶段概率和置信度；`compute_rabc_weights.py` 会把 SARM 跑在整套数据上，生成 parquet 供 RA-BC 过滤或重加权使用。仓库里对“最终取观测窗哪一帧作为 reward”存在不同默认值，因此实际行为要以具体调用点为准。

## 关键超参数及影响

- `annotation_mode`：决定是否需要子任务标注，以及启用单头还是双头。
- `n_obs_steps`、`frame_gap`、`max_rewind_steps`：决定时序窗口长度与历史回看强度。
- `num_sparse_stages`、`num_dense_stages`、`*_temporal_proportions`：决定阶段粒度和 reward 分段方式。
- `hidden_dim`、`num_layers`、`num_heads`、`dropout`：决定表达能力、速度和显存。
- `rewind_probability`、`language_perturbation_probability`：影响时间鲁棒性与语言鲁棒性。

## 适用场景、优势与局限

SARM 适合“可以自然拆成若干阶段、演示时长差异明显、需要离线估计任务进度”的长时程操作任务。它的优势是 reward 更可解释，比纯帧序号或单一匹配分数更容易稳定对齐完成度，也便于拿去做 RA-BC 样本重加权。局限同样明显：依赖子任务标注质量，阶段错误会传给 `τ` 回归，当前实现默认单相机且固定使用 CLIP，而且它本身只是 reward model，并不直接输出动作。

## 参考资料

1. SARM 论文: https://arxiv.org/abs/2509.25358
2. SARM 项目页: https://qianzhong-chen.github.io/sarm.github.io/
3. LeRobot 官方 SARM 文档: https://huggingface.co/docs/lerobot/en/sarm
4. Deep Reinforcement Learning from Human Preferences: https://arxiv.org/abs/1706.03741
