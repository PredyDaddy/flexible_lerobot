# ACT Pro 设计文档可行性评审报告（结合当前代码仓库）

> 目的：评审 `ACT_PRO_DESIGN.md` 与 `ACT_PRO_IMPLEMENTATION_PLAN.md` 中提出的优化点，结合本仓库现有实现给出“哪些已具备/只需改配置/需要改代码/高风险不建议优先做”的结论，并给出可执行的落地路线。
>
> 评审范围：仅基于当前仓库代码结构与实现方式（`src/lerobot/`），不引入外部依赖与未经验证的新组件。

---

## 0. TL;DR（结论摘要）

- **整体方向有用**：视觉骨干升级、训练稳定性、显存/吞吐优化、推理延迟优化这些方向是对的。
- **但“新建一套 act_pro 目录”会重复造轮子**：仓库已经有 `act_dinov2`（ACT + DINOv2 backbone），以及通用的数据增强与 AMP 机制；更建议在现有 `act_dinov2`/`act` 上做增量改造与 ablation。
- **真正缺口主要在 3 类**：
  1) **多帧观测 `n_obs_steps>1` 的端到端支持**（数据管线 + 模型结构 + 融合策略），当前 ACT 明确不支持。  
  2) **RoPE / Flash-Attention 这类 attention 内核级改动**：侵入性强、收益不确定，建议后置。  
  3) **训练过程策略**（如“训练到某步解冻 backbone”）与 **显存优化（checkpointing）**：可做、收益明确，但需要实现训练期逻辑。

---

## 1. 设计文档提案逐项对照（仓库现状）

下表按“是否已实现/是否只需改配置/是否需要改代码/风险”归类。

| 设计点 | 文档意图 | 当前仓库支持情况 | 建议 |
|---|---|---|---|
| DINOv2 视觉骨干（grid 输出） | 用更强视觉特征提升精细操作泛化 | **已实现**：`act_dinov2`（`src/lerobot/policies/act_dinov2/*`）支持 `dinov2_image_size`、`output_mode=grid`、`freeze_backbone` 等 | 直接复用 `act_dinov2`，不要另起炉灶 |
| “Decoder 层数 bug 修复”（1→4） | 增强动作序列建模能力 | **本仓库按 `n_decoder_layers` 构建多层**，默认设为 1 是为了复现原 repo 的行为（`src/lerobot/policies/act/configuration_act.py:118`） | **直接改配置**：`--policy.n_decoder_layers=4`（对 `act`/`act_dinov2` 都适用） |
| Pre-Norm | 训练更稳定、可加深网络 | **已支持**：`pre_norm`（`src/lerobot/policies/act/configuration_act.py:112`），encoder/decoder layer 已实现 pre-norm 分支 | 直接改配置做 ablation |
| 多步观测 `n_obs_steps=2` + 时序融合 | 利用短期时序提升精细控制 | **当前不支持**：ACT 配置里直接报错（`src/lerobot/policies/act/configuration_act.py:158`），且 `observation_delta_indices` 为 `None`（`src/lerobot/policies/act/configuration_act.py:176`）导致数据集不会采样历史帧（`src/lerobot/datasets/factory.py:62`） | 这是“真正的大工程”，建议单独分支实施，先明确融合策略与时延预算 |
| RoPE | 更好的相对位置建模 | **未实现**：当前使用正弦/learned pos embed（`src/lerobot/policies/act/modeling_act.py`）且注意力是 `nn.MultiheadAttention` | 高侵入性，建议后置；先用更低成本手段（decoder 层数 + pre-norm + 数据增强 + 解冻策略） |
| AMP（混合精度） | 降显存提速度 | **已有通用开关**：`use_amp` 在 `PreTrainedConfig`（`src/lerobot/configs/policies.py:59`）；train 用 accelerate autocast（`src/lerobot/scripts/lerobot_train.py:89`）；eval 用 autocast（`src/lerobot/scripts/lerobot_eval.py:542`） | 直接开：`--policy.use_amp=true`（cuda 设备） |
| 梯度检查点 | 进一步降显存换算力 | **ACT/ACTDinov2 未实现**（仓库其他策略如 `pi0/pi05` 有相关实现经验） | 建议做（收益明确），但要设计好 checkpoint 位置与训练速度折衷 |
| 学习率策略（分层 LR、cosine+warmup） | 更快收敛 | 本仓库 scheduler 体系存在（`src/lerobot/optim/schedulers.py`），但 ACT 默认 `get_scheduler_preset()` 返回 `None`（`src/lerobot/policies/act/configuration_act.py:169`） | 建议为 ACT 增加 scheduler preset（或训练时显式指定 scheduler config） |
| 数据增强（颜色/裁剪/动作噪声） | 提升泛化、抗 domain shift | **图像增强已内置**：dataset 层 transforms（`src/lerobot/configs/default.py:23`、`src/lerobot/datasets/transforms.py:136`） | 直接用 dataset.image_transforms；动作噪声可考虑在训练 batch 或 loss 前注入（需实现） |
| Flash Attention | 推理提速 | 当前 attention 使用 `nn.MultiheadAttention`，且在 `act_dinov2` 下 DINOv2 backbone 往往是主要耗时 | 先做 profiling 再决定；优先优化 Python token 拼接与 `torch.compile` 友好度 |
| FP16 推理 + torch.compile | 部署提速 | eval 端已支持 autocast；`torch.compile` 在 policy factory 里有注释但未启用（`src/lerobot/policies/factory.py:428`） | 建议先在单机推理脚本内做 opt-in（避免影响全局） |

---

## 2. 立即可落地（不写新代码：只改训练配置）

### 2.1 建议的“低风险高收益”配置组合（先做 ablation）

以 `act_dinov2` 为基底（已支持 grid 特征与冻结骨干）：

- `--policy.type=act_dinov2`
- `--policy.pre_norm=true`
- `--policy.n_decoder_layers=4`（注意：`act` 默认是 1 只是为了复现原实现行为）
- `--policy.use_amp=true`（cuda）
- `--dataset.image_transforms.enable=true`（并适当调 `max_num_transforms`/各项 weight）
- backbone 冻结与分层 LR：`--policy.freeze_backbone=true` + 合理设置 `optimizer_lr_backbone`（`ACTDinov2Config` 已支持）

> 建议做的最小 ablation 顺序：  
> (1) baseline `act_dinov2`（n_decoder=1、pre_norm=false）  
> (2) + `pre_norm=true`  
> (3) + `n_decoder_layers=4`  
> (4) + `dataset.image_transforms.enable=true`  
> (5) + 训练后半程解冻（见第 3 节）

### 2.2 数据增强落地点（仓库已提供）

- dataset 级 transforms 配置入口：`DatasetConfig.image_transforms`（`src/lerobot/configs/default.py:23`）
- transforms 实现：`src/lerobot/datasets/transforms.py`

优点：无需侵入 policy；缺点：只覆盖图像增强，不包含动作噪声（动作噪声需要你明确“加在 target 还是 prediction 上、是否影响归一化”）。

---

## 3. 建议优先做的“中等改动”（收益明确，工程可控）

### 3.1 “冻结→解冻最后 N 层”训练策略（当前缺失但很值得）

现状：
- `ACTDinov2Config` 有 `freeze_backbone` 与单独 `optimizer_lr_backbone`（`src/lerobot/policies/act_dinov2/configuration_act_dinov2.py:34`）。
- 但**没有**“训练到某 step 自动解冻”的机制。

可行落地方式（推荐）：
- 训练循环每 step 会调用 `policy.update()`（如果存在）（`src/lerobot/scripts/lerobot_train.py:115`）。  
  这给了我们一个“把训练期 schedule 写进 policy 内部”的 hook 点，而无需侵入 train loop 太多。

建议设计：
- 在 `ACTDinov2Policy` 增加可选配置：`unfreeze_at_step`、`unfreeze_last_n_layers`、`backbone_eval_when_frozen`。
- `update()` 内部维护 step 计数，当到达阈值后：
  - 将 backbone 里指定层 `requires_grad_(True)`  
  - （可选）切换 backbone 到 train mode（或只打开指定层）
  - 确保 optimizer param group 已经包含这些参数（必要时需要重新构建 optimizer 或提前把 param group 放进去并控制 requires_grad）

为什么推荐先做它：相比 RoPE、多帧观测，这个改动更小，但在“数据不够、domain shift 强”的精细操作任务上通常更实用。

### 3.2 LR scheduler（cosine + warmup）对 ACT 的补齐

现状：
- 仓库有 scheduler 配置体系（`src/lerobot/optim/schedulers.py`）。
- ACT config 的 `get_scheduler_preset()` 目前返回 `None`（`src/lerobot/policies/act/configuration_act.py:169`），等于默认没有 warmup/cosine。

建议：
- 给 ACT 增加一个轻量默认 scheduler preset（例如 `DiffuserSchedulerConfig(name="cosine", num_warmup_steps=...)`），或者你在训练命令里显式提供 scheduler config。

---

## 4. 高成本/高风险项（建议后置或先小规模验证）

### 4.1 多帧观测 `n_obs_steps>1`（需要同时改“数据管线+模型+融合”）

当前阻塞点（必须一起解决）：

1) **配置层硬限制**：ACT 直接禁止 `n_obs_steps!=1`（`src/lerobot/policies/act/configuration_act.py:158`）。  
2) **数据层不会给历史帧**：delta timestamps 是由 `PreTrainedConfig.observation_delta_indices` 决定的（`src/lerobot/datasets/factory.py:62`），而 ACT 返回 `None`（`src/lerobot/policies/act/configuration_act.py:176`）。  
3) **模型 forward 输入形状假设**：ACT/ACTDinov2 当前假设每个相机只给单帧（`OBS_IMAGES` 是 list[Tensor] 且每个 Tensor 形状类似 `(B,C,H,W)`）。

更现实的实现路径（建议你在实现前先确定）：
- 你想要的“2 帧观测”到底是：
  - A) 只给每个 camera 两张帧，做简单 concat（token 维度变 2 倍，时延变大）  
  - B) 先对每帧独立提特征，然后用轻量 temporal attention / gating 做融合（更合理）  
  - C) 直接把 “时间”当成额外维度做 3D attention（代价最大）

如果目标是“桌面级精细操作 + 实时控制”，建议优先做 B，并把融合模块限制为：
- 只在 image token 上做时序融合（不要让 state/env token 参与复杂时序注意力）
- 控制 token 增量（例如只融合最后一层 patch tokens，或在 spatial 上先做降采样）

### 4.2 RoPE / Flash Attention（侵入性强，收益不确定）

原因：
- 当前 attention 使用 `nn.MultiheadAttention`，要 RoPE/Flash-Attn 往往意味着：
  - 换 attention 实现（引入/依赖新内核 or 自己写 attention）
  - 处理 token 结构（latent/state/image patch 混合序列）时的 RoPE 维度定义
- 对 `act_dinov2`，端到端速度瓶颈常常在 backbone 而非 decoder 的 attention。

建议策略：
- 先通过 profiling 明确瓶颈，再决定是否值得做 flash-attn。
- RoPE 建议等你已经证明“decoder 层数 + pre-norm + 解冻策略 + 数据增强”仍不够时再做。

---

## 5. 额外可优化点（文档未提，但结合代码更关键）

### 5.1 Python 层 token 拼接影响吞吐与 `torch.compile`

现状：
- ACT 在 forward 中把每个 patch token `(h*w, b, c)` 转成 Python list，然后 `extend`（`src/lerobot/policies/act/modeling_act.py:457`）。
- ACTDinov2 也同样 `extend(list(cam_features))`（`src/lerobot/policies/act_dinov2/modeling_act_dinov2.py:320`）。

问题：
- 这会引入大量 Python 开销与动态图行为，对吞吐与 `torch.compile` 友好度都不好。

建议（高优先级的性能改动）：
- 改为 tensor 级 concat：把 1D token（latent/state/env）与 image tokens 在 tensor 维度上直接 `torch.cat`，避免 list-of-tensor。
- 同时把 pos_embed 的构造也做成 tensor concat（避免 list extend）。

### 5.2 decoder_in 每次 forward 都 new tensor

现状：
- `decoder_in = torch.zeros((chunk_size, batch, dim))` 每步创建（`src/lerobot/policies/act/modeling_act.py:491`，`src/lerobot/policies/act_dinov2/modeling_act_dinov2.py:336`）。

建议：
- 在不破坏 batch size 动态的前提下，考虑缓存一个最大 batch 的 buffer 或用 `new_zeros`（小收益，但属于“顺手优化”）。

---

## 6. 推荐落地路线图（按性价比排序）

### Stage A（1–3 天）：只做配置 + ablation，快速找增益
1. 以 `act_dinov2` 为 baseline 训练跑通（确保可复现）。
2. 打开 `pre_norm=true`、`n_decoder_layers=4`。
3. 打开 `dataset.image_transforms.enable=true`（先用默认 transforms；必要时调权重）。
4. 记录成功率/动作误差/推理时延与显存占用，决定下一步是否需要“工程改动”。

### Stage B（3–7 天）：中等工程改动（训练策略 + 性能）
1. 实现“冻结→解冻最后 N 层”训练 schedule（优先）。
2. 给 ACT 增加 scheduler preset（cosine+warmup）或在训练命令显式指定。
3. 重构 token 拼接（去 Python list-of-tensor），为吞吐/compile 打基础。

### Stage C（>1 周）：高成本架构实验
1. `n_obs_steps=2` 的端到端支持 + 轻量 temporal fusion（先小规模数据集验证）。
2. 再考虑 RoPE/flash-attn（必须先 profiling 证明瓶颈在 attention）。

---

## 7. 关键代码索引（便于你实现/排查）

- ACT 默认 decoder 层数与 `n_obs_steps` 限制：`src/lerobot/policies/act/configuration_act.py:118`、`src/lerobot/policies/act/configuration_act.py:158`
- ACT decoder 按 `n_decoder_layers` 构建：`src/lerobot/policies/act/modeling_act.py:570`
- ACTDinov2 配置入口（freeze_backbone、image_size 等）：`src/lerobot/policies/act_dinov2/configuration_act_dinov2.py:25`
- 训练 loop 每步调用 `policy.update()` 的 hook：`src/lerobot/scripts/lerobot_train.py:115`
- AMP：train autocast（`src/lerobot/scripts/lerobot_train.py:89`）、eval autocast（`src/lerobot/scripts/lerobot_eval.py:542`）
- dataset 的 delta timestamps 生成依赖 `observation_delta_indices`：`src/lerobot/datasets/factory.py:40`
- dataset 图像增强配置与实现：`src/lerobot/configs/default.py:23`、`src/lerobot/datasets/transforms.py:136`

---

## 8. 下一步需要你补充的信息（我才能给更“落地”的最终方案）

为避免“为了优化而优化”，建议你提供以下 6 个信息（任意形式都行）：
1) 任务类型与成功率瓶颈（例如：插拔/对齐/旋钮/卡扣等）  
2) 控制频率（fps）与实际推理时延预算（例如 <20ms / <50ms）  
3) 相机数量、分辨率、是否有 `observation.environment_state`  
4) action 维度、`chunk_size`、`n_action_steps` 的设置  
5) 数据规模（episode 数/总时长）与 domain shift 情况  
6) 你是否更关注“成功率/泛化”还是“实时性/部署”优先

拿到这些信息后，可以把 Stage A/B 的 ablation 组合细化为一份“明确的训练命令 + 配置模板 + 指标表格”。

