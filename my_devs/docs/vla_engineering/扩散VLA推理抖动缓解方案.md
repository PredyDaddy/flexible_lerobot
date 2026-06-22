# 扩散 VLA 推理抖动缓解方案

## 1. 问题背景

在真实机器人部署中，常见现象是：

- 自回归模型输出动作时，看起来更连续、更柔顺；
- 扩散/flow matching 类 VLA，例如 PI0、PI0.5、GR00T，推理出来的动作在真实机械臂上更容易出现抖动、顿挫或 chunk 边界跳变；
- 这种抖动通常不是模型“不会输出平滑动作”，而是推理执行链路没有处理好连续控制问题。

在本仓库中，相关模型主要位于：

- `src/lerobot/policies/pi0/`
- `src/lerobot/policies/pi05/`
- `src/lerobot/policies/groot/`
- ACT 作为对照模型位于 `src/lerobot/policies/act/`

其中 ACT 是直接回归 action chunk，PI0、PI0.5、GR00T 是扩散/flow matching 风格，推理时从噪声开始，通过多步去噪或速度场积分得到动作 chunk。

## 2. 抖动的主要来源

### 2.1 Chunk 与 chunk 之间不连续

扩散/flow VLA 每次推理通常输出一个动作序列：

```text
action_chunk = [a0, a1, a2, ..., aH-1]
```

如果真实执行时每个控制周期都重新推理一个新 chunk，并且只执行新 chunk 的第一个动作，就会出现：

```text
第 t 次推理:     a_t[0], a_t[1], a_t[2], ...
第 t+1 次推理:   a_{t+1}[0], a_{t+1}[1], ...
真实执行序列:    a_t[0] -> a_{t+1}[0] -> a_{t+2}[0] ...
```

这些 `a_t[0]` 来自不同采样过程，可能存在模式跳变，所以真实机器人会表现为抖动。

### 2.2 每次重新采样噪声导致模式跳变

扩散/flow 推理一般从随机噪声开始：

```python
x_t = noise
```

如果每次重新规划都使用完全独立的随机噪声，模型在多种可行动作模式之间切换的概率会变大。对离线评估来说这可能只是多模态能力，对真实机器人来说就可能变成高频抖动。

### 2.3 Execution horizon 太短

如果 action chunk 长度是 50，但真实执行时每次只拿 1 步，然后立刻重新推理，机器人闭环频率虽然高，但动作稳定性会变差。

更合理的方式是 receding horizon：

```text
预测 50 步，只执行前 4/8/10 步，然后重新观测和规划。
```

这样可以在柔顺性和闭环响应之间折中。

### 2.4 推理延迟和控制周期不同步

VLA 推理耗时通常明显高于传统控制器。如果机器人控制周期是 30 Hz 或 50 Hz，但模型推理是异步返回的，就会出现：

- 新动作 chunk 到达时，旧 chunk 已经执行了一部分；
- 新 chunk 的开头动作和当前机器人状态不完全匹配；
- 控制层突然切换目标，造成顿挫。

### 2.5 控制层过硬

如果最终 action 是绝对关节位置目标，模型输出的一点点跳变都会直接变成电机目标跳变。尤其是位置伺服机械臂，缺少速度/加速度限制时更容易抖。

### 2.6 数据和归一化问题

如果训练数据本身存在遥操作手抖、时间戳错位、action 尖峰，扩散模型可能会诚实地拟合这种分布。另外，反归一化统计量不匹配也会放大动作跳变。

## 3. 本仓库中三个 VLA 的推理特点

### 3.1 PI0

默认配置：

```python
num_inference_steps = 10
chunk_size = 50
n_action_steps = 50
max_action_dim = 32
```

推理路径：

```text
noise -> denoise_step 循环 -> Euler 积分 -> action chunk
```

后处理：

- 输出 shape 先是 `(B, chunk_size, max_action_dim)`；
- 再裁剪到真实 `action_dim`；
- policy 层通过 action queue 执行；
- processor 层做 `UnnormalizerProcessorStep`，把归一化动作还原到真实动作空间。

### 3.2 PI0.5

默认配置：

```python
num_inference_steps = 10
chunk_size = 50
n_action_steps = 50
max_action_dim = 32
```

PI0.5 和 PI0 类似，也是 flow matching 推理。主要差异是 PI0.5 会把 normalized state 编入 prompt/token 中，状态信息不是像 PI0 那样作为单独连续输入传入 `sample_actions()`。

部署抖动处理上，PI0.5 和 PI0 可以使用同一套策略：

- execution horizon；
- temporal ensemble；
- RTC；
- 控制层限速/限加速度。

### 3.3 GR00T

GR00T 的去噪步数不在 `GrootConfig` 中写死，而是由 checkpoint 的：

```text
action_head_cfg["num_inference_timesteps"]
```

决定。

推理路径：

```text
backbone 编码视觉/语言 -> action head 从随机噪声开始 -> 多步 velocity 积分 -> action_pred
```

GR00T 的 postprocessor 比 PI0/PI0.5 更特殊：

- 如果输出仍是 3D chunk，会取最后一个 timestep；
- 裁剪到 `env_action_dim`；
- 使用 min/max stats 从 `[-1, 1]` 反归一化回真实动作空间；
- 再移动到 CPU。

因此 GR00T 的抖动缓解尤其要注意：不要过早丢掉完整 chunk，否则无法做 chunk-level temporal ensemble。

## 4. 推荐解决方案总览

优先级建议如下：

```text
P0: 确认 execution horizon，不要每次只执行 1 步
P1: 对 overlapping action chunks 做 temporal ensemble
P2: 启用或完善 RTC / chunk inpainting
P3: 最终控制层加速度/加速度/jerk 限制
P4: 根据延迟预算调整 denoise steps
P5: 检查动作空间、归一化统计量、数据平滑和时间对齐
```

其中 P0、P1、P2 是解决 chunk 边界跳变的关键；P3 是机器人安全层兜底；P4、P5 是进一步优化。

## 5. 方案一：调整 Execution Horizon

### 5.1 原理

不要每次推理只执行第一个动作。更稳的做法是：

```text
模型预测 H 步 action chunk；
真实机器人只执行前 K 步；
K 步之后重新观测、重新推理。
```

其中：

```text
1 < K < H
```

K 太小，动作反馈快但容易抖；K 太大，动作更顺但反馈变慢。

### 5.2 推荐初始参数

如果控制频率是 30 Hz：

```text
n_action_steps = 4 或 8
```

如果控制频率是 50 Hz：

```text
n_action_steps = 5 或 10
```

如果控制频率是 100 Hz：

```text
n_action_steps = 8 或 16
```

### 5.3 在本仓库中的位置

PI0：

```text
src/lerobot/policies/pi0/configuration_pi0.py
```

PI0.5：

```text
src/lerobot/policies/pi05/configuration_pi05.py
```

GR00T：

```text
src/lerobot/policies/groot/configuration_groot.py
```

相关字段：

```python
chunk_size: int = 50
n_action_steps: int = 50
```

如果实际运行脚本中覆盖了 `n_action_steps=1`，应优先改为 4、8、10 这类中等 horizon。

## 6. 方案二：Temporal Ensemble

### 6.1 原理

Temporal ensemble 的核心是：多个连续推理得到的 action chunks 会在未来时间上重叠，对同一个真实执行时刻的多个预测做加权平均。

示例：

```text
第 t 次推理预测:
    a_t[0], a_t[1], a_t[2], a_t[3]

第 t+1 次推理预测:
    a_{t+1}[0], a_{t+1}[1], a_{t+1}[2], a_{t+1}[3]

对于真实时刻 t+1，可以融合:
    a_t[1] 和 a_{t+1}[0]
```

这样可以缓解 chunk 边界处的跳变。

### 6.2 加权方式

可以使用指数权重：

```python
w_i = exp(-alpha * age_i)
```

推荐初始值：

```text
alpha = 0.01 到 0.05
```

调参方向：

- `alpha` 越小，越平滑，响应越慢；
- `alpha` 越大，越灵敏，抖动抑制越弱。

### 6.3 应该在哪个空间做 ensemble

推荐在归一化动作空间做 temporal ensemble，然后再反归一化。

原因：

- PI0/PI0.5 的 policy 输出还在 normalized action 空间；
- processor 后处理才做 unnormalize；
- 在归一化空间做平均，不会被不同维度量纲影响。

对 GR00T 要特别注意：

- 不要先让 postprocessor 取最后 timestep；
- 应保留完整 action chunk；
- 在 chunk 级别完成 ensemble 后，再做 slice 和 min-max 反归一化。

### 6.4 和 ACT 的关系

ACT 已经有类似机制：

```text
ACTTemporalEnsembler
```

扩散/flow VLA 可以复用这个思路。也就是说，ACT 的柔顺性有一部分不是因为它是自回归或非扩散，而是因为 action chunking + temporal ensembling 的执行方式天然更稳。

## 7. 方案三：RTC / Chunk Inpainting

### 7.1 原理

RTC，即 Real-Time Chunking，适合异步推理场景。它的核心思想是：

```text
新 chunk 生成时，不是完全从头生成；
而是把已经决定要执行的旧 chunk 前段固定住；
只对后面的未知部分做 inpainting / denoise。
```

这样可以让新旧 chunk 在边界处自然衔接，减少突然切换。

### 7.2 本仓库已有接口

PI0 和 PI0.5 配置中有：

```python
rtc_config: RTCConfig | None = None
```

推理中也有：

```python
if self._rtc_enabled():
    v_t = self.rtc_processor.denoise_step(...)
else:
    v_t = denoise_step_partial_call(x_t)
```

这说明本仓库已经预留了 RTC 链路。对于真实机器人异步推理，优先建议启用 RTC，而不是只在最后动作上做低通滤波。

### 7.3 RTC 适合解决的问题

RTC 主要解决：

- 推理延迟导致的新旧 chunk 不对齐；
- 每次重采样导致的 chunk 边界跳变；
- 短 execution horizon 下的 mode jumping；
- 异步 server/client 推理下的动作续接问题。

## 8. 方案四：控制层限速、限加速度、低通滤波

### 8.1 为什么还需要控制层保护

即使模型输出比较平滑，真实机器人仍需要安全控制层。VLA 输出不应该直接无限制地下发到电机。

控制层建议至少包含：

- action delta limit；
- velocity limit；
- acceleration limit；
- gripper 单独限幅；
- workspace 和 joint limit safety clamp。

### 8.2 最小限速方案

如果 action 是关节位置目标，可以使用：

```python
delta = raw_action - prev_action
delta = torch.clamp(delta, -max_delta_per_step, max_delta_per_step)
safe_action = prev_action + delta
```

### 8.3 加速度限制方案

更稳的方式：

```python
velocity = raw_action - prev_action
velocity = torch.clamp(velocity, prev_velocity - max_acc, prev_velocity + max_acc)
velocity = torch.clamp(velocity, -max_vel, max_vel)
safe_action = prev_action + velocity
```

### 8.4 一阶低通滤波

简单低通：

```python
filtered_action = beta * raw_action + (1.0 - beta) * prev_action
```

推荐初始值：

```text
beta = 0.2 到 0.5
```

调参方向：

- `beta = 0.2`：更稳，但响应慢；
- `beta = 0.5`：更灵敏，但抖动抑制弱一些。

### 8.5 注意事项

低通滤波和限速只能作为兜底，不应该作为唯一方案。因为它会引入滞后，可能影响接触、插入、抓取这类需要快速修正的任务。

更推荐的顺序是：

```text
先解决 chunk 边界连续性；
再加控制层安全约束。
```

## 9. 方案五：调整去噪步数

PI0 和 PI0.5 默认：

```python
num_inference_steps = 10
```

可以尝试：

```text
10 -> 20
```

预期效果：

- 单个 chunk 内的采样质量可能更好；
- 动作可能更稳定；
- 推理延迟会增加。

注意：

```text
如果问题主要来自 chunk 与 chunk 之间跳变，
单纯增加 denoise steps 不一定能解决。
```

这种情况下，execution horizon、temporal ensemble、RTC 更有效。

GR00T 的去噪步数来自 checkpoint 中的：

```text
action_head_cfg["num_inference_timesteps"]
```

需要先读取实际 checkpoint config，再决定是否调整。

## 10. 方案六：固定随机性或 Warm Start

### 10.1 固定随机种子

最简单的方式是 eval 时固定随机种子，让采样更可复现。

优点：

- 实现简单；
- 有助于排查抖动是否来自随机采样。

缺点：

- 可能降低策略多样性；
- 不一定适合长期真实部署。

### 10.2 Warm Start

更好的方式是使用上一轮剩余 chunk 初始化当前动作轨迹的一部分：

```text
上一轮 chunk 剩余动作 -> 当前轮 x_t 的已知部分
当前轮只补全后续未知动作
```

这实际上接近 RTC / inpainting 的思想。

## 11. 方案七：训练数据和动作空间检查

如果部署侧处理后仍然抖，需要回到训练数据检查：

- action 是否有尖峰；
- 遥操作数据是否手抖严重；
- observation 和 action 时间戳是否对齐；
- 控制频率是否稳定；
- action 是 absolute position、delta position、velocity 还是 end-effector delta pose；
- dataset stats 是否和部署机器人一致；
- PI0/PI0.5 的 quantile 或 mean/std 反归一化是否正确；
- GR00T 的 min/max stats 是否正确。

可以考虑在训练中加入平滑正则：

```python
vel_loss = ((actions[:, 1:] - actions[:, :-1]) ** 2).mean()
acc_loss = ((actions[:, 2:] - 2 * actions[:, 1:-1] + actions[:, :-2]) ** 2).mean()
loss = imitation_loss + lambda_vel * vel_loss + lambda_acc * acc_loss
```

但训练侧改动成本更高，应放在部署策略优化之后。

## 12. 推荐落地路线

### 12.1 第一阶段：不改模型，先改推理执行

目标：最快判断抖动是不是执行策略造成的。

建议：

```text
1. 确认实际 n_action_steps，不要只执行 1 步；
2. 试 n_action_steps = 4、8、10；
3. 保持 num_inference_steps = 10；
4. 加最小 action delta limit；
5. 记录机器人动作曲线和 raw action 曲线。
```

验收标准：

```text
chunk 边界处动作跳变明显下降；
机器人没有明显高频抖动；
任务成功率不明显下降。
```

### 12.2 第二阶段：加入 Temporal Ensemble

目标：进一步降低 overlapping chunk 的不连续。

建议：

```text
1. 在 normalized action chunk 上做 ensemble；
2. 初始 alpha = 0.02；
3. 对比 alpha = 0.01、0.02、0.05；
4. ensemble 后再进入 unnormalize。
```

验收标准：

```text
动作曲线一阶差分和二阶差分下降；
机器人末端轨迹更连续；
不会因为过度平滑导致明显滞后。
```

### 12.3 第三阶段：启用 RTC

目标：解决异步推理和 chunk inpainting 问题。

建议：

```text
1. 优先在 PI0/PI0.5 上启用已有 rtc_config；
2. 传入 inference_delay、prev_chunk_left_over、execution_horizon；
3. 对比普通采样、temporal ensemble、RTC 三种模式；
4. 记录推理延迟和控制延迟。
```

验收标准：

```text
新旧 chunk 衔接自然；
异步推理下动作不突然跳变；
延迟可接受。
```

### 12.4 第四阶段：优化去噪步数和数据

目标：在稳定执行的基础上提升单 chunk 质量。

建议：

```text
1. PI0/PI0.5 试 num_inference_steps = 20；
2. GR00T 读取 checkpoint 的 num_inference_timesteps 后再调；
3. 检查数据动作尖峰；
4. 必要时加入训练侧平滑正则。
```

## 13. 建议记录的指标

为了判断方案是否有效，建议每次实验记录：

```text
raw_action
filtered_action
executed_action
joint_position
joint_velocity
policy_latency_ms
control_period_ms
chunk_id
action_index_in_chunk
```

动作平滑指标：

```python
delta = action[1:] - action[:-1]
acc = action[2:] - 2 * action[1:-1] + action[:-2]

mean_abs_delta = delta.abs().mean()
max_abs_delta = delta.abs().max()
mean_abs_acc = acc.abs().mean()
max_abs_acc = acc.abs().max()
```

重点看：

- chunk 边界处 `delta` 是否异常大；
- 反归一化前后动作是否被放大；
- gripper 维度是否单独抖动；
- 某些关节是否比其他关节更容易抖。

## 14. 最终建议

扩散/flow VLA 的抖动问题，优先不要理解成“模型不柔顺”。更准确的判断是：

```text
扩散模型一次生成 chunk；
真实机器人需要连续控制；
如果 chunk 执行、重规划、异步延迟、控制限幅没有处理好，
多模态采样就会在真实机器人上表现为抖动。
```

推荐最终组合方案：

```text
PI0 / PI0.5:
    n_action_steps = 4 到 10
    temporal ensemble
    RTC
    action delta / acceleration limit
    必要时 num_inference_steps = 20

GR00T:
    读取 checkpoint 去噪步数
    保留完整 action chunk 做 ensemble
    避免过早取最后 timestep
    min-max 反归一化后加安全限幅

ACT:
    作为柔顺性对照，重点参考它的 temporal ensemble 机制
```

一句话总结：

```text
先解决 chunk 边界连续性，再做控制层平滑兜底；不要只靠低通滤波掩盖扩散 VLA 的执行链路问题。
```
