# Curated 42 Episodes 训练资格与 Smoke Test 结果

验证日期：2026-07-13。

## 结论

`tests/outputs/jz_robot_pin_timed_curated_42eps_20260713` 达到仓库当前定义的离线训练准入标准，
可以进入 ACT 20 Epoch 正式训练。该结论表示数据结构、数值、时序、视频、显式 gripper 语义、
raw18/model16 边界和训练技术链路通过，不表示 42 episodes 已足以保证模型泛化或实机成功。

## 数据审计

- 42 episodes、8370 frames、20 FPS、单任务。
- raw `observation.state` / `action` 均为 18D；模型投影均为 16D。
- 8370/8370 帧均有合法 `source_timing v1`；state reuse 为 0。
- 三路相机 source FPS 均约 30 FPS，ZMQ sequence gap 为 0。
- H.264 / CRF18；合并后的 9 个 MP4 已全部用 ffmpeg 解码，退出码均为 0。
- gripper generation 左右均逐 session 严格推进；force 索引 15/17 未进入模型。
- curation 状态 `MERGE_PASS`；颜色人工复核状态 `PASS_WITH_REVIEW`。
- 全部 42 个 episode 的 best-lag MAE/P95 均通过 0.01/0.05 rad 门限；多数 best lag 为 6 帧，
  episode 14/27 为 5 帧。

完整重跑报告：

```text
tests/outputs/audits/jz_robot_pin_timed_curated_42eps_20260713_20260713_202636/
```

## 记录但不阻断训练的事项

- camera observation reuse：head 6/8370（0.07%）、left 17/8370（0.20%）、right 22/8370
  （0.26%）。当前质量规范只记录，不设硬失败门限。
- episode 13/32 的 initial action/state delta 分别约 0.465/0.468 rad；episode 20 的最大 action
  step 约 0.146 rad。它们低于历史正式采集使用的 10 rad guard，且 episode lag/motion 检查通过，
  因而保留；如果后续任务需要更严格速度/起始姿态策略，应另设门限重新筛选。
- 右夹爪 observation 是 `commanded_opening`（command echo），不是硬件 measured feedback。
  manifest 已如实记录；模型不能据此学习右夹爪真实跟踪误差。
- 当前 `splits` 只有全部 42 条训练数据，没有独立 validation/test split。loss 只能用于训练过程
  监控，不能当作泛化指标。正式实机前仍建议保留独立 episode 做离线比较，并在获得授权后先
  dry-run，再做受控 armed evaluation。

## `lerobot-train` Smoke Test

实际工具：

```text
/home/luzhuang/miniconda3/envs/lerobot_flex/bin/lerobot-train
```

实际执行 2 个 GPU optimizer updates，batch size 2：

| Step | Loss | Grad norm |
|---:|---:|---:|
| 1 | 87.065 | 1478.184 |
| 2 | 76.291 | 1152.130 |

Smoke checkpoint：

```text
tests/outputs/smoke_act_jz_robot_pin_timed_curated_42eps_20260713_20260713_202603/
```

验收结果：

- checkpoint 保存并可离线加载；
- ACT config 输入/输出均为 model16；
- 三路图像 processor 均 resize 为 224x224；
- raw18→model16 projection 位于 normalization 前；
- model16→raw18 expansion 位于 unnormalization 后；
- 一条真实数据经 checkpoint 离线推理，模型 action shape `(1, 16)`，最终 raw action shape
  `(1, 18)`，所有值 finite。

Smoke Test 只证明训练/保存/加载/边界推理链路正确，两步 loss 无模型质量意义。

## 20 Epoch 配置

- batch size：16
- steps per epoch：`ceil(8370 / 16) = 524`
- total steps：10480
- checkpoint：每 5 epochs，即每 2620 steps
- ACT `chunk_size=50`，`n_action_steps=25`
- 三路图像：224x224
- W&B：关闭
- 默认 AMP：关闭，可显式设置 `USE_AMP=true` 后另做对照

正式命令：

```bash
cd /home/luzhuang/cqy/aaa/flexible_lerobot
bash my_devs/jz_robot_pin_timed/train/train_act_20_epochs.sh
```
