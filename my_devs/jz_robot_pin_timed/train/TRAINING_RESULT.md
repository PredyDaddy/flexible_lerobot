# ACT 20 Epoch 训练结果

训练日期：2026-07-11。

> 这是简化 schema 之前的历史 raw18 checkpoint 结果。该 checkpoint 的 state/action head 和
> normalization stats 都是 18D，不能直接续训或冒充新的 model16 checkpoint；原文件保留用于
> 兼容审计。新训练契约见 `../docs/raw18_model16_compatibility.md`。

## 输入与配置

- 数据集：`tests/outputs/jz_robot_pin_timed_real_20260711_190502`
- 数据量：3 episodes、894 frames、30 FPS
- 输入：3 路相机统一 resize 到 224x224，加 18D `observation.state`
- 输出：18D `action`
- ACT：ResNet18 ImageNet 预训练 backbone，`chunk_size=50`，`n_action_steps=25`
- batch size：16
- 训练长度：56 updates/epoch，1120 updates，共 20 epochs
- conda 环境：`lerobot_flex`

## 结果

训练正常完成，主要 epoch loss：

| Epoch | Loss |
|---:|---:|
| 1 | 12.608 |
| 5 | 2.754 |
| 10 | 2.099 |
| 15 | 1.770 |
| 20 | 1.449 |

输出目录：

```text
tests/outputs/act_jz_robot_pin_timed_real_20260711_190502_e20_b16_r224
```

最终模型：

```text
tests/outputs/act_jz_robot_pin_timed_real_20260711_190502_e20_b16_r224/checkpoints/last/pretrained_model
```

训练保存了 5、10、15、20 epoch 四个 checkpoint，总大小约 2.4 GB。自动验收确认：

- final checkpoint 配置为 1120 steps；
- 三路相机和 18D state/action 均保留；
- `ImageCropResizeProcessorStep(224, 224)` 已序列化进 preprocessor；
- final checkpoint 能在 CUDA 上离线加载；
- 数据集单帧经 preprocessor 后三路图像均为 `(1, 3, 224, 224)`；
- `select_action()` 输出 shape `(1, 18)`，所有值均为 finite。

本次没有连接或控制机器人。3 条 episode 只足以验证完整训练技术链路，loss 下降不能证明模型
已经具备泛化能力。进入实机前还需要独立数据上的离线评估和明确授权的 dry-run。
