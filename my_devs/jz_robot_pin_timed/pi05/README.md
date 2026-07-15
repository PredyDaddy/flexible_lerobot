# JZ Robot Pin Timed - PI0.5

本目录是 `jz_robot_pin_timed` 的 PI0.5 独立训练模块。所有新增训练脚本、缓存、日志和输出均限制在本目录；脚本只读取原始数据、已有 PI0.5 基模和 tokenizer，不会连接或操控机器人。

## 固定输入

- 数据集：`data/jz_robot_pin_timed_curated_42eps_20260713`
- 数据规模：42 episodes、8370 frames、20 FPS、3 cameras
- 原始边界：18D state/action
- 模型边界：16D state/action（丢弃两路 force，并按 schema 统一夹爪 opening 方向）
- Conda 环境：`lerobot_flex`
- PI0.5 基模：`/data/cqy_workspace/flexible_lerobot/assets/modelscope/lerobot/pi05_base`
- PaliGemma tokenizer：`/data/cqy_workspace/flexible_lerobot/assets/modelscope/google/paligemma-3b-pt-224`

## Smoke train

```bash
cd /data/cqy_workspace/jz_robot/flexible_lerobot
bash my_devs/jz_robot_pin_timed/pi05/smoke_train.sh
```

默认用 batch size 1 跑 2 个真实的 forward/backward/update step，不保存 14GB checkpoint。日志写入本目录的 `logs/`，训练运行目录写入 `outputs/`。

## 正式训练 15 epochs

```bash
cd /data/cqy_workspace/jz_robot/flexible_lerobot
bash my_devs/jz_robot_pin_timed/pi05/train_15_epochs.sh
```

默认 batch size 8：`ceil(8370 / 8) = 1047` steps/epoch，总计 `15705` steps；每 5 epochs 保存一次 checkpoint。可在命令前覆盖参数，例如：

```bash
BATCH_SIZE=4 NUM_WORKERS=2 bash my_devs/jz_robot_pin_timed/pi05/train_15_epochs.sh
```

只检查 15 epoch 参数换算、资产、数据解码和最终命令，而不启动大模型训练：

```bash
DRY_RUN=true bash my_devs/jz_robot_pin_timed/pi05/train_15_epochs.sh
```

## 关键处理

1. 原数据集保持 raw18 不变，训练时通过只读 view 投影为 model16。
2. 合并数据集自带的 quantile 是 episode quantile 的聚合值，不等于全量帧 quantile。本模块每次启动前从 parquet 重算全量 model16 统计到 `runtime/model16_stats.json`，不回写数据集。
3. 默认使用 PI0.5 推荐的 `QUANTILES` normalization；如需复刻旧 SO101 脚本，可设置 `NORMALIZATION_MODE=MEAN_STD`。
4. tokenizer 通过本目录 `runtime/google/` 下的软链接离线路由，不在仓库其他位置创建文件。
5. checkpoint 的 pre/post processor 会保存 raw18→model16 和 model16→raw18 边界，后续部署仍需先离线验证，未经许可不得上真机。
6. 当前基模省略了与 `lm_head` tied 的 `embed_tokens.weight` 独立副本，加载器会打印一条 missing-key 警告；两者在 PaliGemma 配置中共享权重，smoke forward/backward 已验证可用。
