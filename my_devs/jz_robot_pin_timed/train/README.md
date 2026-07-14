# JZ Robot Pin Timed ACT 训练工作流

本目录只做离线数据审计、ACT 训练和 checkpoint 验收，不会连接或控制机器人。所有 Python
命令都必须运行在 `lerobot_flex` conda 环境。

## 当前训练输入

默认数据集：

```text
tests/outputs/jz_robot_pin_timed_curated_42eps_20260713
```

训练前提：42 episodes、8370 frames、20 FPS、三路 H.264/CRF18 视频；数据集边界继续保留
raw18，ACT 模型只接收 model16（14 个关节 + 左右 opening）。raw force 索引 15/17 不进入模型。
manifest 必须明确左 observation 为 `measured_opening`、右 observation 为
`commanded_opening`，并且逐帧 gripper generation 严格推进。

## 为什么仍能直接使用 `lerobot-train`

`train_act.sh` 实际执行已安装的 `lerobot-train` console entry script，并强制使用
`lerobot_flex` 的 Python 解释器。`cli_hook/sitecustomize.py` 是显式开启的启动扩展，只替换
LeRobot trainer 已导入的 dataset/processor factory：

- raw dataset 只读投影为 model16，原始 Parquet 不修改；
- normalization 前插入 raw18→model16；
- 三路相机在 processor 中统一 resize 到 224x224；
- unnormalization 后插入 model16→raw18，并按 manifest 写回固定 wire force。

训练循环、optimizer、dataloader、checkpoint、日志和 ACT 实现全部仍由 `lerobot-train` 提供。

## 1. 完整数据审计

```bash
cd /home/luzhuang/cqy/aaa/flexible_lerobot
bash my_devs/jz_robot_pin_timed/train/audit_dataset.sh
```

该命令重跑数据、严格时序、训练投影、LeRobot runtime 读取，并用 ffmpeg 全量解码所有合并后
MP4。报告写到 `tests/outputs/audits/`。只跳过全视频解码时：

```bash
FULL_VIDEO_DECODE=0 bash my_devs/jz_robot_pin_timed/train/audit_dataset.sh
```

## 2. Smoke Test

```bash
cd /home/luzhuang/cqy/aaa/flexible_lerobot
bash my_devs/jz_robot_pin_timed/train/smoke_test.sh
```

默认先做快速审计，再用 batch 2 运行 2 个真实 GPU update，保存 checkpoint，并完成一次
raw18 sample → preprocessor → ACT model16 action → postprocessor raw18 action 的离线推理。输出不会
自动删除，便于审计。Smoke Test 不代表模型具备任务能力。

## 3. 正式训练 20 Epoch

```bash
cd /home/luzhuang/cqy/aaa/flexible_lerobot
bash my_devs/jz_robot_pin_timed/train/train_act_20_epochs.sh
```

默认参数：batch 16、ACT chunk 50、n_action_steps 25、resize 224x224、4 workers、每 5 epochs
保存一次 checkpoint、W&B 关闭。8370 帧对应每 epoch 524 updates，20 epochs 共 10480 updates。

可选覆盖示例：

```bash
BATCH_SIZE=8 NUM_WORKERS=2 USE_AMP=true \
OUTPUT_DIR=/home/luzhuang/cqy/aaa/flexible_lerobot/tests/outputs/my_act_run \
bash my_devs/jz_robot_pin_timed/train/train_act_20_epochs.sh
```

输出目录必须不存在，脚本不会覆盖已有训练。最终 checkpoint：

```text
<output>/checkpoints/last/pretrained_model
```

## 文件说明

- `audit_dataset.sh`：完整训练资格审计。
- `check_training_readiness.py`：LeRobot runtime、schema view 和抽样解码检查。
- `smoke_test.sh`：2-step 真实 GPU Smoke Test。
- `train_act.sh`：通用 `lerobot-train` 启动器。
- `train_act_20_epochs.sh`：锁定当前 42 episodes 数据的正式入口。
- `jz_lerobot_train_hook.py`：只提供 raw18/model16 和 resize factory 扩展，不实现训练循环。
- `verify_act_training.py`：checkpoint 结构、processor 顺序和离线推理验收。
- `TRAINING_RESULT.md`：2026-07-11 历史 raw18 训练结果，仅作兼容记录。
- `VALIDATION_RESULT_20260713.md`：当前 42 episodes 数据审计与 `lerobot-train` Smoke 结果。
- `infer/`：训练后 checkpoint 离线推理、policy dry-run 和受保护的 armed 评估入口。
- `manifests/`：旧数据的显式 legacy manifest；当前 curated 数据优先使用数据集内 manifest。
