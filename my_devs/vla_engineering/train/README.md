# VLASH PI0.5 LoRA smoke 验证

本目录用于隔离验证 `my_devs/vla_engineering/vlash-main` 的 PI0.5 LoRA fine-tuning 与高效推理入口。

## 约束

- 只在 `my_devs/vla_engineering/train/` 下写输出、日志、临时缓存。
- 原始数据集只读使用：`datasets/desk_cleanup_v1/eraser_cup_multi_task`。
- 原始基座权重只读使用：`assets/modelscope/lerobot/pi05_base`。
- 默认使用独立 conda 环境：`vlash_train`。
- 训练视频后端使用 `pyav`，因为当前 `lerobot_flex` 中 `torchcodec` 缺 FFmpeg 动态库。

## 一键流程

```bash
cd /data/cqy_workspace/flexible_lerobot
bash my_devs/vla_engineering/train/env_setup.sh
bash my_devs/vla_engineering/train/run_vlash_pi05_lora_smoke.sh
```

## 分步探针

```bash
source /home/cqy/miniconda3/etc/profile.d/conda.sh
conda activate vlash_train
export PYTHONPATH=/data/cqy_workspace/flexible_lerobot/my_devs/vla_engineering/vlash-main:/data/cqy_workspace/flexible_lerobot/src:${PYTHONPATH}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python my_devs/vla_engineering/train/probe_vlash_dataset.py
python -m vlash.train --config_path=my_devs/vla_engineering/train/pi05_so101_vlash_lora_smoke.yaml
python my_devs/vla_engineering/train/smoke_vlash_pi05_checkpoint.py \
  --policy-path my_devs/vla_engineering/train/outputs/pi05_so101_vlash_lora_smoke/checkpoints/last/pretrained_model \
  --device cuda \
  --num-inference-steps 2
```

## 12GiB 显存探针

下面两个命令会完整构建数据集、策略、LoRA、优化器，并跑一个 batch 的 forward/backward，用
`torch.cuda.max_memory_allocated/reserved` 记录训练计算峰值。它们不会保存 checkpoint。

```bash
source /home/cqy/miniconda3/etc/profile.d/conda.sh
conda activate vlash_train
export PYTHONPATH=/data/cqy_workspace/flexible_lerobot/my_devs/vla_engineering/vlash-main:/data/cqy_workspace/flexible_lerobot/src:${PYTHONPATH}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python my_devs/vla_engineering/train/probe_vlash_train_memory.py \
  --config-path my_devs/vla_engineering/train/pi05_so101_vlash_lora_smoke.yaml

python my_devs/vla_engineering/train/probe_vlash_train_memory.py \
  --config-path my_devs/vla_engineering/train/pi0_so101_vlash_lora_memory_probe.yaml
```

本机验证结果：

- PI0.5 LoRA：`max_allocated=10.84GiB`，`max_reserved=11.09GiB`。
- PI0 LoRA：`max_allocated=9.91GiB`，`max_reserved=10.21GiB`。

注意：完整训练脚本在保存 checkpoint 时会额外创建 CPU merge 目标并重新加载推理 checkpoint；
这部分不是训练 step 本身的 12GiB 评估口径。

## 真实机器人推理

推荐使用分离式推理：VLASH 模型服务端使用 `vlash_train`，机器人客户端使用 `lerobot_flex`。

先启动策略服务端：

```bash
cd /data/cqy_workspace/flexible_lerobot
DEFAULT_PROMPT="Put the eraser into the small box" \
PORT=8005 \
POLICY_PATH=/data/cqy_workspace/flexible_lerobot/my_devs/vla_engineering/train/outputs/pi05_so101_vlash_lora_10epochs_r192_bs1_accum8/checkpoints/033275/pretrained_model \
bash my_devs/vla_engineering/train/serve_vlash_policy.sh
```

另开一个终端，先 dry-run，不执行动作：

```bash
cd /data/cqy_workspace/flexible_lerobot
HOST=localhost \
PORT=8005 \
bash my_devs/vla_engineering/train/run_vlash_remote_client.sh \
  --task "Put the eraser into the small box" \
  --robot-id hfy_follower \
  --robot-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00 \
  --calib-dir /home/cqy/.cache/huggingface/lerobot/calibration/robots/so_follower \
  --top-cam /dev/video4 \
  --wrist-cam /dev/video6 \
  --top-cam-fourcc YUYV \
  --wrist-cam-fourcc MJPG \
  --img-width 640 \
  --img-height 480 \
  --fps 30 \
  --run-time-s 20 \
  --max-relative-target 10 \
  --motor-io-retries 10 \
  --execute-actions false
```

确认服务端、相机、串口和日志正常后，再打开动作：

```bash
cd /data/cqy_workspace/flexible_lerobot
HOST=localhost \
PORT=8005 \
EXECUTE_ACTIONS=true \
bash my_devs/vla_engineering/train/run_vlash_remote_client.sh \
  --task "Put the eraser into the small box" \
  --robot-id hfy_follower \
  --robot-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00 \
  --calib-dir /home/cqy/.cache/huggingface/lerobot/calibration/robots/so_follower \
  --top-cam /dev/video4 \
  --wrist-cam /dev/video6 \
  --top-cam-fourcc YUYV \
  --wrist-cam-fourcc MJPG \
  --img-width 640 \
  --img-height 480 \
  --fps 30 \
  --run-time-s 60 \
  --max-relative-target 10 \
  --motor-io-retries 10 \
  --action-chunk-steps 30
```

如果仍想使用 VLASH 自带的一体式 `vlash run`，训练完成后也可以使用：

```bash
source /home/cqy/miniconda3/etc/profile.d/conda.sh
conda activate vlash_train
export PYTHONPATH=/data/cqy_workspace/flexible_lerobot/my_devs/vla_engineering/vlash-main:/data/cqy_workspace/flexible_lerobot/src:${PYTHONPATH}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

vlash run my_devs/vla_engineering/train/pi05_so101_vlash_async_infer.yaml
```

如实际串口或摄像头不同，修改 `pi05_so101_vlash_async_infer.yaml` 中的 `robot.port`、`robot.cameras.top.index_or_path`、`robot.cameras.wrist.index_or_path`。
