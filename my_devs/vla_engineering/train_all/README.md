# train_all

`train_all` is the isolated mainline for the complete VLASH delay-aware training and deployment loop.

The first phase keeps data, model, task, and environment aligned with the historical `my_devs/vla_engineering/train` baseline, while moving all new outputs and scripts under:

```text
my_devs/vla_engineering/train_all
```

## Baseline

- Dataset: `/data/cqy_workspace/flexible_lerobot/datasets/desk_cleanup_v1/eraser_cup_multi_task`
- Dataset repo id: `desk_cleanup_v1/eraser_cup_multi_task`
- Base policy: `/data/cqy_workspace/flexible_lerobot/assets/modelscope/lerobot/pi05_base`
- Train environment: `vlash_train`
- Robot client environment: `lerobot_flex`
- Main task: `Put the eraser into the small box`

## Smoke Train

```bash
cd /data/cqy_workspace/flexible_lerobot
bash my_devs/vla_engineering/train_all/scripts/run_smoke_train.sh
```

The smoke run uses:

```text
configs/train/pi05_so101_delay_lora_smoke.yaml
steps=2
lora.r=4
max_delay_steps=1
```

## Full Train

After the smoke run passes, start the full r192 baseline manually:

```bash
cd /data/cqy_workspace/flexible_lerobot
FORCE_RESTART=true \
bash my_devs/vla_engineering/train_all/scripts/train_delay_lora.sh
```

To change only the delay setting without editing the baseline config:

```bash
cd /data/cqy_workspace/flexible_lerobot
FORCE_RESTART=true \
bash my_devs/vla_engineering/train_all/scripts/train_delay_lora.sh \
  --max_delay_steps=8
```

Outputs stay under:

```text
my_devs/vla_engineering/train_all/outputs/
my_devs/vla_engineering/train_all/logs/
my_devs/vla_engineering/train_all/reports/
```

